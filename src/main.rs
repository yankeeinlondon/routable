use std::{env, error::Error, sync::Arc, time::Duration};

use async_trait::async_trait;
use pingora::{
    prelude::{Opt, Server, background_service},
    server::ShutdownWatch,
    services::{background::BackgroundService, listening::Service},
};
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, warn};
use vllm_router::{
    PoolManager,
    api::{ControlPlane, TokenAuth},
    config::{ConfigStore, RouterConfig},
    metrics,
    pool::{ManagedVllmRunner, Pool, PoolError, managed_pool_from_config},
    proxy::DataPlane,
    state::Lifecycle,
};

struct Arguments {
    pingora: Opt,
    router_config: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let arguments = parse_arguments()?;
    // Validate application configuration before Pingora handles `-t` and
    // exits, so config-check mode covers both files.
    let router_config = RouterConfig::load(&arguments.router_config)?;
    let mut server = Server::new(Some(arguments.pingora))?;
    ensure_foreground_mode(server.configuration.daemon)?;
    server.bootstrap();

    let config = Arc::new(ConfigStore::new(router_config.clone())?);
    let auth = TokenAuth::load(&router_config.listeners.management_bearer_token_file)?;
    // zbus owns I/O tasks on the runtime where its connection is constructed.
    // Keep this dedicated runtime alive for the process, while allowing Pingora
    // to construct and block on its own runtimes in `run_forever`.
    let pool_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("vllm-router-pool")
        .build()?;
    let pool = Arc::new(pool_runtime.block_on(managed_pool_from_config(&router_config))?);
    let facade: Arc<dyn PoolManager> = pool.clone();

    let mut data_plane = Service::new(
        "vLLM router data plane".into(),
        DataPlane::new(Arc::clone(&config), Arc::clone(&facade))?,
    );
    data_plane.add_tcp(&router_config.listeners.data_plane.to_string());

    let mut management = Service::new(
        "vLLM router management".into(),
        ControlPlane::new(Arc::clone(&config), facade, auth, 1_024),
    );
    management.add_tcp(&router_config.listeners.management.to_string());

    let mut prometheus = Service::prometheus_http_service();
    prometheus.add_tcp(&router_config.listeners.metrics.to_string());

    info!(
        data_plane = %router_config.listeners.data_plane,
        management = %router_config.listeners.management,
        metrics = %router_config.listeners.metrics,
        "listeners configured"
    );

    server.add_service(data_plane);
    server.add_service(management);
    server.add_service(prometheus);
    server.add_service(background_service(
        "pool observation and idle policy",
        PoolBackground {
            pool,
            interval: observation_interval(&router_config),
        },
    ));
    server.run_forever();
}

struct PoolBackground {
    pool: Arc<Pool<ManagedVllmRunner>>,
    interval: Duration,
}

#[async_trait]
impl BackgroundService for PoolBackground {
    async fn start(&self, mut shutdown: ShutdownWatch) {
        let mut ticker = tokio::time::interval(self.interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
                _ = ticker.tick() => {
                    for result in self.pool.reconcile_all().await {
                        if let Err(error) = result {
                            if should_warn_reconciliation_error(&error) {
                                warn!(error = %error, "service reconciliation failed");
                            } else {
                                debug!(
                                    error = %error,
                                    "service reconciliation deferred during lifecycle transition"
                                );
                            }
                        }
                    }
                    self.pool.reap_idle_once().await;
                    match self.pool.status(Default::default()).await {
                        Ok(status) => publish_state_metrics(&status.services),
                        Err(error) => error!(error = %error, "pool status collection failed"),
                    }
                }
            }
        }
    }
}

fn should_warn_reconciliation_error(error: &PoolError) -> bool {
    !matches!(error, PoolError::TransitionInProgress { .. })
}

fn publish_state_metrics(services: &[vllm_router::state::ServiceStatus]) {
    const STATES: &[(&str, Lifecycle)] = &[
        ("reconciling", Lifecycle::Reconciling),
        ("stopped", Lifecycle::Stopped),
        ("starting", Lifecycle::Starting),
        ("running", Lifecycle::Running),
        ("draining", Lifecycle::Draining),
        ("drained", Lifecycle::Drained),
        ("asleep_l1", Lifecycle::AsleepL1),
        ("asleep_l2", Lifecycle::AsleepL2),
        ("waking", Lifecycle::Waking),
        ("stopping", Lifecycle::Stopping),
        ("failed", Lifecycle::Failed),
        ("backoff", Lifecycle::Backoff),
        ("stale", Lifecycle::Stale),
        ("unknown", Lifecycle::Unknown),
    ];
    for service in services {
        for (name, lifecycle) in STATES {
            metrics::global()
                .service_state
                .with_label_values(&[service.service.as_str(), *name])
                .set(i64::from(service.lifecycle == *lifecycle));
        }
        metrics::global()
            .active
            .with_label_values(&[&service.service])
            .set(i64::try_from(service.active_requests).unwrap_or(i64::MAX));
    }
}

fn observation_interval(config: &RouterConfig) -> Duration {
    config
        .services
        .iter()
        .map(|service| Duration::from_millis(service.readiness.interval_ms))
        .min()
        .unwrap_or(Duration::from_secs(2))
}

fn parse_arguments() -> Result<Arguments, String> {
    parse_arguments_from(
        env::args().skip(1),
        env::var("VLLM_ROUTER_CONFIG").unwrap_or_else(|_| "/etc/vllm-router/router.yaml".into()),
    )
}

fn parse_arguments_from(
    arguments: impl IntoIterator<Item = String>,
    mut router_config: String,
) -> Result<Arguments, String> {
    let mut pingora = Opt::default();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-c" | "--conf" => {
                pingora.conf = Some(
                    arguments
                        .next()
                        .ok_or_else(|| format!("{argument} requires a path"))?,
                );
            }
            "--router-config" => {
                router_config = arguments
                    .next()
                    .ok_or_else(|| "--router-config requires a path".to_string())?;
            }
            "-d" | "--daemon" => return Err(daemon_mode_error()),
            "-t" | "--test" => pingora.test = true,
            "-u" | "--upgrade" => pingora.upgrade = true,
            "--nocapture" => pingora.nocapture = true,
            "-h" | "--help" => {
                return Err(
                    "usage: vllm-router [-tu] -c <pingora.yaml> --router-config <router.yaml>"
                        .into(),
                );
            }
            unknown => return Err(format!("unknown argument `{unknown}`")),
        }
    }
    if pingora.conf.is_none() {
        return Err("a Pingora configuration is required with -c/--conf".into());
    }
    Ok(Arguments {
        pingora,
        router_config,
    })
}

fn ensure_foreground_mode(daemon: bool) -> Result<(), String> {
    if daemon {
        return Err(daemon_mode_error());
    }
    Ok(())
}

fn daemon_mode_error() -> String {
    "daemon mode is unsupported: run vllm-router in the foreground under a process supervisor \
     (the supplied systemd unit uses Type=simple)"
        .into()
}

#[cfg(test)]
mod tests {
    use pingora::server::configuration::ServerConf;
    use vllm_router::pool::PoolError;

    use super::{ensure_foreground_mode, parse_arguments_from, should_warn_reconciliation_error};

    #[test]
    fn rejects_daemon_command_line_modes() {
        for daemon_flag in ["-d", "--daemon"] {
            let error = parse_arguments_from(
                vec![
                    daemon_flag.to_string(),
                    "-c".to_string(),
                    "pingora.yaml".to_string(),
                ],
                "router.yaml".to_string(),
            )
            .err()
            .expect("daemon mode must be rejected");

            assert!(error.contains("daemon mode is unsupported"));
        }
    }

    #[test]
    fn rejects_daemon_mode_from_pingora_configuration() {
        let configuration = ServerConf::from_yaml("---\nversion: 1\ndaemon: true\n")
            .expect("test Pingora configuration should parse");

        let error =
            ensure_foreground_mode(configuration.daemon).expect_err("daemon mode must be rejected");
        assert!(error.contains("Type=simple"));
    }

    #[test]
    fn accepts_foreground_mode() {
        ensure_foreground_mode(false).expect("foreground mode must remain supported");
    }

    #[test]
    fn expected_reconciliation_overlap_is_not_warned_but_real_failures_are() {
        assert!(!should_warn_reconciliation_error(
            &PoolError::TransitionInProgress {
                service: "model-a".into(),
            }
        ));
        assert!(should_warn_reconciliation_error(&PoolError::Runner {
            service: "model-a".into(),
            message: "systemd observation failed".into(),
        }));
    }
}
