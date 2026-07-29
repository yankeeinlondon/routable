use std::{env, error::Error, path::Path, sync::Arc, time::Duration};

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
    gpu::GpuInventory,
    metrics,
    pool::{ManagedVllmRunner, Pool, PoolError, managed_pool_from_config},
    proxy::DataPlane,
    state::Lifecycle,
    sysd::dry_run_generated_inputs,
};

struct Arguments {
    pingora: Opt,
    router_config: String,
    host_gpu_check: HostGpuCheck,
}

/// Whether configuration is being checked on the host it will run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HostGpuCheck {
    Required,
    Skipped,
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let arguments = parse_arguments()?;
    // Everything that decides whether this process can serve has to be decided
    // above `bootstrap()`: Pingora's `-t` exits inside it, so a check placed
    // later cannot fail a configuration that will not start. Only checks owning
    // no runtime state belong up here — a token is bytes and NVML is a
    // synchronous library handle dropped again immediately, while the D-Bus
    // connection and the pool stay below because zbus binds its I/O tasks to
    // the runtime that created the connection.
    let router_config = RouterConfig::load(&arguments.router_config)?;
    let (management_auth, inference_auth) =
        preflight_host_checks(&router_config, arguments.host_gpu_check)?;
    let mut server = Server::new(Some(arguments.pingora))?;
    ensure_foreground_mode(server.configuration.daemon)?;
    server.bootstrap();

    let config = Arc::new(ConfigStore::new(router_config.clone())?);
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
        DataPlane::new(Arc::clone(&config), Arc::clone(&facade), inference_auth)?,
    );
    data_plane.add_tcp(&router_config.listeners.data_plane.to_string());

    let mut management = Service::new(
        "vLLM router management".into(),
        ControlPlane::new(Arc::clone(&config), facade, management_auth, 1_024),
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
                    // A managed observation can be inside D-Bus, NVML, or a
                    // readiness probe when shutdown begins. Keep the entire
                    // tick cancellable so Pingora never has to wait for that
                    // external I/O before its background runtime can exit.
                    tokio::select! {
                        biased;
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() {
                                break;
                            }
                        }
                        _ = self.tick() => {}
                    }
                }
            }
        }
    }
}

impl PoolBackground {
    async fn tick(&self) {
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

/// The loop period must stay the value `validate_probe_freshness` budgeted against, so
/// it is derived from the configuration rather than recomputed here.
fn observation_interval(config: &RouterConfig) -> Duration {
    Duration::from_millis(config.observation_interval_ms())
}

/// Checks that must hold before the process can serve, run before `-t` exits.
///
/// Every problem found is reported, not just the first: an operator validating a
/// replacement configuration during an upgrade window should learn about all of
/// them in one run.
fn preflight_host_checks(
    config: &RouterConfig,
    host_gpu_check: HostGpuCheck,
) -> Result<(TokenAuth, TokenAuth), String> {
    let mut errors = Vec::new();
    let management = load_bearer_token(
        "management_bearer_token_file",
        &config.listeners.management_bearer_token_file,
        &mut errors,
    );
    let inference = load_bearer_token(
        "inference_bearer_token_file",
        &config.listeners.inference_bearer_token_file,
        &mut errors,
    );
    check_host_gpus(config, host_gpu_check, &mut errors);

    if let (true, Some(management), Some(inference)) = (errors.is_empty(), management, inference) {
        return Ok((management, inference));
    }
    for problem in &errors {
        error!(problem = %problem, "configuration cannot start on this host");
    }
    Err(format!(
        "configuration cannot start on this host ({} problem(s)): {}",
        errors.len(),
        errors.join("; ")
    ))
}

fn load_bearer_token(setting: &str, path: &Path, errors: &mut Vec<String>) -> Option<TokenAuth> {
    match TokenAuth::load(path) {
        Ok(auth) => Some(auth),
        Err(error) => {
            errors.push(format!("listeners.{setting}: {error}"));
            None
        }
    }
}

/// Confirm the configured GPUs exist on this host and that the units this
/// configuration would generate can actually be rendered from them.
///
/// Static validation only proves the placement UUIDs match `inventory.gpus`, so
/// one typo repeated in both sections passes it; only NVML can reject that.
fn check_host_gpus(config: &RouterConfig, check: HostGpuCheck, errors: &mut Vec<String>) {
    if check == HostGpuCheck::Skipped {
        warn!(
            "--skip-host-gpu-check: configured GPU UUIDs were NOT validated against this host, \
             and this configuration may still fail to start"
        );
        return;
    }

    let devices = match GpuInventory::initialize().inventory() {
        Ok(devices) => devices,
        Err(error) => {
            // Startup resolves every placement UUID through NVML and fails
            // without it, so passing here would be the same false assurance
            // this check exists to remove. Off-host validation stays possible,
            // but only by asking for it.
            errors.push(format!(
                "{error}; startup resolves GPU UUIDs through NVML, so pass \
                 --skip-host-gpu-check to validate this configuration off-host"
            ));
            return;
        }
    };

    let present = devices.keys().cloned().collect::<Vec<_>>();
    for gpu in &config.inventory.gpus {
        if !devices.contains_key(&gpu.uuid) {
            errors.push(format!(
                "inventory.gpus lists GPU UUID {:?}, which this host does not have (present: {})",
                gpu.uuid,
                present.join(", ")
            ));
        }
    }

    let indices = devices
        .into_iter()
        .map(|(uuid, device)| (uuid, device.index))
        .collect();
    if let Err(error) = dry_run_generated_inputs(config, &indices) {
        errors.push(error.to_string());
    }
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
    let mut host_gpu_check = HostGpuCheck::Required;
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
            "--skip-host-gpu-check" => host_gpu_check = HostGpuCheck::Skipped,
            "--nocapture" => pingora.nocapture = true,
            "-h" | "--help" => {
                return Err(
                    "usage: vllm-router [-tu] [--skip-host-gpu-check] -c <pingora.yaml> \
                     --router-config <router.yaml>"
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
        host_gpu_check,
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

    use super::{
        HostGpuCheck, check_host_gpus, ensure_foreground_mode, parse_arguments_from,
        should_warn_reconciliation_error,
    };

    #[test]
    fn host_gpu_validation_is_required_unless_explicitly_skipped() {
        for (flags, expected) in [
            (vec!["-t"], HostGpuCheck::Required),
            (vec!["-t", "--skip-host-gpu-check"], HostGpuCheck::Skipped),
        ] {
            let mut command_line: Vec<String> =
                flags.into_iter().map(ToString::to_string).collect();
            command_line.extend(["-c".to_string(), "pingora.yaml".to_string()]);

            let arguments = parse_arguments_from(command_line, "router.yaml".to_string())
                .expect("command line should parse");

            assert_eq!(arguments.host_gpu_check, expected);
        }
    }

    /// The off-host escape hatch must stay usable where NVML cannot answer, which
    /// is the only reason it exists.
    #[test]
    fn skipping_host_gpu_validation_reports_no_problems() {
        let config: vllm_router::config::RouterConfig =
            serde_yaml::from_str(include_str!("config/example.yaml"))
                .expect("example configuration should parse");
        let mut errors = Vec::new();

        check_host_gpus(&config, HostGpuCheck::Skipped, &mut errors);

        assert!(errors.is_empty(), "unexpected problems: {errors:?}");
    }

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
    fn deployment_shutdown_window_covers_long_stream_target() {
        const LONG_STREAM_TARGET_SECONDS: u64 = 10 * 60;

        let configuration = ServerConf::from_yaml(include_str!("../deploy/pingora.yaml"))
            .expect("deployment Pingora configuration should parse");
        let grace = configuration
            .grace_period_seconds
            .expect("deployment must configure an explicit request-drain grace");
        let runtime_shutdown = configuration
            .graceful_shutdown_timeout_seconds
            .expect("deployment must configure an explicit runtime shutdown timeout");

        assert!(
            grace > LONG_STREAM_TARGET_SECONDS,
            "deployment grace must cover the ten-minute stream target with handoff margin"
        );
        assert_eq!(
            runtime_shutdown, 0,
            "Pingora 0.8 waits the full runtime shutdown timeout after request drain"
        );

        let unit = include_str!("../deploy/vllm-router.service");
        let supervisor_timeout = unit
            .lines()
            .find_map(|line| line.strip_prefix("TimeoutStopSec="))
            .expect("deployment unit must configure TimeoutStopSec")
            .parse::<u64>()
            .expect("deployment TimeoutStopSec must be an integer number of seconds");
        assert!(
            supervisor_timeout > grace + runtime_shutdown,
            "systemd TimeoutStopSec must exceed Pingora's complete shutdown window"
        );

        // Pingora's graceful path sleeps for the entire grace period whether or not
        // anything is still in flight, so a grace window sized for long streams must not
        // also be the cost of `systemctl stop`. SIGINT selects fast shutdown; SIGQUIT and
        // the -u handoff remain the stream-preserving path.
        let kill_signal = unit
            .lines()
            .find_map(|line| line.strip_prefix("KillSignal="))
            .expect("deployment unit must pin an explicit administrative stop signal");
        assert_eq!(
            kill_signal, "SIGINT",
            "a long stream grace window must not become the cost of every administrative stop"
        );
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
