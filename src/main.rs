use std::{env, error::Error, sync::Arc, time::Duration};

use async_trait::async_trait;
use pingora::{
    prelude::{Opt, Server, background_service},
    server::ShutdownWatch,
    services::{background::BackgroundService, listening::Service},
};
use tokio::time::MissedTickBehavior;
use tracing::{error, warn};
use vllm_router::{
    PoolManager,
    api::{ControlPlane, TokenAuth},
    config::{ConfigStore, RouterConfig},
    metrics,
    pool::{ManagedVllmRunner, Pool, managed_pool_from_config},
    proxy::DataPlane,
    state::Lifecycle,
};

struct Arguments {
    pingora: Opt,
    router_config: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
    server.bootstrap();

    let config = Arc::new(ConfigStore::new(router_config.clone())?);
    let auth = TokenAuth::load(&router_config.listeners.management_bearer_token_file)?;
    let pool = Arc::new(managed_pool_from_config(&router_config).await?);
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
                            warn!(error = %error, "service reconciliation failed");
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

fn publish_state_metrics(services: &[vllm_router::state::ServiceStatus]) {
    const STATES: &[(&str, Lifecycle)] = &[
        ("reconciling", Lifecycle::Reconciling),
        ("stopped", Lifecycle::Stopped),
        ("starting", Lifecycle::Starting),
        ("running", Lifecycle::Running),
        ("draining", Lifecycle::Draining),
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
    let mut pingora = Opt::default();
    let mut router_config =
        env::var("VLLM_ROUTER_CONFIG").unwrap_or_else(|_| "/etc/vllm-router/router.yaml".into());
    let mut arguments = env::args().skip(1);
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
            "-d" | "--daemon" => pingora.daemon = true,
            "-t" | "--test" => pingora.test = true,
            "-u" | "--upgrade" => pingora.upgrade = true,
            "--nocapture" => pingora.nocapture = true,
            "-h" | "--help" => {
                return Err(
                    "usage: vllm-router [-dtu] -c <pingora.yaml> --router-config <router.yaml>"
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
