//! Production composition of the generic pool runner seam.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use thiserror::Error;

use super::{
    GpuStatus, IdlePolicy, Pool, PoolError, ProbeStatus, Runner, RunnerError, RunnerObservation,
    ServiceDefinition, SleepLevel, SystemdState, SystemdStatus,
};
use crate::config::{self, RouterConfig};
use crate::gpu::{ExpectedGpuState, GpuInventory};
use crate::runner::{
    Runner as VllmApi, SleepLevel as VllmSleepLevel, VllmRunner, VllmRunnerConfig,
    WarmupRequest as VllmWarmupRequest,
};
use crate::sysd::{SystemdError, SystemdManager, UnitActiveState};

struct ManagedService {
    config: config::ServiceDefinition,
    vllm: VllmRunner,
    probe: Mutex<ProbeDebounce>,
    commanded_sleep: Mutex<Option<SleepLevel>>,
}

#[derive(Default)]
struct ProbeDebounce {
    successes: u32,
    failures: u32,
    ready: bool,
}

/// Runner implementation composed from the host's systemd, vLLM HTTP, and NVML adapters.
pub struct ManagedVllmRunner {
    systemd: SystemdManager,
    gpu: GpuInventory,
    services: HashMap<String, ManagedService>,
}

#[derive(Debug, Error)]
pub enum ManagedBuildError {
    #[error(transparent)]
    Systemd(#[from] SystemdError),
    #[error("could not construct vLLM adapter for `{service}`: {source}")]
    Vllm {
        service: String,
        #[source]
        source: crate::runner::RunnerError,
    },
    #[error(transparent)]
    Pool(#[from] PoolError),
}

impl ManagedVllmRunner {
    /// Connect to the system bus and construct one bounded HTTP adapter per service.
    pub async fn connect(config: &RouterConfig) -> Result<Self, ManagedBuildError> {
        let systemd_timeout = config
            .services
            .iter()
            .map(|service| Duration::from_millis(service.deadlines.systemd_job_ms))
            .max()
            .unwrap_or(Duration::from_secs(30));
        let systemd = SystemdManager::connect_system(systemd_timeout).await?;
        Self::from_parts(config, systemd, GpuInventory::initialize())
    }

    /// Dependency-injected constructor useful for host integration tests.
    pub fn from_parts(
        config: &RouterConfig,
        systemd: SystemdManager,
        gpu: GpuInventory,
    ) -> Result<Self, ManagedBuildError> {
        let mut services = HashMap::new();
        for service in &config.services {
            let request_timeout = Duration::from_millis(service.readiness.timeout_ms);
            let vllm = VllmRunner::new(VllmRunnerConfig {
                base_url: format!("http://{}", service.backend.socket_addr()),
                request_timeout,
                api_key: None,
            })
            .map_err(|source| ManagedBuildError::Vllm {
                service: service.name.clone(),
                source,
            })?;
            services.insert(
                service.name.clone(),
                ManagedService {
                    config: service.clone(),
                    vllm,
                    probe: Mutex::new(ProbeDebounce::default()),
                    commanded_sleep: Mutex::new(None),
                },
            );
        }
        Ok(Self {
            systemd,
            gpu,
            services,
        })
    }

    fn service(&self, name: &str) -> Result<&ManagedService, RunnerError> {
        self.services
            .get(name)
            .ok_or_else(|| RunnerError::new(format!("unknown managed service `{name}`")))
    }

    fn update_probe(
        service: &ManagedService,
        successful: bool,
        sleeping_level: Option<u8>,
    ) -> ProbeStatus {
        let mut probe = service.probe.lock().expect("probe lock poisoned");
        if sleeping_level.is_some() {
            probe.successes = 0;
            probe.failures = 0;
            probe.ready = false;
        } else if successful {
            probe.successes = probe.successes.saturating_add(1);
            probe.failures = 0;
            if probe.successes >= service.config.readiness.consecutive_successes {
                probe.ready = true;
            }
        } else {
            probe.successes = 0;
            probe.failures = probe.failures.saturating_add(1);
            if probe.failures >= service.config.readiness.consecutive_failures {
                probe.ready = false;
            }
        }
        ProbeStatus {
            healthy: probe.ready,
            ready: probe.ready,
            served_model_matches: probe.ready,
            sleeping_level,
            consecutive_successes: probe.successes,
            consecutive_failures: probe.failures,
            age: Some(Duration::ZERO),
        }
    }
}

/// Convert the validated configuration schema into the generic scheduler schema.
pub fn pool_definition(service: &config::ServiceDefinition) -> ServiceDefinition {
    let mut definition = ServiceDefinition::new(
        service.name.clone(),
        service.served_model.clone(),
        service.placement.gpu_uuids.clone(),
    );
    definition.endpoint = service.backend.socket_addr();
    definition.enabled = service.enabled;
    definition.priority = service.priority;
    definition.evictable = service.evictable;
    definition.automatic_start = service.automatic_start;
    definition.parking_limit = u64::try_from(service.parking.queue_limit).unwrap_or(u64::MAX);
    definition.sleep_levels = if service.sleep.enabled && service.sleep.hardware_validated {
        service
            .sleep
            .levels
            .iter()
            .copied()
            .map(|level| match level {
                config::SleepLevel::L1 => SleepLevel::L1,
                config::SleepLevel::L2 => SleepLevel::L2,
            })
            .collect()
    } else {
        Vec::new()
    };
    definition.startup_deadline = Duration::from_millis(service.deadlines.startup_ms);
    definition.wake_deadline = Duration::from_millis(service.deadlines.wake_ms);
    definition.readiness_poll_interval = Duration::from_millis(service.readiness.interval_ms);
    definition.drain_deadline = Duration::from_millis(service.deadlines.drain_ms);
    definition.stop_deadline = Duration::from_millis(service.deadlines.systemd_job_ms);
    definition.backoff_initial = Duration::from_millis(service.retry.initial_backoff_ms);
    definition.backoff_max = Duration::from_millis(service.retry.maximum_backoff_ms);
    definition.idle = IdlePolicy {
        sleep_l1_after: service.idle.sleep_l1_after_ms.map(Duration::from_millis),
        sleep_l2_after: service.idle.sleep_l2_after_ms.map(Duration::from_millis),
        stop_after: service.idle.stop_after_ms.map(Duration::from_millis),
    };
    definition
}

/// Build the production pool directly from validated router configuration.
pub async fn managed_pool_from_config(
    config: &RouterConfig,
) -> Result<Pool<ManagedVllmRunner>, ManagedBuildError> {
    let runner = ManagedVllmRunner::connect(config).await?;
    let definitions = config.services.iter().map(pool_definition);
    Ok(Pool::new(definitions, runner)?)
}

#[async_trait]
impl Runner for ManagedVllmRunner {
    async fn observe(
        &self,
        definition: &ServiceDefinition,
    ) -> Result<RunnerObservation, RunnerError> {
        let service = self.service(&definition.name)?;
        let unit = self
            .systemd
            .unit_status(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        let systemd = SystemdStatus {
            state: systemd_state(&unit.active_state),
            sub_state: Some(unit.sub_state),
            main_pid: None,
        };

        let gpu_snapshot = self.gpu.snapshot_async(ExpectedGpuState::default()).await;
        let gpus = definition
            .gpu_ids
            .iter()
            .map(|uuid| {
                let observed = gpu_snapshot.devices.get(uuid);
                GpuStatus {
                    uuid: uuid.clone(),
                    used_memory_bytes: observed.map(|gpu| gpu.used_memory_bytes),
                    process_ids: observed
                        .map(|gpu| {
                            gpu.compute_processes
                                .iter()
                                .map(|process| process.pid)
                                .collect()
                        })
                        .unwrap_or_default(),
                    healthy: gpu_snapshot.available && observed.is_some(),
                }
            })
            .collect();

        if systemd.state != SystemdState::Active {
            return Ok(RunnerObservation {
                systemd,
                probe: Self::update_probe(service, false, None),
                gpus,
            });
        }

        let sleeping = if service.config.sleep.enabled {
            VllmApi::is_sleeping(&service.vllm).await.ok()
        } else {
            None
        };
        let sleeping_level = if sleeping == Some(true) {
            let remembered = *service
                .commanded_sleep
                .lock()
                .expect("sleep-level lock poisoned");
            Some(match remembered.unwrap_or(SleepLevel::L1) {
                SleepLevel::L1 => 1,
                SleepLevel::L2 => 2,
            })
        } else {
            None
        };
        let successful = if sleeping_level.is_some() {
            false
        } else {
            let health = VllmApi::health(&service.vllm).await;
            let models = if health.is_ok() {
                VllmApi::models(&service.vllm).await.ok()
            } else {
                None
            };
            health.is_ok()
                && models.is_some_and(|models| {
                    models
                        .iter()
                        .any(|model| model == &service.config.served_model)
                })
        };
        Ok(RunnerObservation {
            systemd,
            probe: Self::update_probe(service, successful, sleeping_level),
            gpus,
        })
    }

    async fn start(&self, definition: &ServiceDefinition) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        self.systemd
            .start_unit(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .commanded_sleep
            .lock()
            .expect("sleep-level lock poisoned") = None;
        Ok(())
    }

    async fn stop(&self, definition: &ServiceDefinition, _force: bool) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        self.systemd
            .stop_unit(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .commanded_sleep
            .lock()
            .expect("sleep-level lock poisoned") = None;
        Ok(())
    }

    async fn sleep(
        &self,
        definition: &ServiceDefinition,
        level: SleepLevel,
    ) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        VllmApi::sleep(&service.vllm, vllm_sleep_level(level))
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .commanded_sleep
            .lock()
            .expect("sleep-level lock poisoned") = Some(level);
        Ok(())
    }

    async fn wake(
        &self,
        definition: &ServiceDefinition,
        level: SleepLevel,
    ) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        VllmApi::wake(&service.vllm, vllm_sleep_level(level))
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .commanded_sleep
            .lock()
            .expect("sleep-level lock poisoned") = None;
        Ok(())
    }

    async fn warmup(&self, definition: &ServiceDefinition) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        let Some(configured) = &service.config.warmup else {
            return Ok(());
        };
        let prompt = configured
            .body
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("warmup")
            .to_string();
        let max_tokens = configured
            .body
            .get("max_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(1);
        let temperature = configured
            .body
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        VllmApi::warmup(
            &service.vllm,
            &VllmWarmupRequest {
                model: service.config.served_model.clone(),
                prompt,
                max_tokens,
                temperature,
            },
        )
        .await
        .map_err(|error| RunnerError::new(error.to_string()))
    }
}

fn vllm_sleep_level(level: SleepLevel) -> VllmSleepLevel {
    match level {
        SleepLevel::L1 => VllmSleepLevel::L1,
        SleepLevel::L2 => VllmSleepLevel::L2,
    }
}

fn systemd_state(state: &UnitActiveState) -> SystemdState {
    match state {
        UnitActiveState::Active | UnitActiveState::Reloading | UnitActiveState::Refreshing => {
            SystemdState::Active
        }
        UnitActiveState::Inactive => SystemdState::Inactive,
        UnitActiveState::Failed | UnitActiveState::Maintenance => SystemdState::Failed,
        UnitActiveState::Activating => SystemdState::Activating,
        UnitActiveState::Deactivating => SystemdState::Deactivating,
        UnitActiveState::Unknown(_) => SystemdState::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_validated_service_policy_to_generic_definition() {
        let yaml = include_str!("../config/example.yaml");
        let config: RouterConfig = serde_yaml::from_str(yaml).unwrap();
        config.validate().unwrap();
        let source = &config.services[0];
        let converted = pool_definition(source);
        assert_eq!(converted.name, source.name);
        assert_eq!(converted.served_model, source.served_model);
        assert_eq!(converted.gpu_ids, source.placement.gpu_uuids);
        assert_eq!(converted.endpoint, source.backend.socket_addr());
        assert_eq!(
            converted.parking_limit,
            u64::try_from(source.parking.queue_limit).unwrap()
        );
    }
}
