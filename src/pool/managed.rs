//! Production composition of the generic pool runner seam.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use thiserror::Error;

use super::{
    GpuStatus, IdlePolicy, Pool, PoolError, ProbeStatus, Runner, RunnerError, RunnerObservation,
    ServiceDefinition, SleepLevel, SystemdState, SystemdStatus,
};
use crate::config::{self, RouterConfig};
use crate::gpu::{
    ExpectedGpuState, ExpectedGpuUsage, GpuDrift, GpuError, GpuInventory, GpuSnapshot,
};
use crate::metrics;
use crate::runner::{
    Runner as VllmApi, SleepLevel as VllmSleepLevel, VllmRunner, VllmRunnerConfig, VllmTimeouts,
    WarmupRequest as VllmWarmupRequest,
};
use crate::sysd::{
    GeneratedSystemdError, SystemdError, SystemdManager, UnitActiveState, UnitStatus,
    install_generated_inputs,
};

struct ManagedService {
    config: config::ServiceDefinition,
    vllm: VllmRunner,
    probe: Mutex<ProbeDebounce>,
    observed_systemd: Mutex<Option<SystemdStatus>>,
    sleep: Mutex<SleepObservationState>,
    gpu_warning: Mutex<GpuWarningState>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct GpuWarningState {
    degraded_reason: Option<String>,
    drift: Vec<GpuDrift>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SleepObservationState {
    commanded: Option<SleepLevel>,
    observed_sleeping: bool,
}

impl SleepObservationState {
    /// Apply an authoritative `/is_sleeping` response.
    ///
    /// A failed probe proves nothing and therefore never reaches this state: the last
    /// authoritative fact is retained by `SleepProof` together with the timestamp that
    /// proved it.
    fn observe(&mut self, sleeping: bool) -> Option<u8> {
        if sleeping {
            self.observed_sleeping = true;
        } else {
            self.observed_sleeping = false;
            self.commanded = None;
        }

        self.observed_sleeping
            .then(|| match reported_sleep_level(self.commanded) {
                SleepLevel::L1 => 1,
                SleepLevel::L2 => 2,
            })
    }
}

/// Provenance of the sleep fact carried by one observation.
///
/// `sleeping_level: None` has three distinct causes — sleep is disabled, the backend
/// answered "awake", or `/is_sleeping` could not be reached — and admission must not
/// treat the third like the first two. This makes the difference explicit at the one
/// place that decides whether the observation may be stamped as freshly proven.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SleepFact {
    /// Sleep is disabled for this service, so there is nothing to observe and the
    /// configuration itself is the authority.
    NotApplicable,
    /// `/is_sleeping` answered authoritatively.
    Observed(Option<u8>),
    /// `/is_sleeping` could not be observed, so this observation proves nothing and
    /// carries no value of its own.
    Unobserved,
}

/// One probe cycle's sleep reading: what `/is_sleeping` said, which vLLM process
/// generation was asked, and when. These travel together so a fact can never be
/// recorded against another cycle's generation or timestamp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SleepReading {
    fact: SleepFact,
    generation: u64,
    observed_at: Instant,
}

/// One authoritative sleep observation: the value, and the instant that proved it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProvenSleep {
    level: Option<u8>,
    observed_at: Instant,
}

/// The sleep fact this runner currently stands behind, bound to the vLLM process
/// generation it describes.
///
/// Value, timestamp, and generation move as one unit: `prove` is the only way to
/// install any of them, so a fact can never be published under a timestamp that proved
/// a different fact. Start, stop, sleep, and wake begin a new generation, which retires
/// the previous proof until `/is_sleeping` speaks for the process now behind the
/// endpoint — an old "the previous process is absent or asleep" observation must never
/// be presented as a fresh "this process is awake".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SleepProof {
    generation: u64,
    proven: Option<ProvenSleep>,
}

impl SleepProof {
    /// Retire the current proof because a new process (or a commanded sleep
    /// transition) means the previous observation no longer describes the backend.
    fn begin_generation(&mut self) -> u64 {
        self.generation = self.generation.saturating_add(1);
        self.proven = None;
        self.generation
    }

    /// Install an authoritative observation of `generation`.
    ///
    /// An observation of a superseded generation is dropped rather than published: it
    /// was taken against a process that has since been started, stopped, slept, or
    /// woken, so it proves nothing about the current one.
    fn prove(&mut self, generation: u64, level: Option<u8>, observed_at: Instant) {
        if generation < self.generation {
            return;
        }
        self.proven = Some(ProvenSleep { level, observed_at });
    }

    /// Prove a fact the runner itself just established, such as a successful sleep
    /// command, for the generation that command begins.
    fn prove_new_generation(&mut self, level: Option<u8>, observed_at: Instant) {
        let generation = self.begin_generation();
        self.prove(generation, level, observed_at);
    }

    fn level(self) -> Option<u8> {
        self.proven.and_then(|proven| proven.level)
    }

    /// How long ago the standing proof was taken, or `None` while no proof describes
    /// the current generation.
    fn age_at(self, now: Instant) -> Option<Duration> {
        self.proven
            .map(|proven| now.saturating_duration_since(proven.observed_at))
    }
}

#[derive(Default)]
struct ProbeDebounce {
    successes: u32,
    failures: u32,
    ready: bool,
    healthy: bool,
    served_model_matches: bool,
    sleep: SleepProof,
}

impl ProbeDebounce {
    fn record(
        &mut self,
        healthy: bool,
        served_model_matches: bool,
        sleep: SleepReading,
        consecutive_successes: u32,
        consecutive_failures: u32,
    ) -> ProbeStatus {
        self.healthy = healthy;
        self.served_model_matches = served_model_matches;
        // Only a proven sleep fact may refresh the observation timestamp. Admission's
        // freshness gate is the sole barrier between a sleeping backend and a request
        // that hangs until the inter-chunk timeout, so stamping a retained-but-unproven
        // "awake" here would defeat it. Leaving the standing proof alone ages the fact
        // out instead: a single failed probe is absorbed, and sustained failure closes
        // admission once the age passes `readiness.maximum_probe_age_ms`.
        match sleep.fact {
            // Sleep is disabled, so the configuration itself proves there is nothing to
            // be asleep in and every observation is authoritative.
            SleepFact::NotApplicable => self.sleep.prove(sleep.generation, None, sleep.observed_at),
            SleepFact::Observed(level) => {
                self.sleep.prove(sleep.generation, level, sleep.observed_at)
            }
            SleepFact::Unobserved => {}
        }
        let sleeping_level = self.sleep.level();
        if sleeping_level.is_some() {
            self.successes = 0;
            self.failures = 0;
            self.ready = false;
        } else if healthy && served_model_matches {
            self.successes = self.successes.saturating_add(1);
            self.failures = 0;
            if self.successes >= consecutive_successes {
                self.ready = true;
            }
        } else {
            self.successes = 0;
            self.failures = self.failures.saturating_add(1);
            if self.failures >= consecutive_failures {
                self.ready = false;
            }
        }
        self.status_at(Instant::now())
    }

    fn status_at(&self, now: Instant) -> ProbeStatus {
        ProbeStatus {
            healthy: self.healthy,
            ready: self.ready,
            served_model_matches: self.served_model_matches,
            sleeping_level: self.sleep.level(),
            consecutive_successes: self.successes,
            consecutive_failures: self.failures,
            age: self.sleep.age_at(now),
        }
    }
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
    GeneratedSystemd(#[from] GeneratedSystemdError),
    #[error(transparent)]
    Systemd(#[from] SystemdError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
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
        let gpu = GpuInventory::initialize();
        let gpu_indices = gpu
            .inventory()?
            .into_iter()
            .map(|(uuid, device)| (uuid, device.index))
            .collect();
        install_generated_inputs(config, &gpu_indices)?;
        let systemd_timeout = config
            .services
            .iter()
            .map(|service| Duration::from_millis(service.deadlines.systemd_job_ms))
            .max()
            .unwrap_or(Duration::from_secs(30));
        let systemd = SystemdManager::connect_system(systemd_timeout).await?;
        systemd.reload().await?;
        Self::from_parts(config, systemd, gpu)
    }

    /// Dependency-injected constructor useful for host integration tests.
    pub fn from_parts(
        config: &RouterConfig,
        systemd: SystemdManager,
        gpu: GpuInventory,
    ) -> Result<Self, ManagedBuildError> {
        let mut services = HashMap::new();
        for service in &config.services {
            let vllm = VllmRunner::new(VllmRunnerConfig {
                base_url: format!("http://{}", service.backend.socket_addr()),
                timeouts: vllm_timeouts(service),
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
                    observed_systemd: Mutex::new(None),
                    sleep: Mutex::new(SleepObservationState::default()),
                    gpu_warning: Mutex::new(GpuWarningState::default()),
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
        healthy: bool,
        served_model_matches: bool,
        sleep: SleepReading,
    ) -> ProbeStatus {
        service.probe.lock().expect("probe lock poisoned").record(
            healthy,
            served_model_matches,
            sleep,
            service.config.readiness.consecutive_successes,
            service.config.readiness.consecutive_failures,
        )
    }

    /// The generation an observation that starts now is allowed to speak for.
    ///
    /// Read before any probing so a lifecycle command that lands mid-observation
    /// supersedes it: the probe answered for a process that is no longer there.
    fn sleep_generation(service: &ManagedService) -> u64 {
        service
            .probe
            .lock()
            .expect("probe lock poisoned")
            .sleep
            .generation
    }

    /// Retire the standing sleep proof after a lifecycle command changed which process
    /// (or which sleep state) is behind the endpoint.
    fn begin_sleep_generation(service: &ManagedService) {
        service
            .probe
            .lock()
            .expect("probe lock poisoned")
            .sleep
            .begin_generation();
    }

    /// Record the sleep state a successful sleep command just established.
    ///
    /// The command is itself authoritative, so the new generation starts proven rather
    /// than waiting for `/is_sleeping` — and it starts proven asleep, which admission
    /// rejects outright.
    fn prove_commanded_sleep(service: &ManagedService, level: SleepLevel) {
        service
            .probe
            .lock()
            .expect("probe lock poisoned")
            .sleep
            .prove_new_generation(
                Some(match level {
                    SleepLevel::L1 => 1,
                    SleepLevel::L2 => 2,
                }),
                Instant::now(),
            );
    }

    fn expected_gpu_state(&self) -> ExpectedGpuState {
        expected_gpu_state(self.services.values().map(|service| {
            let systemd = service
                .observed_systemd
                .lock()
                .expect("systemd-observation lock poisoned")
                .clone();
            let sleeping_level = service
                .probe
                .lock()
                .expect("probe lock poisoned")
                .sleep
                .level();
            (&service.config, systemd, sleeping_level)
        }))
    }
}

fn expected_gpu_state<'a>(
    services: impl IntoIterator<
        Item = (
            &'a config::ServiceDefinition,
            Option<SystemdStatus>,
            Option<u8>,
        ),
    >,
) -> ExpectedGpuState {
    let mut expected = ExpectedGpuState::default();
    for (service, systemd, sleeping_level) in services {
        let maximum_used_memory_bytes =
            expected_service_memory(service, systemd.as_ref(), sleeping_level);
        let process_facts_known = expected_process_facts_known(systemd.as_ref());
        let main_pid = systemd.as_ref().and_then(|status| status.main_pid);
        for uuid in &service.placement.gpu_uuids {
            let usage = expected
                .devices
                .entry(uuid.clone())
                .or_insert_with(|| ExpectedGpuUsage {
                    maximum_used_memory_bytes: Some(0),
                    allowed_compute_pids: BTreeSet::new(),
                    allow_unexpected_compute_pids: false,
                });
            usage.maximum_used_memory_bytes =
                match (usage.maximum_used_memory_bytes, maximum_used_memory_bytes) {
                    (Some(total), Some(service_maximum)) => total.checked_add(service_maximum),
                    _ => None,
                };
            if let Some(main_pid) = main_pid {
                usage
                    .allowed_compute_pids
                    .extend(process_tree_pids(main_pid));
            }
            if !process_facts_known {
                usage.allow_unexpected_compute_pids = true;
            }
        }
    }
    expected
}

fn expected_service_memory(
    service: &config::ServiceDefinition,
    systemd: Option<&SystemdStatus>,
    sleeping_level: Option<u8>,
) -> Option<u64> {
    match systemd {
        Some(SystemdStatus {
            state: SystemdState::Active,
            ..
        }) => Some(if sleeping_level.is_some() && service.sleep.enabled {
            service.sleep.residual_vram_bytes_per_gpu
        } else {
            service.placement.vram_reservation_bytes_per_gpu
        }),
        Some(SystemdStatus {
            state: SystemdState::Inactive | SystemdState::Failed,
            main_pid: None,
            ..
        }) => Some(0),
        Some(_) | None => None,
    }
}

fn expected_process_facts_known(systemd: Option<&SystemdStatus>) -> bool {
    matches!(
        systemd,
        Some(SystemdStatus {
            state: SystemdState::Active,
            main_pid: Some(_),
            ..
        }) | Some(SystemdStatus {
            state: SystemdState::Inactive | SystemdState::Failed,
            main_pid: None,
            ..
        })
    )
}

fn process_tree_pids(main_pid: u32) -> BTreeSet<u32> {
    let mut discovered = BTreeSet::from([main_pid]);
    let mut pending = vec![main_pid];
    while let Some(pid) = pending.pop() {
        let path = format!("/proc/{pid}/task/{pid}/children");
        let Ok(children) = fs::read_to_string(path) else {
            continue;
        };
        for child in children
            .split_whitespace()
            .filter_map(|child| child.parse::<u32>().ok())
        {
            if discovered.insert(child) {
                pending.push(child);
            }
        }
    }
    discovered
}

fn gpu_statuses(definition: &ServiceDefinition, snapshot: &GpuSnapshot) -> Vec<GpuStatus> {
    definition
        .gpu_ids
        .iter()
        .map(|uuid| {
            let observed = snapshot.devices.get(uuid);
            let drift: Vec<_> = snapshot
                .drift
                .iter()
                .filter(|drift| drift.uuid() == uuid)
                .cloned()
                .collect();
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
                healthy: snapshot.available && observed.is_some() && drift.is_empty(),
                drift,
            }
        })
        .collect()
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
    definition.maximum_probe_age = Duration::from_millis(service.readiness.maximum_probe_age_ms);
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
        let generation = Self::sleep_generation(service);
        let unit = self
            .systemd
            .unit_status(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        let systemd = runner_systemd_status(unit);

        let probe = if systemd.state != SystemdState::Active {
            // Sleep observations only describe the current vLLM process. Once
            // systemd says it is no longer active, a later process must
            // rediscover sleep state (and therefore retain unknown wake
            // provenance) rather than inherit stale command history.
            *service
                .sleep
                .lock()
                .expect("sleep-observation lock poisoned") = SleepObservationState::default();
            // An inactive unit has no vLLM process to be asleep, and the retained
            // observation state was just reset, so this is an authoritative fact.
            Self::update_probe(
                service,
                false,
                false,
                SleepReading {
                    fact: SleepFact::Observed(None),
                    generation,
                    observed_at: Instant::now(),
                },
            )
        } else {
            let fact = if service.config.sleep.enabled {
                match VllmApi::is_sleeping(&service.vllm).await {
                    Ok(sleeping) => SleepFact::Observed(
                        service
                            .sleep
                            .lock()
                            .expect("sleep-observation lock poisoned")
                            .observe(sleeping),
                    ),
                    Err(error) => {
                        metrics::global()
                            .probe_failures
                            .with_label_values(&[&definition.name, "is_sleeping"])
                            .inc();
                        tracing::warn!(
                            service = %definition.name,
                            error = %error,
                            "sleep observation failed; retaining last authoritative state, which will age out of admission"
                        );
                        SleepFact::Unobserved
                    }
                }
            } else {
                SleepFact::NotApplicable
            };
            let sleeping_level = match fact {
                SleepFact::NotApplicable => None,
                SleepFact::Observed(level) => level,
                // Nothing new was proved, so the standing proof still describes this
                // process: do not probe a backend the runner believes is asleep.
                SleepFact::Unobserved => service
                    .probe
                    .lock()
                    .expect("probe lock poisoned")
                    .sleep
                    .level(),
            };
            let (healthy, served_model_matches) = if sleeping_level.is_some() {
                (false, false)
            } else {
                let health = VllmApi::health(&service.vllm).await;
                let models = if health.is_ok() {
                    VllmApi::models(&service.vllm).await.ok()
                } else {
                    None
                };
                (
                    health.is_ok(),
                    models.is_some_and(|models| {
                        served_model_matches(&models, &service.config.served_model)
                    }),
                )
            };
            Self::update_probe(
                service,
                healthy,
                served_model_matches,
                SleepReading {
                    fact,
                    generation,
                    observed_at: Instant::now(),
                },
            )
        };
        *service
            .observed_systemd
            .lock()
            .expect("systemd-observation lock poisoned") = Some(systemd.clone());

        let gpu_snapshot = self.gpu.snapshot_async(self.expected_gpu_state()).await;
        let assigned_drift: Vec<_> = gpu_snapshot
            .drift
            .iter()
            .filter(|drift| definition.gpu_ids.iter().any(|uuid| uuid == drift.uuid()))
            .collect();
        let gpu_observation_healthy = gpu_snapshot.available
            && definition
                .gpu_ids
                .iter()
                .all(|uuid| gpu_snapshot.devices.contains_key(uuid))
            && assigned_drift.is_empty();
        let degraded_reason = (!gpu_observation_healthy).then(|| {
            gpu_snapshot
                .unavailable_reason
                .clone()
                .unwrap_or_else(|| "one or more assigned GPUs are missing or have drift".into())
        });
        let warning_state = GpuWarningState {
            degraded_reason,
            drift: assigned_drift
                .iter()
                .map(|drift| (*drift).clone())
                .collect(),
        };
        let previous_warning_state = {
            let mut previous = service
                .gpu_warning
                .lock()
                .expect("GPU-warning lock poisoned");
            std::mem::replace(&mut *previous, warning_state.clone())
        };
        if warning_state == previous_warning_state {
            if warning_state.degraded_reason.is_some() {
                tracing::debug!(
                    service = %definition.name,
                    reason = warning_state.degraded_reason.as_deref(),
                    drift = ?warning_state.drift,
                    "GPU observation remains degraded"
                );
            }
        } else if let Some(reason) = warning_state.degraded_reason.as_deref() {
            metrics::global()
                .probe_failures
                .with_label_values(&[&definition.name, "nvml"])
                .inc();
            tracing::warn!(
                service = %definition.name,
                reason,
                "GPU observation degraded"
            );
        } else if previous_warning_state.degraded_reason.is_some() {
            tracing::warn!(
                service = %definition.name,
                "GPU observation recovered"
            );
        }
        for drift in &warning_state.drift {
            metrics::global()
                .probe_failures
                .with_label_values(&[&definition.name, "nvml_drift"])
                .inc();
            if !previous_warning_state.drift.contains(drift) {
                tracing::warn!(
                    service = %definition.name,
                    gpu_uuid = drift.uuid(),
                    drift_kind = drift.kind(),
                    drift = ?drift,
                    advisory = true,
                    "NVML drift detected"
                );
            }
        }
        for drift in &previous_warning_state.drift {
            if !warning_state.drift.contains(drift) {
                tracing::warn!(
                    service = %definition.name,
                    gpu_uuid = drift.uuid(),
                    drift_kind = drift.kind(),
                    drift = ?drift,
                    advisory = true,
                    "NVML drift cleared"
                );
            }
        }
        let gpus = gpu_statuses(definition, &gpu_snapshot);
        let placement_error = if probe.ready
            && probe.healthy
            && probe.served_model_matches
            && systemd.state == SystemdState::Active
        {
            verify_main_pid_placement(definition, &systemd, &gpu_snapshot).err()
        } else {
            None
        };

        Ok(RunnerObservation {
            systemd,
            probe,
            gpus,
            placement_error,
        })
    }

    async fn start(&self, definition: &ServiceDefinition) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        self.systemd
            .start_unit(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned") = SleepObservationState::default();
        // Retire after the command, never before: a proof taken between the command and
        // this point described the previous process, and only an observation that starts
        // after this point can speak for the new one.
        Self::begin_sleep_generation(service);
        Ok(())
    }

    async fn stop(&self, definition: &ServiceDefinition, _force: bool) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        // StopUnit is intentional systemd lifecycle control: it suppresses the
        // generated unit's Restart=on-failure policy for this stop transaction.
        // `force` governs the pool's drain decision; using KillUnit here would
        // instead look like a crash and invite the configured restart policy.
        self.systemd
            .stop_unit(&service.config.systemd.unit)
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned") = SleepObservationState::default();
        Self::begin_sleep_generation(service);
        Ok(())
    }

    async fn sleep(
        &self,
        definition: &ServiceDefinition,
        level: SleepLevel,
    ) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        let sleep = *service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned");
        service
            .vllm
            .sleep_from(
                sleep.commanded.map(vllm_sleep_level),
                sleep.observed_sleeping,
                vllm_sleep_level(level),
            )
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned") = SleepObservationState {
            commanded: Some(level),
            observed_sleeping: true,
        };
        Self::prove_commanded_sleep(service, level);
        Ok(())
    }

    async fn wake(
        &self,
        definition: &ServiceDefinition,
        _level: SleepLevel,
    ) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        let current = service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned")
            .commanded;
        service
            .vllm
            .wake_from(current.map(vllm_sleep_level))
            .await
            .map_err(|error| RunnerError::new(error.to_string()))?;
        *service
            .sleep
            .lock()
            .expect("sleep-observation lock poisoned") = SleepObservationState::default();
        // A successful wake command is not the specified sleep proof: the endpoint must
        // answer `/is_sleeping` for this generation before readiness or admission may
        // treat the backend as awake.
        Self::begin_sleep_generation(service);
        Ok(())
    }

    async fn warmup(&self, definition: &ServiceDefinition) -> Result<(), RunnerError> {
        let service = self.service(&definition.name)?;
        let Some(request) = vllm_warmup_request(&service.config)? else {
            return Ok(());
        };
        VllmApi::warmup(&service.vllm, &request)
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

fn vllm_timeouts(service: &config::ServiceDefinition) -> VllmTimeouts {
    VllmTimeouts {
        probe: Duration::from_millis(service.readiness.timeout_ms),
        sleep: Duration::from_millis(service.deadlines.drain_ms),
        wake: Duration::from_millis(service.deadlines.wake_ms),
        warmup: Duration::from_millis(service.deadlines.startup_ms.max(service.deadlines.wake_ms)),
    }
}

fn vllm_warmup_request(
    service: &config::ServiceDefinition,
) -> Result<Option<VllmWarmupRequest>, RunnerError> {
    let Some(configured) = &service.warmup else {
        return Ok(None);
    };
    let body = configured
        .body
        .as_object()
        .cloned()
        .ok_or_else(|| RunnerError::new("configured warmup body must be a JSON object"))?;
    Ok(Some(VllmWarmupRequest {
        path: configured.path.clone(),
        model: service.served_model.clone(),
        body,
    }))
}

fn reported_sleep_level(commanded: Option<SleepLevel>) -> SleepLevel {
    // `/is_sleeping` has no level field. Report unknown sleepers as L1 so
    // capacity accounting reserves the possible pinned host RAM, but do not
    // write that reporting assumption back into command history.
    commanded.unwrap_or(SleepLevel::L1)
}

fn served_model_matches(models: &[String], canonical: &str) -> bool {
    models.iter().any(|model| model == canonical)
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

fn runner_systemd_status(unit: UnitStatus) -> SystemdStatus {
    SystemdStatus {
        state: systemd_state(&unit.active_state),
        sub_state: Some(unit.sub_state),
        main_pid: unit.main_pid,
    }
}

fn verify_main_pid_placement(
    definition: &ServiceDefinition,
    systemd: &SystemdStatus,
    snapshot: &GpuSnapshot,
) -> Result<(), String> {
    if !snapshot.available {
        return Err(format!(
            "cannot verify GPU placement after readiness: {}",
            snapshot
                .unavailable_reason
                .as_deref()
                .unwrap_or("NVML observation unavailable")
        ));
    }
    let main_pid = systemd.main_pid.ok_or_else(|| {
        "cannot verify GPU placement after readiness: MainPID is absent".to_string()
    })?;
    let unit_processes = process_tree_pids(main_pid);
    verify_unit_process_placement(definition, main_pid, &unit_processes, snapshot)
}

fn verify_unit_process_placement(
    definition: &ServiceDefinition,
    main_pid: u32,
    unit_processes: &BTreeSet<u32>,
    snapshot: &GpuSnapshot,
) -> Result<(), String> {
    let assigned: BTreeSet<_> = definition.gpu_ids.iter().cloned().collect();
    let observed: BTreeSet<_> = snapshot
        .devices
        .iter()
        .filter(|(_, gpu)| {
            gpu.compute_processes
                .iter()
                .any(|process| unit_processes.contains(&process.pid))
        })
        .map(|(uuid, _)| uuid.clone())
        .collect();
    if observed == assigned {
        Ok(())
    } else {
        Err(format!(
            "unit process tree rooted at MainPID {main_pid} has a GPU placement mismatch: assigned {assigned:?}, observed {observed:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuDevice, GpuDrift, GpuObservation, GpuProcess};

    fn reading(fact: SleepFact, generation: u64, observed_at: Instant) -> SleepReading {
        SleepReading {
            fact,
            generation,
            observed_at,
        }
    }

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
        assert_eq!(
            converted.maximum_probe_age,
            Duration::from_millis(source.readiness.maximum_probe_age_ms)
        );
        assert_eq!(
            vllm_timeouts(source),
            VllmTimeouts {
                probe: Duration::from_millis(source.readiness.timeout_ms),
                sleep: Duration::from_millis(source.deadlines.drain_ms),
                wake: Duration::from_millis(source.deadlines.wake_ms),
                warmup: Duration::from_millis(
                    source.deadlines.startup_ms.max(source.deadlines.wake_ms)
                ),
            }
        );

        let mut wake_dominates = source.clone();
        wake_dominates.deadlines.startup_ms = 10;
        wake_dominates.deadlines.wake_ms = 20;
        assert_eq!(
            vllm_timeouts(&wake_dominates).warmup,
            Duration::from_millis(20)
        );

        let mut arbitrary_warmup = source.clone();
        let body = serde_json::json!({
            "model": "must-be-overridden",
            "stream": true,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "warm"}]}],
            "guided_json": {"type": "object"},
            "custom": {"nested": [1, 2, 3]}
        });
        arbitrary_warmup.warmup.as_mut().unwrap().path = "/v1/chat/completions".into();
        arbitrary_warmup.warmup.as_mut().unwrap().body = body.clone();
        let request = vllm_warmup_request(&arbitrary_warmup).unwrap().unwrap();
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.model, arbitrary_warmup.served_model);
        assert_eq!(request.body, body.as_object().unwrap().clone());
    }

    #[test]
    fn unknown_sleep_level_reports_l1_without_losing_unknown_provenance() {
        let unknown = None;
        assert_eq!(reported_sleep_level(unknown), SleepLevel::L1);
        assert_eq!(unknown, None);
        assert_eq!(reported_sleep_level(Some(SleepLevel::L1)), SleepLevel::L1);
        assert_eq!(reported_sleep_level(Some(SleepLevel::L2)), SleepLevel::L2);
    }

    #[test]
    fn failed_sleep_observation_preserves_the_authoritative_proof_it_could_not_refresh() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();
        let generation = probe.sleep.generation;

        probe.record(
            false,
            false,
            reading(SleepFact::Observed(Some(2)), generation, started),
            1,
            1,
        );
        let unproven = probe.record(
            false,
            false,
            reading(
                SleepFact::Unobserved,
                generation,
                started + Duration::from_secs(5),
            ),
            1,
            1,
        );

        // Both halves of the proof are retained: the level it established, and the
        // timestamp that established it.
        assert_eq!(unproven.sleeping_level, Some(2));
        assert_eq!(
            probe.status_at(started + Duration::from_secs(5)).age,
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn awake_sleep_observation_clears_remembered_sleep_state() {
        let mut state = SleepObservationState {
            commanded: Some(SleepLevel::L2),
            observed_sleeping: true,
        };

        assert_eq!(state.observe(false), None);
        assert_eq!(state, SleepObservationState::default());
    }

    #[test]
    fn restart_discovered_sleeper_keeps_unknown_command_provenance() {
        let mut state = SleepObservationState::default();

        assert_eq!(state.observe(true), Some(1));
        assert_eq!(state.commanded, None);
        assert!(state.observed_sleeping);
    }

    #[test]
    fn healthy_wrong_model_is_distinct_and_probe_age_uses_observation_time() {
        let observed_at = Instant::now();
        let mut probe = ProbeDebounce::default();
        let generation = probe.sleep.generation;

        let wrong_model = probe.record(
            true,
            false,
            reading(SleepFact::NotApplicable, generation, observed_at),
            1,
            1,
        );

        assert!(wrong_model.healthy);
        assert!(!wrong_model.served_model_matches);
        assert!(!wrong_model.ready);
        assert_eq!(wrong_model.consecutive_failures, 1);
        let later = probe.status_at(observed_at + Duration::from_secs(5));
        assert_eq!(later.age, Some(Duration::from_secs(5)));
        assert!(later.age > wrong_model.age);
    }

    #[test]
    fn an_unobserved_sleep_fact_ages_out_instead_of_looking_freshly_proven() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();
        let generation = probe.sleep.generation;

        // A proven awake observation is admissible and freshly stamped.
        let proven = probe.record(
            true,
            true,
            reading(SleepFact::Observed(None), generation, started),
            1,
            1,
        );
        assert_eq!(proven.sleeping_level, None);
        assert!(proven.ready);
        assert_eq!(probe.status_at(started).age, Some(Duration::ZERO));

        // `/is_sleeping` then fails. The proof from the same process still stands, but
        // its timestamp must not move: the backend may have slept out of band, and a
        // fresh-looking "awake" would pass admission's only sleep barrier.
        let unproven = probe.record(
            true,
            true,
            reading(
                SleepFact::Unobserved,
                generation,
                started + Duration::from_secs(30),
            ),
            1,
            1,
        );
        assert_eq!(unproven.sleeping_level, None);
        assert_eq!(
            probe.status_at(started + Duration::from_secs(30)).age,
            Some(Duration::from_secs(30)),
            "a failed sleep probe must let the fact age past maximum_probe_age"
        );

        // One successful probe re-proves the fact and restores admissibility.
        probe.record(
            true,
            true,
            reading(
                SleepFact::Observed(None),
                generation,
                started + Duration::from_secs(60),
            ),
            1,
            1,
        );
        assert_eq!(
            probe.status_at(started + Duration::from_secs(60)).age,
            Some(Duration::ZERO),
            "a recovered sleep probe must restore freshness without a restart"
        );
    }

    #[test]
    fn a_service_without_sleep_enabled_keeps_a_fresh_observation() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();
        let generation = probe.sleep.generation;

        // `NotApplicable` and `Unobserved` both carry `sleeping_level: None`; only the
        // latter is unproven, so sleep-disabled services must not age out.
        probe.record(
            true,
            true,
            reading(SleepFact::NotApplicable, generation, started),
            1,
            1,
        );
        assert_eq!(probe.status_at(started).age, Some(Duration::ZERO));

        // A service with no `/is_sleeping` to consult must also re-prove itself on the
        // first observation after a restart, rather than waiting out a deadline.
        let restarted = probe.sleep.begin_generation();
        assert_eq!(probe.status_at(started).age, None);
        probe.record(
            true,
            true,
            reading(
                SleepFact::NotApplicable,
                restarted,
                started + Duration::from_secs(1),
            ),
            1,
            1,
        );

        assert_eq!(
            probe.status_at(started + Duration::from_secs(1)).age,
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn a_started_process_cannot_inherit_the_proof_that_the_previous_one_was_absent() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();

        // The last authoritative probe of the previous process: systemd inactive, so
        // nothing is asleep and the fact is stamped now.
        probe.record(
            false,
            false,
            reading(SleepFact::Observed(None), probe.sleep.generation, started),
            1,
            1,
        );
        assert_eq!(probe.status_at(started).age, Some(Duration::ZERO));

        // `start` retires that proof; `/is_sleeping` then fails against the new process
        // while `/health` and `/v1/models` both succeed.
        let generation = probe.sleep.begin_generation();
        let unproven = probe.record(
            true,
            true,
            reading(
                SleepFact::Unobserved,
                generation,
                started + Duration::from_millis(50),
            ),
            1,
            1,
        );

        assert!(unproven.ready);
        assert_eq!(unproven.sleeping_level, None);
        assert_eq!(
            unproven.age, None,
            "the pre-start timestamp must not present an unproven process as freshly awake; \
             admission rejects a missing age as stale_probe"
        );

        // Only `/is_sleeping` itself restores admissibility.
        probe.record(
            true,
            true,
            reading(
                SleepFact::Observed(None),
                generation,
                started + Duration::from_millis(100),
            ),
            1,
            1,
        );
        assert_eq!(
            probe.status_at(started + Duration::from_millis(100)).age,
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn a_woken_process_cannot_inherit_the_proof_that_it_was_asleep() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();

        let asleep = probe.record(
            false,
            false,
            reading(
                SleepFact::Observed(Some(1)),
                probe.sleep.generation,
                started,
            ),
            1,
            1,
        );
        assert_eq!(asleep.sleeping_level, Some(1));
        assert_eq!(probe.status_at(started).age, Some(Duration::ZERO));

        // `wake` retires the asleep proof. A failed `/is_sleeping` with healthy HTTP
        // must not resolve to "awake, proved a moment ago": the awake claim would be
        // carrying the timestamp that proved the service asleep.
        let generation = probe.sleep.begin_generation();
        let unproven = probe.record(
            true,
            true,
            reading(
                SleepFact::Unobserved,
                generation,
                started + Duration::from_millis(50),
            ),
            1,
            1,
        );

        assert_eq!(unproven.sleeping_level, None);
        assert_eq!(unproven.age, None);
    }

    #[test]
    fn an_observation_of_a_superseded_process_is_discarded() {
        let started = Instant::now();
        let mut probe = ProbeDebounce::default();
        // An observation reads the generation before it probes, so a lifecycle command
        // that lands while `/is_sleeping` is in flight supersedes its answer.
        let in_flight = probe.sleep.generation;
        probe.sleep.begin_generation();

        let superseded = probe.record(
            true,
            true,
            reading(SleepFact::Observed(None), in_flight, started),
            1,
            1,
        );

        assert_eq!(
            superseded.age, None,
            "an answer from the previous process must not prove the current one"
        );
    }

    #[test]
    fn a_sleep_proof_never_moves_its_value_and_timestamp_apart() {
        let started = Instant::now();
        let mut proof = SleepProof::default();
        let generation = proof.generation;

        proof.prove(generation, Some(1), started);
        assert_eq!(proof.level(), Some(1));
        assert_eq!(proof.age_at(started), Some(Duration::ZERO));

        // Every published value carries the instant that proved it, and vice versa:
        // there is no state in which one has advanced without the other.
        proof.prove(generation, None, started + Duration::from_secs(1));
        assert_eq!(proof.level(), None);
        assert_eq!(
            proof.age_at(started + Duration::from_secs(1)),
            Some(Duration::ZERO)
        );

        // A commanded sleep supersedes the awake proof with one of its own.
        proof.prove_new_generation(Some(2), started + Duration::from_secs(2));
        assert_eq!(proof.level(), Some(2));
        assert_eq!(
            proof.age_at(started + Duration::from_secs(2)),
            Some(Duration::ZERO)
        );

        // Retiring a generation drops the value and its timestamp together.
        proof.begin_generation();
        assert_eq!(proof.level(), None);
        assert_eq!(proof.age_at(started + Duration::from_secs(3)), None);
    }

    #[test]
    fn served_model_matches_canonical_among_additional_names() {
        let models = vec![
            "short-name".to_owned(),
            "example/model-30b".to_owned(),
            "legacy-name".to_owned(),
        ];

        assert!(served_model_matches(&models, "example/model-30b"));
    }

    #[test]
    fn served_model_does_not_match_aliases_without_canonical() {
        let models = vec!["short-name".to_owned(), "legacy-name".to_owned()];

        assert!(!served_model_matches(&models, "example/model-30b"));
    }

    #[test]
    fn systemd_main_pid_is_carried_into_runner_status() {
        let status = runner_systemd_status(UnitStatus {
            name: "vllm@example.service".into(),
            load_state: "loaded".into(),
            active_state: UnitActiveState::Active,
            sub_state: "running".into(),
            main_pid: Some(4242),
        });

        assert_eq!(status.state, SystemdState::Active);
        assert_eq!(status.sub_state.as_deref(), Some("running"));
        assert_eq!(status.main_pid, Some(4242));
    }

    #[test]
    fn post_readiness_unit_process_tree_must_use_exactly_assigned_gpus() {
        let config: RouterConfig =
            serde_yaml::from_str(include_str!("../config/example.yaml")).unwrap();
        let service = pool_definition(&config.services[0]);
        let assigned_uuid = service.gpu_ids[0].clone();
        let other_uuid = "GPU-other".to_string();
        let observation = |uuid: &str, pids: &[u32]| GpuObservation {
            device: GpuDevice {
                uuid: uuid.into(),
                index: 0,
                name: "GPU".into(),
                total_memory_bytes: 100,
            },
            used_memory_bytes: 1,
            free_memory_bytes: 99,
            reserved_memory_bytes: 0,
            compute_processes: pids
                .iter()
                .map(|pid| GpuProcess {
                    pid: *pid,
                    used_memory_bytes: None,
                })
                .collect(),
        };

        let correct = GpuSnapshot {
            available: true,
            unavailable_reason: None,
            devices: [
                (assigned_uuid.clone(), observation(&assigned_uuid, &[5000])),
                (other_uuid.clone(), observation(&other_uuid, &[])),
            ]
            .into(),
            drift: Vec::new(),
        };
        let unit_processes = BTreeSet::from([4242, 5000]);
        verify_unit_process_placement(&service, 4242, &unit_processes, &correct).unwrap();

        let wrong = GpuSnapshot {
            available: true,
            unavailable_reason: None,
            devices: [
                (assigned_uuid.clone(), observation(&assigned_uuid, &[])),
                (other_uuid.clone(), observation(&other_uuid, &[5000])),
            ]
            .into(),
            drift: Vec::new(),
        };
        let error =
            verify_unit_process_placement(&service, 4242, &unit_processes, &wrong).unwrap_err();
        assert!(error.contains("placement mismatch"));
        assert!(error.contains("GPU-other"));
    }

    #[test]
    fn expected_gpu_state_combines_reservations_and_allows_managed_main_pids() {
        let config: RouterConfig =
            serde_yaml::from_str(include_str!("../config/example.yaml")).unwrap();
        let awake = config.services[0].clone();
        let mut sleeping = awake.clone();
        sleeping.name = "sleeping-replica".into();
        sleeping.sleep.enabled = true;
        sleeping.sleep.residual_vram_bytes_per_gpu = 5;
        let mut missing = awake.clone();
        missing.name = "missing-replica".into();
        missing.placement.gpu_uuids = vec!["GPU-missing".into()];
        let awake_status = SystemdStatus {
            state: SystemdState::Active,
            sub_state: Some("running".into()),
            main_pid: Some(101),
        };
        let sleeping_status = SystemdStatus {
            state: SystemdState::Active,
            sub_state: Some("running".into()),
            main_pid: Some(202),
        };

        let expected = expected_gpu_state([
            (&awake, Some(awake_status), None),
            (&sleeping, Some(sleeping_status), Some(1)),
            (
                &missing,
                Some(SystemdStatus {
                    state: SystemdState::Inactive,
                    sub_state: Some("dead".into()),
                    main_pid: None,
                }),
                None,
            ),
        ]);
        let uuid = awake.placement.gpu_uuids[0].clone();
        let usage = &expected.devices[&uuid];

        assert_eq!(
            usage.maximum_used_memory_bytes,
            Some(awake.placement.vram_reservation_bytes_per_gpu + 5)
        );
        assert_eq!(usage.allowed_compute_pids, BTreeSet::from([101, 202]));
        assert!(!usage.allow_unexpected_compute_pids);

        let maximum = usage.maximum_used_memory_bytes.unwrap();
        let observed = [(
            uuid.clone(),
            GpuObservation {
                device: GpuDevice {
                    uuid: uuid.clone(),
                    index: 0,
                    name: "GPU".into(),
                    total_memory_bytes: maximum + 100,
                },
                used_memory_bytes: maximum + 1,
                free_memory_bytes: 99,
                reserved_memory_bytes: 0,
                compute_processes: [101, 202, 999]
                    .into_iter()
                    .map(|pid| GpuProcess {
                        pid,
                        used_memory_bytes: None,
                    })
                    .collect(),
            },
        )]
        .into();
        let drift = crate::gpu::detect_drift(&observed, &expected);
        assert!(drift.contains(&GpuDrift::MemoryReservationExceeded {
            uuid: uuid.clone(),
            observed_bytes: maximum + 1,
            maximum_bytes: maximum,
        }));
        assert!(drift.contains(&GpuDrift::UnexpectedComputeProcess {
            uuid: uuid.clone(),
            pid: 999,
        }));
        assert!(!drift.iter().any(|drift| matches!(
            drift,
            GpuDrift::UnexpectedComputeProcess { pid: 101 | 202, .. }
        )));
        assert!(drift.contains(&GpuDrift::MissingGpu {
            uuid: "GPU-missing".into(),
        }));

        let unknown = expected_gpu_state([(&awake, None, None)]);
        let unknown_usage = &unknown.devices[&awake.placement.gpu_uuids[0]];
        assert_eq!(unknown_usage.maximum_used_memory_bytes, None);
        assert!(unknown_usage.allow_unexpected_compute_pids);
    }

    #[test]
    fn managed_gpu_status_exposes_memory_process_and_missing_drift() {
        let config: RouterConfig =
            serde_yaml::from_str(include_str!("../config/example.yaml")).unwrap();
        let mut definition = pool_definition(&config.services[0]);
        definition.gpu_ids.push("GPU-missing".into());
        let uuid = definition.gpu_ids[0].clone();
        let observed = GpuObservation {
            device: GpuDevice {
                uuid: uuid.clone(),
                index: 0,
                name: "GPU".into(),
                total_memory_bytes: 100,
            },
            used_memory_bytes: 90,
            free_memory_bytes: 10,
            reserved_memory_bytes: 0,
            compute_processes: vec![
                GpuProcess {
                    pid: 101,
                    used_memory_bytes: Some(80),
                },
                GpuProcess {
                    pid: 999,
                    used_memory_bytes: Some(10),
                },
            ],
        };
        let snapshot = GpuSnapshot {
            available: true,
            unavailable_reason: None,
            devices: [(uuid.clone(), observed)].into(),
            drift: vec![
                GpuDrift::MemoryReservationExceeded {
                    uuid: uuid.clone(),
                    observed_bytes: 90,
                    maximum_bytes: 80,
                },
                GpuDrift::UnexpectedComputeProcess {
                    uuid: uuid.clone(),
                    pid: 999,
                },
                GpuDrift::MissingGpu {
                    uuid: "GPU-missing".into(),
                },
            ],
        };

        let statuses = gpu_statuses(&definition, &snapshot);
        let present = statuses.iter().find(|status| status.uuid == uuid).unwrap();
        assert_eq!(present.process_ids, [101, 999]);
        assert!(!present.healthy);
        assert_eq!(present.drift.len(), 2);
        let missing = statuses
            .iter()
            .find(|status| status.uuid == "GPU-missing")
            .unwrap();
        assert_eq!(
            missing.drift,
            vec![GpuDrift::MissingGpu {
                uuid: "GPU-missing".into()
            }]
        );
        assert!(!missing.healthy);
    }
}
