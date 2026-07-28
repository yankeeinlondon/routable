//! Generic lifecycle engine and placement scheduler.
//!
//! Runner adapters implement [`Runner`].  The proxy and management planes should only use
//! [`PoolManager`], keeping process supervision and vLLM-specific commands behind this seam.

mod managed;

pub use managed::{
    ManagedBuildError, ManagedVllmRunner, managed_pool_from_config, pool_definition,
};

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, watch};
use uuid::Uuid;

use crate::state::{
    ActiveGuard, Activity, GpuStatus, Lifecycle, ObservedState, ParkedGuard, ProbeStatus,
    ServiceStatus, StateStore, SystemdState, SystemdStatus,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepLevel {
    L1,
    L2,
}

impl SleepLevel {
    fn lifecycle(self) -> Lifecycle {
        match self {
            Self::L1 => Lifecycle::AsleepL1,
            Self::L2 => Lifecycle::AsleepL2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IdlePolicy {
    pub sleep_l1_after: Option<Duration>,
    pub sleep_l2_after: Option<Duration>,
    pub stop_after: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct ServiceDefinition {
    pub name: String,
    pub served_model: String,
    pub endpoint: SocketAddr,
    pub enabled: bool,
    /// Stable GPU UUIDs. Any overlap is an awake-placement conflict.
    pub gpu_ids: Vec<String>,
    /// Lower values are evicted first.
    pub priority: i32,
    pub evictable: bool,
    pub automatic_start: bool,
    pub parking_limit: u64,
    pub sleep_levels: Vec<SleepLevel>,
    pub startup_deadline: Duration,
    pub wake_deadline: Duration,
    pub readiness_poll_interval: Duration,
    pub drain_deadline: Duration,
    pub stop_deadline: Duration,
    pub idle: IdlePolicy,
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
}

impl ServiceDefinition {
    pub fn new(
        name: impl Into<String>,
        served_model: impl Into<String>,
        gpu_ids: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            served_model: served_model.into(),
            endpoint: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            enabled: true,
            gpu_ids,
            priority: 0,
            evictable: true,
            automatic_start: true,
            parking_limit: 128,
            sleep_levels: Vec::new(),
            startup_deadline: Duration::from_secs(300),
            wake_deadline: Duration::from_secs(60),
            readiness_poll_interval: Duration::from_millis(10),
            drain_deadline: Duration::from_secs(30),
            stop_deadline: Duration::from_secs(30),
            idle: IdlePolicy::default(),
            backoff_initial: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EnsureReadyOptions {
    pub deadline: Instant,
    /// Permit safe displacement. Data-plane demand normally sets this to true; an
    /// administrative start can set it explicitly.
    pub evict: bool,
}

impl EnsureReadyOptions {
    pub fn by(deadline: Instant) -> Self {
        Self {
            deadline,
            evict: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrainOptions {
    pub deadline: Instant,
}

#[derive(Clone, Debug)]
pub struct SleepOptions {
    pub level: SleepLevel,
    pub deadline: Instant,
}

#[derive(Clone, Debug)]
pub struct StopOptions {
    pub deadline: Instant,
    /// Continue with the runner stop after the drain deadline expires.
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct EnsureReadyRequest {
    pub service: String,
    pub deadline: Instant,
    pub evict: bool,
}

impl EnsureReadyRequest {
    pub fn by(service: impl Into<String>, deadline: Instant) -> Self {
        Self {
            service: service.into(),
            deadline,
            evict: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrainRequest {
    pub service: String,
    pub deadline: Instant,
}

#[derive(Clone, Debug)]
pub struct SleepRequest {
    pub service: String,
    pub level: SleepLevel,
    pub deadline: Instant,
}

#[derive(Clone, Debug)]
pub struct StopRequest {
    pub service: String,
    pub deadline: Instant,
    pub force: bool,
}

#[derive(Clone, Debug, Default)]
pub struct StatusRequest {
    /// `None` returns the whole pool snapshot.
    pub service: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransitionResponse {
    pub operation_id: Option<Uuid>,
    pub status: ServiceStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StatusResponse {
    pub services: Vec<ServiceStatus>,
}

#[derive(Clone, Debug)]
pub struct AdmissionRequest {
    pub model: String,
    pub deadline: Duration,
}

/// A selected ready backend. Holding this value holds its exact active-request slot.
pub struct Admission {
    pub service: String,
    pub endpoint: SocketAddr,
    active_guard: ActiveGuard,
}

impl Admission {
    pub fn active_guard(&self) -> &ActiveGuard {
        &self.active_guard
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunnerObservation {
    pub systemd: SystemdStatus,
    pub probe: ProbeStatus,
    pub gpus: Vec<GpuStatus>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct RunnerError {
    pub message: String,
}

impl RunnerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// The runner-owned lifecycle mechanics.
///
/// `wake` owns any level-specific sequence (including vLLM L2's reload-weights steps).
/// `start`/`wake` need not wait for readiness; the pool verifies readiness through
/// `observe` and only then runs `warmup`.
#[async_trait]
pub trait Runner: Send + Sync {
    async fn observe(&self, service: &ServiceDefinition) -> Result<RunnerObservation, RunnerError>;
    async fn start(&self, service: &ServiceDefinition) -> Result<(), RunnerError>;
    async fn stop(&self, service: &ServiceDefinition, force: bool) -> Result<(), RunnerError>;
    async fn sleep(
        &self,
        service: &ServiceDefinition,
        level: SleepLevel,
    ) -> Result<(), RunnerError>;
    async fn wake(&self, service: &ServiceDefinition, level: SleepLevel)
    -> Result<(), RunnerError>;

    async fn warmup(&self, _service: &ServiceDefinition) -> Result<(), RunnerError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize)]
pub enum PoolError {
    #[error("unknown service `{0}`")]
    UnknownService(String),
    #[error("automatic start is disabled for `{0}`")]
    AutomaticStartDisabled(String),
    #[error("service `{service}` has an incompatible transition in progress")]
    TransitionInProgress { service: String },
    #[error("service `{service}` cannot be placed: {reason}")]
    PlacementUnavailable { service: String, reason: String },
    #[error("{operation} for `{service}` exceeded its deadline")]
    Deadline {
        service: String,
        operation: &'static str,
    },
    #[error("runner operation for `{service}` failed: {message}")]
    Runner { service: String, message: String },
    #[error("sleep level {level:?} is not enabled for `{service}`")]
    SleepUnsupported { service: String, level: SleepLevel },
    #[error("service `{service}` failed: {message}")]
    TerminalFailure { service: String, message: String },
    #[error("parking queue for `{service}` is full")]
    ParkingFull { service: String },
}

impl PoolError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownService(_) => "unknown_service",
            Self::AutomaticStartDisabled(_) => "automatic_start_disabled",
            Self::TransitionInProgress { .. } => "transition_in_progress",
            Self::PlacementUnavailable { .. } => "placement_unavailable",
            Self::Deadline { .. } => "deadline_exceeded",
            Self::Runner { .. } => "runner_failure",
            Self::SleepUnsupported { .. } => "sleep_unsupported",
            Self::TerminalFailure { .. } => "terminal_failure",
            Self::ParkingFull { .. } => "parking_full",
        }
    }
}

#[async_trait]
pub trait PoolManager: Send + Sync {
    async fn ensure_ready(
        &self,
        request: EnsureReadyRequest,
    ) -> Result<TransitionResponse, PoolError>;
    async fn drain(&self, request: DrainRequest) -> Result<TransitionResponse, PoolError>;
    async fn sleep(&self, request: SleepRequest) -> Result<TransitionResponse, PoolError>;
    async fn stop(&self, request: StopRequest) -> Result<TransitionResponse, PoolError>;
    async fn status(&self, request: StatusRequest) -> Result<StatusResponse, PoolError>;
    async fn reconcile(&self, service: Option<String>) -> Result<StatusResponse, PoolError>;
    fn subscribe(&self, service: String) -> Result<watch::Receiver<ServiceStatus>, PoolError>;

    /// Atomically verify readiness and increment exact active accounting.
    async fn acquire_active(&self, service: String) -> Result<ActiveGuard, PoolError>;
    fn acquire_parked(&self, service: String) -> Result<ParkedGuard, PoolError>;
    async fn admit(&self, request: AdmissionRequest) -> Result<Admission, PoolError>;
}

pub struct Pool<R> {
    inner: Arc<PoolInner<R>>,
}

impl<R> Clone for Pool<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

struct PoolInner<R> {
    runner: Arc<R>,
    state: StateStore,
    services: HashMap<String, Arc<ServiceEntry>>,
    /// Tokio's mutex acquisition is FIFO. Holding this guard gives cold-placement demands
    /// a stable arrival order, including while an eligible victim is drained.
    placement_fifo: Mutex<()>,
    readiness_poll_interval: Duration,
}

struct ServiceEntry {
    definition: ServiceDefinition,
    activity: Activity,
    admission: Mutex<()>,
    flight: Mutex<Option<Flight>>,
    failures: AtomicU32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransitionKind {
    EnsureReady,
    Drain,
    Sleep(SleepLevel),
    Stop,
}

#[derive(Clone)]
struct Flight {
    kind: TransitionKind,
    operation_id: Uuid,
    result: watch::Receiver<FlightResult>,
}

#[derive(Clone, Debug)]
enum FlightResult {
    Pending,
    Complete(Box<Result<ObservedState, PoolError>>),
}

impl<R: Runner + 'static> Pool<R> {
    pub fn new(
        definitions: impl IntoIterator<Item = ServiceDefinition>,
        runner: R,
    ) -> Result<Self, PoolError> {
        let state = StateStore::new();
        let mut services = HashMap::new();
        for definition in definitions {
            let name = definition.name.clone();
            if services.contains_key(&name) {
                return Err(PoolError::PlacementUnavailable {
                    service: name,
                    reason: "duplicate service definition".to_string(),
                });
            }
            state.register(name.clone());
            services.insert(
                name,
                Arc::new(ServiceEntry {
                    definition,
                    activity: Activity::new(),
                    admission: Mutex::new(()),
                    flight: Mutex::new(None),
                    failures: AtomicU32::new(0),
                }),
            );
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                runner: Arc::new(runner),
                state,
                services,
                placement_fifo: Mutex::new(()),
                readiness_poll_interval: Duration::from_millis(10),
            }),
        })
    }

    pub fn definitions(&self) -> Vec<ServiceDefinition> {
        let mut definitions: Vec<_> = self
            .inner
            .services
            .values()
            .map(|entry| entry.definition.clone())
            .collect();
        definitions.sort_by(|a, b| a.name.cmp(&b.name));
        definitions
    }

    pub fn activity(&self, service: &str) -> Result<Activity, PoolError> {
        Ok(self.entry(service)?.activity.clone())
    }

    /// Concrete convenience API; the object-safe facade accepts owned request structs.
    pub async fn ensure_ready_status(
        &self,
        service: &str,
        options: EnsureReadyOptions,
    ) -> Result<ObservedState, PoolError> {
        let status = self.current_status(service)?;
        if status.lifecycle == Lifecycle::Running {
            return Ok(status);
        }
        let service_name = service.to_string();
        let task_options = options.clone();
        let receiver = self
            .get_or_spawn_flight(
                service,
                TransitionKind::EnsureReady,
                move |pool, operation_id| async move {
                    pool.do_ensure_ready(&service_name, task_options, operation_id)
                        .await
                },
            )
            .await?;
        Self::wait_for_flight(receiver, options.deadline, service, "ensure ready").await
    }

    pub async fn drain_status(
        &self,
        service: &str,
        options: DrainOptions,
    ) -> Result<ObservedState, PoolError> {
        let service_name = service.to_string();
        let task_options = options.clone();
        let receiver = self
            .get_or_spawn_flight(
                service,
                TransitionKind::Drain,
                move |pool, operation_id| async move {
                    pool.do_drain(&service_name, task_options, operation_id)
                        .await
                },
            )
            .await?;
        Self::wait_for_flight(receiver, options.deadline, service, "drain").await
    }

    pub async fn sleep_status(
        &self,
        service: &str,
        options: SleepOptions,
    ) -> Result<ObservedState, PoolError> {
        let service_name = service.to_string();
        let task_options = options.clone();
        let receiver = self
            .get_or_spawn_flight(
                service,
                TransitionKind::Sleep(options.level),
                move |pool, operation_id| async move {
                    pool.do_sleep(&service_name, task_options, operation_id)
                        .await
                },
            )
            .await?;
        Self::wait_for_flight(receiver, options.deadline, service, "sleep").await
    }

    pub async fn stop_status(
        &self,
        service: &str,
        options: StopOptions,
    ) -> Result<ObservedState, PoolError> {
        let service_name = service.to_string();
        let task_options = options.clone();
        let receiver = self
            .get_or_spawn_flight(
                service,
                TransitionKind::Stop,
                move |pool, operation_id| async move {
                    pool.do_stop(&service_name, task_options, operation_id)
                        .await
                },
            )
            .await?;
        Self::wait_for_flight(receiver, options.deadline, service, "stop").await
    }

    pub async fn reconcile(&self, service: &str) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        if entry.flight.lock().await.is_some() {
            return Err(PoolError::TransitionInProgress {
                service: service.to_string(),
            });
        }
        let operation_id = Uuid::new_v4();
        let generation = self
            .inner
            .state
            .begin_transition(service, Lifecycle::Reconciling, operation_id)
            .expect("registered service missing from state store");
        let observation = self
            .inner
            .runner
            .observe(&entry.definition)
            .await
            .map_err(|error| self.runner_error(service, error))?;
        let activity = entry.activity.snapshot();
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = lifecycle_from_observation(&observation);
                status.probe = observation.probe;
                status.systemd = observation.systemd;
                status.gpus = observation.gpus;
                status.active_requests = activity.active;
                status.parked_requests = activity.parked;
                status.idle_for = activity.idle_for;
                status.last_error = None;
            });
        self.current_status(service)
    }

    pub async fn reconcile_all(&self) -> Vec<Result<ObservedState, PoolError>> {
        let mut names: Vec<_> = self.inner.services.keys().cloned().collect();
        names.sort();
        futures_util::future::join_all(names.iter().map(|name| self.reconcile(name))).await
    }

    /// Apply one pass of the configured idle tiers.
    pub async fn reap_idle_once(&self) {
        let now = Instant::now();
        for entry in self.inner.services.values() {
            let activity = entry.activity.snapshot();
            if activity.active != 0 || activity.parked != 0 {
                continue;
            }
            let Ok(status) = self.current_status(&entry.definition.name) else {
                continue;
            };
            let idle = activity.idle_for;
            let policy = &entry.definition.idle;
            if policy.stop_after.is_some_and(|threshold| idle >= threshold)
                && matches!(
                    status.lifecycle,
                    Lifecycle::Running | Lifecycle::AsleepL1 | Lifecycle::AsleepL2
                )
            {
                let _ = self
                    .stop_status(
                        &entry.definition.name,
                        StopOptions {
                            deadline: now + entry.definition.stop_deadline,
                            force: false,
                        },
                    )
                    .await;
                continue;
            }
            if status.lifecycle == Lifecycle::AsleepL1
                && policy
                    .sleep_l2_after
                    .is_some_and(|threshold| idle >= threshold)
                && entry.definition.sleep_levels.contains(&SleepLevel::L2)
            {
                let _ = self
                    .sleep_status(
                        &entry.definition.name,
                        SleepOptions {
                            level: SleepLevel::L2,
                            deadline: now + entry.definition.drain_deadline,
                        },
                    )
                    .await;
                continue;
            }
            if status.lifecycle == Lifecycle::Running
                && policy
                    .sleep_l1_after
                    .is_some_and(|threshold| idle >= threshold)
                && entry.definition.sleep_levels.contains(&SleepLevel::L1)
            {
                let _ = self
                    .sleep_status(
                        &entry.definition.name,
                        SleepOptions {
                            level: SleepLevel::L1,
                            deadline: now + entry.definition.drain_deadline,
                        },
                    )
                    .await;
            }
        }
    }

    fn entry(&self, service: &str) -> Result<Arc<ServiceEntry>, PoolError> {
        self.inner
            .services
            .get(service)
            .cloned()
            .ok_or_else(|| PoolError::UnknownService(service.to_string()))
    }

    fn current_status(&self, service: &str) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        self.inner.state.refresh_activity(service, &entry.activity);
        self.inner
            .state
            .status(service)
            .ok_or_else(|| PoolError::UnknownService(service.to_string()))
    }

    fn runner_error(&self, service: &str, error: RunnerError) -> PoolError {
        PoolError::Runner {
            service: service.to_string(),
            message: error.message,
        }
    }

    async fn wait_for_flight(
        mut result: watch::Receiver<FlightResult>,
        deadline: Instant,
        service: &str,
        operation: &'static str,
    ) -> Result<ObservedState, PoolError> {
        loop {
            let current = result.borrow().clone();
            if let FlightResult::Complete(outcome) = current {
                return *outcome;
            }
            if tokio::time::timeout_at(deadline.into(), result.changed())
                .await
                .is_err()
            {
                // The executor owns the same deadline and may be restoring a pre-transition
                // state. Give that bounded cleanup a scheduling turn before returning.
                if tokio::time::timeout(Duration::from_millis(25), result.changed())
                    .await
                    .is_ok()
                {
                    if let FlightResult::Complete(outcome) = result.borrow().clone() {
                        return *outcome;
                    }
                }
                return Err(PoolError::Deadline {
                    service: service.to_string(),
                    operation,
                });
            }
        }
    }

    async fn get_or_spawn_flight<F, Fut>(
        &self,
        service: &str,
        kind: TransitionKind,
        task: F,
    ) -> Result<watch::Receiver<FlightResult>, PoolError>
    where
        F: FnOnce(Self, Uuid) -> Fut + Send + 'static,
        Fut: Future<Output = Result<ObservedState, PoolError>> + Send + 'static,
    {
        let entry = self.entry(service)?;
        let mut slot = entry.flight.lock().await;
        if let Some(existing) = slot.as_ref() {
            if existing.kind == kind {
                return Ok(existing.result.clone());
            }
            return Err(PoolError::TransitionInProgress {
                service: service.to_string(),
            });
        }

        let operation_id = Uuid::new_v4();
        let (sender, receiver) = watch::channel(FlightResult::Pending);
        *slot = Some(Flight {
            kind,
            operation_id,
            result: receiver.clone(),
        });
        drop(slot);

        let pool = self.clone();
        let service_name = service.to_string();
        tokio::spawn(async move {
            let outcome = task(pool.clone(), operation_id).await;
            let entry = pool
                .inner
                .services
                .get(&service_name)
                .expect("flight service disappeared");
            let mut slot = entry.flight.lock().await;
            if slot
                .as_ref()
                .is_some_and(|flight| flight.operation_id == operation_id)
            {
                *slot = None;
            }
            drop(slot);
            sender.send_replace(FlightResult::Complete(Box::new(outcome)));
        });
        Ok(receiver)
    }

    async fn do_ensure_ready(
        &self,
        service: &str,
        options: EnsureReadyOptions,
        operation_id: Uuid,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let initial = self.current_status(service)?;
        if initial.lifecycle == Lifecycle::Running {
            return Ok(initial);
        }
        if !entry.definition.automatic_start
            && matches!(
                initial.lifecycle,
                Lifecycle::Stopped | Lifecycle::Unknown | Lifecycle::Failed
            )
        {
            return Err(PoolError::AutomaticStartDisabled(service.to_string()));
        }

        let from_sleep = match initial.lifecycle {
            Lifecycle::AsleepL1 => Some(SleepLevel::L1),
            Lifecycle::AsleepL2 => Some(SleepLevel::L2),
            Lifecycle::Failed | Lifecycle::Backoff => {
                return Err(PoolError::TerminalFailure {
                    service: service.to_string(),
                    message: initial
                        .last_error
                        .unwrap_or_else(|| "service is in failure backoff".to_string()),
                });
            }
            _ => None,
        };

        // The FIFO guard intentionally spans placement, safe eviction, and the cold command.
        // This makes contention order deterministic instead of allowing later starts to barge.
        let _placement_turn = timeout_future(
            options.deadline,
            self.inner.placement_fifo.lock(),
            service,
            "placement",
        )
        .await?;
        self.make_placement(service, &options).await?;

        // Queued demands do not claim their GPUs. The FIFO holder publishes its claim only
        // after safe placement, preventing two queued starts for one GPU from blocking each
        // other as phantom `starting` owners.
        let transition = if from_sleep.is_some() {
            Lifecycle::Waking
        } else {
            Lifecycle::Starting
        };
        let generation = self
            .inner
            .state
            .begin_transition(service, transition, operation_id)
            .expect("registered service missing from state store");

        let runner_result = if let Some(level) = from_sleep {
            timeout_future(
                options.deadline,
                self.inner.runner.wake(&entry.definition, level),
                service,
                "wake",
            )
            .await
        } else {
            timeout_future(
                options.deadline,
                self.inner.runner.start(&entry.definition),
                service,
                "start",
            )
            .await
        };
        let runner_result = runner_result
            .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
        if let Err(error) = runner_result {
            self.record_failure(service, generation, operation_id, &error);
            self.schedule_backoff(service, operation_id);
            return Err(error);
        }

        let ready_observation = match self
            .wait_until_ready(service, &entry.definition, options.deadline)
            .await
        {
            Ok(observation) => observation,
            Err(error) => {
                self.record_failure(service, generation, operation_id, &error);
                self.schedule_backoff(service, operation_id);
                return Err(error);
            }
        };
        let warmup_result = timeout_future(
            options.deadline,
            self.inner.runner.warmup(&entry.definition),
            service,
            "warmup",
        )
        .await
        .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
        if let Err(error) = warmup_result {
            self.record_failure(service, generation, operation_id, &error);
            self.schedule_backoff(service, operation_id);
            return Err(error);
        }

        entry.failures.store(0, Ordering::Release);
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = Lifecycle::Running;
                status.probe = ready_observation.probe;
                status.systemd = ready_observation.systemd;
                status.gpus = ready_observation.gpus;
                status.last_error = None;
            });
        self.current_status(service)
    }

    async fn make_placement(
        &self,
        service: &str,
        options: &EnsureReadyOptions,
    ) -> Result<(), PoolError> {
        loop {
            let entry = self.entry(service)?;
            let mut blockers: Vec<_> = self
                .inner
                .services
                .values()
                .filter(|candidate| candidate.definition.name != service)
                .filter(|candidate| {
                    candidate
                        .definition
                        .gpu_ids
                        .iter()
                        .any(|gpu| entry.definition.gpu_ids.contains(gpu))
                })
                .filter_map(|candidate| {
                    let status = self.current_status(&candidate.definition.name).ok()?;
                    status
                        .lifecycle
                        .owns_awake_placement()
                        .then_some((candidate.clone(), status))
                })
                .collect();
            if blockers.is_empty() {
                return Ok(());
            }

            blockers.sort_by(|(a_entry, a_status), (b_entry, b_status)| {
                a_entry
                    .definition
                    .priority
                    .cmp(&b_entry.definition.priority)
                    .then_with(|| b_status.idle_for.cmp(&a_status.idle_for))
                    .then_with(|| a_entry.definition.name.cmp(&b_entry.definition.name))
            });
            let victim = blockers.iter().find(|(candidate, status)| {
                candidate.definition.evictable
                    && candidate.activity.is_idle()
                    && !status.lifecycle.is_transitioning()
            });

            if options.evict {
                if let Some((victim, _)) = victim {
                    self.evict_for_placement(victim.clone(), options.deadline)
                        .await?;
                    continue;
                }
                // The oldest demand retains the FIFO turn while an ineligible owner drains
                // naturally; later cold starts cannot leapfrog it.
                if Instant::now() >= options.deadline {
                    return Err(PoolError::PlacementUnavailable {
                        service: service.to_string(),
                        reason: blocker_reason(&blockers),
                    });
                }
                tokio::time::sleep(
                    self.inner
                        .readiness_poll_interval
                        .min(options.deadline.saturating_duration_since(Instant::now())),
                )
                .await;
                continue;
            }

            return Err(PoolError::PlacementUnavailable {
                service: service.to_string(),
                reason: blocker_reason(&blockers),
            });
        }
    }

    async fn evict_for_placement(
        &self,
        victim: Arc<ServiceEntry>,
        deadline: Instant,
    ) -> Result<(), PoolError> {
        let service = victim.definition.name.clone();
        let mut flight_slot = victim.flight.lock().await;
        if flight_slot.is_some() || !victim.activity.is_idle() {
            return Err(PoolError::PlacementUnavailable {
                service,
                reason: "chosen victim became busy".to_string(),
            });
        }
        let operation_id = Uuid::new_v4();
        let (_sender, receiver) = watch::channel(FlightResult::Pending);
        *flight_slot = Some(Flight {
            kind: TransitionKind::Stop,
            operation_id,
            result: receiver,
        });
        drop(flight_slot);

        let outcome = self
            .do_stop(
                &service,
                StopOptions {
                    deadline,
                    force: false,
                },
                operation_id,
            )
            .await;
        let mut flight_slot = victim.flight.lock().await;
        if flight_slot
            .as_ref()
            .is_some_and(|flight| flight.operation_id == operation_id)
        {
            *flight_slot = None;
        }
        outcome.map(|_| ())
    }

    async fn wait_until_ready(
        &self,
        service: &str,
        definition: &ServiceDefinition,
        deadline: Instant,
    ) -> Result<RunnerObservation, PoolError> {
        loop {
            let observation = timeout_future(
                deadline,
                self.inner.runner.observe(definition),
                service,
                "readiness",
            )
            .await?
            .map_err(|error| self.runner_error(service, error))?;
            if observation.systemd.state == SystemdState::Active
                && observation.probe.ready
                && observation.probe.healthy
                && observation.probe.served_model_matches
                && observation.gpus.iter().all(|gpu| gpu.healthy)
            {
                return Ok(observation);
            }
            tokio::time::sleep(
                definition
                    .readiness_poll_interval
                    .min(deadline.saturating_duration_since(Instant::now())),
            )
            .await;
        }
    }

    async fn do_drain(
        &self,
        service: &str,
        options: DrainOptions,
        operation_id: Uuid,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let previous = self.current_status(service)?.lifecycle;
        if matches!(
            previous,
            Lifecycle::Stopped | Lifecycle::AsleepL1 | Lifecycle::AsleepL2
        ) {
            return self.current_status(service);
        }
        let admission = entry.admission.lock().await;
        let generation = self
            .inner
            .state
            .begin_transition(service, Lifecycle::Draining, operation_id)
            .expect("registered service missing from state store");
        drop(admission);
        if !entry.activity.wait_until_idle(options.deadline).await {
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = previous
                });
            return Err(PoolError::Deadline {
                service: service.to_string(),
                operation: "drain",
            });
        }
        self.current_status(service)
    }

    async fn do_sleep(
        &self,
        service: &str,
        options: SleepOptions,
        operation_id: Uuid,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        if !entry.definition.sleep_levels.contains(&options.level) {
            return Err(PoolError::SleepUnsupported {
                service: service.to_string(),
                level: options.level,
            });
        }
        let previous = self.current_status(service)?.lifecycle;
        if previous == options.level.lifecycle() {
            return self.current_status(service);
        }
        let admission = entry.admission.lock().await;
        let generation = self
            .inner
            .state
            .begin_transition(service, Lifecycle::Draining, operation_id)
            .expect("registered service missing from state store");
        drop(admission);
        if !entry.activity.wait_until_idle(options.deadline).await {
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = previous
                });
            return Err(PoolError::Deadline {
                service: service.to_string(),
                operation: "drain before sleep",
            });
        }
        let result = timeout_future(
            options.deadline,
            self.inner.runner.sleep(&entry.definition, options.level),
            service,
            "sleep",
        )
        .await
        .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
        if let Err(error) = result {
            self.record_failure(service, generation, operation_id, &error);
            return Err(error);
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = options.level.lifecycle()
            });
        self.current_status(service)
    }

    async fn do_stop(
        &self,
        service: &str,
        options: StopOptions,
        operation_id: Uuid,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let previous = self.current_status(service)?.lifecycle;
        if previous == Lifecycle::Stopped {
            return self.current_status(service);
        }
        let admission = entry.admission.lock().await;
        let generation = self
            .inner
            .state
            .begin_transition(service, Lifecycle::Draining, operation_id)
            .expect("registered service missing from state store");
        drop(admission);
        let drained = entry.activity.wait_until_idle(options.deadline).await;
        if !drained && !options.force {
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = previous
                });
            return Err(PoolError::Deadline {
                service: service.to_string(),
                operation: "drain before stop",
            });
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = Lifecycle::Stopping
            });
        let runner_deadline = if !drained && options.force {
            Instant::now() + entry.definition.stop_deadline
        } else {
            options.deadline
        };
        let result = timeout_future(
            runner_deadline,
            self.inner.runner.stop(&entry.definition, options.force),
            service,
            "stop",
        )
        .await
        .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
        if let Err(error) = result {
            self.record_failure(service, generation, operation_id, &error);
            return Err(error);
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = Lifecycle::Stopped;
                status.probe.ready = false;
                status.probe.healthy = false;
                status.systemd.state = SystemdState::Inactive;
            });
        self.current_status(service)
    }

    fn record_failure(
        &self,
        service: &str,
        generation: u64,
        operation_id: Uuid,
        error: &PoolError,
    ) {
        if let Some(entry) = self.inner.services.get(service) {
            entry.failures.fetch_add(1, Ordering::AcqRel);
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = Lifecycle::Failed;
                status.last_error = Some(error.to_string());
            });
    }

    fn schedule_backoff(&self, service: &str, failed_operation: Uuid) {
        let Some(entry) = self.inner.services.get(service).cloned() else {
            return;
        };
        let failures = entry.failures.load(Ordering::Acquire).min(31);
        let multiplier = 1_u32
            .checked_shl(failures.saturating_sub(1))
            .unwrap_or(u32::MAX);
        let backoff = entry
            .definition
            .backoff_initial
            .saturating_mul(multiplier)
            .min(entry.definition.backoff_max);
        // Equal failures should not synchronize a fleet into retrying at the same instant.
        let backoff = backoff.mul_f64(0.8 + rand::random::<f64>() * 0.4);
        let pool = self.clone();
        let service = service.to_string();
        tokio::spawn(async move {
            let Some(failed) = pool.inner.state.status(&service) else {
                return;
            };
            if failed.operation_id != Some(failed_operation)
                || failed.lifecycle != Lifecycle::Failed
            {
                return;
            }
            let generation = failed.generation;
            pool.inner.state.update_if_current(
                &service,
                generation,
                Some(failed_operation),
                |status| status.lifecycle = Lifecycle::Backoff,
            );
            tokio::time::sleep(backoff).await;
            let stopped = tokio::time::timeout(
                entry.definition.stop_deadline,
                pool.inner.runner.stop(&entry.definition, true),
            )
            .await;
            match stopped {
                Ok(Ok(())) => {
                    pool.inner.state.update_if_current(
                        &service,
                        generation,
                        Some(failed_operation),
                        |status| {
                            status.lifecycle = Lifecycle::Stopped;
                            status.systemd.state = SystemdState::Inactive;
                            status.probe.ready = false;
                            status.probe.healthy = false;
                        },
                    );
                }
                Ok(Err(error)) => {
                    pool.inner.state.update_if_current(
                        &service,
                        generation,
                        Some(failed_operation),
                        |status| {
                            status.lifecycle = Lifecycle::Failed;
                            status.last_error =
                                Some(format!("failure cleanup stop failed: {error}"));
                        },
                    );
                }
                Err(_) => {
                    pool.inner.state.update_if_current(
                        &service,
                        generation,
                        Some(failed_operation),
                        |status| {
                            status.lifecycle = Lifecycle::Failed;
                            status.last_error =
                                Some("failure cleanup stop exceeded its deadline".to_string());
                        },
                    );
                }
            }
        });
    }
}

#[async_trait]
impl<R: Runner + 'static> PoolManager for Pool<R> {
    async fn ensure_ready(
        &self,
        request: EnsureReadyRequest,
    ) -> Result<TransitionResponse, PoolError> {
        let status = self
            .ensure_ready_status(
                &request.service,
                EnsureReadyOptions {
                    deadline: request.deadline,
                    evict: request.evict,
                },
            )
            .await?;
        Ok(TransitionResponse {
            operation_id: status.operation_id,
            status,
        })
    }

    async fn drain(&self, request: DrainRequest) -> Result<TransitionResponse, PoolError> {
        let status = self
            .drain_status(
                &request.service,
                DrainOptions {
                    deadline: request.deadline,
                },
            )
            .await?;
        Ok(TransitionResponse {
            operation_id: status.operation_id,
            status,
        })
    }

    async fn sleep(&self, request: SleepRequest) -> Result<TransitionResponse, PoolError> {
        let status = self
            .sleep_status(
                &request.service,
                SleepOptions {
                    level: request.level,
                    deadline: request.deadline,
                },
            )
            .await?;
        Ok(TransitionResponse {
            operation_id: status.operation_id,
            status,
        })
    }

    async fn stop(&self, request: StopRequest) -> Result<TransitionResponse, PoolError> {
        let status = self
            .stop_status(
                &request.service,
                StopOptions {
                    deadline: request.deadline,
                    force: request.force,
                },
            )
            .await?;
        Ok(TransitionResponse {
            operation_id: status.operation_id,
            status,
        })
    }

    async fn status(&self, request: StatusRequest) -> Result<StatusResponse, PoolError> {
        let services = if let Some(service) = request.service {
            vec![self.current_status(&service)?]
        } else {
            for (name, entry) in &self.inner.services {
                self.inner.state.refresh_activity(name, &entry.activity);
            }
            self.inner.state.snapshot()
        };
        Ok(StatusResponse { services })
    }

    async fn reconcile(&self, service: Option<String>) -> Result<StatusResponse, PoolError> {
        let services = if let Some(service) = service {
            vec![Pool::reconcile(self, &service).await?]
        } else {
            let results = self.reconcile_all().await;
            let mut services = Vec::with_capacity(results.len());
            for result in results {
                services.push(result?);
            }
            services
        };
        Ok(StatusResponse { services })
    }

    fn subscribe(&self, service: String) -> Result<watch::Receiver<ServiceStatus>, PoolError> {
        self.inner
            .state
            .subscribe(&service)
            .ok_or(PoolError::UnknownService(service))
    }

    async fn acquire_active(&self, service: String) -> Result<ActiveGuard, PoolError> {
        let entry = self.entry(&service)?;
        let _admission = entry.admission.lock().await;
        let status = self.current_status(&service)?;
        if status.lifecycle != Lifecycle::Running {
            if status.lifecycle.is_transitioning() {
                return Err(PoolError::TransitionInProgress { service });
            }
            return Err(PoolError::PlacementUnavailable {
                service,
                reason: format!("service is {:?}", status.lifecycle),
            });
        }
        Ok(entry.activity.acquire_active())
    }

    fn acquire_parked(&self, service: String) -> Result<ParkedGuard, PoolError> {
        let entry = self.entry(&service)?;
        entry
            .activity
            .try_acquire_parked(entry.definition.parking_limit)
            .ok_or(PoolError::ParkingFull { service })
    }

    async fn admit(&self, request: AdmissionRequest) -> Result<Admission, PoolError> {
        let deadline = Instant::now() + request.deadline;
        let mut candidates: Vec<_> = self
            .inner
            .services
            .values()
            .filter(|entry| {
                entry.definition.enabled && entry.definition.served_model == request.model
            })
            .cloned()
            .collect();
        if candidates.is_empty() {
            return Err(PoolError::UnknownService(request.model));
        }
        candidates.sort_by(|a, b| {
            a.activity
                .active()
                .cmp(&b.activity.active())
                .then_with(|| a.definition.name.cmp(&b.definition.name))
        });

        // Fast path: least-inflight among replicas that remain ready while admission is
        // serialized against drain.
        for entry in &candidates {
            let admission = entry.admission.lock().await;
            let status = self.current_status(&entry.definition.name)?;
            if status.lifecycle == Lifecycle::Running {
                let active_guard = entry.activity.acquire_active();
                drop(admission);
                return Ok(Admission {
                    service: entry.definition.name.clone(),
                    endpoint: entry.definition.endpoint,
                    active_guard,
                });
            }
        }

        // Cold path: park against a deterministic replica. The parked guard is independent
        // of the detached transition flight, so caller cancellation cannot cancel startup.
        candidates.sort_by(|a, b| {
            a.activity
                .parked()
                .cmp(&b.activity.parked())
                .then_with(|| a.definition.name.cmp(&b.definition.name))
        });
        let target = candidates
            .into_iter()
            .find(|entry| entry.definition.automatic_start)
            .ok_or_else(|| PoolError::AutomaticStartDisabled(request.model.clone()))?;
        let parked = target
            .activity
            .try_acquire_parked(target.definition.parking_limit)
            .ok_or_else(|| PoolError::ParkingFull {
                service: target.definition.name.clone(),
            })?;
        self.ensure_ready_status(
            &target.definition.name,
            EnsureReadyOptions {
                deadline,
                evict: true,
            },
        )
        .await?;
        drop(parked);
        self.acquire_active(target.definition.name.clone())
            .await
            .map(|active_guard| Admission {
                service: target.definition.name.clone(),
                endpoint: target.definition.endpoint,
                active_guard,
            })
    }
}

fn lifecycle_from_observation(observation: &RunnerObservation) -> Lifecycle {
    match observation.systemd.state {
        SystemdState::Inactive => Lifecycle::Stopped,
        SystemdState::Failed => Lifecycle::Failed,
        SystemdState::Activating => Lifecycle::Starting,
        SystemdState::Deactivating => Lifecycle::Stopping,
        SystemdState::Active => match observation.probe.sleeping_level {
            Some(1) => Lifecycle::AsleepL1,
            Some(2) => Lifecycle::AsleepL2,
            _ if observation.probe.ready
                && observation.probe.healthy
                && observation.probe.served_model_matches
                && observation.gpus.iter().all(|gpu| gpu.healthy) =>
            {
                Lifecycle::Running
            }
            _ => Lifecycle::Stale,
        },
        SystemdState::Unknown => Lifecycle::Unknown,
    }
}

fn blocker_reason(blockers: &[(Arc<ServiceEntry>, ObservedState)]) -> String {
    blockers
        .iter()
        .map(|(entry, status)| {
            format!(
                "{} is {:?} (active={}, parked={})",
                entry.definition.name,
                status.lifecycle,
                entry.activity.active(),
                entry.activity.parked()
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

async fn timeout_future<F, T>(
    deadline: Instant,
    future: F,
    service: &str,
    operation: &'static str,
) -> Result<T, PoolError>
where
    F: Future<Output = T>,
{
    tokio::time::timeout_at(deadline.into(), future)
        .await
        .map_err(|_| PoolError::Deadline {
            service: service.to_string(),
            operation,
        })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use super::*;

    struct MockService {
        running: AtomicBool,
        sleeping: Mutex<Option<SleepLevel>>,
    }

    #[derive(Clone)]
    struct MockRunner {
        services: Arc<HashMap<String, Arc<MockService>>>,
        starts: Arc<AtomicUsize>,
        start_order: Arc<Mutex<Vec<String>>>,
        stops: Arc<AtomicUsize>,
        stop_order: Arc<Mutex<Vec<String>>>,
        start_delay: Duration,
        fail_start: Arc<AtomicBool>,
    }

    impl MockRunner {
        fn new(names: &[&str], running: &[&str]) -> Self {
            let services = names
                .iter()
                .map(|name| {
                    (
                        (*name).to_string(),
                        Arc::new(MockService {
                            running: AtomicBool::new(running.contains(name)),
                            sleeping: Mutex::new(None),
                        }),
                    )
                })
                .collect();
            Self {
                services: Arc::new(services),
                starts: Arc::new(AtomicUsize::new(0)),
                start_order: Arc::new(Mutex::new(Vec::new())),
                stops: Arc::new(AtomicUsize::new(0)),
                stop_order: Arc::new(Mutex::new(Vec::new())),
                start_delay: Duration::from_millis(20),
                fail_start: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    #[async_trait]
    impl Runner for MockRunner {
        async fn observe(
            &self,
            service: &ServiceDefinition,
        ) -> Result<RunnerObservation, RunnerError> {
            let state = &self.services[&service.name];
            let running = state.running.load(Ordering::Acquire);
            let sleeping = *state.sleeping.lock().await;
            Ok(RunnerObservation {
                systemd: SystemdStatus {
                    state: if running {
                        SystemdState::Active
                    } else {
                        SystemdState::Inactive
                    },
                    sub_state: None,
                    main_pid: running.then_some(42),
                },
                probe: ProbeStatus {
                    healthy: running && sleeping.is_none(),
                    ready: running && sleeping.is_none(),
                    served_model_matches: running,
                    sleeping_level: sleeping.map(|level| match level {
                        SleepLevel::L1 => 1,
                        SleepLevel::L2 => 2,
                    }),
                    consecutive_successes: u32::from(running),
                    consecutive_failures: u32::from(!running),
                    age: Some(Duration::ZERO),
                },
                gpus: service
                    .gpu_ids
                    .iter()
                    .map(|uuid| GpuStatus {
                        uuid: uuid.clone(),
                        used_memory_bytes: Some(if running { 1 } else { 0 }),
                        process_ids: if running { vec![42] } else { vec![] },
                        healthy: true,
                    })
                    .collect(),
            })
        }

        async fn start(&self, service: &ServiceDefinition) -> Result<(), RunnerError> {
            self.starts.fetch_add(1, Ordering::AcqRel);
            self.start_order.lock().await.push(service.name.clone());
            tokio::time::sleep(self.start_delay).await;
            if self.fail_start.load(Ordering::Acquire) {
                return Err(RunnerError::new("injected start failure"));
            }
            self.services[&service.name]
                .running
                .store(true, Ordering::Release);
            Ok(())
        }

        async fn stop(&self, service: &ServiceDefinition, _force: bool) -> Result<(), RunnerError> {
            self.stops.fetch_add(1, Ordering::AcqRel);
            self.stop_order.lock().await.push(service.name.clone());
            self.services[&service.name]
                .running
                .store(false, Ordering::Release);
            *self.services[&service.name].sleeping.lock().await = None;
            Ok(())
        }

        async fn sleep(
            &self,
            service: &ServiceDefinition,
            level: SleepLevel,
        ) -> Result<(), RunnerError> {
            *self.services[&service.name].sleeping.lock().await = Some(level);
            Ok(())
        }

        async fn wake(
            &self,
            service: &ServiceDefinition,
            _level: SleepLevel,
        ) -> Result<(), RunnerError> {
            *self.services[&service.name].sleeping.lock().await = None;
            Ok(())
        }
    }

    fn definition(name: &str, gpu: &str) -> ServiceDefinition {
        let mut definition = ServiceDefinition::new(name, name, vec![gpu.to_string()]);
        definition.sleep_levels = vec![SleepLevel::L1, SleepLevel::L2];
        definition.backoff_initial = Duration::from_secs(30);
        definition
    }

    async fn reconciled_pool(
        definitions: Vec<ServiceDefinition>,
        runner: MockRunner,
    ) -> Pool<MockRunner> {
        let pool = Pool::new(definitions, runner).unwrap();
        for result in pool.reconcile_all().await {
            result.unwrap();
        }
        pool
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fifty_concurrent_ensure_ready_calls_are_single_flight() {
        let runner = MockRunner::new(&["a"], &[]);
        let starts = runner.starts.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut tasks = Vec::new();
        for _ in 0..50 {
            let pool = pool.clone();
            tasks.push(tokio::spawn(async move {
                pool.ensure_ready_status("a", EnsureReadyOptions::by(deadline))
                    .await
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap().unwrap().lifecycle, Lifecycle::Running);
        }
        assert_eq!(starts.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn drain_waits_for_guards_and_restores_state_on_deadline() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let guard = pool.activity("a").unwrap().acquire_active();
        let error = pool
            .drain_status(
                "a",
                DrainOptions {
                    deadline: Instant::now() + Duration::from_millis(30),
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, PoolError::Deadline { .. }));
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );

        let deadline = Instant::now() + Duration::from_secs(1);
        let pool_for_drain = pool.clone();
        let task = tokio::spawn(async move {
            pool_for_drain
                .drain_status("a", DrainOptions { deadline })
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished());
        drop(guard);
        assert_eq!(task.await.unwrap().unwrap().lifecycle, Lifecycle::Draining);
    }

    #[tokio::test]
    async fn active_or_parked_owner_is_never_evicted() {
        let runner = MockRunner::new(&["owner", "new"], &["owner"]);
        let stops = runner.stops.clone();
        let mut owner = definition("owner", "gpu-0");
        owner.priority = -10;
        let pool = reconciled_pool(vec![owner, definition("new", "gpu-0")], runner).await;
        let active = pool.activity("owner").unwrap().acquire_active();
        let parked = pool.activity("owner").unwrap().acquire_parked();
        let result = pool
            .ensure_ready_status(
                "new",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_millis(40),
                    evict: true,
                },
            )
            .await;
        assert!(result.is_err());
        assert_eq!(stops.load(Ordering::Acquire), 0);
        assert_eq!(
            pool.current_status("owner").unwrap().lifecycle,
            Lifecycle::Running
        );
        drop(active);
        drop(parked);
    }

    #[tokio::test]
    async fn eligible_victim_uses_priority_then_lru_and_never_busy_service() {
        let runner = MockRunner::new(&["high", "low", "new"], &["high", "low"]);
        let stop_order = runner.stop_order.clone();
        let mut high = definition("high", "gpu-0");
        high.priority = 100;
        let mut low = definition("low", "gpu-0");
        low.priority = -100;
        let pool = reconciled_pool(vec![high, low, definition("new", "gpu-0")], runner).await;

        // This deliberately constructs drift (two observed owners) to prove victim ordering.
        let status = pool
            .ensure_ready_status(
                "new",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(status.lifecycle, Lifecycle::Running);
        assert_eq!(
            pool.current_status("low").unwrap().lifecycle,
            Lifecycle::Stopped
        );
        assert_eq!(
            pool.current_status("high").unwrap().lifecycle,
            Lifecycle::Stopped
        );
        assert_eq!(*stop_order.lock().await, vec!["low", "high"]);
    }

    #[tokio::test]
    async fn subscription_reports_reconciliation_and_transitions() {
        let runner = MockRunner::new(&["a"], &[]);
        let pool = Pool::new(vec![definition("a", "gpu-0")], runner).unwrap();
        let mut status = PoolManager::subscribe(&pool, "a".to_string()).unwrap();
        assert_eq!(status.borrow().lifecycle, Lifecycle::Reconciling);
        pool.reconcile("a").await.unwrap();
        status.changed().await.unwrap();
        assert_eq!(status.borrow().lifecycle, Lifecycle::Stopped);

        let deadline = Instant::now() + Duration::from_secs(1);
        pool.ensure_ready_status("a", EnsureReadyOptions::by(deadline))
            .await
            .unwrap();
        status.changed().await.unwrap();
        assert_eq!(status.borrow().lifecycle, Lifecycle::Running);
        assert!(status.borrow().generation >= 2);
        assert!(status.borrow().operation_id.is_some());
    }

    #[tokio::test]
    async fn terminal_start_failure_is_fanned_out_to_all_waiters() {
        let runner = MockRunner::new(&["a"], &[]);
        runner.fail_start.store(true, Ordering::Release);
        let starts = runner.starts.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut tasks = Vec::new();
        for _ in 0..12 {
            let pool = pool.clone();
            tasks.push(tokio::spawn(async move {
                pool.ensure_ready_status("a", EnsureReadyOptions::by(deadline))
                    .await
                    .unwrap_err()
            }));
        }
        let first = tasks.remove(0).await.unwrap();
        for task in tasks {
            assert_eq!(task.await.unwrap(), first);
        }
        assert_eq!(starts.load(Ordering::Acquire), 1);
        assert!(matches!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Failed | Lifecycle::Backoff
        ));
    }

    #[tokio::test]
    async fn object_safe_admission_selects_least_inflight_and_holds_raii_slot() {
        let runner = MockRunner::new(&["a", "b"], &["a", "b"]);
        let mut a = definition("a", "gpu-0");
        a.served_model = "shared".to_string();
        a.endpoint = "127.0.0.1:8101".parse().unwrap();
        let mut b = definition("b", "gpu-1");
        b.served_model = "shared".to_string();
        b.endpoint = "127.0.0.1:8102".parse().unwrap();
        let pool = reconciled_pool(vec![a, b], runner).await;
        let busy = PoolManager::acquire_active(&pool, "a".to_string())
            .await
            .unwrap();
        let manager: Arc<dyn PoolManager> = Arc::new(pool.clone());

        let admission = manager
            .admit(AdmissionRequest {
                model: "shared".to_string(),
                deadline: Duration::from_secs(1),
            })
            .await
            .unwrap();
        assert_eq!(admission.service, "b");
        assert_eq!(admission.endpoint.port(), 8102);
        assert_eq!(pool.activity("b").unwrap().active(), 1);
        drop(admission);
        assert_eq!(pool.activity("b").unwrap().active(), 0);
        drop(busy);

        let all = manager
            .status(StatusRequest { service: None })
            .await
            .unwrap();
        assert_eq!(all.services.len(), 2);
    }

    #[tokio::test]
    async fn idle_reaper_moves_through_configured_sleep_tier() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let mut service = definition("a", "gpu-0");
        service.idle.sleep_l1_after = Some(Duration::ZERO);
        let pool = reconciled_pool(vec![service], runner).await;
        pool.reap_idle_once().await;
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::AsleepL1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_gpu_contention_is_fifo_without_phantom_starting_deadlock() {
        let runner = MockRunner::new(&["first", "second"], &[]);
        let starts = runner.starts.clone();
        let start_order = runner.start_order.clone();
        let pool = reconciled_pool(
            vec![definition("first", "gpu-0"), definition("second", "gpu-0")],
            runner,
        )
        .await;
        let deadline = Instant::now() + Duration::from_secs(2);
        let first_pool = pool.clone();
        let first = tokio::spawn(async move {
            first_pool
                .ensure_ready_status("first", EnsureReadyOptions::by(deadline))
                .await
        });
        while starts.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let second_pool = pool.clone();
        let second = tokio::spawn(async move {
            second_pool
                .ensure_ready_status("second", EnsureReadyOptions::by(deadline))
                .await
        });

        assert_eq!(first.await.unwrap().unwrap().service, "first");
        assert_eq!(second.await.unwrap().unwrap().service, "second");
        assert_eq!(*start_order.lock().await, vec!["first", "second"]);
        assert_eq!(
            pool.current_status("first").unwrap().lifecycle,
            Lifecycle::Stopped
        );
        assert_eq!(
            pool.current_status("second").unwrap().lifecycle,
            Lifecycle::Running
        );
    }
}
