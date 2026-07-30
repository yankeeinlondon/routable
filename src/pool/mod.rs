//! Generic lifecycle engine and placement scheduler.
//!
//! Runner adapters implement [`Runner`].  The proxy and management planes should only use
//! [`PoolManager`], keeping process supervision and vLLM-specific commands behind this seam.

mod managed;

pub use managed::{
    ManagedBuildError, ManagedVllmRunner, managed_pool_from_config, pool_definition,
};

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::runtime::Handle;
use tokio::sync::{Mutex, watch};
use tracing::info;
use uuid::Uuid;

use crate::metrics;
use crate::state::{
    ActiveGuard, Activity, GpuStatus, Lifecycle, ObservedState, ParkedGuard, ProbeStatus,
    ServiceStatus, StateStore, SystemdState, SystemdStatus,
};

const MAX_CONCURRENT_RECONCILES: usize = 8;

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
    pub maximum_probe_age: Duration,
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
            maximum_probe_age: Duration::from_secs(10),
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
    /// Permit an explicit administrative start/wake to clear a terminal drain.
    pub reactivate_drained: bool,
}

impl EnsureReadyOptions {
    pub fn by(deadline: Instant) -> Self {
        Self {
            deadline,
            evict: true,
            reactivate_drained: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrainOptions {
    pub deadline: Instant,
    /// Complete the drain after the deadline even if active work remains.
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct SleepOptions {
    pub level: SleepLevel,
    pub deadline: Instant,
    /// After the drain deadline, force-stop instead of sleeping with active work.
    pub force: bool,
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
    /// Data-plane demand must leave this false; explicit administration may reactivate.
    pub reactivate_drained: bool,
}

impl EnsureReadyRequest {
    pub fn by(service: impl Into<String>, deadline: Instant) -> Self {
        Self {
            service: service.into(),
            deadline,
            evict: true,
            reactivate_drained: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrainRequest {
    pub service: String,
    pub deadline: Instant,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct SleepRequest {
    pub service: String,
    pub level: SleepLevel,
    pub deadline: Instant,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct StopRequest {
    pub service: String,
    pub deadline: Instant,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub enum LifecycleRequest {
    EnsureReady(EnsureReadyRequest),
    Drain(DrainRequest),
    Sleep(SleepRequest),
    Stop(StopRequest),
}

#[derive(Clone, Debug, Default)]
pub struct StatusRequest {
    /// `None` returns the whole pool snapshot.
    pub service: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransitionResponse {
    pub operation_id: Option<Uuid>,
    /// Active requests displaced or left attached by a forced lifecycle transition.
    pub affected_requests: u64,
    pub status: ServiceStatus,
}

pub struct TransitionHandle {
    operation_id: Uuid,
    service: String,
    result: watch::Receiver<FlightResult>,
    affected_requests: AffectedRequests,
    joined_existing: bool,
}

impl TransitionHandle {
    pub fn operation_id(&self) -> Uuid {
        self.operation_id
    }

    pub fn joined_existing(&self) -> bool {
        self.joined_existing
    }

    /// Wait for the terminal lifecycle result while retaining force-decision metadata,
    /// including when the transition itself fails.
    pub async fn wait_outcome(mut self) -> TransitionOutcome {
        loop {
            if let FlightResult::Complete(result) = self.result.borrow().clone() {
                return TransitionOutcome {
                    operation_id: self.operation_id,
                    affected_requests: self.affected_requests.get(),
                    result: *result,
                };
            }
            if self.result.changed().await.is_err() {
                return TransitionOutcome {
                    operation_id: self.operation_id,
                    affected_requests: self.affected_requests.get(),
                    result: Err(PoolError::TerminalFailure {
                        service: self.service,
                        message: "lifecycle transition ended without publishing a result".into(),
                    }),
                };
            }
        }
    }

    pub async fn wait(self) -> Result<TransitionResponse, PoolError> {
        let outcome = self.wait_outcome().await;
        outcome.result.map(|status| TransitionResponse {
            operation_id: Some(outcome.operation_id),
            affected_requests: outcome.affected_requests,
            status,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionOutcome {
    pub operation_id: Uuid,
    pub affected_requests: u64,
    pub result: Result<ServiceStatus, PoolError>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StatusResponse {
    pub services: Vec<ServiceStatus>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunnerObservation {
    pub systemd: SystemdStatus,
    pub probe: ProbeStatus,
    pub gpus: Vec<GpuStatus>,
    /// Set once HTTP readiness is reached but NVML shows the unit MainPID on
    /// GPUs other than exactly those assigned to the service.
    pub placement_error: Option<String>,
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
    #[error("service `{0}` is administratively drained")]
    ServiceDrained(String),
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
            Self::ServiceDrained(_) => "service_drained",
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
    /// Start or join a serialized lifecycle transition without waiting for completion.
    async fn begin_lifecycle(
        &self,
        request: LifecycleRequest,
    ) -> Result<TransitionHandle, PoolError>;
    async fn ensure_ready(
        &self,
        request: EnsureReadyRequest,
    ) -> Result<TransitionResponse, PoolError>;
    async fn drain(&self, request: DrainRequest) -> Result<TransitionResponse, PoolError>;
    async fn sleep(&self, request: SleepRequest) -> Result<TransitionResponse, PoolError>;
    async fn stop(&self, request: StopRequest) -> Result<TransitionResponse, PoolError>;
    async fn status(&self, request: StatusRequest) -> Result<StatusResponse, PoolError>;
    /// Read the published facts without touching the store.
    ///
    /// `status` refreshes activity accounting, which writes to the state store and wakes
    /// every watcher. A readiness probe runs at load-balancer frequency and must not push
    /// that work into the data plane, so it reads through here instead.
    async fn status_snapshot(&self) -> Result<StatusResponse, PoolError> {
        self.status(StatusRequest::default()).await
    }
    async fn reconcile(&self, service: Option<String>) -> Result<StatusResponse, PoolError>;
    fn subscribe(&self, service: String) -> Result<watch::Receiver<ServiceStatus>, PoolError>;

    /// Atomically verify readiness and increment exact active accounting.
    async fn acquire_active(&self, service: String) -> Result<ActiveGuard, PoolError>;
    fn acquire_parked(&self, service: String) -> Result<ParkedGuard, PoolError>;
}

/// What one snapshot of cached facts means for active admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionVerdict {
    Admissible,
    /// A lifecycle transition owns the service; a request may wait for it.
    Transitioning,
    NotRunning,
    /// Running, but the facts contradict themselves or are too old to trust.
    Inconsistent {
        /// Stable metric label for the rejection reason.
        label: &'static str,
        reason: &'static str,
    },
}

/// Decide whether cached facts admit an inference request, without touching any state.
///
/// The supported lifecycle path publishes Draining under the service's admission lock
/// before issuing a sleep command, so a sufficiently fresh ready observation is the
/// hot-path lease. This avoids serializing every inference request on `/is_sleeping`;
/// stale or internally inconsistent facts fail closed until bounded reconciliation
/// refreshes them.
///
/// Readiness reporting shares this predicate so `/readyz` can never advertise a fleet
/// that admission would reject. It therefore takes `now` and its own copy of the facts
/// instead of reading through the state store, whose refresh path mutates and notifies.
pub fn admission_verdict(
    status: &ObservedState,
    maximum_probe_age: Duration,
    now: SystemTime,
) -> AdmissionVerdict {
    if status.lifecycle != Lifecycle::Running {
        return if status.lifecycle.is_transitioning() {
            AdmissionVerdict::Transitioning
        } else {
            AdmissionVerdict::NotRunning
        };
    }
    // Ordered so the reported reason names the operator's actual problem: a
    // fleet-wide stale-observation outage must not be reported as "not ready".
    let probe_is_current = status
        .probe_age_at(now)
        .is_some_and(|age| age <= maximum_probe_age);
    if status.probe.sleeping_level.is_some() {
        AdmissionVerdict::Inconsistent {
            label: "asleep",
            reason: "cached runner readiness reports a sleeping backend",
        }
    } else if !probe_is_current {
        AdmissionVerdict::Inconsistent {
            label: "stale_probe",
            reason: "cached runner readiness is older than the configured maximum probe age",
        }
    } else if status.systemd.state != SystemdState::Active {
        AdmissionVerdict::Inconsistent {
            label: "systemd_not_active",
            reason: "cached systemd state is not active for a running service",
        }
    } else if !status.probe.ready || !status.probe.healthy {
        AdmissionVerdict::Inconsistent {
            label: "not_ready",
            reason: "cached runner readiness is incomplete for a running service",
        }
    } else if !status.probe.served_model_matches {
        AdmissionVerdict::Inconsistent {
            label: "served_model_mismatch",
            reason: "cached runner readiness reports an unexpected served model",
        }
    } else {
        AdmissionVerdict::Admissible
    }
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
    /// Detached lifecycle work stays on the runtime where the pool was
    /// constructed. In production that runtime lives for the whole process,
    /// independent of Pingora's service runtimes and their shutdown.
    task_runtime: Option<Handle>,
    state: StateStore,
    services: HashMap<String, Arc<ServiceEntry>>,
    /// Arrival-ordered demands. Only overlapping GPU sets constrain one another.
    placement_queue: Mutex<VecDeque<PlacementDemand>>,
    readiness_poll_interval: Duration,
}

struct PlacementDemand {
    operation_id: Uuid,
    service: String,
}

struct ServiceEntry {
    definition: ServiceDefinition,
    activity: Activity,
    admission: Mutex<()>,
    flight: Mutex<Option<Flight>>,
    failures: AtomicU32,
    /// Generation whose failure cleanup is currently armed, or zero.
    backoff_generation: AtomicU64,
}

/// The complete behaviour-changing identity of one in-flight lifecycle transition.
///
/// A joining caller does not get its own executor: it inherits whichever policy the
/// running flight was spawned with. So every option that changes what the flight does
/// to a backend — a force fallback that terminates active work, displacing a resident,
/// clearing a terminal drain latch — belongs in this record rather than in the caller's
/// options alone, and a caller asking for a different policy is refused instead of
/// silently adopting (or silently losing) one. Deadlines are deliberately excluded:
/// each caller waits on its own deadline, and no two arrivals ever compute the same
/// `Instant`, so including them would dissolve coalescing without protecting anything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransitionPolicy {
    EnsureReady {
        evict: bool,
        reactivate_drained: bool,
    },
    Drain {
        force: bool,
    },
    Sleep {
        level: SleepLevel,
        force: bool,
    },
    Stop {
        force: bool,
    },
}

impl TransitionPolicy {
    /// The pool cannot distinguish a `start` from a `wake` — both are one `EnsureReady`
    /// flight — so joined records correlate to the accepted record by `operation_id`
    /// rather than by matching the control plane's action vocabulary.
    fn audit_action(self) -> &'static str {
        match self {
            Self::EnsureReady { .. } => "ensure_ready",
            Self::Drain { .. } => "drain",
            Self::Sleep { .. } => "sleep",
            Self::Stop { .. } => "stop",
        }
    }

    /// Discriminating detail the joined record would otherwise drop. `None` omits the
    /// field entirely, so each record carries only what applies to its transition.
    fn audit_sleep_level(self) -> Option<&'static str> {
        match self {
            Self::Sleep {
                level: SleepLevel::L1,
                ..
            } => Some("l1"),
            Self::Sleep {
                level: SleepLevel::L2,
                ..
            } => Some("l2"),
            _ => None,
        }
    }

    fn audit_force(self) -> Option<bool> {
        match self {
            Self::Drain { force } | Self::Sleep { force, .. } | Self::Stop { force } => Some(force),
            Self::EnsureReady { .. } => None,
        }
    }

    fn audit_evict(self) -> Option<bool> {
        match self {
            Self::EnsureReady { evict, .. } => Some(evict),
            _ => None,
        }
    }

    fn audit_reactivate_drained(self) -> Option<bool> {
        match self {
            Self::EnsureReady {
                reactivate_drained, ..
            } => Some(reactivate_drained),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Flight {
    policy: TransitionPolicy,
    operation_id: Uuid,
    result: watch::Receiver<FlightResult>,
    affected_requests: AffectedRequests,
}

#[derive(Clone, Debug)]
enum FlightResult {
    Pending,
    Complete(Box<Result<ObservedState, PoolError>>),
}

#[derive(Clone, Default)]
struct AffectedRequests(Arc<AtomicU64>);

impl AffectedRequests {
    fn capture(&self, active_requests: u64) {
        self.0.store(active_requests, Ordering::Release);
    }

    fn get(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
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
                    backoff_generation: AtomicU64::new(0),
                }),
            );
        }
        Ok(Self {
            inner: Arc::new(PoolInner {
                runner: Arc::new(runner),
                task_runtime: Handle::try_current().ok(),
                state,
                services,
                placement_queue: Mutex::new(VecDeque::new()),
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
        let handle = self.ensure_ready_flight(service, options.clone()).await?;
        Self::wait_for_flight(handle.result, options.deadline, service, "ensure ready").await
    }

    async fn ensure_ready_flight(
        &self,
        service: &str,
        options: EnsureReadyOptions,
    ) -> Result<TransitionHandle, PoolError> {
        let service_name = service.to_string();
        let task_options = options;
        self.get_or_spawn_flight(
            service,
            TransitionPolicy::EnsureReady {
                evict: task_options.evict,
                reactivate_drained: task_options.reactivate_drained,
            },
            move |pool, operation_id, _affected_requests| async move {
                pool.do_ensure_ready(&service_name, task_options, operation_id)
                    .await
            },
        )
        .await
    }

    pub async fn drain_status(
        &self,
        service: &str,
        options: DrainOptions,
    ) -> Result<ObservedState, PoolError> {
        let handle = self.drain_flight(service, options.clone()).await?;
        Self::wait_for_flight(handle.result, options.deadline, service, "drain").await
    }

    async fn drain_flight(
        &self,
        service: &str,
        options: DrainOptions,
    ) -> Result<TransitionHandle, PoolError> {
        let service_name = service.to_string();
        let task_options = options;
        self.get_or_spawn_flight(
            service,
            TransitionPolicy::Drain {
                force: task_options.force,
            },
            move |pool, operation_id, affected_requests| async move {
                pool.do_drain(&service_name, task_options, operation_id, affected_requests)
                    .await
            },
        )
        .await
    }

    pub async fn sleep_status(
        &self,
        service: &str,
        options: SleepOptions,
    ) -> Result<ObservedState, PoolError> {
        let handle = self.sleep_flight(service, options.clone()).await?;
        Self::wait_for_flight(handle.result, options.deadline, service, "sleep").await
    }

    pub async fn stop_status(
        &self,
        service: &str,
        options: StopOptions,
    ) -> Result<ObservedState, PoolError> {
        let handle = self.stop_flight(service, options.clone()).await?;
        Self::wait_for_flight(handle.result, options.deadline, service, "stop").await
    }

    async fn sleep_flight(
        &self,
        service: &str,
        options: SleepOptions,
    ) -> Result<TransitionHandle, PoolError> {
        let service_name = service.to_string();
        let task_options = options;
        self.get_or_spawn_flight(
            service,
            TransitionPolicy::Sleep {
                level: task_options.level,
                force: task_options.force,
            },
            move |pool, operation_id, affected_requests| async move {
                pool.do_sleep(&service_name, task_options, operation_id, affected_requests)
                    .await
            },
        )
        .await
    }

    async fn stop_flight(
        &self,
        service: &str,
        options: StopOptions,
    ) -> Result<TransitionHandle, PoolError> {
        let service_name = service.to_string();
        let task_options = options;
        self.get_or_spawn_flight(
            service,
            TransitionPolicy::Stop {
                force: task_options.force,
            },
            move |pool, operation_id, affected_requests| async move {
                pool.do_stop(&service_name, task_options, operation_id, affected_requests)
                    .await
            },
        )
        .await
    }

    pub async fn reconcile(&self, service: &str) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        if entry.flight.lock().await.is_some() {
            return Err(PoolError::TransitionInProgress {
                service: service.to_string(),
            });
        }
        let current = self.current_status(service)?;
        if matches!(current.lifecycle, Lifecycle::Failed | Lifecycle::Backoff) {
            let observation = match self.inner.runner.observe(&entry.definition).await {
                Ok(observation) => observation,
                Err(error) => {
                    let error = self.runner_error(service, error);
                    self.inner.state.update_if_current(
                        service,
                        current.generation,
                        current.operation_id,
                        |status| status.last_error = Some(error.to_string()),
                    );
                    return Err(error);
                }
            };
            let activity = entry.activity.snapshot();
            let recovery_is_pending =
                entry.backoff_generation.load(Ordering::Acquire) == current.generation;
            let observed_inactive = observation.systemd.state == SystemdState::Inactive;
            self.inner.state.update_if_current(
                service,
                current.generation,
                current.operation_id,
                |status| {
                    if matches!(status.lifecycle, Lifecycle::Failed | Lifecycle::Backoff) {
                        status.probe = observation.probe;
                        status.systemd = observation.systemd;
                        status.gpus = observation.gpus;
                        status.active_requests = activity.active;
                        status.parked_requests = activity.parked;
                        status.idle_for = activity.idle_for;
                        if observed_inactive && !recovery_is_pending {
                            status.lifecycle = Lifecycle::Stopped;
                            status.probe.ready = false;
                            status.probe.healthy = false;
                        }
                    }
                },
            );
            return self.current_status(service);
        }
        // Serialize the publication of Reconciling with active admission. Once this state is
        // visible, no request can acquire a guard from the observation being replaced; if an
        // admission already holds the lock, it publishes its exact active count first.
        let admission = entry.admission.lock().await;
        let current = self.current_status(service)?;
        if matches!(current.lifecycle, Lifecycle::Failed | Lifecycle::Backoff) {
            drop(admission);
            return Ok(current);
        }
        let prior_lifecycle = current.lifecycle;
        let preserve_drained = prior_lifecycle == Lifecycle::Drained;
        let operation_id = Uuid::new_v4();
        let generation = self
            .inner
            .state
            .begin_transition(
                service,
                if preserve_drained {
                    Lifecycle::Drained
                } else {
                    Lifecycle::Reconciling
                },
                operation_id,
            )
            .expect("registered service missing from state store");
        drop(admission);
        let observation = match self.inner.runner.observe(&entry.definition).await {
            Ok(observation) => observation,
            Err(error) => {
                let error = self.runner_error(service, error);
                self.inner.state.update_if_current(
                    service,
                    generation,
                    Some(operation_id),
                    |status| {
                        status.lifecycle = prior_lifecycle;
                        status.last_error = Some(error.to_string());
                    },
                );
                return Err(error);
            }
        };
        let activity = entry.activity.snapshot();
        let placement_error = observation.placement_error.clone();
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = if preserve_drained {
                    Lifecycle::Drained
                } else {
                    lifecycle_from_observation(&observation)
                };
                status.probe = observation.probe;
                status.systemd = observation.systemd;
                status.gpus = observation.gpus;
                status.active_requests = activity.active;
                status.parked_requests = activity.parked;
                status.idle_for = activity.idle_for;
                status.last_error = placement_error;
            });
        self.current_status(service)
    }

    pub async fn reconcile_all(&self) -> Vec<Result<ObservedState, PoolError>> {
        let mut names: Vec<_> = self.inner.services.keys().cloned().collect();
        names.sort();
        let mut results: Vec<_> = futures_util::stream::iter(
            names
                .into_iter()
                .enumerate()
                .map(|(index, name)| async move { (index, self.reconcile(&name).await) }),
        )
        .buffer_unordered(MAX_CONCURRENT_RECONCILES)
        .collect()
        .await;
        results.sort_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, result)| result).collect()
    }

    /// Trigger one pass of the configured idle tiers.
    ///
    /// Lifecycle work runs in the existing per-service single-flight executor. Dropping the
    /// receiver here deliberately detaches the background policy from drain and runner
    /// deadlines, while the flight slot prevents later ticks from spawning duplicate work.
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
                && !(status.lifecycle == Lifecycle::Drained
                    && status.systemd.state == SystemdState::Inactive)
                && matches!(
                    status.lifecycle,
                    Lifecycle::Running
                        | Lifecycle::Drained
                        | Lifecycle::AsleepL1
                        | Lifecycle::AsleepL2
                )
            {
                let _receiver = self
                    .stop_flight(
                        &entry.definition.name,
                        StopOptions {
                            deadline: now + entry.definition.stop_deadline,
                            force: false,
                        },
                    )
                    .await;
                continue;
            }
            if (status.lifecycle == Lifecycle::AsleepL1
                || (status.lifecycle == Lifecycle::Drained
                    && status.probe.sleeping_level == Some(1)))
                && policy
                    .sleep_l2_after
                    .is_some_and(|threshold| idle >= threshold)
                && entry.definition.sleep_levels.contains(&SleepLevel::L2)
            {
                let _receiver = self
                    .sleep_flight(
                        &entry.definition.name,
                        SleepOptions {
                            level: SleepLevel::L2,
                            deadline: now + entry.definition.drain_deadline,
                            force: false,
                        },
                    )
                    .await;
                continue;
            }
            if (status.lifecycle == Lifecycle::Running
                || (status.lifecycle == Lifecycle::Drained && status_owns_awake_placement(&status)))
                && policy
                    .sleep_l1_after
                    .is_some_and(|threshold| idle >= threshold)
                && entry.definition.sleep_levels.contains(&SleepLevel::L1)
            {
                let _receiver = self
                    .sleep_flight(
                        &entry.definition.name,
                        SleepOptions {
                            level: SleepLevel::L1,
                            deadline: now + entry.definition.drain_deadline,
                            force: false,
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

    fn require_admissible_status(
        &self,
        service: &str,
        entry: &ServiceEntry,
    ) -> Result<ObservedState, PoolError> {
        let status = self.current_status(service)?;
        match admission_verdict(
            &status,
            entry.definition.maximum_probe_age,
            SystemTime::now(),
        ) {
            AdmissionVerdict::Admissible => Ok(status),
            AdmissionVerdict::Transitioning => {
                self.count_admission_rejection(service, "transition_in_progress");
                Err(PoolError::TransitionInProgress {
                    service: service.to_string(),
                })
            }
            AdmissionVerdict::NotRunning => {
                self.count_admission_rejection(service, "lifecycle_not_running");
                Err(PoolError::PlacementUnavailable {
                    service: service.to_string(),
                    reason: format!("service is {:?}", status.lifecycle),
                })
            }
            AdmissionVerdict::Inconsistent { label, reason } => {
                self.count_admission_rejection(service, label);
                Err(PoolError::PlacementUnavailable {
                    service: service.to_string(),
                    reason: reason.into(),
                })
            }
        }
    }

    fn count_admission_rejection(&self, service: &str, reason: &str) {
        metrics::global()
            .admission_rejections
            .with_label_values(&[service, reason])
            .inc();
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
        policy: TransitionPolicy,
        task: F,
    ) -> Result<TransitionHandle, PoolError>
    where
        F: FnOnce(Self, Uuid, AffectedRequests) -> Fut + Send + 'static,
        Fut: Future<Output = Result<ObservedState, PoolError>> + Send + 'static,
    {
        let entry = self.entry(service)?;
        let mut slot = entry.flight.lock().await;
        if let Some(existing) = slot.as_ref() {
            // Only a caller asking for exactly the running operation may join it; anything
            // else would execute a policy nobody requested. Data-plane demand is uniform,
            // so a stampede still collapses onto one flight.
            if existing.policy == policy {
                // The record describes the operation that will actually run, which is the
                // stored policy — never the joining request's copy of it.
                let stored = existing.policy;
                info!(
                    audit = true,
                    operation_id = %existing.operation_id,
                    service,
                    action = stored.audit_action(),
                    sleep_level = stored.audit_sleep_level(),
                    force = stored.audit_force(),
                    evict = stored.audit_evict(),
                    reactivate_drained = stored.audit_reactivate_drained(),
                    joined_existing = true,
                    "pool lifecycle request joined existing operation"
                );
                return Ok(TransitionHandle {
                    operation_id: existing.operation_id,
                    service: service.to_string(),
                    result: existing.result.clone(),
                    affected_requests: existing.affected_requests.clone(),
                    joined_existing: true,
                });
            }
            return Err(PoolError::TransitionInProgress {
                service: service.to_string(),
            });
        }

        let operation_id = Uuid::new_v4();
        let affected_requests = AffectedRequests::default();
        let (sender, receiver) = watch::channel(FlightResult::Pending);
        *slot = Some(Flight {
            policy,
            operation_id,
            result: receiver.clone(),
            affected_requests: affected_requests.clone(),
        });
        drop(slot);

        let pool = self.clone();
        let service_name = service.to_string();
        let task_affected_requests = affected_requests.clone();
        let task = async move {
            let outcome = task(pool.clone(), operation_id, task_affected_requests).await;
            // Counted here rather than at the call sites because this is the only place a
            // transition actually executes: joining callers inherit a running flight and
            // must not each register a transition that ran once. `result` reuses the
            // `PoolError` code vocabulary so a failure says why without a second metric.
            metrics::global()
                .transitions
                .with_label_values(&[
                    &service_name,
                    policy.audit_action(),
                    match &outcome {
                        Ok(_) => "success",
                        Err(error) => error.code(),
                    },
                ])
                .inc();
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
        };
        if let Some(runtime) = &self.inner.task_runtime {
            runtime.spawn(task);
        } else {
            tokio::spawn(task);
        }
        Ok(TransitionHandle {
            operation_id,
            service: service.to_string(),
            result: receiver,
            affected_requests,
            joined_existing: false,
        })
    }

    async fn do_ensure_ready(
        &self,
        service: &str,
        options: EnsureReadyOptions,
        operation_id: Uuid,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let initial = self.current_status(service)?;
        if initial.lifecycle == Lifecycle::Drained && !options.reactivate_drained {
            return Err(PoolError::ServiceDrained(service.to_string()));
        }
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
            Lifecycle::Drained => match initial.probe.sleeping_level {
                Some(1) => Some(SleepLevel::L1),
                Some(2) => Some(SleepLevel::L2),
                _ => None,
            },
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

        let transition = if from_sleep.is_some() {
            Lifecycle::Waking
        } else {
            Lifecycle::Starting
        };
        // Placement inspection and claim publication are atomic. The queue retains FIFO
        // only among overlapping GPU sets; the mutex is released before every slow runner,
        // readiness, warmup, eviction, or contention wait.
        let generation = self
            .claim_placement(service, &options, transition, operation_id)
            .await?;

        let runner_phase_started = Instant::now();
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
        if from_sleep.is_none() {
            observe_start_phase(
                service,
                "systemd_to_active",
                runner_phase_started,
                runner_result.is_ok(),
            );
        }
        if let Err(error) = runner_result {
            self.record_failure_and_schedule_backoff(service, generation, operation_id, &error);
            return Err(error);
        }

        let readiness_phase_started = Instant::now();
        let ready_result = self
            .wait_until_ready(service, &entry.definition, options.deadline)
            .await;
        if from_sleep.is_none() {
            observe_start_phase(
                service,
                "active_to_ready",
                readiness_phase_started,
                ready_result.is_ok(),
            );
        }
        let ready_observation = match ready_result {
            Ok(observation) => observation,
            Err(error) => {
                self.record_failure_and_schedule_backoff(service, generation, operation_id, &error);
                return Err(error);
            }
        };
        let warmup_phase_started = Instant::now();
        let warmup_result = timeout_future(
            options.deadline,
            self.inner.runner.warmup(&entry.definition),
            service,
            "warmup",
        )
        .await
        .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
        if from_sleep.is_none() {
            observe_start_phase(
                service,
                "warmup",
                warmup_phase_started,
                warmup_result.is_ok(),
            );
        }
        if let Err(error) = warmup_result {
            self.record_failure_and_schedule_backoff(service, generation, operation_id, &error);
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

    async fn claim_placement(
        &self,
        service: &str,
        options: &EnsureReadyOptions,
        transition: Lifecycle,
        operation_id: Uuid,
    ) -> Result<u64, PoolError> {
        let mut queue = timeout_future(
            options.deadline,
            self.inner.placement_queue.lock(),
            service,
            "placement",
        )
        .await?;
        queue.push_back(PlacementDemand {
            operation_id,
            service: service.to_string(),
        });
        drop(queue);

        loop {
            let entry = self.entry(service)?;
            let mut queue = match timeout_future(
                options.deadline,
                self.inner.placement_queue.lock(),
                service,
                "placement",
            )
            .await
            {
                Ok(queue) => queue,
                Err(error) => {
                    self.cancel_placement_demand(operation_id).await;
                    return Err(error);
                }
            };
            let position = queue
                .iter()
                .position(|demand| demand.operation_id == operation_id)
                .expect("active placement demand disappeared");
            let waiting_for_earlier_overlap = queue
                .iter()
                .take(position)
                .any(|demand| self.services_overlap(service, &demand.service));
            if waiting_for_earlier_overlap {
                drop(queue);
                if Instant::now() >= options.deadline {
                    self.cancel_placement_demand(operation_id).await;
                    return Err(PoolError::Deadline {
                        service: service.to_string(),
                        operation: "placement",
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
                    status_owns_awake_placement(&status).then_some((candidate.clone(), status))
                })
                .collect();
            if blockers.is_empty() {
                let generation = self
                    .inner
                    .state
                    .begin_transition(service, transition, operation_id)
                    .expect("registered service missing from state store");
                remove_placement_demand(&mut queue, operation_id);
                return Ok(generation);
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
                    let victim = victim.clone();
                    drop(queue);
                    if let Err(error) = self.evict_for_placement(victim, options.deadline).await {
                        self.cancel_placement_demand(operation_id).await;
                        return Err(error);
                    }
                    continue;
                }
                if Instant::now() >= options.deadline {
                    let reason = blocker_reason(&blockers);
                    remove_placement_demand(&mut queue, operation_id);
                    return Err(PoolError::PlacementUnavailable {
                        service: service.to_string(),
                        reason,
                    });
                }
                drop(queue);
                tokio::time::sleep(
                    self.inner
                        .readiness_poll_interval
                        .min(options.deadline.saturating_duration_since(Instant::now())),
                )
                .await;
                continue;
            }

            let reason = blocker_reason(&blockers);
            remove_placement_demand(&mut queue, operation_id);
            return Err(PoolError::PlacementUnavailable {
                service: service.to_string(),
                reason,
            });
        }
    }

    fn services_overlap(&self, first: &str, second: &str) -> bool {
        let Some(first) = self.inner.services.get(first) else {
            return false;
        };
        let Some(second) = self.inner.services.get(second) else {
            return false;
        };
        first
            .definition
            .gpu_ids
            .iter()
            .any(|gpu| second.definition.gpu_ids.contains(gpu))
    }

    async fn cancel_placement_demand(&self, operation_id: Uuid) {
        let mut queue = self.inner.placement_queue.lock().await;
        remove_placement_demand(&mut queue, operation_id);
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
        let affected_requests = AffectedRequests::default();
        let (_sender, receiver) = watch::channel(FlightResult::Pending);
        *flight_slot = Some(Flight {
            policy: TransitionPolicy::Stop { force: false },
            operation_id,
            result: receiver,
            affected_requests: affected_requests.clone(),
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
                affected_requests,
            )
            .await;
        // Eviction owns its flight slot directly instead of going through
        // `get_or_spawn_flight`, so it must count itself or a model that vanished to make
        // room for another would leave no transition record at all. It is labelled
        // `evict` rather than `stop` because "something displaced my model" and "my model
        // idled out" are the two questions this counter exists to tell apart.
        metrics::global()
            .transitions
            .with_label_values(&[
                &service,
                "evict",
                match &outcome {
                    Ok(_) => "success",
                    Err(error) => error.code(),
                },
            ])
            .inc();
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
            // `timeout_future` only fires while the observation itself is pending, so a
            // readiness condition that never becomes true would otherwise poll a fast
            // runner forever. The deadline bounds the wait, not just each observation.
            if Instant::now() >= deadline {
                return Err(PoolError::Deadline {
                    service: service.to_string(),
                    operation: "readiness",
                });
            }
            let observation = timeout_future(
                deadline,
                self.inner.runner.observe(definition),
                service,
                "readiness",
            )
            .await?
            .map_err(|error| self.runner_error(service, error))?;
            // A `None` age means no authoritative observation describes the process this
            // start or wake just produced. Completing readiness on health and model
            // alone would publish `Running` on a fact proved by a previous process, so
            // wait for the runner to re-prove it (bounded by `deadline`).
            if observation.systemd.state == SystemdState::Active
                && observation.probe.age.is_some()
                && observation.probe.ready
                && observation.probe.healthy
                && observation.probe.served_model_matches
            {
                if let Some(message) = &observation.placement_error {
                    return Err(PoolError::Runner {
                        service: service.to_string(),
                        message: message.clone(),
                    });
                }
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
        affected_requests: AffectedRequests,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let previous = self.current_status(service)?.lifecycle;
        if previous == Lifecycle::Drained {
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
                operation: "drain",
            });
        }
        if !drained {
            affected_requests.capture(entry.activity.active());
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = Lifecycle::Drained
            });
        self.current_status(service)
    }

    async fn do_sleep(
        &self,
        service: &str,
        options: SleepOptions,
        operation_id: Uuid,
        affected_requests: AffectedRequests,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        if !entry.definition.sleep_levels.contains(&options.level) {
            return Err(PoolError::SleepUnsupported {
                service: service.to_string(),
                level: options.level,
            });
        }
        let previous_status = self.current_status(service)?;
        let previous = previous_status.lifecycle;
        let requested_level = match options.level {
            SleepLevel::L1 => 1,
            SleepLevel::L2 => 2,
        };
        if previous == options.level.lifecycle()
            || (previous == Lifecycle::Drained
                && previous_status.systemd.state == SystemdState::Active
                && previous_status.probe.sleeping_level == Some(requested_level))
        {
            return Ok(previous_status);
        }
        let valid_source = match previous {
            Lifecycle::Running => true,
            Lifecycle::AsleepL1 => options.level == SleepLevel::L2,
            Lifecycle::Drained => {
                previous_status.systemd.state == SystemdState::Active
                    && match previous_status.probe.sleeping_level {
                        None => true,
                        Some(1) => options.level == SleepLevel::L2,
                        Some(_) => false,
                    }
            }
            _ => false,
        };
        if !valid_source {
            return Err(PoolError::PlacementUnavailable {
                service: service.to_string(),
                reason: format!(
                    "cannot enter sleep level {requested_level} from {:?}",
                    previous
                ),
            });
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
                operation: "drain before sleep",
            });
        }
        if !drained {
            affected_requests.capture(entry.activity.active());
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = Lifecycle::Stopping
                });
            let force_stop = tokio::time::timeout(
                entry.definition.stop_deadline,
                self.inner.runner.stop(&entry.definition, true),
            )
            .await
            .map_err(|_| PoolError::Deadline {
                service: service.to_string(),
                operation: "forced stop instead of unsafe sleep",
            })
            .and_then(|result| result.map_err(|error| self.runner_error(service, error)));
            if let Err(error) = force_stop {
                self.inner.state.update_if_current(
                    service,
                    generation,
                    Some(operation_id),
                    |status| {
                        status.lifecycle = Lifecycle::Drained;
                        status.last_error = Some(error.to_string());
                    },
                );
                return Err(error);
            }
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = Lifecycle::Drained;
                    status.probe.ready = false;
                    status.probe.healthy = false;
                    status.probe.sleeping_level = None;
                    status.systemd.state = SystemdState::Inactive;
                });
            return self.current_status(service);
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
            if previous == Lifecycle::Drained {
                self.inner.state.update_if_current(
                    service,
                    generation,
                    Some(operation_id),
                    |status| {
                        status.lifecycle = Lifecycle::Drained;
                        status.last_error = Some(error.to_string());
                    },
                );
            } else {
                self.record_failure_and_schedule_backoff(service, generation, operation_id, &error);
            }
            return Err(error);
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = if previous == Lifecycle::Drained {
                    Lifecycle::Drained
                } else {
                    options.level.lifecycle()
                };
                status.probe.sleeping_level = Some(requested_level);
                status.probe.ready = false;
                status.probe.healthy = false;
            });
        self.current_status(service)
    }

    async fn do_stop(
        &self,
        service: &str,
        options: StopOptions,
        operation_id: Uuid,
        affected_requests: AffectedRequests,
    ) -> Result<ObservedState, PoolError> {
        let entry = self.entry(service)?;
        let previous_status = self.current_status(service)?;
        let previous = previous_status.lifecycle;
        if previous == Lifecycle::Stopped
            || (previous == Lifecycle::Drained
                && previous_status.systemd.state == SystemdState::Inactive)
        {
            return Ok(previous_status);
        }
        let admission = entry.admission.lock().await;
        // Capture the exact force-stop blast radius while admission is closed.
        // Requests may finish during the drain window, but every request present
        // at this boundary was subjected to the administrative force decision.
        if options.force {
            affected_requests.capture(entry.activity.active());
        }
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
            if options.force || previous == Lifecycle::Drained {
                self.inner.state.update_if_current(
                    service,
                    generation,
                    Some(operation_id),
                    |status| {
                        status.lifecycle = Lifecycle::Drained;
                        status.last_error = Some(error.to_string());
                    },
                );
            } else {
                self.record_failure_and_schedule_backoff(service, generation, operation_id, &error);
            }
            return Err(error);
        }
        self.inner
            .state
            .update_if_current(service, generation, Some(operation_id), |status| {
                status.lifecycle = if options.force || previous == Lifecycle::Drained {
                    Lifecycle::Drained
                } else {
                    Lifecycle::Stopped
                };
                status.probe.ready = false;
                status.probe.healthy = false;
                status.probe.sleeping_level = None;
                status.systemd.state = SystemdState::Inactive;
            });
        self.current_status(service)
    }

    fn record_failure_and_schedule_backoff(
        &self,
        service: &str,
        generation: u64,
        operation_id: Uuid,
        error: &PoolError,
    ) {
        let Some(entry) = self.inner.services.get(service) else {
            return;
        };
        // Publish the recovery intent before Failed becomes observable so
        // reconciliation cannot mistake the handoff to the backoff task for an
        // orphaned failure.
        entry
            .backoff_generation
            .store(generation, Ordering::Release);
        let recorded =
            self.inner
                .state
                .update_if_current(service, generation, Some(operation_id), |status| {
                    status.lifecycle = Lifecycle::Failed;
                    status.last_error = Some(error.to_string());
                });
        if !recorded {
            let _ = entry.backoff_generation.compare_exchange(
                generation,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return;
        }
        entry.failures.fetch_add(1, Ordering::AcqRel);
        self.schedule_backoff(service, generation, operation_id);
    }

    fn schedule_backoff(&self, service: &str, generation: u64, failed_operation: Uuid) {
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
        let task = async move {
            struct PendingBackoff {
                generation: u64,
                marker: Arc<ServiceEntry>,
            }
            impl Drop for PendingBackoff {
                fn drop(&mut self) {
                    let _ = self.marker.backoff_generation.compare_exchange(
                        self.generation,
                        0,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                }
            }
            let _pending = PendingBackoff {
                generation,
                marker: entry.clone(),
            };
            let Some(failed) = pool.inner.state.status(&service) else {
                return;
            };
            if failed.operation_id != Some(failed_operation)
                || failed.generation != generation
                || failed.lifecycle != Lifecycle::Failed
            {
                return;
            }
            pool.inner.state.update_if_current(
                &service,
                generation,
                Some(failed_operation),
                |status| status.lifecycle = Lifecycle::Backoff,
            );
            tokio::time::sleep(backoff).await;
            let Some(backoff_status) = pool.inner.state.status(&service) else {
                return;
            };
            if backoff_status.generation != generation
                || backoff_status.operation_id != Some(failed_operation)
                || backoff_status.lifecycle != Lifecycle::Backoff
            {
                return;
            }
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
        };
        if let Some(runtime) = &self.inner.task_runtime {
            runtime.spawn(task);
        } else {
            tokio::spawn(task);
        }
    }
}

#[async_trait]
impl<R: Runner + 'static> PoolManager for Pool<R> {
    async fn begin_lifecycle(
        &self,
        request: LifecycleRequest,
    ) -> Result<TransitionHandle, PoolError> {
        match request {
            LifecycleRequest::EnsureReady(request) => {
                self.ensure_ready_flight(
                    &request.service,
                    EnsureReadyOptions {
                        deadline: request.deadline,
                        evict: request.evict,
                        reactivate_drained: request.reactivate_drained,
                    },
                )
                .await
            }
            LifecycleRequest::Drain(request) => {
                self.drain_flight(
                    &request.service,
                    DrainOptions {
                        deadline: request.deadline,
                        force: request.force,
                    },
                )
                .await
            }
            LifecycleRequest::Sleep(request) => {
                self.sleep_flight(
                    &request.service,
                    SleepOptions {
                        level: request.level,
                        deadline: request.deadline,
                        force: request.force,
                    },
                )
                .await
            }
            LifecycleRequest::Stop(request) => {
                self.stop_flight(
                    &request.service,
                    StopOptions {
                        deadline: request.deadline,
                        force: request.force,
                    },
                )
                .await
            }
        }
    }

    async fn ensure_ready(
        &self,
        request: EnsureReadyRequest,
    ) -> Result<TransitionResponse, PoolError> {
        self.ensure_ready_flight(
            &request.service,
            EnsureReadyOptions {
                deadline: request.deadline,
                evict: request.evict,
                reactivate_drained: request.reactivate_drained,
            },
        )
        .await?
        .wait()
        .await
    }

    async fn drain(&self, request: DrainRequest) -> Result<TransitionResponse, PoolError> {
        self.drain_flight(
            &request.service,
            DrainOptions {
                deadline: request.deadline,
                force: request.force,
            },
        )
        .await?
        .wait()
        .await
    }

    async fn sleep(&self, request: SleepRequest) -> Result<TransitionResponse, PoolError> {
        self.sleep_flight(
            &request.service,
            SleepOptions {
                level: request.level,
                deadline: request.deadline,
                force: request.force,
            },
        )
        .await?
        .wait()
        .await
    }

    async fn stop(&self, request: StopRequest) -> Result<TransitionResponse, PoolError> {
        self.stop_flight(
            &request.service,
            StopOptions {
                deadline: request.deadline,
                force: request.force,
            },
        )
        .await?
        .wait()
        .await
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

    async fn status_snapshot(&self) -> Result<StatusResponse, PoolError> {
        Ok(StatusResponse {
            services: self.inner.state.snapshot(),
        })
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
        self.require_admissible_status(&service, &entry)?;
        Ok(entry.activity.acquire_active())
    }

    fn acquire_parked(&self, service: String) -> Result<ParkedGuard, PoolError> {
        let entry = self.entry(&service)?;
        entry
            .activity
            .try_acquire_parked(entry.definition.parking_limit)
            .ok_or(PoolError::ParkingFull { service })
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
            _ if observation.placement_error.is_some() => Lifecycle::Failed,
            _ if observation.probe.ready
                && observation.probe.healthy
                && observation.probe.served_model_matches =>
            {
                Lifecycle::Running
            }
            _ => Lifecycle::Stale,
        },
        SystemdState::Unknown => Lifecycle::Unknown,
    }
}

fn status_owns_awake_placement(status: &ObservedState) -> bool {
    if status.lifecycle.owns_awake_placement() {
        return true;
    }
    // Lifecycle publication and runner observations are not one atomic operation. Release
    // placement only on an authoritative inactive or asleep fact; an inconsistent stopped,
    // drained, or asleep label with an awake/unknown underlying process must fail closed.
    status.systemd.state != SystemdState::Inactive && status.probe.sleeping_level.is_none()
}

fn remove_placement_demand(queue: &mut VecDeque<PlacementDemand>, operation_id: Uuid) {
    if let Some(position) = queue
        .iter()
        .position(|demand| demand.operation_id == operation_id)
    {
        queue.remove(position);
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

fn observe_start_phase(service: &str, phase: &'static str, started: Instant, succeeded: bool) {
    let result = if succeeded { "success" } else { "error" };
    metrics::global()
        .start_phase_seconds
        .with_label_values(&[service, phase, result])
        .observe(started.elapsed().as_secs_f64());
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
    use std::{
        io::{self, Write},
        sync::atomic::{AtomicBool, AtomicUsize},
    };

    use tokio::sync::Notify;

    use super::*;

    struct AuditWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for AuditWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("audit capture lock poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// Run `future` with a JSON audit subscriber installed and return what it emitted.
    ///
    /// `set_default` is thread-local, so callers must use a current-thread runtime; a
    /// work-stealing runtime could resume the awaited work on a thread without it.
    async fn capture_audit<F: Future>(future: F) -> String {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(move || AuditWriter(Arc::clone(&sink)))
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        future.await;
        drop(guard);
        let bytes = captured
            .lock()
            .expect("audit capture lock poisoned")
            .clone();
        String::from_utf8(bytes).expect("audit output should be UTF-8")
    }

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
        stop_forces: Arc<Mutex<Vec<bool>>>,
        sleep_order: Arc<Mutex<Vec<(String, SleepLevel)>>>,
        wakes: Arc<AtomicUsize>,
        observes: Arc<AtomicUsize>,
        observe_started: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        observe_release: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        start_started: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        start_release: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        stop_started: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        stop_release: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        start_delay: Duration,
        fail_observe: Arc<AtomicBool>,
        fail_start: Arc<AtomicBool>,
        fail_sleep: Arc<AtomicBool>,
        fail_stop: Arc<AtomicBool>,
        gpu_healthy: Arc<AtomicBool>,
        /// Report observations no authoritative sleep proof stands behind, as the
        /// managed runner does between a start or wake and the next `/is_sleeping`.
        sleep_unproven: Arc<AtomicBool>,
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
                stop_forces: Arc::new(Mutex::new(Vec::new())),
                sleep_order: Arc::new(Mutex::new(Vec::new())),
                wakes: Arc::new(AtomicUsize::new(0)),
                observes: Arc::new(AtomicUsize::new(0)),
                observe_started: Arc::new(Mutex::new(HashMap::new())),
                observe_release: Arc::new(Mutex::new(HashMap::new())),
                start_started: Arc::new(Mutex::new(HashMap::new())),
                start_release: Arc::new(Mutex::new(HashMap::new())),
                stop_started: Arc::new(Mutex::new(HashMap::new())),
                stop_release: Arc::new(Mutex::new(HashMap::new())),
                start_delay: Duration::from_millis(20),
                fail_observe: Arc::new(AtomicBool::new(false)),
                fail_start: Arc::new(AtomicBool::new(false)),
                fail_sleep: Arc::new(AtomicBool::new(false)),
                fail_stop: Arc::new(AtomicBool::new(false)),
                gpu_healthy: Arc::new(AtomicBool::new(true)),
                sleep_unproven: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    #[async_trait]
    impl Runner for MockRunner {
        async fn observe(
            &self,
            service: &ServiceDefinition,
        ) -> Result<RunnerObservation, RunnerError> {
            self.observes.fetch_add(1, Ordering::AcqRel);
            let started = self
                .observe_started
                .lock()
                .await
                .get(&service.name)
                .cloned();
            if let Some(started) = started {
                started.notify_one();
            }
            let release = self
                .observe_release
                .lock()
                .await
                .get(&service.name)
                .cloned();
            if let Some(release) = release {
                release.notified().await;
            }
            if self.fail_observe.load(Ordering::Acquire) {
                return Err(RunnerError::new("injected observation failure"));
            }
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
                    age: (!self.sleep_unproven.load(Ordering::Acquire)).then_some(Duration::ZERO),
                },
                gpus: service
                    .gpu_ids
                    .iter()
                    .map(|uuid| GpuStatus {
                        uuid: uuid.clone(),
                        used_memory_bytes: Some(if running { 1 } else { 0 }),
                        process_ids: if running { vec![42] } else { vec![] },
                        healthy: self.gpu_healthy.load(Ordering::Acquire),
                        drift: Vec::new(),
                    })
                    .collect(),
                placement_error: None,
            })
        }

        async fn start(&self, service: &ServiceDefinition) -> Result<(), RunnerError> {
            self.starts.fetch_add(1, Ordering::AcqRel);
            self.start_order.lock().await.push(service.name.clone());
            let started = self.start_started.lock().await.get(&service.name).cloned();
            if let Some(started) = started {
                started.notify_one();
            }
            let release = self.start_release.lock().await.get(&service.name).cloned();
            if let Some(release) = release {
                release.notified().await;
            }
            tokio::time::sleep(self.start_delay).await;
            if self.fail_start.load(Ordering::Acquire) {
                return Err(RunnerError::new("injected start failure"));
            }
            self.services[&service.name]
                .running
                .store(true, Ordering::Release);
            Ok(())
        }

        async fn stop(&self, service: &ServiceDefinition, force: bool) -> Result<(), RunnerError> {
            self.stops.fetch_add(1, Ordering::AcqRel);
            self.stop_order.lock().await.push(service.name.clone());
            self.stop_forces.lock().await.push(force);
            if self.fail_stop.swap(false, Ordering::AcqRel) {
                return Err(RunnerError::new("injected stop failure"));
            }
            let started = self.stop_started.lock().await.get(&service.name).cloned();
            if let Some(started) = started {
                started.notify_one();
            }
            let release = self.stop_release.lock().await.get(&service.name).cloned();
            if let Some(release) = release {
                release.notified().await;
            }
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
            self.sleep_order
                .lock()
                .await
                .push((service.name.clone(), level));
            if self.fail_sleep.load(Ordering::Acquire) {
                return Err(RunnerError::new("injected sleep failure"));
            }
            *self.services[&service.name].sleeping.lock().await = Some(level);
            Ok(())
        }

        async fn wake(
            &self,
            service: &ServiceDefinition,
            _level: SleepLevel,
        ) -> Result<(), RunnerError> {
            self.wakes.fetch_add(1, Ordering::AcqRel);
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

    async fn wait_for_status(
        pool: &Pool<MockRunner>,
        service: &str,
        predicate: impl Fn(&ObservedState) -> bool,
    ) -> ObservedState {
        let mut statuses = pool
            .inner
            .state
            .subscribe(service)
            .expect("test service must be registered");
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let current = statuses.borrow().clone();
                if predicate(&current) {
                    return current;
                }
                statuses
                    .changed()
                    .await
                    .expect("test pool must retain the status sender");
            }
        })
        .await
        .expect("expected service status was not published")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_reconcile_does_not_block_fast_service_observation() {
        let runner = MockRunner::new(&["fast", "slow"], &["fast", "slow"]);
        let controls = runner.clone();
        let pool = reconciled_pool(
            vec![definition("fast", "gpu-0"), definition("slow", "gpu-1")],
            runner,
        )
        .await;
        let initial_fast_generation = pool.current_status("fast").unwrap().generation;
        let slow_started = Arc::new(Notify::new());
        let slow_release = Arc::new(Notify::new());
        controls
            .observe_started
            .lock()
            .await
            .insert("slow".into(), slow_started.clone());
        controls
            .observe_release
            .lock()
            .await
            .insert("slow".into(), slow_release.clone());

        let reconcile_pool = pool.clone();
        let reconcile = tokio::spawn(async move { reconcile_pool.reconcile_all().await });
        tokio::time::timeout(Duration::from_secs(1), slow_started.notified())
            .await
            .expect("the slow observation should start");

        let fast = wait_for_status(&pool, "fast", |status| {
            status.generation > initial_fast_generation && status.lifecycle == Lifecycle::Running
        })
        .await;
        assert_eq!(fast.lifecycle, Lifecycle::Running);
        assert!(
            !reconcile.is_finished(),
            "the slow observation must still be held independently"
        );

        slow_release.notify_one();
        let results = reconcile.await.unwrap();
        assert!(results.into_iter().all(|result| result.is_ok()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconcile_all_bounds_in_flight_observations() {
        let names = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
        let runner = MockRunner::new(&names, &names);
        let controls = runner.clone();
        let definitions = names
            .iter()
            .enumerate()
            .map(|(index, name)| definition(name, &format!("gpu-{index}")))
            .collect();
        let pool = reconciled_pool(definitions, runner).await;
        controls.observes.store(0, Ordering::Release);

        let mut releases = Vec::new();
        for name in names {
            let release = Arc::new(Notify::new());
            controls
                .observe_release
                .lock()
                .await
                .insert(name.into(), release.clone());
            releases.push(release);
        }

        let reconcile_pool = pool.clone();
        let reconcile = tokio::spawn(async move { reconcile_pool.reconcile_all().await });
        tokio::time::timeout(Duration::from_secs(1), async {
            while controls.observes.load(Ordering::Acquire) < MAX_CONCURRENT_RECONCILES {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the reconcile window should fill");
        assert_eq!(
            controls.observes.load(Ordering::Acquire),
            MAX_CONCURRENT_RECONCILES
        );

        releases[0].notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while controls.observes.load(Ordering::Acquire) < names.len() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completing one observation should admit the next");
        for release in &releases[1..] {
            release.notify_one();
        }

        let results = reconcile.await.unwrap();
        assert!(results.into_iter().all(|result| result.is_ok()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn detached_idle_reap_does_not_block_tick_or_fast_service() {
        let runner = MockRunner::new(&["fast", "slow"], &["fast", "slow"]);
        let controls = runner.clone();
        let mut fast = definition("fast", "gpu-0");
        fast.idle.stop_after = Some(Duration::ZERO);
        let mut slow = definition("slow", "gpu-1");
        slow.idle.stop_after = Some(Duration::ZERO);
        slow.stop_deadline = Duration::from_secs(30);
        let pool = reconciled_pool(vec![fast, slow], runner).await;

        let slow_started = Arc::new(Notify::new());
        let slow_release = Arc::new(Notify::new());
        controls
            .stop_started
            .lock()
            .await
            .insert("slow".into(), slow_started.clone());
        controls
            .stop_release
            .lock()
            .await
            .insert("slow".into(), slow_release.clone());

        tokio::time::timeout(Duration::from_millis(500), pool.reap_idle_once())
            .await
            .expect("the tick must detach idle lifecycle work");
        tokio::time::timeout(Duration::from_secs(1), slow_started.notified())
            .await
            .expect("the slow stop should start in the background");
        let fast = wait_for_status(&pool, "fast", |status| {
            status.lifecycle == Lifecycle::Stopped
        })
        .await;
        assert_eq!(fast.systemd.state, SystemdState::Inactive);
        assert_eq!(
            pool.current_status("slow").unwrap().lifecycle,
            Lifecycle::Stopping
        );

        for _ in 0..16 {
            pool.reap_idle_once().await;
        }
        let stop_order = controls.stop_order.lock().await;
        assert_eq!(
            stop_order
                .iter()
                .filter(|service| *service == "slow")
                .count(),
            1
        );
        assert_eq!(
            stop_order
                .iter()
                .filter(|service| *service == "fast")
                .count(),
            1
        );
        drop(stop_order);

        slow_release.notify_one();
        wait_for_status(&pool, "slow", |status| {
            status.lifecycle == Lifecycle::Stopped
        })
        .await;
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
    async fn cold_start_records_each_successful_phase_once() {
        let service = "phase-metrics-success";
        let runner = MockRunner::new(&[service], &[]);
        let pool = reconciled_pool(vec![definition(service, "phase-metrics-gpu")], runner).await;

        pool.ensure_ready_status(
            service,
            EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
        )
        .await
        .unwrap();

        for phase in ["systemd_to_active", "active_to_ready", "warmup"] {
            assert_eq!(start_phase_sample_count(service, phase, "success"), 1);
            assert_eq!(start_phase_sample_count(service, phase, "error"), 0);
        }
    }

    #[tokio::test]
    async fn failed_cold_start_records_only_the_failed_completed_phase() {
        let service = "phase-metrics-start-error";
        let runner = MockRunner::new(&[service], &[]);
        runner.fail_start.store(true, Ordering::Release);
        let pool =
            reconciled_pool(vec![definition(service, "phase-metrics-error-gpu")], runner).await;

        let result = pool
            .ensure_ready_status(
                service,
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await;
        assert!(matches!(result, Err(PoolError::Runner { .. })));
        assert_eq!(
            start_phase_sample_count(service, "systemd_to_active", "error"),
            1
        );
        assert_eq!(
            start_phase_sample_count(service, "active_to_ready", "success"),
            0
        );
        assert_eq!(start_phase_sample_count(service, "warmup", "success"), 0);
    }

    #[tokio::test]
    async fn a_completed_transition_is_counted_once_with_its_outcome() {
        let service = "transition-metrics-success";
        let runner = MockRunner::new(&[service], &[]);
        let pool =
            reconciled_pool(vec![definition(service, "transition-metrics-gpu")], runner).await;

        pool.ensure_ready_status(
            service,
            EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
        )
        .await
        .unwrap();

        assert_eq!(transition_count(service, "ensure_ready", "success"), 1);

        pool.stop_status(
            service,
            StopOptions {
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();

        // Each transition kind is counted under its own label, so a service that started
        // and later stopped is distinguishable from one that only started.
        assert_eq!(transition_count(service, "stop", "success"), 1);
        assert_eq!(transition_count(service, "ensure_ready", "success"), 1);
    }

    #[tokio::test]
    async fn a_failed_transition_is_counted_under_its_error_code() {
        let service = "transition-metrics-failure";
        let runner = MockRunner::new(&[service], &[]);
        runner.fail_start.store(true, Ordering::Release);
        let pool = reconciled_pool(
            vec![definition(service, "transition-metrics-failure-gpu")],
            runner,
        )
        .await;

        let result = pool
            .ensure_ready_status(
                service,
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await;

        assert!(matches!(result, Err(PoolError::Runner { .. })));
        assert_eq!(
            transition_count(service, "ensure_ready", "runner_failure"),
            1
        );
        assert_eq!(transition_count(service, "ensure_ready", "success"), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joining_callers_do_not_inflate_the_transition_count() {
        let service = "transition-metrics-join";
        let runner = MockRunner::new(&[service], &[]);
        let pool = reconciled_pool(
            vec![definition(service, "transition-metrics-join-gpu")],
            runner,
        )
        .await;
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut tasks = Vec::new();
        for _ in 0..50 {
            let pool = pool.clone();
            tasks.push(tokio::spawn(async move {
                pool.ensure_ready_status(service, EnsureReadyOptions::by(deadline))
                    .await
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap().unwrap().lifecycle, Lifecycle::Running);
        }

        // The counter measures transitions executed, not callers served: fifty arrivals
        // collapse onto one flight and must read as one transition.
        assert_eq!(transition_count(service, "ensure_ready", "success"), 1);
    }

    fn transition_count(service: &str, transition: &str, result: &str) -> u64 {
        counter_value(
            "vllm_router_transitions_total",
            &[
                ("service", service),
                ("transition", transition),
                ("result", result),
            ],
        )
    }

    fn counter_value(name: &str, expected: &[(&str, &str)]) -> u64 {
        prometheus::gather()
            .into_iter()
            .find(|family| family.get_name() == name)
            .and_then(|family| {
                family
                    .get_metric()
                    .iter()
                    .find(|metric| {
                        let labels: HashMap<_, _> = metric
                            .get_label()
                            .iter()
                            .map(|label| (label.get_name(), label.get_value()))
                            .collect();
                        expected
                            .iter()
                            .all(|(name, value)| labels.get(name) == Some(value))
                    })
                    .cloned()
            })
            .map(|metric| metric.get_counter().get_value() as u64)
            .unwrap_or(0)
    }

    fn start_phase_sample_count(service: &str, phase: &str, result: &str) -> u64 {
        prometheus::gather()
            .into_iter()
            .find(|family| family.get_name() == "vllm_router_start_phase_duration_seconds")
            .and_then(|family| {
                family
                    .get_metric()
                    .iter()
                    .find(|metric| {
                        let labels: HashMap<_, _> = metric
                            .get_label()
                            .iter()
                            .map(|label| (label.get_name(), label.get_value()))
                            .collect();
                        labels.get("service") == Some(&service)
                            && labels.get("phase") == Some(&phase)
                            && labels.get("result") == Some(&result)
                    })
                    .cloned()
            })
            .map(|metric| metric.get_histogram().get_sample_count())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn lifecycle_handles_join_one_pool_id_and_reject_a_different_kind() {
        let runner = MockRunner::new(&["a"], &[]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let start_started = Arc::new(Notify::new());
        let start_release = Arc::new(Notify::new());
        controls
            .start_started
            .lock()
            .await
            .insert("a".into(), start_started.clone());
        controls
            .start_release
            .lock()
            .await
            .insert("a".into(), start_release.clone());
        let request = EnsureReadyRequest {
            service: "a".into(),
            deadline: Instant::now() + Duration::from_secs(1),
            evict: true,
            reactivate_drained: true,
        };

        let first =
            PoolManager::begin_lifecycle(&pool, LifecycleRequest::EnsureReady(request.clone()))
                .await
                .unwrap();
        assert!(!first.joined_existing());
        tokio::time::timeout(Duration::from_secs(1), start_started.notified())
            .await
            .expect("the authoritative pool flight should start");
        let mut second = None;
        let audit = capture_audit(async {
            second = Some(
                PoolManager::begin_lifecycle(&pool, LifecycleRequest::EnsureReady(request))
                    .await
                    .unwrap(),
            );
        })
        .await;
        let second = second.expect("the joining request should have completed");
        assert_eq!(first.operation_id(), second.operation_id());
        assert!(second.joined_existing());
        assert!(audit.contains(r#""audit":true"#), "{audit}");
        assert!(audit.contains(r#""action":"ensure_ready""#), "{audit}");
        assert!(audit.contains(r#""joined_existing":true"#), "{audit}");
        assert!(
            audit.contains(&format!(r#""operation_id":"{}""#, first.operation_id())),
            "{audit}"
        );
        // Discriminating fields are emitted only where they apply, so an ensure-ready
        // record carries its placement policy but neither a sleep level nor a force
        // decision.
        assert!(audit.contains(r#""evict":true"#), "{audit}");
        assert!(audit.contains(r#""reactivate_drained":true"#), "{audit}");
        assert!(!audit.contains(r#""sleep_level""#), "{audit}");
        assert!(!audit.contains(r#""force""#), "{audit}");

        let rejected = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            }),
        )
        .await;
        assert!(matches!(
            rejected,
            Err(PoolError::TransitionInProgress { .. })
        ));
        let in_progress = pool.current_status("a").unwrap();
        assert_eq!(in_progress.lifecycle, Lifecycle::Starting);
        assert_eq!(in_progress.operation_id, Some(first.operation_id()));

        let operation_id = first.operation_id();
        start_release.notify_one();
        let first_terminal = first.wait().await.unwrap();
        let second_terminal = second.wait().await.unwrap();
        assert_eq!(first_terminal.operation_id, Some(operation_id));
        assert_eq!(second_terminal.operation_id, Some(operation_id));
        assert_eq!(first_terminal.status.lifecycle, Lifecycle::Running);
        assert_eq!(second_terminal.status.operation_id, Some(operation_id));
    }

    // Single-threaded on purpose: `set_default` installs a thread-local subscriber, so a
    // work-stealing runtime could resume the audited await on a thread that cannot see it.
    #[tokio::test]
    async fn joined_audit_records_carry_their_discriminating_decision() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let stop_started = Arc::new(Notify::new());
        let stop_release = Arc::new(Notify::new());
        controls
            .stop_started
            .lock()
            .await
            .insert("a".into(), stop_started.clone());
        controls
            .stop_release
            .lock()
            .await
            .insert("a".into(), stop_release.clone());
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;

        // A sleep flight parked on its drain, joined by an identical request.
        let held = pool.activity("a").unwrap().acquire_active();
        let sleeping = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L2,
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            }),
        )
        .await
        .unwrap();
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;

        let audit = capture_audit(async {
            PoolManager::begin_lifecycle(
                &pool,
                LifecycleRequest::Sleep(SleepRequest {
                    service: "a".into(),
                    level: SleepLevel::L2,
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                }),
            )
            .await
            .unwrap()
        })
        .await;
        assert!(audit.contains(r#""action":"sleep""#), "{audit}");
        assert!(audit.contains(r#""sleep_level":"l2""#), "{audit}");
        assert!(audit.contains(r#""joined_existing":true"#), "{audit}");
        // The force decision is what the flight will do to active work after the drain
        // deadline, so it is part of the joined operation's identity and must appear.
        assert!(
            audit.contains(r#""force":false"#),
            "a sleep record must report the stored force decision: {audit}"
        );
        assert!(!audit.contains(r#""evict""#), "{audit}");
        drop(held);
        sleeping.wait_outcome().await;

        // A forced stop blocked inside the runner, joined by an identical request.
        let stopping = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Stop(StopRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_secs(1),
                force: true,
            }),
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(1), stop_started.notified())
            .await
            .expect("the forced stop should reach the blocked runner");

        let audit = capture_audit(async {
            PoolManager::begin_lifecycle(
                &pool,
                LifecycleRequest::Stop(StopRequest {
                    service: "a".into(),
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: true,
                }),
            )
            .await
            .unwrap()
        })
        .await;
        assert!(audit.contains(r#""action":"stop""#), "{audit}");
        assert!(audit.contains(r#""force":true"#), "{audit}");
        assert!(
            !audit.contains(r#""sleep_level""#),
            "a stop record must not claim a sleep level: {audit}"
        );
        stop_release.notify_one();
        stopping.wait_outcome().await;
    }

    #[tokio::test]
    async fn a_force_sleep_cannot_join_and_cannot_weaken_a_running_non_force_sleep() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let stops = runner.stops.clone();
        let sleep_order = runner.sleep_order.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let held = pool.activity("a").unwrap().acquire_active();

        let parked = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await
        .unwrap();
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;

        let escalated = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(120),
                force: true,
            }),
        )
        .await;
        assert!(
            matches!(escalated, Err(PoolError::TransitionInProgress { .. })),
            "a force sleep must not silently lose its fallback to a running non-force sleep"
        );
        let identical = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await
        .unwrap();
        assert!(identical.joined_existing());
        assert_eq!(identical.operation_id(), parked.operation_id());

        // The refused escalation left the running flight non-force: it expires without
        // ever force-stopping the request still holding the service.
        assert!(matches!(
            parked.wait_outcome().await.result,
            Err(PoolError::Deadline { .. })
        ));
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );
        assert_eq!(stops.load(Ordering::Acquire), 0);
        assert!(sleep_order.lock().await.is_empty());
        drop(held);
    }

    #[tokio::test]
    async fn a_non_force_sleep_cannot_join_a_running_force_sleep() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let stops = runner.stops.clone();
        let stop_forces = runner.stop_forces.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let held = pool.activity("a").unwrap().acquire_active();

        let forced = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(120),
                force: true,
            }),
        )
        .await
        .unwrap();
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;

        let safe = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Sleep(SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await;
        assert!(
            matches!(safe, Err(PoolError::TransitionInProgress { .. })),
            "a nominally safe sleep must not share an operation that force-stops active work"
        );

        // The force policy stayed with the caller that asked for it.
        let outcome = forced.wait_outcome().await;
        assert_eq!(outcome.result.unwrap().lifecycle, Lifecycle::Drained);
        assert_eq!(outcome.affected_requests, 1);
        assert_eq!(stops.load(Ordering::Acquire), 1);
        assert_eq!(*stop_forces.lock().await, vec![true]);
        drop(held);
    }

    #[tokio::test]
    async fn a_force_drain_cannot_join_and_cannot_weaken_a_running_non_force_drain() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let held = pool.activity("a").unwrap().acquire_active();

        let parked = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await
        .unwrap();
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;

        let escalated = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(120),
                force: true,
            }),
        )
        .await;
        assert!(matches!(
            escalated,
            Err(PoolError::TransitionInProgress { .. })
        ));
        let identical = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await
        .unwrap();
        assert!(identical.joined_existing());
        assert_eq!(identical.operation_id(), parked.operation_id());

        let outcome = parked.wait_outcome().await;
        assert!(matches!(outcome.result, Err(PoolError::Deadline { .. })));
        assert_eq!(outcome.affected_requests, 0);
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );
        drop(held);
    }

    #[tokio::test]
    async fn a_non_force_drain_cannot_join_a_running_force_drain() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let held = pool.activity("a").unwrap().acquire_active();

        let forced = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(120),
                force: true,
            }),
        )
        .await
        .unwrap();
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;

        let safe = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Drain(DrainRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(120),
                force: false,
            }),
        )
        .await;
        assert!(
            matches!(safe, Err(PoolError::TransitionInProgress { .. })),
            "a non-force drain must not inherit a drain that abandons active work"
        );

        let outcome = forced.wait_outcome().await;
        assert_eq!(outcome.result.unwrap().lifecycle, Lifecycle::Drained);
        assert_eq!(outcome.affected_requests, 1);
        drop(held);
    }

    #[tokio::test]
    async fn administrative_start_policies_never_share_data_plane_demand() {
        let runner = MockRunner::new(&["a"], &[]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let start_started = Arc::new(Notify::new());
        let start_release = Arc::new(Notify::new());
        controls
            .start_started
            .lock()
            .await
            .insert("a".into(), start_started.clone());
        controls
            .start_release
            .lock()
            .await
            .insert("a".into(), start_release.clone());

        let demand = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::EnsureReady(EnsureReadyRequest::by(
                "a",
                Instant::now() + Duration::from_secs(1),
            )),
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(1), start_started.notified())
            .await
            .expect("data-plane demand should own the flight");

        // The spec requires a non-evicting administrative start to fail fast rather than
        // ride demand that is allowed to displace a resident.
        for (evict, reactivate_drained) in [(false, true), (true, true), (false, false)] {
            let refused = PoolManager::begin_lifecycle(
                &pool,
                LifecycleRequest::EnsureReady(EnsureReadyRequest {
                    service: "a".into(),
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict,
                    reactivate_drained,
                }),
            )
            .await;
            assert!(
                matches!(refused, Err(PoolError::TransitionInProgress { .. })),
                "evict={evict} reactivate_drained={reactivate_drained} must not join demand"
            );
        }

        start_release.notify_one();
        assert_eq!(
            demand.wait().await.unwrap().status.lifecycle,
            Lifecycle::Running
        );
    }

    #[tokio::test]
    async fn data_plane_demand_never_joins_an_administrative_start() {
        let runner = MockRunner::new(&["a"], &[]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let start_started = Arc::new(Notify::new());
        let start_release = Arc::new(Notify::new());
        controls
            .start_started
            .lock()
            .await
            .insert("a".into(), start_started.clone());
        controls
            .start_release
            .lock()
            .await
            .insert("a".into(), start_release.clone());

        let administrative = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::EnsureReady(EnsureReadyRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_secs(1),
                evict: false,
                reactivate_drained: true,
            }),
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(1), start_started.notified())
            .await
            .expect("the administrative start should own the flight");

        // Demand must not acquire eviction rights it was refused, nor inherit the
        // administrative permission to clear a drain latch.
        let refused = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::EnsureReady(EnsureReadyRequest::by(
                "a",
                Instant::now() + Duration::from_secs(1),
            )),
        )
        .await;
        assert!(matches!(
            refused,
            Err(PoolError::TransitionInProgress { .. })
        ));
        let identical = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::EnsureReady(EnsureReadyRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_secs(1),
                evict: false,
                reactivate_drained: true,
            }),
        )
        .await
        .unwrap();
        assert!(identical.joined_existing());
        assert_eq!(identical.operation_id(), administrative.operation_id());

        start_release.notify_one();
        assert_eq!(
            administrative.wait().await.unwrap().status.lifecycle,
            Lifecycle::Running
        );
        identical.wait().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn identical_demand_still_collapses_onto_one_operation() {
        let runner = MockRunner::new(&["a"], &[]);
        let controls = runner.clone();
        let starts = runner.starts.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let start_started = Arc::new(Notify::new());
        let start_release = Arc::new(Notify::new());
        controls
            .start_started
            .lock()
            .await
            .insert("a".into(), start_started.clone());
        controls
            .start_release
            .lock()
            .await
            .insert("a".into(), start_release.clone());

        let first = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::EnsureReady(EnsureReadyRequest::by(
                "a",
                Instant::now() + Duration::from_secs(5),
            )),
        )
        .await
        .unwrap();
        assert!(!first.joined_existing());
        tokio::time::timeout(Duration::from_secs(1), start_started.notified())
            .await
            .expect("the first demand should own the flight");

        // Distinct arrival deadlines on purpose: a deadline is a per-caller wait, not part
        // of the operation policy, so it must never split the stampede.
        let mut joined = Vec::new();
        for _ in 0..49 {
            joined.push(
                PoolManager::begin_lifecycle(
                    &pool,
                    LifecycleRequest::EnsureReady(EnsureReadyRequest::by(
                        "a",
                        Instant::now() + Duration::from_secs(5),
                    )),
                )
                .await
                .unwrap(),
            );
        }
        let operation_id = first.operation_id();
        assert!(
            joined
                .iter()
                .all(|handle| handle.joined_existing() && handle.operation_id() == operation_id),
            "49 identical demands must share one operation id"
        );

        start_release.notify_one();
        assert_eq!(
            first.wait().await.unwrap().status.lifecycle,
            Lifecycle::Running
        );
        for handle in joined {
            assert_eq!(
                handle.wait().await.unwrap().operation_id,
                Some(operation_id)
            );
        }
        assert_eq!(starts.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn degraded_gpu_observation_does_not_block_readiness_or_active_acquisition() {
        let runner = MockRunner::new(&["a"], &[]);
        runner.gpu_healthy.store(false, Ordering::Release);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;

        pool.ensure_ready_status(
            "a",
            EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
        )
        .await
        .unwrap();
        let active = PoolManager::acquire_active(&pool, "a".into())
            .await
            .unwrap();
        let status = pool.current_status("a").unwrap();

        assert_eq!(status.lifecycle, Lifecycle::Running);
        assert!(!status.gpus[0].healthy);
        assert_eq!(pool.activity("a").unwrap().active(), 1);
        drop(active);

        let reconciled = pool.reconcile("a").await.unwrap();
        assert_eq!(reconciled.lifecycle, Lifecycle::Running);
        assert!(!reconciled.gpus[0].healthy);
        PoolManager::acquire_active(&pool, "a".into())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn active_admission_rejects_stale_or_sleeping_running_facts() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let mut service = definition("a", "gpu-0");
        service.maximum_probe_age = Duration::from_millis(20);
        let pool = reconciled_pool(vec![service], runner).await;

        let current = pool.current_status("a").unwrap();
        pool.inner.state.update_if_current(
            "a",
            current.generation,
            current.operation_id,
            |status| status.probe.age = Some(Duration::from_secs(1)),
        );
        assert!(matches!(
            PoolManager::acquire_active(&pool, "a".into()).await,
            Err(PoolError::PlacementUnavailable { .. })
        ));
        assert_eq!(pool.activity("a").unwrap().active(), 0);

        *controls.services["a"].sleeping.lock().await = Some(SleepLevel::L1);
        let asleep = pool.reconcile("a").await.unwrap();
        assert_eq!(asleep.lifecycle, Lifecycle::AsleepL1);
        pool.inner
            .state
            .update_if_current("a", asleep.generation, asleep.operation_id, |status| {
                status.lifecycle = Lifecycle::Running
            });
        assert!(matches!(
            PoolManager::acquire_active(&pool, "a".into()).await,
            Err(PoolError::PlacementUnavailable { .. })
        ));
        assert_eq!(pool.activity("a").unwrap().active(), 0);
    }

    #[tokio::test]
    async fn active_admission_rejects_a_running_service_with_no_standing_proof() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;

        // A start or wake retires the runner's sleep proof until `/is_sleeping` answers
        // for the new process. Healthy HTTP alone must not lease the backend: the fact
        // admission would be trusting was proved by a process that is gone.
        controls.sleep_unproven.store(true, Ordering::Release);
        let observed = pool.reconcile("a").await.unwrap();
        assert_eq!(observed.probe.age, None);

        let rejected = PoolManager::acquire_active(&pool, "a".into()).await;

        assert!(matches!(
            rejected,
            Err(PoolError::PlacementUnavailable { ref reason, .. })
                if reason.contains("maximum probe age")
        ));
        assert_eq!(pool.activity("a").unwrap().active(), 0);
    }

    #[tokio::test]
    async fn readiness_does_not_complete_without_a_current_process_sleep_proof() {
        let runner = MockRunner::new(&["a"], &[]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        controls.sleep_unproven.store(true, Ordering::Release);

        // The unit starts and serves `/health` and `/v1/models`, but nothing proves the
        // new process is awake, so readiness must run out its deadline instead of
        // publishing Running on the pre-start fact.
        let failed = PoolManager::ensure_ready(
            &pool,
            EnsureReadyRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(300),
                evict: false,
                reactivate_drained: false,
            },
        )
        .await;

        assert!(matches!(
            failed,
            Err(PoolError::Deadline {
                operation: "readiness",
                ..
            })
        ));
        assert_ne!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );
        assert!(controls.starts.load(Ordering::Acquire) >= 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconciliation_cannot_publish_over_an_admission_critical_section() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let observe_started = Arc::new(Notify::new());
        let observe_release = Arc::new(Notify::new());
        controls
            .observe_started
            .lock()
            .await
            .insert("a".into(), observe_started.clone());
        controls
            .observe_release
            .lock()
            .await
            .insert("a".into(), observe_release.clone());

        let entry = pool.entry("a").unwrap();
        let admission = entry.admission.lock().await;
        let reconcile_pool = pool.clone();
        let reconcile = tokio::spawn(async move { reconcile_pool.reconcile("a").await });
        assert!(
            tokio::time::timeout(Duration::from_millis(30), observe_started.notified())
                .await
                .is_err(),
            "reconciliation must not replace Running while admission is deciding"
        );
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );

        drop(admission);
        tokio::time::timeout(Duration::from_secs(1), observe_started.notified())
            .await
            .expect("reconciliation should proceed once admission releases");
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Reconciling
        );
        assert!(matches!(
            PoolManager::acquire_active(&pool, "a".into()).await,
            Err(PoolError::TransitionInProgress { .. })
        ));
        assert_eq!(pool.activity("a").unwrap().active(), 0);

        observe_release.notify_one();
        assert_eq!(
            reconcile.await.unwrap().unwrap().lifecycle,
            Lifecycle::Running
        );
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
                    force: false,
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
                .drain_status(
                    "a",
                    DrainOptions {
                        deadline,
                        force: false,
                    },
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished());
        drop(guard);
        assert_eq!(task.await.unwrap().unwrap().lifecycle, Lifecycle::Drained);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn force_drain_waits_for_deadline_then_preserves_active_work_and_reports_it() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let stops = runner.stops.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let guard = pool.activity("a").unwrap().acquire_active();
        let drain_pool = pool.clone();
        let drain = tokio::spawn(async move {
            PoolManager::drain(
                &drain_pool,
                DrainRequest {
                    service: "a".into(),
                    deadline: Instant::now() + Duration::from_millis(40),
                    force: true,
                },
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            !drain.is_finished(),
            "force must not skip the drain deadline"
        );
        let response = drain.await.unwrap().unwrap();
        assert_eq!(response.status.lifecycle, Lifecycle::Drained);
        assert_eq!(response.affected_requests, 1);
        assert_eq!(pool.activity("a").unwrap().active(), 1);
        assert_eq!(stops.load(Ordering::Acquire), 0);
        assert!(matches!(
            PoolManager::acquire_active(&pool, "a".into()).await,
            Err(PoolError::PlacementUnavailable { .. })
        ));
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn force_sleep_waits_then_force_stops_without_sleeping_active_work() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let stops = runner.stops.clone();
        let stop_forces = runner.stop_forces.clone();
        let sleep_order = runner.sleep_order.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let guard = pool.activity("a").unwrap().acquire_active();

        let refused = PoolManager::sleep(
            &pool,
            SleepRequest {
                service: "a".into(),
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_millis(20),
                force: false,
            },
        )
        .await;
        assert!(matches!(refused, Err(PoolError::Deadline { .. })));
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Running
        );
        assert!(sleep_order.lock().await.is_empty());
        assert_eq!(stops.load(Ordering::Acquire), 0);

        let sleep_pool = pool.clone();
        let forced = tokio::spawn(async move {
            PoolManager::sleep(
                &sleep_pool,
                SleepRequest {
                    service: "a".into(),
                    level: SleepLevel::L1,
                    deadline: Instant::now() + Duration::from_millis(40),
                    force: true,
                },
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            !forced.is_finished(),
            "force must not skip the drain deadline"
        );
        let response = forced.await.unwrap().unwrap();

        assert_eq!(response.status.lifecycle, Lifecycle::Drained);
        assert_eq!(response.status.systemd.state, SystemdState::Inactive);
        assert_eq!(response.affected_requests, 1);
        assert_eq!(pool.activity("a").unwrap().active(), 1);
        assert!(sleep_order.lock().await.is_empty());
        assert_eq!(stops.load(Ordering::Acquire), 1);
        assert_eq!(*stop_forces.lock().await, vec![true]);
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forced_stop_reports_initial_blast_radius_and_remains_durably_drained() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let starts = runner.starts.clone();
        let stop_started = Arc::new(Notify::new());
        let stop_release = Arc::new(Notify::new());
        runner
            .stop_started
            .lock()
            .await
            .insert("a".into(), stop_started.clone());
        runner
            .stop_release
            .lock()
            .await
            .insert("a".into(), stop_release.clone());
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let mut active: Vec<_> = (0..8)
            .map(|_| pool.activity("a").unwrap().acquire_active())
            .collect();
        let handle = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Stop(StopRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_millis(40),
                force: true,
            }),
        )
        .await
        .unwrap();
        let operation_id = handle.operation_id();

        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Draining).await;
        // Completion during the drain window must not rewrite the exact
        // admission-boundary count recorded for the force decision.
        active.drain(..7);
        tokio::time::timeout(Duration::from_secs(1), stop_started.notified())
            .await
            .expect("forced stop should reach the blocked runner");
        drop(active);
        assert_eq!(pool.activity("a").unwrap().active(), 0);
        stop_release.notify_one();

        let outcome = handle.wait_outcome().await;
        assert_eq!(outcome.operation_id, operation_id);
        assert_eq!(
            outcome.affected_requests, 8,
            "terminal metadata must retain the count captured when admission closed"
        );
        let stopped = outcome.result.unwrap();
        assert_eq!(stopped.lifecycle, Lifecycle::Drained);
        assert_eq!(stopped.systemd.state, SystemdState::Inactive);
        assert!(!controls.services["a"].running.load(Ordering::Acquire));

        assert_eq!(
            pool.reconcile("a").await.unwrap().lifecycle,
            Lifecycle::Drained,
            "observation must preserve the administrative stop latch"
        );
        let automatic_restart = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: false,
                },
            )
            .await;
        assert!(matches!(
            automatic_restart,
            Err(PoolError::ServiceDrained(service)) if service == "a"
        ));
        assert_eq!(
            starts.load(Ordering::Acquire),
            0,
            "automatic demand must not undo an administrative force-stop"
        );
    }

    #[tokio::test]
    async fn forced_stop_failure_retains_count_and_durable_stop_intent() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let stops = runner.stops.clone();
        runner.fail_stop.store(true, Ordering::Release);
        let mut service = definition("a", "gpu-0");
        service.backoff_initial = Duration::ZERO;
        service.backoff_max = Duration::ZERO;
        let pool = reconciled_pool(vec![service], runner).await;
        let active = pool.activity("a").unwrap().acquire_active();
        let handle = PoolManager::begin_lifecycle(
            &pool,
            LifecycleRequest::Stop(StopRequest {
                service: "a".into(),
                deadline: Instant::now(),
                force: true,
            }),
        )
        .await
        .unwrap();

        let outcome = handle.wait_outcome().await;
        assert_eq!(outcome.affected_requests, 1);
        assert!(matches!(outcome.result, Err(PoolError::Runner { .. })));
        drop(active);

        tokio::time::sleep(Duration::from_millis(20)).await;
        let status = pool.current_status("a").unwrap();
        assert_eq!(status.lifecycle, Lifecycle::Drained);
        assert_eq!(
            stops.load(Ordering::Acquire),
            1,
            "durable administrative stop failure must not schedule automatic recovery"
        );
        assert!(matches!(
            pool.ensure_ready_status(
                "a",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: false,
                },
            )
            .await,
            Err(PoolError::ServiceDrained(service)) if service == "a"
        ));
    }

    #[tokio::test]
    async fn non_force_noop_reports_zero_affected_requests_even_with_active_work() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let active = pool.activity("a").unwrap().acquire_active();

        let response = PoolManager::ensure_ready(
            &pool,
            EnsureReadyRequest {
                service: "a".into(),
                deadline: Instant::now() + Duration::from_secs(1),
                evict: false,
                reactivate_drained: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(response.status.active_requests, 1);
        assert_eq!(response.affected_requests, 0);
        drop(active);
    }

    #[tokio::test]
    async fn automatic_demand_cannot_undo_drain_but_explicit_reactivation_can() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let starts = runner.starts.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        pool.drain_status(
            "a",
            DrainOptions {
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            pool.reconcile("a").await.unwrap().lifecycle,
            Lifecycle::Drained
        );

        let automatic = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: false,
                },
            )
            .await;
        assert_eq!(automatic, Err(PoolError::ServiceDrained("a".into())));
        assert_eq!(starts.load(Ordering::Acquire), 0);
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::Drained
        );

        let explicit = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(explicit.lifecycle, Lifecycle::Running);
        assert_eq!(starts.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn drained_intent_preserves_refreshed_underlying_ownership_facts() {
        let runner = MockRunner::new(&["a"], &[]);
        let observed = runner.services["a"].clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;

        let stopped = pool
            .drain_status(
                "a",
                DrainOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(stopped.lifecycle, Lifecycle::Drained);
        assert_eq!(stopped.systemd.state, SystemdState::Inactive);
        assert!(!status_owns_awake_placement(&stopped));

        observed.running.store(true, Ordering::Release);
        let awake = pool.reconcile("a").await.unwrap();
        assert_eq!(awake.lifecycle, Lifecycle::Drained);
        assert_eq!(awake.systemd.state, SystemdState::Active);
        assert!(status_owns_awake_placement(&awake));

        *observed.sleeping.lock().await = Some(SleepLevel::L1);
        let asleep = pool.reconcile("a").await.unwrap();
        assert_eq!(asleep.lifecycle, Lifecycle::Drained);
        assert_eq!(asleep.probe.sleeping_level, Some(1));
        assert!(!status_owns_awake_placement(&asleep));

        let runner = MockRunner::new(&["b"], &["b"]);
        let wakes = runner.wakes.clone();
        let pool = reconciled_pool(vec![definition("b", "gpu-0")], runner).await;
        pool.sleep_status(
            "b",
            SleepOptions {
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();
        let drained_asleep = pool
            .drain_status(
                "b",
                DrainOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(drained_asleep.lifecycle, Lifecycle::Drained);
        assert_eq!(drained_asleep.probe.sleeping_level, Some(1));
        assert!(!status_owns_awake_placement(&drained_asleep));

        let reactivated = pool
            .ensure_ready_status(
                "b",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(reactivated.lifecycle, Lifecycle::Running);
        assert_eq!(wakes.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn drained_owner_is_idle_reapable_and_evictable() {
        let runner = MockRunner::new(&["owner", "new"], &["owner"]);
        let stops = runner.stops.clone();
        let mut owner = definition("owner", "gpu-0");
        owner.idle.stop_after = Some(Duration::ZERO);
        let pool = reconciled_pool(vec![owner.clone(), definition("new", "gpu-0")], runner).await;
        pool.drain_status(
            "owner",
            DrainOptions {
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();
        pool.reap_idle_once().await;
        wait_for_status(&pool, "owner", |status| {
            status.lifecycle == Lifecycle::Drained && status.systemd.state == SystemdState::Inactive
        })
        .await;
        assert_eq!(
            pool.current_status("owner").unwrap().lifecycle,
            Lifecycle::Drained
        );
        assert_eq!(
            pool.current_status("owner").unwrap().systemd.state,
            SystemdState::Inactive
        );
        pool.reap_idle_once().await;
        assert_eq!(stops.load(Ordering::Acquire), 1);

        let runner = MockRunner::new(&["owner", "new"], &["owner"]);
        let pool = reconciled_pool(vec![owner, definition("new", "gpu-0")], runner).await;
        pool.drain_status(
            "owner",
            DrainOptions {
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();
        let started = pool
            .ensure_ready_status(
                "new",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    evict: true,
                    reactivate_drained: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(started.lifecycle, Lifecycle::Running);
        assert_eq!(
            pool.current_status("owner").unwrap().lifecycle,
            Lifecycle::Drained
        );
        assert_eq!(
            pool.current_status("owner").unwrap().systemd.state,
            SystemdState::Inactive
        );
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
                    reactivate_drained: false,
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn busy_owner_keeps_placement_while_reconciliation_is_in_flight() {
        let runner = MockRunner::new(&["owner", "new"], &["owner"]);
        let controls = runner.clone();
        let starts = runner.starts.clone();
        let stops = runner.stops.clone();
        let pool = reconciled_pool(
            vec![definition("owner", "gpu-0"), definition("new", "gpu-0")],
            runner,
        )
        .await;
        let observe_started = Arc::new(Notify::new());
        let observe_release = Arc::new(Notify::new());
        controls
            .observe_started
            .lock()
            .await
            .insert("owner".into(), observe_started.clone());
        controls
            .observe_release
            .lock()
            .await
            .insert("owner".into(), observe_release.clone());
        let mut active = Vec::new();
        for _ in 0..8 {
            active.push(
                PoolManager::acquire_active(&pool, "owner".into())
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(pool.activity("owner").unwrap().active(), 8);

        let reconcile_pool = pool.clone();
        let reconcile =
            tokio::spawn(async move { reconcile_pool.reconcile("owner").await.unwrap() });
        tokio::time::timeout(Duration::from_secs(1), observe_started.notified())
            .await
            .expect("owner reconciliation should enter its observation window");
        assert_eq!(
            pool.current_status("owner").unwrap().lifecycle,
            Lifecycle::Reconciling
        );

        let result = pool
            .ensure_ready_status(
                "new",
                EnsureReadyOptions {
                    deadline: Instant::now() + Duration::from_millis(40),
                    evict: true,
                    reactivate_drained: false,
                },
            )
            .await;

        assert!(
            matches!(
                result,
                Err(PoolError::PlacementUnavailable { .. } | PoolError::Deadline { .. })
            ),
            "concurrent reconciliation must not hide an awake busy owner: {result:?}"
        );
        assert_eq!(starts.load(Ordering::Acquire), 0);
        assert_eq!(stops.load(Ordering::Acquire), 0);
        assert!(
            controls.services["owner"].running.load(Ordering::Acquire),
            "the busy owner must remain awake"
        );
        assert!(
            !controls.services["new"].running.load(Ordering::Acquire),
            "the contender must never be double-placed"
        );

        observe_release.notify_one();
        assert_eq!(reconcile.await.unwrap().lifecycle, Lifecycle::Running);
        drop(active);
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
                    reactivate_drained: false,
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
        // Eviction does not run through `get_or_spawn_flight`, so without its own counter
        // a displaced model would leave no transition record. `evict` keeps "something
        // took my GPU" separate from an idle reap, which is recorded as `stop`.
        assert_eq!(transition_count("low", "evict", "success"), 1);
        assert_eq!(transition_count("high", "evict", "success"), 1);
        assert_eq!(transition_count("low", "stop", "success"), 0);
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
    async fn status_reads_advance_probe_age_without_another_observation() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let observes = Arc::clone(&runner.observes);
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let current = pool.current_status("a").unwrap();
        pool.inner.state.update_if_current(
            "a",
            current.generation,
            current.operation_id,
            |status| status.probe.age = Some(Duration::from_secs(3)),
        );
        let observations_before_status_reads = observes.load(Ordering::Acquire);

        let first = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some("a".into()),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let second = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some("a".into()),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();

        assert!(first.probe.age >= Some(Duration::from_secs(3)));
        assert!(
            second.probe.age >= first.probe.age.map(|age| age + Duration::from_millis(10)),
            "probe age did not advance across status reads: first={:?}, second={:?}",
            first.probe.age,
            second.probe.age
        );
        assert_eq!(
            observes.load(Ordering::Acquire),
            observations_before_status_reads,
            "status reads must not run another probe"
        );
    }

    #[tokio::test]
    async fn reconcile_error_restores_prior_lifecycle_with_new_attempt_identity() {
        for prior_lifecycle in [Lifecycle::Stopped, Lifecycle::Running, Lifecycle::Drained] {
            let runner = MockRunner::new(&["a"], &[]);
            let controls = runner.clone();
            let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
            let seeded_operation = Uuid::new_v4();
            pool.inner
                .state
                .begin_transition("a", prior_lifecycle, seeded_operation)
                .unwrap();
            let before = pool.current_status("a").unwrap();
            controls.fail_observe.store(true, Ordering::Release);

            let error = pool.reconcile("a").await.unwrap_err();
            let restored = pool.current_status("a").unwrap();

            assert!(matches!(error, PoolError::Runner { .. }));
            assert_eq!(restored.lifecycle, prior_lifecycle);
            assert_eq!(restored.generation, before.generation + 1);
            assert_ne!(restored.operation_id, before.operation_id);
            assert!(restored.operation_id.is_some());
            assert!(
                restored
                    .last_error
                    .as_deref()
                    .is_some_and(|error| error.contains("injected observation failure"))
            );
        }
    }

    #[tokio::test]
    async fn reconcile_error_does_not_obsolete_failed_or_backoff_authority() {
        for prior_lifecycle in [Lifecycle::Failed, Lifecycle::Backoff] {
            let runner = MockRunner::new(&["a"], &[]);
            let controls = runner.clone();
            let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
            let seeded_operation = Uuid::new_v4();
            pool.inner
                .state
                .begin_transition("a", prior_lifecycle, seeded_operation)
                .unwrap();
            let before = pool.current_status("a").unwrap();
            controls.fail_observe.store(true, Ordering::Release);

            pool.reconcile("a").await.unwrap_err();
            let restored = pool.current_status("a").unwrap();

            assert_eq!(restored.lifecycle, prior_lifecycle);
            assert_eq!(restored.generation, before.generation);
            assert_eq!(restored.operation_id, before.operation_id);
            assert!(
                restored
                    .last_error
                    .as_deref()
                    .is_some_and(|error| error.contains("injected observation failure"))
            );
        }
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
    async fn sleeping_a_stopped_service_is_rejected_without_runner_or_state_mutation() {
        let runner = MockRunner::new(&["a"], &[]);
        let sleep_order = runner.sleep_order.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        let before = pool.current_status("a").unwrap();

        let error = pool
            .sleep_status(
                "a",
                SleepOptions {
                    level: SleepLevel::L1,
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(error, PoolError::PlacementUnavailable { .. }));
        assert!(sleep_order.lock().await.is_empty());
        let after = pool.current_status("a").unwrap();
        assert_eq!(after.lifecycle, Lifecycle::Stopped);
        assert_eq!(after.generation, before.generation);
        assert_eq!(after.operation_id, before.operation_id);
        assert_eq!(after.last_error, before.last_error);
        assert_eq!(pool.entry("a").unwrap().failures.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn failed_sleep_enters_backoff_gates_retry_and_cleans_up() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let starts = runner.starts.clone();
        let stops = runner.stops.clone();
        let mut service = definition("a", "gpu-0");
        service.backoff_initial = Duration::ZERO;
        service.backoff_max = Duration::ZERO;
        let pool = reconciled_pool(vec![service], runner).await;
        let cleanup_release = Arc::new(Notify::new());
        controls
            .stop_release
            .lock()
            .await
            .insert("a".into(), cleanup_release.clone());
        controls.fail_sleep.store(true, Ordering::Release);

        let error = pool
            .sleep_status(
                "a",
                SleepOptions {
                    level: SleepLevel::L1,
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, PoolError::Runner { .. }));
        let failed_operation =
            wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Backoff)
                .await
                .operation_id;

        let retry = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await;
        assert!(matches!(retry, Err(PoolError::TerminalFailure { .. })));
        assert_eq!(starts.load(Ordering::Acquire), 0);

        cleanup_release.notify_one();
        let stopped =
            wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Stopped).await;
        assert_eq!(stopped.operation_id, failed_operation);
        assert_eq!(stops.load(Ordering::Acquire), 1);

        controls.fail_sleep.store(false, Ordering::Release);
        let recovered = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await
            .unwrap();
        assert_eq!(recovered.lifecycle, Lifecycle::Running);
        assert_eq!(starts.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn failed_stop_enters_backoff_gates_retry_and_retries_cleanup_once() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let starts = runner.starts.clone();
        let stops = runner.stops.clone();
        let mut service = definition("a", "gpu-0");
        service.backoff_initial = Duration::ZERO;
        service.backoff_max = Duration::ZERO;
        let pool = reconciled_pool(vec![service], runner).await;
        let cleanup_release = Arc::new(Notify::new());
        controls
            .stop_release
            .lock()
            .await
            .insert("a".into(), cleanup_release.clone());
        controls.fail_stop.store(true, Ordering::Release);

        let error = pool
            .stop_status(
                "a",
                StopOptions {
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(error, PoolError::Runner { .. }));
        let failed_operation =
            wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Backoff)
                .await
                .operation_id;

        let retry = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await;
        assert!(matches!(retry, Err(PoolError::TerminalFailure { .. })));
        assert_eq!(starts.load(Ordering::Acquire), 0);

        cleanup_release.notify_one();
        let stopped =
            wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Stopped).await;
        assert_eq!(stopped.operation_id, failed_operation);
        assert_eq!(stops.load(Ordering::Acquire), 2);

        let recovered = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await
            .unwrap();
        assert_eq!(recovered.lifecycle, Lifecycle::Running);
        assert_eq!(starts.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn failed_sleep_from_drained_preserves_drain_intent_and_can_be_retried() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let controls = runner.clone();
        let pool = reconciled_pool(vec![definition("a", "gpu-0")], runner).await;
        pool.drain_status(
            "a",
            DrainOptions {
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap();
        controls.fail_sleep.store(true, Ordering::Release);

        pool.sleep_status(
            "a",
            SleepOptions {
                level: SleepLevel::L1,
                deadline: Instant::now() + Duration::from_secs(1),
                force: false,
            },
        )
        .await
        .unwrap_err();
        let failed = pool.current_status("a").unwrap();
        assert_eq!(failed.lifecycle, Lifecycle::Drained);
        assert!(failed.last_error.is_some());
        assert_eq!(pool.entry("a").unwrap().failures.load(Ordering::Acquire), 0);

        controls.fail_sleep.store(false, Ordering::Release);
        let slept = pool
            .sleep_status(
                "a",
                SleepOptions {
                    level: SleepLevel::L1,
                    deadline: Instant::now() + Duration::from_secs(1),
                    force: false,
                },
            )
            .await
            .unwrap();
        assert_eq!(slept.lifecycle, Lifecycle::Drained);
        assert_eq!(slept.probe.sleeping_level, Some(1));
    }

    #[tokio::test]
    async fn reconcile_cannot_neutralize_or_obsolete_failure_backoff() {
        let runner = MockRunner::new(&["a"], &[]);
        runner.fail_start.store(true, Ordering::Release);
        let starts = runner.starts.clone();
        let stops = runner.stops.clone();
        let mut service = definition("a", "gpu-0");
        service.backoff_initial = Duration::from_millis(250);
        service.backoff_max = Duration::from_millis(250);
        let pool = reconciled_pool(vec![service], runner).await;

        pool.ensure_ready_status(
            "a",
            EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
        )
        .await
        .unwrap_err();
        let failure = pool.current_status("a").unwrap();
        assert!(matches!(
            failure.lifecycle,
            Lifecycle::Failed | Lifecycle::Backoff
        ));
        assert!(failure.operation_id.is_some());
        assert!(failure.last_error.is_some());

        for _ in 0..8 {
            let reconciled = pool.reconcile("a").await.unwrap();
            assert!(matches!(
                reconciled.lifecycle,
                Lifecycle::Failed | Lifecycle::Backoff
            ));
            assert_eq!(reconciled.generation, failure.generation);
            assert_eq!(reconciled.operation_id, failure.operation_id);
            assert_eq!(reconciled.last_error, failure.last_error);
        }

        let retry = pool
            .ensure_ready_status(
                "a",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            )
            .await;
        assert!(matches!(retry, Err(PoolError::TerminalFailure { .. })));
        assert_eq!(starts.load(Ordering::Acquire), 1);

        let stopped = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let status = pool.current_status("a").unwrap();
                if status.lifecycle == Lifecycle::Stopped {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("scheduled backoff cleanup should remain authoritative");
        assert_eq!(stopped.generation, failure.generation);
        assert_eq!(stopped.operation_id, failure.operation_id);
        assert_eq!(stops.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn overlapping_failure_generations_converge_to_stopped() {
        let runner = MockRunner::new(&["a"], &[]);
        let stops = runner.stops.clone();
        let mut service = definition("a", "gpu-0");
        service.backoff_initial = Duration::from_millis(40);
        service.backoff_max = Duration::from_millis(40);
        let pool = reconciled_pool(vec![service], runner).await;
        let first_operation = Uuid::new_v4();
        let first_generation = pool
            .inner
            .state
            .begin_transition("a", Lifecycle::Starting, first_operation)
            .unwrap();
        pool.record_failure_and_schedule_backoff(
            "a",
            first_generation,
            first_operation,
            &PoolError::Runner {
                service: "a".into(),
                message: "first failure".into(),
            },
        );

        let second_operation = Uuid::new_v4();
        let second_generation = pool
            .inner
            .state
            .begin_transition("a", Lifecycle::Starting, second_operation)
            .unwrap();
        pool.record_failure_and_schedule_backoff(
            "a",
            second_generation,
            second_operation,
            &PoolError::Runner {
                service: "a".into(),
                message: "second failure".into(),
            },
        );

        let stopped =
            wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::Stopped).await;
        assert_eq!(stopped.generation, second_generation);
        assert_eq!(stopped.operation_id, Some(second_operation));
        assert_eq!(stops.load(Ordering::Acquire), 1);
        assert_eq!(
            pool.entry("a")
                .unwrap()
                .backoff_generation
                .load(Ordering::Acquire),
            0
        );
    }

    #[tokio::test]
    async fn idle_reaper_requests_l2_after_l1_and_updates_state() {
        let runner = MockRunner::new(&["a"], &["a"]);
        let sleep_order = runner.sleep_order.clone();
        let wakes = runner.wakes.clone();
        let mut service = definition("a", "gpu-0");
        service.idle.sleep_l1_after = Some(Duration::ZERO);
        service.idle.sleep_l2_after = Some(Duration::ZERO);
        let pool = reconciled_pool(vec![service], runner).await;

        pool.reap_idle_once().await;
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::AsleepL1).await;
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::AsleepL1
        );

        pool.reap_idle_once().await;
        wait_for_status(&pool, "a", |status| status.lifecycle == Lifecycle::AsleepL2).await;
        assert_eq!(
            pool.current_status("a").unwrap().lifecycle,
            Lifecycle::AsleepL2
        );
        assert_eq!(
            *sleep_order.lock().await,
            vec![
                ("a".to_string(), SleepLevel::L1),
                ("a".to_string(), SleepLevel::L2)
            ]
        );
        // The adapter owns vLLM's required wake/resleep sequence; the pool keeps
        // this as one draining-to-asleep transition.
        assert_eq!(wakes.load(Ordering::Acquire), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_cold_start_does_not_block_disjoint_gpu_wake() {
        let mut runner = MockRunner::new(&["slow", "queued", "fast"], &["fast"]);
        *runner.services["fast"].sleeping.lock().await = Some(SleepLevel::L1);
        runner.start_delay = Duration::from_millis(300);
        let starts = runner.starts.clone();
        let start_order = runner.start_order.clone();
        let wakes = runner.wakes.clone();
        let pool = reconciled_pool(
            vec![
                definition("slow", "gpu-a"),
                definition("queued", "gpu-a"),
                definition("fast", "gpu-b"),
            ],
            runner,
        )
        .await;
        let slow_pool = pool.clone();
        let slow = tokio::spawn(async move {
            slow_pool
                .ensure_ready_status(
                    "slow",
                    EnsureReadyOptions::by(Instant::now() + Duration::from_secs(2)),
                )
                .await
        });
        while starts.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let queued_pool = pool.clone();
        let queued = tokio::spawn(async move {
            queued_pool
                .ensure_ready_status(
                    "queued",
                    EnsureReadyOptions::by(Instant::now() + Duration::from_secs(2)),
                )
                .await
        });
        loop {
            if pool
                .inner
                .placement_queue
                .lock()
                .await
                .iter()
                .any(|demand| demand.service == "queued")
            {
                break;
            }
            tokio::task::yield_now().await;
        }

        let fast = tokio::time::timeout(
            Duration::from_millis(100),
            pool.ensure_ready_status(
                "fast",
                EnsureReadyOptions::by(Instant::now() + Duration::from_secs(1)),
            ),
        )
        .await
        .expect("disjoint wake must not wait for the slow cold start")
        .unwrap();
        assert_eq!(fast.lifecycle, Lifecycle::Running);
        assert_eq!(wakes.load(Ordering::Acquire), 1);
        assert!(!slow.is_finished());
        assert!(!queued.is_finished());
        assert_eq!(slow.await.unwrap().unwrap().lifecycle, Lifecycle::Running);
        assert_eq!(queued.await.unwrap().unwrap().lifecycle, Lifecycle::Running);
        assert_eq!(*start_order.lock().await, vec!["slow", "queued"]);
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
