//! Observed lifecycle state and exact request accounting.
//!
//! This module deliberately contains no runner- or transport-specific code.  Observers
//! publish facts here and consumers subscribe through `tokio::sync::watch`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, watch};
use uuid::Uuid;

use crate::gpu::GpuDrift;

/// The externally visible lifecycle of a managed service.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lifecycle {
    Reconciling,
    Stopped,
    Starting,
    Running,
    Draining,
    Drained,
    AsleepL1,
    AsleepL2,
    Waking,
    Stopping,
    Failed,
    Backoff,
    Stale,
    #[default]
    Unknown,
}

impl Lifecycle {
    pub fn is_transitioning(self) -> bool {
        matches!(
            self,
            Self::Reconciling
                | Self::Starting
                | Self::Draining
                | Self::Waking
                | Self::Stopping
                | Self::Backoff
        )
    }

    pub fn is_ready(self) -> bool {
        self == Self::Running
    }

    /// States that conservatively reserve an awake placement.
    ///
    /// Reconciliation and unknown observations must fail closed: either can describe an
    /// already-running process whose last authoritative observation is temporarily hidden.
    pub fn owns_awake_placement(self) -> bool {
        matches!(
            self,
            Self::Reconciling
                | Self::Starting
                | Self::Running
                | Self::Draining
                | Self::Waking
                | Self::Stopping
                | Self::Failed
                | Self::Backoff
                | Self::Stale
                | Self::Unknown
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemdState {
    Inactive,
    Activating,
    Active,
    Deactivating,
    Failed,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SystemdStatus {
    pub state: SystemdState,
    pub sub_state: Option<String>,
    pub main_pid: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProbeStatus {
    pub healthy: bool,
    pub ready: bool,
    pub served_model_matches: bool,
    pub sleeping_level: Option<u8>,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    /// Age of the newest authoritative observation the runner still stands behind, at
    /// the time this status was produced. `None` means no such observation describes
    /// the process currently behind the endpoint, which fails admission closed.
    pub age: Option<Duration>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuStatus {
    pub uuid: String,
    pub used_memory_bytes: Option<u64>,
    pub process_ids: Vec<u32>,
    pub healthy: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drift: Vec<GpuDrift>,
}

/// A complete, point-in-time status record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ObservedState {
    pub service: String,
    pub lifecycle: Lifecycle,
    pub generation: u64,
    pub operation_id: Option<Uuid>,
    pub observed_at: SystemTime,
    pub probe: ProbeStatus,
    pub systemd: SystemdStatus,
    pub gpus: Vec<GpuStatus>,
    pub active_requests: u64,
    pub parked_requests: u64,
    pub idle_for: Duration,
    pub last_error: Option<String>,
}

/// Public facade name used by the pool trait.
pub type ServiceStatus = ObservedState;

impl ObservedState {
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            lifecycle: Lifecycle::Reconciling,
            generation: 0,
            operation_id: None,
            observed_at: SystemTime::now(),
            probe: ProbeStatus::default(),
            systemd: SystemdStatus::default(),
            gpus: Vec::new(),
            active_requests: 0,
            parked_requests: 0,
            idle_for: Duration::ZERO,
            last_error: None,
        }
    }

    /// Probe age extrapolated to `now`.
    ///
    /// The stored age only advances when the state store is written, so a reader that
    /// must not mutate the store — a readiness probe, for instance — has to extrapolate
    /// for itself instead of trusting the recorded value.
    pub fn probe_age_at(&self, now: SystemTime) -> Option<Duration> {
        let age = self.probe.age?;
        let elapsed = now
            .duration_since(self.observed_at)
            .unwrap_or(Duration::ZERO);
        Some(age.saturating_add(elapsed))
    }
}

/// Generation-aware storage for observed service state.
///
/// A transition first calls [`begin_transition`](Self::begin_transition).  Late probe or
/// command completions then use [`update_if_current`](Self::update_if_current), which rejects
/// an update if either its generation or operation ID is obsolete.
#[derive(Default)]
pub struct StateStore {
    services: RwLock<HashMap<String, watch::Sender<ObservedState>>>,
}

impl StateStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, service: impl Into<String>) -> watch::Receiver<ObservedState> {
        let service = service.into();
        let mut services = self.services.write().expect("state store lock poisoned");
        services
            .entry(service.clone())
            .or_insert_with(|| watch::channel(ObservedState::new(service)).0)
            .subscribe()
    }

    pub fn contains(&self, service: &str) -> bool {
        self.services
            .read()
            .expect("state store lock poisoned")
            .contains_key(service)
    }

    pub fn status(&self, service: &str) -> Option<ObservedState> {
        self.services
            .read()
            .expect("state store lock poisoned")
            .get(service)
            .map(|sender| sender.borrow().clone())
    }

    pub fn snapshot(&self) -> Vec<ObservedState> {
        let mut result: Vec<_> = self
            .services
            .read()
            .expect("state store lock poisoned")
            .values()
            .map(|sender| sender.borrow().clone())
            .collect();
        result.sort_by(|a, b| a.service.cmp(&b.service));
        result
    }

    pub fn subscribe(&self, service: &str) -> Option<watch::Receiver<ObservedState>> {
        self.services
            .read()
            .expect("state store lock poisoned")
            .get(service)
            .map(watch::Sender::subscribe)
    }

    pub fn begin_transition(
        &self,
        service: &str,
        lifecycle: Lifecycle,
        operation_id: Uuid,
    ) -> Option<u64> {
        let sender = self
            .services
            .read()
            .expect("state store lock poisoned")
            .get(service)
            .cloned()?;
        let mut generation = 0;
        sender.send_modify(|next| {
            let now = SystemTime::now();
            age_probe(next, now);
            next.generation = next.generation.saturating_add(1);
            next.operation_id = Some(operation_id);
            next.lifecycle = lifecycle;
            next.observed_at = now;
            next.last_error = None;
            generation = next.generation;
        });
        Some(generation)
    }

    pub fn update_if_current(
        &self,
        service: &str,
        generation: u64,
        operation_id: Option<Uuid>,
        update: impl FnOnce(&mut ObservedState),
    ) -> bool {
        let Some(sender) = self
            .services
            .read()
            .expect("state store lock poisoned")
            .get(service)
            .cloned()
        else {
            return false;
        };
        let mut update = Some(update);
        let mut applied = false;
        sender.send_if_modified(|next| {
            if next.generation != generation || next.operation_id != operation_id {
                return false;
            }
            if let Some(update) = update.take() {
                let now = SystemTime::now();
                age_probe(next, now);
                update(next);
                next.observed_at = now;
                applied = true;
                true
            } else {
                false
            }
        });
        applied
    }

    pub fn refresh_activity(&self, service: &str, activity: &Activity) -> bool {
        let Some(sender) = self
            .services
            .read()
            .expect("state store lock poisoned")
            .get(service)
            .cloned()
        else {
            return false;
        };
        let snapshot = activity.snapshot();
        sender.send_modify(|next| {
            let now = SystemTime::now();
            age_probe(next, now);
            next.active_requests = snapshot.active;
            next.parked_requests = snapshot.parked;
            next.idle_for = snapshot.idle_for;
            next.observed_at = now;
        });
        true
    }
}

fn age_probe(status: &mut ObservedState, now: SystemTime) {
    if let Some(aged) = status.probe_age_at(now) {
        status.probe.age = Some(aged);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivitySnapshot {
    pub active: u64,
    pub parked: u64,
    pub idle_for: Duration,
}

#[derive(Clone)]
pub struct Activity {
    inner: Arc<ActivityInner>,
}

struct ActivityInner {
    active: AtomicU64,
    parked: AtomicU64,
    last_used: Mutex<Instant>,
    changed: Notify,
}

impl Default for Activity {
    fn default() -> Self {
        Self::new()
    }
}

impl Activity {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ActivityInner {
                active: AtomicU64::new(0),
                parked: AtomicU64::new(0),
                last_used: Mutex::new(Instant::now()),
                changed: Notify::new(),
            }),
        }
    }

    /// Admit an upstream request. Dropping the returned guard decrements the exact count.
    pub fn acquire_active(&self) -> ActiveGuard {
        self.touch();
        self.inner.active.fetch_add(1, Ordering::AcqRel);
        self.inner.changed.notify_waiters();
        ActiveGuard {
            inner: Some(self.inner.clone()),
        }
    }

    /// Account for a request waiting for readiness.
    pub fn acquire_parked(&self) -> ParkedGuard {
        self.touch();
        self.inner.parked.fetch_add(1, Ordering::AcqRel);
        self.inner.changed.notify_waiters();
        ParkedGuard {
            inner: Some(self.inner.clone()),
        }
    }

    pub fn try_acquire_parked(&self, limit: u64) -> Option<ParkedGuard> {
        let mut current = self.inner.parked.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return None;
            }
            match self.inner.parked.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.touch();
                    self.inner.changed.notify_waiters();
                    return Some(ParkedGuard {
                        inner: Some(self.inner.clone()),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }

    pub fn active(&self) -> u64 {
        self.inner.active.load(Ordering::Acquire)
    }

    pub fn parked(&self) -> u64 {
        self.inner.parked.load(Ordering::Acquire)
    }

    pub fn is_idle(&self) -> bool {
        self.active() == 0 && self.parked() == 0
    }

    pub fn touch(&self) {
        *self.inner.last_used.lock().expect("activity lock poisoned") = Instant::now();
    }

    pub fn snapshot(&self) -> ActivitySnapshot {
        ActivitySnapshot {
            active: self.active(),
            parked: self.parked(),
            idle_for: self
                .inner
                .last_used
                .lock()
                .expect("activity lock poisoned")
                .elapsed(),
        }
    }

    pub async fn wait_until_idle(&self, deadline: Instant) -> bool {
        self.wait_until_idle_with_registration_hook(deadline, || {})
            .await
    }

    async fn wait_until_idle_with_registration_hook(
        &self,
        deadline: Instant,
        mut registration_hook: impl FnMut(),
    ) -> bool {
        loop {
            if self.is_idle() {
                return true;
            }
            let notified = self.inner.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_idle() {
                return true;
            }
            registration_hook();
            if tokio::time::timeout_at(deadline.into(), notified)
                .await
                .is_err()
            {
                return self.is_idle();
            }
        }
    }
}

pub struct ActiveGuard {
    inner: Option<Arc<ActivityInner>>,
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            // Keep the timestamp lock through the decrement so a snapshot that observes
            // zero active requests cannot also observe the pre-completion timestamp.
            let mut last_used = inner.last_used.lock().expect("activity lock poisoned");
            *last_used = Instant::now();
            let previous = inner.active.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(previous > 0, "active request counter underflow");
            drop(last_used);
            inner.changed.notify_waiters();
        }
    }
}

pub struct ParkedGuard {
    inner: Option<Arc<ActivityInner>>,
}

impl Drop for ParkedGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let previous = inner.parked.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(previous > 0, "parked request counter underflow");
            inner.changed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drained_is_a_serialized_terminal_routing_state() {
        assert_eq!(
            serde_json::to_string(&Lifecycle::Drained).unwrap(),
            "\"drained\""
        );
        assert!(!Lifecycle::Drained.is_ready());
        assert!(!Lifecycle::Drained.is_transitioning());
        assert!(!Lifecycle::Drained.owns_awake_placement());
    }

    #[test]
    fn uncertain_lifecycle_states_reserve_placement() {
        assert!(Lifecycle::Reconciling.owns_awake_placement());
        assert!(Lifecycle::Unknown.owns_awake_placement());
    }

    #[test]
    fn obsolete_transition_updates_are_ignored() {
        let store = StateStore::new();
        store.register("model");
        let old_op = Uuid::new_v4();
        let old_generation = store
            .begin_transition("model", Lifecycle::Starting, old_op)
            .unwrap();
        let new_op = Uuid::new_v4();
        store.begin_transition("model", Lifecycle::Stopping, new_op);

        assert!(
            !store.update_if_current("model", old_generation, Some(old_op), |state| {
                state.lifecycle = Lifecycle::Running;
            })
        );
        assert_eq!(
            store.status("model").unwrap().lifecycle,
            Lifecycle::Stopping
        );
    }

    #[tokio::test]
    async fn guards_account_exactly_and_wake_idle_waiters() {
        let activity = Activity::new();
        let active = activity.acquire_active();
        let parked = activity.acquire_parked();
        assert_eq!(activity.active(), 1);
        assert_eq!(activity.parked(), 1);

        let waiter = {
            let activity = activity.clone();
            tokio::spawn(async move {
                activity
                    .wait_until_idle(Instant::now() + Duration::from_secs(1))
                    .await
            })
        };
        drop(active);
        assert!(!activity.is_idle());
        drop(parked);
        assert!(waiter.await.unwrap());
        assert!(activity.is_idle());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn idle_waiter_cannot_miss_drop_before_notification_is_polled() {
        let activity = Activity::new();
        let active = activity.acquire_active();
        let registered = Arc::new(std::sync::Barrier::new(2));
        let resume = Arc::new(std::sync::Barrier::new(2));

        let waiter = {
            let activity = activity.clone();
            let registered = registered.clone();
            let resume = resume.clone();
            tokio::spawn(async move {
                activity
                    .wait_until_idle_with_registration_hook(
                        Instant::now() + Duration::from_secs(30),
                        || {
                            registered.wait();
                            resume.wait();
                        },
                    )
                    .await
            })
        };

        registered.wait();
        drop(active);
        resume.wait();

        assert!(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .expect("idle waiter should wake without burning its deadline")
                .unwrap()
        );
    }

    #[test]
    fn completing_active_request_refreshes_last_used() {
        let activity = Activity::new();
        let active = activity.acquire_active();
        let admitted_at = *activity
            .inner
            .last_used
            .lock()
            .expect("activity lock poisoned");

        while admitted_at.elapsed() < Duration::from_millis(1) {
            std::thread::yield_now();
        }

        drop(active);
        let completed_at = *activity
            .inner
            .last_used
            .lock()
            .expect("activity lock poisoned");
        assert!(completed_at > admitted_at);
        assert_eq!(activity.active(), 0);
    }

    #[tokio::test]
    async fn watch_subscriber_sees_generation_change() {
        let store = StateStore::new();
        let mut receiver = store.register("model");
        let op = Uuid::new_v4();
        store.begin_transition("model", Lifecycle::Starting, op);
        receiver.changed().await.unwrap();
        assert_eq!(receiver.borrow().lifecycle, Lifecycle::Starting);
        assert_eq!(receiver.borrow().operation_id, Some(op));
    }
}
