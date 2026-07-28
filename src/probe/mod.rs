use std::{future::Future, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    sync::{Mutex, watch},
    task::JoinHandle,
    time::Instant,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeState {
    Unknown,
    Healthy,
    Unhealthy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeThresholds {
    pub successes: u32,
    pub failures: u32,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProbeConfigError {
    #[error("probe success threshold must be at least one")]
    ZeroSuccessThreshold,
    #[error("probe failure threshold must be at least one")]
    ZeroFailureThreshold,
    #[error("probe interval must be non-zero")]
    ZeroInterval,
    #[error("probe timeout must be non-zero")]
    ZeroTimeout,
}

impl ProbeThresholds {
    pub fn new(successes: u32, failures: u32) -> Result<Self, ProbeConfigError> {
        if successes == 0 {
            return Err(ProbeConfigError::ZeroSuccessThreshold);
        }
        if failures == 0 {
            return Err(ProbeConfigError::ZeroFailureThreshold);
        }
        Ok(Self {
            successes,
            failures,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProbeSnapshot {
    pub state: ProbeState,
    pub last_result_ok: Option<bool>,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
    pub observed_at: Option<Instant>,
}

impl ProbeSnapshot {
    pub fn age_at(&self, now: Instant) -> Option<Duration> {
        self.observed_at
            .map(|observed_at| now.saturating_duration_since(observed_at))
    }

    pub fn age(&self) -> Option<Duration> {
        self.age_at(Instant::now())
    }
}

#[derive(Debug)]
pub struct ProbeTracker {
    thresholds: ProbeThresholds,
    snapshot: ProbeSnapshot,
}

impl ProbeTracker {
    pub fn new(thresholds: ProbeThresholds) -> Self {
        Self {
            thresholds,
            snapshot: ProbeSnapshot {
                state: ProbeState::Unknown,
                last_result_ok: None,
                consecutive_successes: 0,
                consecutive_failures: 0,
                last_error: None,
                observed_at: None,
            },
        }
    }

    pub fn record(&mut self, result: Result<(), impl ToString>, now: Instant) -> ProbeSnapshot {
        match result {
            Ok(()) => {
                self.snapshot.last_result_ok = Some(true);
                self.snapshot.consecutive_successes =
                    self.snapshot.consecutive_successes.saturating_add(1);
                self.snapshot.consecutive_failures = 0;
                self.snapshot.last_error = None;
                if self.snapshot.consecutive_successes >= self.thresholds.successes {
                    self.snapshot.state = ProbeState::Healthy;
                }
            }
            Err(error) => {
                self.snapshot.last_result_ok = Some(false);
                self.snapshot.consecutive_failures =
                    self.snapshot.consecutive_failures.saturating_add(1);
                self.snapshot.consecutive_successes = 0;
                self.snapshot.last_error = Some(error.to_string());
                if self.snapshot.consecutive_failures >= self.thresholds.failures {
                    self.snapshot.state = ProbeState::Unhealthy;
                }
            }
        }
        self.snapshot.observed_at = Some(now);
        self.snapshot.clone()
    }

    pub fn snapshot(&self) -> ProbeSnapshot {
        self.snapshot.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PeriodicProbeConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub thresholds: ProbeThresholds,
}

impl PeriodicProbeConfig {
    pub fn validate(self) -> Result<Self, ProbeConfigError> {
        if self.interval.is_zero() {
            return Err(ProbeConfigError::ZeroInterval);
        }
        if self.timeout.is_zero() {
            return Err(ProbeConfigError::ZeroTimeout);
        }
        Ok(self)
    }
}

pub struct PeriodicProbe {
    receiver: watch::Receiver<ProbeSnapshot>,
    task: JoinHandle<()>,
}

impl PeriodicProbe {
    /// Starts an independent task. A hung probe is bounded by `config.timeout`
    /// and cannot delay any other service's observation loop.
    pub fn spawn<F, Fut, E>(config: PeriodicProbeConfig, probe: F) -> Result<Self, ProbeConfigError>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), E>> + Send + 'static,
        E: ToString + Send + 'static,
    {
        let config = config.validate()?;
        let tracker = Arc::new(Mutex::new(ProbeTracker::new(config.thresholds)));
        let initial = ProbeTracker::new(config.thresholds).snapshot();
        let (sender, receiver) = watch::channel(initial);
        let probe = Arc::new(probe);
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let result = match tokio::time::timeout(config.timeout, probe()).await {
                    Ok(result) => result.map_err(|error| error.to_string()),
                    Err(_) => Err(format!("probe timed out after {:?}", config.timeout)),
                };
                let snapshot = tracker.lock().await.record(result, Instant::now());
                if sender.send(snapshot).is_err() {
                    break;
                }
            }
        });
        Ok(Self { receiver, task })
    }

    pub fn subscribe(&self) -> watch::Receiver<ProbeSnapshot> {
        self.receiver.clone()
    }

    pub fn current(&self) -> ProbeSnapshot {
        self.receiver.borrow().clone()
    }

    pub fn abort(&self) {
        self.task.abort();
    }
}

impl Drop for PeriodicProbe {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn thresholds_debounce_both_directions_and_track_age() {
        let thresholds = ProbeThresholds::new(2, 3).unwrap();
        let mut tracker = ProbeTracker::new(thresholds);
        let start = Instant::now();

        assert_eq!(
            tracker.record(Ok::<_, &str>(()), start).state,
            ProbeState::Unknown
        );
        assert_eq!(
            tracker
                .record(Ok::<_, &str>(()), start + Duration::from_secs(1))
                .state,
            ProbeState::Healthy
        );
        assert_eq!(
            tracker
                .record(Err("one"), start + Duration::from_secs(2))
                .state,
            ProbeState::Healthy
        );
        assert_eq!(
            tracker
                .record(Err("two"), start + Duration::from_secs(3))
                .state,
            ProbeState::Healthy
        );
        let snapshot = tracker.record(Err("three"), start + Duration::from_secs(4));
        assert_eq!(snapshot.state, ProbeState::Unhealthy);
        assert_eq!(
            snapshot.age_at(start + Duration::from_secs(9)),
            Some(Duration::from_secs(5))
        );
    }

    #[tokio::test]
    async fn periodic_probe_runs_immediately_and_publishes() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let probe = PeriodicProbe::spawn(
            PeriodicProbeConfig {
                interval: Duration::from_secs(60),
                timeout: Duration::from_secs(1),
                thresholds: ProbeThresholds::new(1, 1).unwrap(),
            },
            move || {
                let observed = Arc::clone(&observed);
                async move {
                    observed.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, String>(())
                }
            },
        )
        .unwrap();
        let mut updates = probe.subscribe();
        tokio::time::timeout(Duration::from_secs(1), updates.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updates.borrow().state, ProbeState::Healthy);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
