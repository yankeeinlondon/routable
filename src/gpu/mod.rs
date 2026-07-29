use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory, error::NvmlError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;

const NVML_SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone)]
pub struct GpuInventory {
    nvml: Result<Arc<Nvml>, Arc<str>>,
    snapshots: NvmlSnapshotSingleFlight,
}

type RawGpuObservation = Result<BTreeMap<String, GpuObservation>, String>;

#[derive(Clone)]
struct NvmlSnapshotSingleFlight {
    in_flight: Arc<Mutex<Option<watch::Receiver<Option<RawGpuObservation>>>>>,
    waiters: Arc<AtomicUsize>,
}

struct NvmlSnapshotFlightGuard {
    in_flight: Arc<Mutex<Option<watch::Receiver<Option<RawGpuObservation>>>>>,
}

struct NvmlSnapshotWaiterGuard {
    waiters: Arc<AtomicUsize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub uuid: String,
    pub index: u32,
    pub name: String,
    pub total_memory_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub used_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuObservation {
    pub device: GpuDevice,
    pub used_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub reserved_memory_bytes: u64,
    pub compute_processes: Vec<GpuProcess>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpectedGpuState {
    /// Expected stable UUID -> reservation and permitted compute PIDs.
    pub devices: BTreeMap<String, ExpectedGpuUsage>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpectedGpuUsage {
    pub maximum_used_memory_bytes: Option<u64>,
    pub allowed_compute_pids: BTreeSet<u32>,
    /// Suppress process drift when one or more configured owners have not yet
    /// produced enough systemd facts to identify their legitimate processes.
    pub allow_unexpected_compute_pids: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuDrift {
    MissingGpu {
        uuid: String,
    },
    UnexpectedComputeProcess {
        uuid: String,
        pid: u32,
    },
    MemoryReservationExceeded {
        uuid: String,
        observed_bytes: u64,
        maximum_bytes: u64,
    },
}

impl GpuDrift {
    pub fn uuid(&self) -> &str {
        match self {
            Self::MissingGpu { uuid }
            | Self::UnexpectedComputeProcess { uuid, .. }
            | Self::MemoryReservationExceeded { uuid, .. } => uuid,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::MissingGpu { .. } => "missing_gpu",
            Self::UnexpectedComputeProcess { .. } => "unexpected_compute_process",
            Self::MemoryReservationExceeded { .. } => "memory_reservation_exceeded",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub available: bool,
    pub unavailable_reason: Option<String>,
    /// Observations are keyed by stable NVIDIA UUID, never CUDA ordinal.
    pub devices: BTreeMap<String, GpuObservation>,
    pub drift: Vec<GpuDrift>,
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("NVML is unavailable: {0}")]
    Unavailable(String),
    #[error("NVML operation {operation} failed: {source}")]
    Nvml {
        operation: &'static str,
        #[source]
        source: NvmlError,
    },
}

impl GpuInventory {
    /// Initializes NVML once. Absence of the library/driver is represented as
    /// availability data rather than a process-fatal error.
    pub fn initialize() -> Self {
        Self {
            nvml: Nvml::init()
                .map(Arc::new)
                .map_err(|error| Arc::<str>::from(error.to_string())),
            snapshots: NvmlSnapshotSingleFlight::default(),
        }
    }

    pub fn unavailable_reason(&self) -> Option<&str> {
        self.nvml.as_ref().err().map(AsRef::as_ref)
    }

    pub fn is_available(&self) -> bool {
        self.nvml.is_ok()
    }

    pub fn inventory(&self) -> Result<BTreeMap<String, GpuDevice>, GpuError> {
        let nvml = self
            .nvml
            .as_ref()
            .map_err(|reason| GpuError::Unavailable(reason.to_string()))?;
        inventory(nvml)
    }

    pub fn snapshot(&self, expected: &ExpectedGpuState) -> GpuSnapshot {
        let observation = self
            .nvml
            .as_ref()
            .map_err(|reason| GpuError::Unavailable(reason.to_string()))
            .and_then(|nvml| observe(nvml));
        snapshot_from_observation(observation, expected)
    }

    /// Runs NVML calls on the blocking pool so periodic async observers do not
    /// occupy a Tokio worker. Driver calls cannot be safely cancelled once
    /// entered, so the async boundary is hard-bounded and reports a degraded
    /// snapshot instead of holding reconciliation open.
    pub async fn snapshot_async(&self, expected: ExpectedGpuState) -> GpuSnapshot {
        let nvml = self.nvml.clone();
        let observation = self
            .snapshots
            .run(NVML_SNAPSHOT_TIMEOUT, move || {
                nvml.as_ref()
                    .map_err(|reason| GpuError::Unavailable(reason.to_string()))
                    .and_then(|nvml| observe(nvml))
                    .map_err(|error| error.to_string())
            })
            .await;
        snapshot_from_raw_observation(observation, &expected)
    }
}

fn snapshot_from_observation(
    observation: Result<BTreeMap<String, GpuObservation>, GpuError>,
    expected: &ExpectedGpuState,
) -> GpuSnapshot {
    snapshot_from_raw_observation(observation.map_err(|error| error.to_string()), expected)
}

fn snapshot_from_raw_observation(
    observation: RawGpuObservation,
    expected: &ExpectedGpuState,
) -> GpuSnapshot {
    match observation {
        Ok(devices) => {
            let drift = detect_drift(&devices, expected);
            GpuSnapshot {
                available: true,
                unavailable_reason: None,
                devices,
                drift,
            }
        }
        Err(error) => unavailable_snapshot(error.to_string()),
    }
}

impl NvmlSnapshotSingleFlight {
    async fn run<F>(&self, timeout: Duration, snapshot: F) -> RawGpuObservation
    where
        F: FnOnce() -> RawGpuObservation + Send + 'static,
    {
        let mut result = {
            let mut in_flight = self.in_flight.lock().expect("NVML flight lock poisoned");
            if let Some(result) = in_flight.as_ref() {
                result.clone()
            } else {
                let (sender, result) = watch::channel(None);
                *in_flight = Some(result.clone());
                let flight = NvmlSnapshotFlightGuard {
                    in_flight: self.in_flight.clone(),
                };
                tokio::task::spawn_blocking(move || {
                    let _flight = flight;
                    sender.send_replace(Some(snapshot()));
                });
                result
            }
        };

        self.waiters.fetch_add(1, Ordering::AcqRel);
        let _waiter = NvmlSnapshotWaiterGuard {
            waiters: self.waiters.clone(),
        };
        match tokio::time::timeout(timeout, async {
            loop {
                if let Some(snapshot) = result.borrow().clone() {
                    return snapshot;
                }
                if result.changed().await.is_err() {
                    return Err("NVML observation task ended without a result".to_string());
                }
            }
        })
        .await
        {
            Ok(snapshot) => snapshot,
            Err(_) => Err(format!(
                "NVML observation timed out after {} ms",
                timeout.as_millis()
            )),
        }
    }
}

impl Default for NvmlSnapshotSingleFlight {
    fn default() -> Self {
        Self {
            in_flight: Arc::new(Mutex::new(None)),
            waiters: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Drop for NvmlSnapshotFlightGuard {
    fn drop(&mut self) {
        *self.in_flight.lock().expect("NVML flight lock poisoned") = None;
    }
}

impl Drop for NvmlSnapshotWaiterGuard {
    fn drop(&mut self) {
        self.waiters.fetch_sub(1, Ordering::AcqRel);
    }
}

fn unavailable_snapshot(reason: impl Into<String>) -> GpuSnapshot {
    GpuSnapshot {
        available: false,
        unavailable_reason: Some(reason.into()),
        devices: BTreeMap::new(),
        drift: Vec::new(),
    }
}

impl Default for GpuInventory {
    fn default() -> Self {
        Self::initialize()
    }
}

fn inventory(nvml: &Nvml) -> Result<BTreeMap<String, GpuDevice>, GpuError> {
    let count = nvml.device_count().map_err(|source| GpuError::Nvml {
        operation: "device_count",
        source,
    })?;
    let mut devices = BTreeMap::new();
    for index in 0..count {
        let device = nvml
            .device_by_index(index)
            .map_err(|source| GpuError::Nvml {
                operation: "device_by_index",
                source,
            })?;
        let uuid = device.uuid().map_err(|source| GpuError::Nvml {
            operation: "device.uuid",
            source,
        })?;
        let name = device.name().map_err(|source| GpuError::Nvml {
            operation: "device.name",
            source,
        })?;
        let memory = device.memory_info().map_err(|source| GpuError::Nvml {
            operation: "device.memory_info",
            source,
        })?;
        devices.insert(
            uuid.clone(),
            GpuDevice {
                uuid,
                index,
                name,
                total_memory_bytes: memory.total,
            },
        );
    }
    Ok(devices)
}

fn observe(nvml: &Nvml) -> Result<BTreeMap<String, GpuObservation>, GpuError> {
    let devices = inventory(nvml)?;
    let mut observations = BTreeMap::new();
    for (uuid, description) in devices {
        let device = nvml
            .device_by_uuid(uuid.as_str())
            .map_err(|source| GpuError::Nvml {
                operation: "device_by_uuid",
                source,
            })?;
        let memory = device.memory_info().map_err(|source| GpuError::Nvml {
            operation: "device.memory_info",
            source,
        })?;
        let processes = device
            .running_compute_processes()
            .map_err(|source| GpuError::Nvml {
                operation: "device.running_compute_processes",
                source,
            })?
            .into_iter()
            .map(|process| GpuProcess {
                pid: process.pid,
                used_memory_bytes: match process.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => Some(bytes),
                    UsedGpuMemory::Unavailable => None,
                },
            })
            .collect();
        observations.insert(
            uuid,
            GpuObservation {
                device: description,
                used_memory_bytes: memory.used,
                free_memory_bytes: memory.free,
                reserved_memory_bytes: memory.reserved,
                compute_processes: processes,
            },
        );
    }
    Ok(observations)
}

pub fn detect_drift(
    observed: &BTreeMap<String, GpuObservation>,
    expected: &ExpectedGpuState,
) -> Vec<GpuDrift> {
    let mut drift = Vec::new();
    for (uuid, expectation) in &expected.devices {
        let Some(observation) = observed.get(uuid) else {
            drift.push(GpuDrift::MissingGpu { uuid: uuid.clone() });
            continue;
        };
        if let Some(maximum_bytes) = expectation.maximum_used_memory_bytes {
            if observation.used_memory_bytes > maximum_bytes {
                drift.push(GpuDrift::MemoryReservationExceeded {
                    uuid: uuid.clone(),
                    observed_bytes: observation.used_memory_bytes,
                    maximum_bytes,
                });
            }
        }
        for process in &observation.compute_processes {
            if !expectation.allow_unexpected_compute_pids
                && !expectation.allowed_compute_pids.contains(&process.pid)
            {
                drift.push(GpuDrift::UnexpectedComputeProcess {
                    uuid: uuid.clone(),
                    pid: process.pid,
                });
            }
        }
    }
    drift
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(uuid: &str, memory: u64, pids: &[u32]) -> GpuObservation {
        GpuObservation {
            device: GpuDevice {
                uuid: uuid.into(),
                index: 0,
                name: "GPU".into(),
                total_memory_bytes: 100,
            },
            used_memory_bytes: memory,
            free_memory_bytes: 100 - memory,
            reserved_memory_bytes: 0,
            compute_processes: pids
                .iter()
                .map(|pid| GpuProcess {
                    pid: *pid,
                    used_memory_bytes: None,
                })
                .collect(),
        }
    }

    #[test]
    fn drift_is_keyed_by_uuid_and_reports_process_and_memory() {
        let observed =
            BTreeMap::from([("GPU-stable".into(), observation("GPU-stable", 90, &[7, 8]))]);
        let expected = ExpectedGpuState {
            devices: BTreeMap::from([
                (
                    "GPU-stable".into(),
                    ExpectedGpuUsage {
                        maximum_used_memory_bytes: Some(80),
                        allowed_compute_pids: BTreeSet::from([7]),
                        allow_unexpected_compute_pids: false,
                    },
                ),
                ("GPU-missing".into(), ExpectedGpuUsage::default()),
            ]),
        };
        let drift = detect_drift(&observed, &expected);
        assert!(drift.contains(&GpuDrift::MemoryReservationExceeded {
            uuid: "GPU-stable".into(),
            observed_bytes: 90,
            maximum_bytes: 80,
        }));
        assert!(drift.contains(&GpuDrift::UnexpectedComputeProcess {
            uuid: "GPU-stable".into(),
            pid: 8,
        }));
        assert!(!drift.contains(&GpuDrift::UnexpectedComputeProcess {
            uuid: "GPU-stable".into(),
            pid: 7,
        }));
        assert!(drift.contains(&GpuDrift::MissingGpu {
            uuid: "GPU-missing".into(),
        }));
    }

    #[test]
    fn unknown_process_facts_suppress_only_process_drift() {
        let observed =
            BTreeMap::from([("GPU-stable".into(), observation("GPU-stable", 90, &[999]))]);
        let expected = ExpectedGpuState {
            devices: BTreeMap::from([(
                "GPU-stable".into(),
                ExpectedGpuUsage {
                    maximum_used_memory_bytes: Some(80),
                    allowed_compute_pids: BTreeSet::new(),
                    allow_unexpected_compute_pids: true,
                },
            )]),
        };

        let drift = detect_drift(&observed, &expected);

        assert_eq!(
            drift,
            vec![GpuDrift::MemoryReservationExceeded {
                uuid: "GPU-stable".into(),
                observed_bytes: 90,
                maximum_bytes: 80,
            }]
        );
    }

    #[test]
    fn unavailable_nvml_is_reported_as_degraded_observation() {
        let inventory = GpuInventory {
            nvml: Err(Arc::from("library missing")),
            snapshots: NvmlSnapshotSingleFlight::default(),
        };

        let snapshot = inventory.snapshot(&ExpectedGpuState::default());

        assert!(!snapshot.available);
        assert_eq!(
            snapshot.unavailable_reason.as_deref(),
            Some("NVML is unavailable: library missing")
        );
        assert!(snapshot.devices.is_empty());
        assert!(snapshot.drift.is_empty());
    }

    #[test]
    fn nvml_operation_error_is_reported_as_degraded_observation() {
        let snapshot = snapshot_from_observation(
            Err(GpuError::Nvml {
                operation: "device.memory_info",
                source: NvmlError::GpuLost,
            }),
            &ExpectedGpuState::default(),
        );

        assert!(!snapshot.available);
        assert!(
            snapshot
                .unavailable_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("device.memory_info"))
        );
        assert!(snapshot.devices.is_empty());
    }

    #[tokio::test]
    async fn concurrent_callers_join_raw_observation_but_derive_distinct_drift() {
        let snapshots = NvmlSnapshotSingleFlight::default();
        let invocations = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let observed = BTreeMap::from([("GPU-stable".into(), observation("GPU-stable", 90, &[7]))]);
        let first_expected = ExpectedGpuState {
            devices: BTreeMap::from([(
                "GPU-stable".into(),
                ExpectedGpuUsage {
                    maximum_used_memory_bytes: Some(80),
                    allowed_compute_pids: BTreeSet::from([7]),
                    allow_unexpected_compute_pids: false,
                },
            )]),
        };
        let second_expected = ExpectedGpuState {
            devices: BTreeMap::from([(
                "GPU-stable".into(),
                ExpectedGpuUsage {
                    maximum_used_memory_bytes: Some(100),
                    allowed_compute_pids: BTreeSet::new(),
                    allow_unexpected_compute_pids: false,
                },
            )]),
        };

        let first_snapshots = snapshots.clone();
        let first_invocations = invocations.clone();
        let first_observed = observed.clone();
        let first = tokio::spawn(async move {
            let raw = first_snapshots
                .run(Duration::from_secs(1), move || {
                    first_invocations.fetch_add(1, Ordering::AcqRel);
                    let _ = started_tx.send(());
                    release_rx.recv().expect("test release sender dropped");
                    Ok(first_observed)
                })
                .await;
            snapshot_from_raw_observation(raw, &first_expected)
        });
        tokio::time::timeout(Duration::from_secs(1), started_rx)
            .await
            .expect("blocking snapshot did not start")
            .expect("blocking snapshot dropped its start signal");

        let second_snapshots = snapshots.clone();
        let second_invocations = invocations.clone();
        let second = tokio::spawn(async move {
            let raw = second_snapshots
                .run(Duration::from_secs(1), move || {
                    second_invocations.fetch_add(1, Ordering::AcqRel);
                    panic!("joiner must not start a second blocking observation");
                })
                .await;
            snapshot_from_raw_observation(raw, &second_expected)
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while snapshots.waiters.load(Ordering::Acquire) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both callers did not join the shared observation");
        release_tx.send(()).unwrap();

        let first = first.await.unwrap();
        let second = second.await.unwrap();
        assert!(first.available);
        assert!(second.available);
        assert_eq!(first.devices, observed);
        assert_eq!(second.devices, observed);
        assert_eq!(
            first.drift,
            vec![GpuDrift::MemoryReservationExceeded {
                uuid: "GPU-stable".into(),
                observed_bytes: 90,
                maximum_bytes: 80,
            }]
        );
        assert_eq!(
            second.drift,
            vec![GpuDrift::UnexpectedComputeProcess {
                uuid: "GPU-stable".into(),
                pid: 7,
            }]
        );
        assert_eq!(invocations.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn timed_out_nvml_observation_returns_promptly_as_degraded() {
        let started = tokio::time::Instant::now();
        let observation = NvmlSnapshotSingleFlight::default()
            .run(Duration::from_millis(20), || {
                std::thread::sleep(Duration::from_millis(250));
                Err("late result".into())
            })
            .await;
        let snapshot = snapshot_from_raw_observation(observation, &ExpectedGpuState::default());

        assert!(!snapshot.available);
        assert_eq!(
            snapshot.unavailable_reason.as_deref(),
            Some("NVML observation timed out after 20 ms")
        );
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "timeout took {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn timed_out_snapshot_remains_single_flight_until_blocking_call_finishes() {
        let snapshots = NvmlSnapshotSingleFlight::default();
        let invocations = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let first_snapshots = snapshots.clone();
        let first_invocations = invocations.clone();
        let first = tokio::spawn(async move {
            first_snapshots
                .run(Duration::from_millis(20), move || {
                    first_invocations.fetch_add(1, Ordering::AcqRel);
                    let _ = started_tx.send(());
                    release_rx.recv().expect("test release sender dropped");
                    Err("first completed".into())
                })
                .await
        });
        tokio::time::timeout(Duration::from_secs(1), started_rx)
            .await
            .expect("blocking snapshot did not start")
            .expect("blocking snapshot dropped its start signal");
        let first = first.await.unwrap();
        assert_eq!(first, Err("NVML observation timed out after 20 ms".into()));

        for _ in 0..3 {
            let later_invocations = invocations.clone();
            let later = snapshots
                .run(Duration::from_millis(20), move || {
                    later_invocations.fetch_add(1, Ordering::AcqRel);
                    panic!("timed-out joiner must not start another blocking task");
                })
                .await;
            assert_eq!(later, Err("NVML observation timed out after 20 ms".into()));
            assert_eq!(invocations.load(Ordering::Acquire), 1);
        }

        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while snapshots
                .in_flight
                .lock()
                .expect("NVML flight lock poisoned")
                .is_some()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking snapshot did not release its permit");

        let third_invocations = invocations.clone();
        let third = snapshots
            .run(Duration::from_millis(20), move || {
                third_invocations.fetch_add(1, Ordering::AcqRel);
                Err("third completed".into())
            })
            .await;
        assert_eq!(third, Err("third completed".into()));
        assert_eq!(invocations.load(Ordering::Acquire), 2);
    }
}
