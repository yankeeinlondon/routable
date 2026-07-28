use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory, error::NvmlError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone)]
pub struct GpuInventory {
    nvml: Result<Arc<Nvml>, Arc<str>>,
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
        let nvml = match &self.nvml {
            Ok(nvml) => nvml,
            Err(reason) => {
                return GpuSnapshot {
                    available: false,
                    unavailable_reason: Some(reason.to_string()),
                    devices: BTreeMap::new(),
                    drift: Vec::new(),
                };
            }
        };
        match observe(nvml) {
            Ok(devices) => {
                let drift = detect_drift(&devices, expected);
                GpuSnapshot {
                    available: true,
                    unavailable_reason: None,
                    devices,
                    drift,
                }
            }
            Err(error) => GpuSnapshot {
                available: false,
                unavailable_reason: Some(error.to_string()),
                devices: BTreeMap::new(),
                drift: Vec::new(),
            },
        }
    }

    /// Runs NVML calls on the blocking pool so periodic async observers do not
    /// occupy a Tokio worker. Failure to join is reported as unavailable data.
    pub async fn snapshot_async(&self, expected: ExpectedGpuState) -> GpuSnapshot {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.snapshot(&expected))
            .await
            .unwrap_or_else(|error| GpuSnapshot {
                available: false,
                unavailable_reason: Some(format!("NVML observation task failed: {error}")),
                devices: BTreeMap::new(),
                drift: Vec::new(),
            })
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
            if !expectation.allowed_compute_pids.contains(&process.pid) {
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
        assert!(drift.contains(&GpuDrift::MissingGpu {
            uuid: "GPU-missing".into(),
        }));
    }
}
