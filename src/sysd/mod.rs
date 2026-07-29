use std::{sync::Arc, time::Duration};

use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::OnceCell;
use zbus::{Connection, Proxy, zvariant::OwnedObjectPath};

mod generator;

pub use generator::{GENERATED_UNIT_DIRECTORY, GeneratedSystemdError, install_generated_inputs};

const SYSTEMD_DESTINATION: &str = "org.freedesktop.systemd1";
const SYSTEMD_MANAGER_PATH: &str = "/org/freedesktop/systemd1";
const SYSTEMD_MANAGER_INTERFACE: &str = "org.freedesktop.systemd1.Manager";
const SYSTEMD_UNIT_INTERFACE: &str = "org.freedesktop.systemd1.Unit";
const SYSTEMD_SERVICE_INTERFACE: &str = "org.freedesktop.systemd1.Service";
const SYSTEMD_UNIT_LOOKUP_OPERATION: &str = "LoadUnit";

#[derive(Clone)]
pub struct SystemdManager {
    connection: Connection,
    timeout: Duration,
    subscribed_to_jobs: Arc<OnceCell<()>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitActiveState {
    Active,
    Reloading,
    Inactive,
    Failed,
    Activating,
    Deactivating,
    Maintenance,
    Refreshing,
    Unknown(String),
}

impl UnitActiveState {
    fn from_systemd(value: String) -> Self {
        match value.as_str() {
            "active" => Self::Active,
            "reloading" => Self::Reloading,
            "inactive" => Self::Inactive,
            "failed" => Self::Failed,
            "activating" => Self::Activating,
            "deactivating" => Self::Deactivating,
            "maintenance" => Self::Maintenance,
            "refreshing" => Self::Refreshing,
            _ => Self::Unknown(value),
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active | Self::Reloading | Self::Refreshing)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct UnitStatus {
    pub name: String,
    pub load_state: String,
    pub active_state: UnitActiveState,
    pub sub_state: String,
    pub main_pid: Option<u32>,
}

#[derive(Debug, Error)]
pub enum SystemdError {
    #[error("failed to connect to the system bus: {source}")]
    Connect {
        #[source]
        source: zbus::Error,
    },
    #[error("systemd {operation} for {unit:?} timed out after {timeout:?}")]
    Timeout {
        operation: &'static str,
        unit: String,
        timeout: Duration,
    },
    #[error("failed to create systemd {interface} proxy: {source}")]
    Proxy {
        interface: &'static str,
        #[source]
        source: zbus::Error,
    },
    #[error("systemd {operation} failed for {unit:?}: {source}")]
    Call {
        operation: &'static str,
        unit: String,
        #[source]
        source: zbus::Error,
    },
    #[error("systemd {operation} job for {unit:?} completed with result {result:?}")]
    JobFailed {
        operation: &'static str,
        unit: String,
        result: String,
    },
    #[error("systemd stopped reporting jobs while waiting for {operation} of {unit:?}")]
    JobSignalEnded {
        operation: &'static str,
        unit: String,
    },
}

#[derive(Debug)]
struct JobRemoved {
    path: OwnedObjectPath,
    result: String,
}

fn optional_main_pid(main_pid: u32) -> Option<u32> {
    (main_pid != 0).then_some(main_pid)
}

async fn wait_for_job_result<S>(
    operation: &'static str,
    unit: &str,
    job_path: &OwnedObjectPath,
    events: S,
) -> Result<(), SystemdError>
where
    S: Stream<Item = Result<JobRemoved, zbus::Error>>,
{
    tokio::pin!(events);
    while let Some(event) = events.next().await {
        let event = event.map_err(|source| SystemdError::Call {
            operation: "decode JobRemoved",
            unit: unit.to_owned(),
            source,
        })?;
        if &event.path != job_path {
            continue;
        }
        return if event.result == "done" {
            Ok(())
        } else {
            Err(SystemdError::JobFailed {
                operation,
                unit: unit.to_owned(),
                result: event.result,
            })
        };
    }
    Err(SystemdError::JobSignalEnded {
        operation,
        unit: unit.to_owned(),
    })
}

impl SystemdManager {
    pub async fn connect_system(timeout: Duration) -> Result<Self, SystemdError> {
        let connection = tokio::time::timeout(timeout, Connection::system())
            .await
            .map_err(|_| SystemdError::Timeout {
                operation: "connect",
                unit: "<system-bus>".into(),
                timeout,
            })?
            .map_err(|source| SystemdError::Connect { source })?;
        Ok(Self {
            connection,
            timeout,
            subscribed_to_jobs: Arc::new(OnceCell::new()),
        })
    }

    pub fn from_connection(connection: Connection, timeout: Duration) -> Self {
        Self {
            connection,
            timeout,
            subscribed_to_jobs: Arc::new(OnceCell::new()),
        }
    }

    async fn manager_proxy(&self) -> Result<Proxy<'_>, SystemdError> {
        Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            SYSTEMD_MANAGER_PATH,
            SYSTEMD_MANAGER_INTERFACE,
        )
        .await
        .map_err(|source| SystemdError::Proxy {
            interface: SYSTEMD_MANAGER_INTERFACE,
            source,
        })
    }

    async fn unit_path(&self, unit: &str) -> Result<OwnedObjectPath, SystemdError> {
        let proxy = self.manager_proxy().await?;
        proxy
            // LoadUnit returns an object even when an inactive unit has been
            // garbage-collected from systemd's in-memory unit set.
            .call(SYSTEMD_UNIT_LOOKUP_OPERATION, &(unit,))
            .await
            .map_err(|source| SystemdError::Call {
                operation: SYSTEMD_UNIT_LOOKUP_OPERATION,
                unit: unit.to_owned(),
                source,
            })
    }

    async fn unit_job(
        &self,
        operation: &'static str,
        unit: &str,
    ) -> Result<OwnedObjectPath, SystemdError> {
        self.ensure_job_subscription().await?;
        let proxy = self.manager_proxy().await?;
        // Register the match before queueing the job so an immediately
        // completing unit cannot race past the listener.
        let signals =
            proxy
                .receive_signal("JobRemoved")
                .await
                .map_err(|source| SystemdError::Call {
                    operation: "subscribe to JobRemoved",
                    unit: unit.to_owned(),
                    source,
                })?;
        let job_path = proxy
            .call(operation, &(unit, "replace"))
            .await
            .map_err(|source| SystemdError::Call {
                operation,
                unit: unit.to_owned(),
                source,
            })?;
        let events = signals.map(|message| {
            message
                .body()
                .deserialize::<(u32, OwnedObjectPath, String, String)>()
                .map(|(_, path, _, result)| JobRemoved { path, result })
        });
        wait_for_job_result(operation, unit, &job_path, events).await?;
        Ok(job_path)
    }

    async fn ensure_job_subscription(&self) -> Result<(), SystemdError> {
        self.subscribed_to_jobs
            .get_or_try_init(|| async {
                let proxy = self.manager_proxy().await?;
                proxy
                    .call("Subscribe", &())
                    .await
                    .map_err(|source| SystemdError::Call {
                        operation: "Subscribe",
                        unit: "<manager>".into(),
                        source,
                    })
            })
            .await
            .map(|_| ())
    }

    pub async fn start_unit(&self, unit: &str) -> Result<OwnedObjectPath, SystemdError> {
        tokio::time::timeout(self.timeout, self.unit_job("StartUnit", unit))
            .await
            .map_err(|_| SystemdError::Timeout {
                operation: "StartUnit",
                unit: unit.to_owned(),
                timeout: self.timeout,
            })?
    }

    pub async fn stop_unit(&self, unit: &str) -> Result<OwnedObjectPath, SystemdError> {
        tokio::time::timeout(self.timeout, self.unit_job("StopUnit", unit))
            .await
            .map_err(|_| SystemdError::Timeout {
                operation: "StopUnit",
                unit: unit.to_owned(),
                timeout: self.timeout,
            })?
    }

    /// Ask systemd to reload unit files after generated inputs are installed.
    pub async fn reload(&self) -> Result<(), SystemdError> {
        tokio::time::timeout(self.timeout, self.reload_inner())
            .await
            .map_err(|_| SystemdError::Timeout {
                operation: "Reload",
                unit: "<manager>".into(),
                timeout: self.timeout,
            })?
    }

    async fn reload_inner(&self) -> Result<(), SystemdError> {
        let proxy = self.manager_proxy().await?;
        proxy
            .call("Reload", &())
            .await
            .map_err(|source| SystemdError::Call {
                operation: "Reload",
                unit: "<manager>".into(),
                source,
            })
    }

    pub async fn unit_status(&self, unit: &str) -> Result<UnitStatus, SystemdError> {
        tokio::time::timeout(self.timeout, self.unit_status_inner(unit))
            .await
            .map_err(|_| SystemdError::Timeout {
                operation: "unit status",
                unit: unit.to_owned(),
                timeout: self.timeout,
            })?
    }

    async fn unit_status_inner(&self, unit: &str) -> Result<UnitStatus, SystemdError> {
        let path = self.unit_path(unit).await?;
        let proxy = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            path.clone(),
            SYSTEMD_UNIT_INTERFACE,
        )
        .await
        .map_err(|source| SystemdError::Proxy {
            interface: SYSTEMD_UNIT_INTERFACE,
            source,
        })?;
        let load_state = proxy
            .get_property::<String>("LoadState")
            .await
            .map_err(|source| SystemdError::Call {
                operation: "read LoadState",
                unit: unit.to_owned(),
                source,
            })?;
        let active_state = proxy
            .get_property::<String>("ActiveState")
            .await
            .map_err(|source| SystemdError::Call {
                operation: "read ActiveState",
                unit: unit.to_owned(),
                source,
            })?;
        let sub_state = proxy
            .get_property::<String>("SubState")
            .await
            .map_err(|source| SystemdError::Call {
                operation: "read SubState",
                unit: unit.to_owned(),
                source,
            })?;
        let service_proxy = Proxy::new(
            &self.connection,
            SYSTEMD_DESTINATION,
            path,
            SYSTEMD_SERVICE_INTERFACE,
        )
        .await
        .map_err(|source| SystemdError::Proxy {
            interface: SYSTEMD_SERVICE_INTERFACE,
            source,
        })?;
        let main_pid = service_proxy
            .get_property::<u32>("MainPID")
            .await
            .map_err(|source| SystemdError::Call {
                operation: "read MainPID",
                unit: unit.to_owned(),
                source,
            })?;
        Ok(UnitStatus {
            name: unit.to_owned(),
            load_state,
            active_state: UnitActiveState::from_systemd(active_state),
            sub_state,
            main_pid: optional_main_pid(main_pid),
        })
    }
}

#[cfg(test)]
mod tests {
    use futures_util::stream;

    use super::*;

    fn job_path(id: u32) -> OwnedObjectPath {
        OwnedObjectPath::try_from(format!("/org/freedesktop/systemd1/job/{id}")).unwrap()
    }

    #[test]
    fn active_state_mapping_is_forward_compatible() {
        assert_eq!(
            UnitActiveState::from_systemd("active".into()),
            UnitActiveState::Active
        );
        assert!(UnitActiveState::from_systemd("reloading".into()).is_active());
        assert_eq!(
            UnitActiveState::from_systemd("future-state".into()),
            UnitActiveState::Unknown("future-state".into())
        );
    }

    #[test]
    fn inactive_unit_lookup_uses_load_unit() {
        assert_eq!(SYSTEMD_UNIT_LOOKUP_OPERATION, "LoadUnit");
    }

    #[test]
    fn main_pid_zero_maps_to_absent_and_process_ids_are_preserved() {
        assert_eq!(optional_main_pid(0), None);
        assert_eq!(optional_main_pid(42), Some(42));
    }

    #[tokio::test]
    async fn job_result_waiter_matches_the_returned_path_and_accepts_done() {
        let expected = job_path(42);
        let events = stream::iter([
            Ok(JobRemoved {
                path: job_path(41),
                result: "failed".into(),
            }),
            Ok(JobRemoved {
                path: expected.clone(),
                result: "done".into(),
            }),
        ]);

        wait_for_job_result("StartUnit", "vllm@example.service", &expected, events)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn job_result_waiter_surfaces_every_non_done_result() {
        for expected_result in [
            "canceled",
            "timeout",
            "failed",
            "dependency",
            "skipped",
            "future-result",
        ] {
            let expected = job_path(42);
            let events = stream::iter([Ok(JobRemoved {
                path: expected.clone(),
                result: expected_result.into(),
            })]);

            let error = wait_for_job_result("StopUnit", "vllm@example.service", &expected, events)
                .await
                .unwrap_err();
            match error {
                SystemdError::JobFailed {
                    operation,
                    unit,
                    result,
                } => {
                    assert_eq!(operation, "StopUnit");
                    assert_eq!(unit, "vllm@example.service");
                    assert_eq!(result, expected_result);
                }
                other => panic!("unexpected error for job result {expected_result:?}: {other}"),
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires a host system bus and systemd"]
    async fn observes_real_systemd_unit() {
        let manager = SystemdManager::connect_system(Duration::from_secs(2))
            .await
            .unwrap();
        let status = manager.unit_status("dbus.service").await.unwrap();
        assert!(status.active_state.is_active());
    }
}
