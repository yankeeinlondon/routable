use std::{future::pending, time::Duration};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    sync::watch,
    task::JoinHandle,
    time::{Instant, MissedTickBehavior},
};
use zbus::{Connection, Proxy, zvariant::OwnedObjectPath};

const SYSTEMD_DESTINATION: &str = "org.freedesktop.systemd1";
const SYSTEMD_MANAGER_PATH: &str = "/org/freedesktop/systemd1";
const SYSTEMD_MANAGER_INTERFACE: &str = "org.freedesktop.systemd1.Manager";
const SYSTEMD_UNIT_INTERFACE: &str = "org.freedesktop.systemd1.Unit";

#[derive(Clone)]
pub struct SystemdManager {
    connection: Connection,
    timeout: Duration,
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
}

#[derive(Clone, Debug)]
pub struct UnitObservation {
    pub status: Option<UnitStatus>,
    pub error: Option<String>,
    pub observed_at: Instant,
}

pub struct UnitObserver {
    receiver: watch::Receiver<UnitObservation>,
    task: JoinHandle<()>,
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
        })
    }

    pub fn from_connection(connection: Connection, timeout: Duration) -> Self {
        Self {
            connection,
            timeout,
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
            .call("GetUnit", &(unit,))
            .await
            .map_err(|source| SystemdError::Call {
                operation: "GetUnit",
                unit: unit.to_owned(),
                source,
            })
    }

    async fn unit_job(
        &self,
        operation: &'static str,
        unit: &str,
    ) -> Result<OwnedObjectPath, SystemdError> {
        let proxy = self.manager_proxy().await?;
        proxy
            .call(operation, &(unit, "replace"))
            .await
            .map_err(|source| SystemdError::Call {
                operation,
                unit: unit.to_owned(),
                source,
            })
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
            path,
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
        Ok(UnitStatus {
            name: unit.to_owned(),
            load_state,
            active_state: UnitActiveState::from_systemd(active_state),
            sub_state,
        })
    }

    /// Publishes D-Bus `ActiveState` property changes through a watch channel.
    /// A bounded periodic refresh catches coalesced/missed signals. Each unit
    /// owns an independent task, so a hung operation cannot stall another
    /// service and the steady-state path never shells out.
    pub fn observe_unit(&self, unit: impl Into<String>, interval: Duration) -> UnitObserver {
        let unit = unit.into();
        let manager = self.clone();
        let initial = UnitObservation {
            status: None,
            error: None,
            observed_at: Instant::now(),
        };
        let (sender, receiver) = watch::channel(initial);
        let task = tokio::spawn(async move {
            let signal_proxy = match manager.unit_path(&unit).await {
                Ok(path) => Proxy::new(
                    &manager.connection,
                    SYSTEMD_DESTINATION,
                    path,
                    SYSTEMD_UNIT_INTERFACE,
                )
                .await
                .ok(),
                Err(_) => None,
            };
            let mut active_state_changes = match signal_proxy.as_ref() {
                Some(proxy) => Some(
                    proxy
                        .receive_property_changed::<String>("ActiveState")
                        .await,
                ),
                None => None,
            };
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut previous: Option<(Option<UnitStatus>, Option<String>)> = None;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {}
                    change = async {
                        match active_state_changes.as_mut() {
                            Some(changes) => changes.next().await,
                            None => pending().await,
                        }
                    } => {
                        if change.is_none() {
                            active_state_changes = None;
                        }
                    }
                }
                let (status, error) = match manager.unit_status(&unit).await {
                    Ok(status) => (Some(status), None),
                    Err(error) => (None, Some(error.to_string())),
                };
                let current = (status, error);
                if previous.as_ref() != Some(&current) {
                    previous = Some(current.clone());
                    let observation = UnitObservation {
                        status: current.0,
                        error: current.1,
                        observed_at: Instant::now(),
                    };
                    if sender.send(observation).is_err() {
                        break;
                    }
                }
            }
        });
        UnitObserver { receiver, task }
    }
}

impl UnitObserver {
    pub fn subscribe(&self) -> watch::Receiver<UnitObservation> {
        self.receiver.clone()
    }

    pub fn current(&self) -> UnitObservation {
        self.receiver.borrow().clone()
    }

    pub fn abort(&self) {
        self.task.abort();
    }
}

impl Drop for UnitObserver {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
