use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Start,
    Wake,
    Drain,
    Sleep,
    Stop,
    Reconcile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum OperationState {
    Pending,
    Running,
    Succeeded,
    Failed { code: String, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: Uuid,
    pub kind: OperationKind,
    pub service: Option<String>,
    pub state: OperationState,
    pub created_unix_ms: u128,
    pub updated_unix_ms: u128,
}

impl Operation {
    fn new(kind: OperationKind, service: Option<String>) -> Self {
        let now = now_ms();
        Self {
            id: Uuid::new_v4(),
            kind,
            service,
            state: OperationState::Pending,
            created_unix_ms: now,
            updated_unix_ms: now,
        }
    }
}

#[derive(Clone)]
pub struct OperationStore {
    capacity: usize,
    inner: Arc<Mutex<VecDeque<Operation>>>,
}

impl OperationStore {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "operation store capacity must be non-zero");
        Self {
            capacity,
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
        }
    }

    pub fn create(&self, kind: OperationKind, service: Option<String>) -> Operation {
        let operation = Operation::new(kind, service);
        let mut operations = self.inner.lock().expect("operation store poisoned");
        if operations.len() == self.capacity {
            operations.pop_front();
        }
        operations.push_back(operation.clone());
        operation
    }

    pub fn update(&self, id: Uuid, state: OperationState) -> Option<Operation> {
        let mut operations = self.inner.lock().expect("operation store poisoned");
        let operation = operations.iter_mut().find(|operation| operation.id == id)?;
        operation.state = state;
        operation.updated_unix_ms = now_ms();
        Some(operation.clone())
    }

    pub fn get(&self, id: Uuid) -> Option<Operation> {
        self.inner
            .lock()
            .expect("operation store poisoned")
            .iter()
            .find(|operation| operation.id == id)
            .cloned()
    }

    pub fn list(&self) -> Vec<Operation> {
        self.inner
            .lock()
            .expect("operation store poisoned")
            .iter()
            .cloned()
            .collect()
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_bounded_and_updates_by_id() {
        let store = OperationStore::new(2);
        let first = store.create(OperationKind::Start, Some("a".into()));
        let second = store.create(OperationKind::Stop, Some("b".into()));
        assert_eq!(
            store
                .update(second.id, OperationState::Succeeded)
                .unwrap()
                .state,
            OperationState::Succeeded
        );
        store.create(OperationKind::Reconcile, None);
        assert!(store.get(first.id).is_none());
        assert_eq!(store.list().len(), 2);
    }
}
