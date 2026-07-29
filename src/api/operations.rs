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
    fn new(id: Uuid, kind: OperationKind, service: Option<String>, state: OperationState) -> Self {
        let now = now_ms();
        Self {
            id,
            kind,
            service,
            state,
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
        let operation = Operation::new(Uuid::new_v4(), kind, service, OperationState::Pending);
        let mut operations = self.inner.lock().expect("operation store poisoned");
        self.make_room_for_live_operation(&mut operations);
        operations.push_back(operation.clone());
        operation
    }

    /// Register a pool-owned lifecycle operation as running.
    ///
    /// The boolean is true only for the caller that inserted this ID. Joiners receive the
    /// existing record and must not spawn another terminal-state tracker.
    pub fn register_running(
        &self,
        id: Uuid,
        kind: OperationKind,
        service: String,
    ) -> (Operation, bool) {
        let mut operations = self.inner.lock().expect("operation store poisoned");
        if let Some(operation) = operations.iter().find(|operation| operation.id == id) {
            return (operation.clone(), false);
        }
        let operation = Operation::new(id, kind, Some(service), OperationState::Running);
        self.make_room_for_live_operation(&mut operations);
        operations.push_back(operation.clone());
        (operation, true)
    }

    pub fn update(&self, id: Uuid, state: OperationState) -> Option<Operation> {
        let mut operations = self.inner.lock().expect("operation store poisoned");
        let operation = operations.iter_mut().find(|operation| operation.id == id)?;
        operation.state = state;
        operation.updated_unix_ms = now_ms();
        let updated = operation.clone();
        if is_terminal(&updated.state) {
            self.trim_terminal_operations(&mut operations);
        }
        Some(updated)
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

    /// Retention is strictly bounded whenever enough terminal records exist to evict. If all
    /// retained records are pending or running, the store temporarily exceeds its configured
    /// capacity rather than making a live operation (and its single-flight ID) disappear.
    fn make_room_for_live_operation(&self, operations: &mut VecDeque<Operation>) {
        while operations.len() >= self.capacity {
            let Some(index) = operations
                .iter()
                .position(|operation| is_terminal(&operation.state))
            else {
                break;
            };
            operations.remove(index);
        }
    }

    fn trim_terminal_operations(&self, operations: &mut VecDeque<Operation>) {
        while operations.len() > self.capacity {
            let Some(index) = operations
                .iter()
                .position(|operation| is_terminal(&operation.state))
            else {
                break;
            };
            operations.remove(index);
        }
    }
}

fn is_terminal(state: &OperationState) -> bool {
    matches!(
        state,
        OperationState::Succeeded | OperationState::Failed { .. }
    )
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
    fn terminal_records_are_evicted_before_live_records() {
        let store = OperationStore::new(2);
        let first = store.create(OperationKind::Start, Some("a".into()));
        let second = store.create(OperationKind::Stop, Some("b".into()));
        assert_eq!(
            store
                .update(first.id, OperationState::Succeeded)
                .unwrap()
                .state,
            OperationState::Succeeded
        );
        let third = store.create(OperationKind::Reconcile, None);
        assert!(store.get(first.id).is_none());
        assert!(store.get(second.id).is_some());
        assert!(store.get(third.id).is_some());
        assert_eq!(store.list().len(), 2);
    }

    #[test]
    fn live_pool_owned_ids_survive_temporary_capacity_overflow_and_still_dedupe() {
        let store = OperationStore::new(1);
        let id = Uuid::new_v4();
        let (first, inserted) = store.register_running(id, OperationKind::Start, "a".into());
        assert!(inserted);
        assert_eq!(first.id, id);
        assert_eq!(first.state, OperationState::Running);

        let (joined, inserted) = store.register_running(id, OperationKind::Wake, "a".into());
        assert!(!inserted);
        assert_eq!(joined.id, id);
        assert_eq!(joined.kind, OperationKind::Start);

        let replacement = Uuid::new_v4();
        store.register_running(replacement, OperationKind::Stop, "b".into());
        assert!(store.get(id).is_some());
        assert!(store.get(replacement).is_some());
        assert_eq!(store.list().len(), 2);

        let (joined, inserted) = store.register_running(id, OperationKind::Wake, "a".into());
        assert!(!inserted);
        assert_eq!(joined.id, id);
        assert_eq!(joined.kind, OperationKind::Start);
    }

    #[test]
    fn temporary_live_overflow_shrinks_as_operations_become_terminal() {
        let store = OperationStore::new(2);
        let ids: Vec<_> = ["a", "b", "c", "d"]
            .into_iter()
            .map(|service| {
                let id = Uuid::new_v4();
                store.register_running(id, OperationKind::Start, service.into());
                id
            })
            .collect();

        assert_eq!(store.list().len(), 4);
        for id in &ids {
            assert!(store.get(*id).is_some());
        }

        store.update(ids[0], OperationState::Succeeded).unwrap();
        assert_eq!(store.list().len(), 3);
        for id in &ids[1..] {
            assert!(store.get(*id).is_some());
        }

        store
            .update(
                ids[1],
                OperationState::Failed {
                    code: "test".into(),
                    message: "terminal".into(),
                },
            )
            .unwrap();
        assert_eq!(store.list().len(), 2);
        assert!(store.get(ids[2]).is_some());
        assert!(store.get(ids[3]).is_some());
    }
}
