use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use http::{Method, Response, StatusCode, header};
use pingora::{apps::http_app::ServeHttp, protocols::http::ServerSession};
use serde::Deserialize;
use serde_json::json;
use tracing::info;
use uuid::Uuid;

use crate::{
    config::{ConfigStore, DeadlineConfig, RouterConfig},
    pool::{
        DrainRequest, EnsureReadyRequest, LifecycleRequest, PoolError, PoolManager, SleepLevel,
        SleepRequest, StatusRequest, StopRequest, TransitionOutcome,
    },
    state::{Lifecycle, ServiceStatus},
};

use super::{OperationKind, OperationState, OperationStore, TokenAuth};

const MAX_CONTROL_BODY: usize = 64 * 1024;

pub struct ControlPlane {
    config: Arc<ConfigStore>,
    pool: Arc<dyn PoolManager>,
    auth: TokenAuth,
    operations: OperationStore,
}

impl ControlPlane {
    pub fn new(
        config: Arc<ConfigStore>,
        pool: Arc<dyn PoolManager>,
        auth: TokenAuth,
        operation_capacity: usize,
    ) -> Self {
        Self {
            config,
            pool,
            auth,
            operations: OperationStore::new(operation_capacity),
        }
    }

    async fn route(&self, http: &mut ServerSession) -> Response<Vec<u8>> {
        let method = http.req_header().method.clone();
        let path = http.req_header().uri.path().to_owned();
        if path == "/livez" {
            return json_response(StatusCode::OK, json!({"status": "live"}));
        }
        if path == "/readyz" {
            return match self.pool.status(StatusRequest::default()).await {
                Ok(status) => readiness_response(&self.config.snapshot(), &status.services),
                Err(error) => api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    error.code(),
                    &error.to_string(),
                ),
            };
        }
        if !path.starts_with("/pool/v1") {
            return api_error(StatusCode::NOT_FOUND, "not_found", "endpoint not found");
        }
        if !self.auth.authorize(
            http.get_header("authorization")
                .map(|value| value.as_bytes()),
        ) {
            return api_error(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "a valid bearer token is required",
            );
        }

        if method == Method::GET && path == "/pool/v1/services" {
            return self.services().await;
        }
        if method == Method::GET {
            if let Some(id) = path.strip_prefix("/pool/v1/operations/") {
                return self.operation_response(id);
            }
        }
        if method == Method::POST && path == "/pool/v1/reconcile" {
            return self.spawn_reconcile().await;
        }
        if method == Method::POST {
            if let Some(rest) = path.strip_prefix("/pool/v1/services/") {
                if let Some((service, action)) = rest.rsplit_once('/') {
                    let body = match read_options(http).await {
                        Ok(body) => body,
                        Err(response) => return response,
                    };
                    return self.spawn_mutation(service, action, body).await;
                }
            }
        }
        if method == Method::GET
            && (path.contains("/start")
                || path.contains("/wake")
                || path.contains("/drain")
                || path.contains("/sleep")
                || path.contains("/stop")
                || path.ends_with("/reconcile"))
        {
            return api_error(
                StatusCode::METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "lifecycle mutations require POST",
            );
        }
        api_error(StatusCode::NOT_FOUND, "not_found", "endpoint not found")
    }

    async fn services(&self) -> Response<Vec<u8>> {
        match self.pool.status(StatusRequest::default()).await {
            Ok(status) => {
                let config = self.config.snapshot();
                let observed: std::collections::HashMap<_, _> = status
                    .services
                    .into_iter()
                    .map(|status| (status.service.clone(), status))
                    .collect();
                let services: Vec<_> = config
                    .services
                    .iter()
                    .map(|desired| {
                        json!({
                            "desired": desired,
                            "observed": observed.get(&desired.name),
                        })
                    })
                    .collect();
                json_response(StatusCode::OK, json!({"services": services}))
            }
            Err(error) => api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                error.code(),
                &error.to_string(),
            ),
        }
    }

    fn operation_response(&self, id: &str) -> Response<Vec<u8>> {
        match Uuid::parse_str(id)
            .ok()
            .and_then(|id| self.operations.get(id))
        {
            Some(operation) => json_response(StatusCode::OK, operation),
            None => api_error(
                StatusCode::NOT_FOUND,
                "operation_not_found",
                "operation does not exist or aged out",
            ),
        }
    }

    async fn spawn_mutation(
        &self,
        service: &str,
        action: &str,
        options: MutationOptions,
    ) -> Response<Vec<u8>> {
        let config = self.config.snapshot();
        let Some(definition) = config.service(service) else {
            return api_error(
                StatusCode::NOT_FOUND,
                "unknown_service",
                "service is not configured",
            );
        };
        let Some(deadline_ms) =
            mutation_deadline_ms(action, options.deadline_ms, &definition.deadlines)
        else {
            return api_error(
                StatusCode::NOT_FOUND,
                "unknown_action",
                "lifecycle action is not supported",
            );
        };
        let deadline = deadline(deadline_ms);
        let force = options.force.unwrap_or(false);
        let (kind, request) = match action {
            "start" | "wake" => {
                let evict = options.evict.unwrap_or(false);
                (
                    if action == "start" {
                        OperationKind::Start
                    } else {
                        OperationKind::Wake
                    },
                    LifecycleRequest::EnsureReady(EnsureReadyRequest {
                        service: service.to_owned(),
                        deadline,
                        evict,
                        reactivate_drained: true,
                    }),
                )
            }
            "drain" => (
                OperationKind::Drain,
                LifecycleRequest::Drain(DrainRequest {
                    service: service.to_owned(),
                    deadline,
                    force,
                }),
            ),
            "sleep" => {
                let level = match options.level.as_deref() {
                    Some("l2" | "L2" | "2") => SleepLevel::L2,
                    Some("l1" | "L1" | "1") | None => SleepLevel::L1,
                    Some(_) => {
                        return api_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_sleep_level",
                            "sleep level must be l1 or l2",
                        );
                    }
                };
                (
                    OperationKind::Sleep,
                    LifecycleRequest::Sleep(SleepRequest {
                        service: service.to_owned(),
                        level,
                        deadline,
                        force,
                    }),
                )
            }
            "stop" => (
                OperationKind::Stop,
                LifecycleRequest::Stop(StopRequest {
                    service: service.to_owned(),
                    deadline,
                    force,
                }),
            ),
            _ => unreachable!("mutation deadline validation rejected unknown action"),
        };
        let handle = match self.pool.begin_lifecycle(request).await {
            Ok(handle) => handle,
            Err(error) => return mutation_start_error(error),
        };
        let id = handle.operation_id();
        let joined_existing = handle.joined_existing();
        let (operation, inserted) = self
            .operations
            .register_running(id, kind, service.to_owned());
        // Joined flights are audited at the pool boundary so data-plane
        // stampedes and management requests share one authoritative record.
        if !joined_existing {
            info!(
                audit = true,
                operation_id = %id,
                service,
                action,
                joined_existing = false,
                force,
                evict = options.evict.unwrap_or(false),
                "pool mutation accepted"
            );
        }
        if inserted {
            let store = self.operations.clone();
            let action = action.to_owned();
            let service = service.to_owned();
            tokio::spawn(async move {
                let outcome = handle.wait_outcome().await;
                let (state, audit) =
                    terminal_mutation_completion(&action, &service, force, outcome);
                audit.emit();
                store.update(id, state);
            });
        }
        json_response(StatusCode::ACCEPTED, json!({"operation_id": operation.id}))
    }

    async fn spawn_reconcile(&self) -> Response<Vec<u8>> {
        let operation = self.operations.create(OperationKind::Reconcile, None);
        self.operations
            .update(operation.id, OperationState::Running);
        let pool = Arc::clone(&self.pool);
        let store = self.operations.clone();
        let id = operation.id;
        info!(audit = true, operation_id = %id, action = "reconcile", "pool mutation accepted");
        tokio::spawn(async move {
            let state = match pool.reconcile(None).await {
                Ok(_) => OperationState::Succeeded,
                Err(error) => OperationState::Failed {
                    code: error.code().into(),
                    message: error.to_string(),
                },
            };
            store.update(id, state);
        });
        json_response(StatusCode::ACCEPTED, json!({"operation_id": id}))
    }
}

#[async_trait]
impl ServeHttp for ControlPlane {
    async fn response(&self, http: &mut ServerSession) -> Response<Vec<u8>> {
        self.route(http).await
    }
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MutationOptions {
    deadline_ms: Option<u64>,
    force: Option<bool>,
    evict: Option<bool>,
    level: Option<String>,
}

#[derive(Debug, Eq, PartialEq)]
struct TerminalMutationAudit {
    operation_id: Uuid,
    action: String,
    service: String,
    force: bool,
    result: &'static str,
    affected_requests: u64,
}

impl TerminalMutationAudit {
    fn emit(&self) {
        info!(
            audit = true,
            operation_id = %self.operation_id,
            service = %self.service,
            action = %self.action,
            force = self.force,
            result = self.result,
            affected_requests = self.affected_requests,
            "pool mutation completed"
        );
    }
}

fn terminal_mutation_completion(
    action: &str,
    service: &str,
    force: bool,
    outcome: TransitionOutcome,
) -> (OperationState, TerminalMutationAudit) {
    let result = if outcome.result.is_ok() {
        "succeeded"
    } else {
        "failed"
    };
    let audit = TerminalMutationAudit {
        operation_id: outcome.operation_id,
        action: action.to_owned(),
        service: service.to_owned(),
        force,
        result,
        affected_requests: outcome.affected_requests,
    };
    let state = match outcome.result {
        Ok(_) => OperationState::Succeeded,
        Err(error) => OperationState::Failed {
            code: error.code().into(),
            message: error.to_string(),
        },
    };
    (state, audit)
}

async fn read_options(http: &mut ServerSession) -> Result<MutationOptions, Response<Vec<u8>>> {
    let mut body = Vec::new();
    loop {
        match http.read_request_body().await {
            Ok(Some(chunk)) => {
                if body.len().saturating_add(chunk.len()) > MAX_CONTROL_BODY {
                    let _ = http.drain_request_body().await;
                    return Err(api_error(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "request_too_large",
                        "control request body exceeds 64 KiB",
                    ));
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(error) => {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "body_read_failed",
                    &error.to_string(),
                ));
            }
        }
    }
    if body.is_empty() {
        return Ok(MutationOptions::default());
    }
    serde_json::from_slice(&body)
        .map_err(|error| api_error(StatusCode::BAD_REQUEST, "invalid_json", &error.to_string()))
}

fn mutation_deadline_ms(
    action: &str,
    requested_ms: Option<u64>,
    configured: &DeadlineConfig,
) -> Option<u64> {
    let configured_ms = match action {
        "start" => configured.startup_ms,
        "wake" => configured.wake_ms,
        "drain" | "sleep" | "stop" => configured.drain_ms,
        _ => return None,
    };
    Some(requested_ms.unwrap_or(configured_ms))
}

fn deadline(milliseconds: u64) -> std::time::Instant {
    std::time::Instant::now() + Duration::from_millis(milliseconds)
}

fn mutation_start_error(error: PoolError) -> Response<Vec<u8>> {
    let status = match error {
        PoolError::UnknownService(_) => StatusCode::NOT_FOUND,
        PoolError::SleepUnsupported { .. } => StatusCode::BAD_REQUEST,
        PoolError::TransitionInProgress { .. } => StatusCode::CONFLICT,
        _ => StatusCode::CONFLICT,
    };
    api_error(status, error.code(), &error.to_string())
}

#[derive(Debug, Eq, PartialEq)]
struct RoutingReadiness {
    enabled_services: usize,
    routable_services: usize,
}

impl RoutingReadiness {
    fn is_ready(&self) -> bool {
        self.routable_services > 0
    }
}

fn routing_readiness(config: &RouterConfig, services: &[ServiceStatus]) -> RoutingReadiness {
    let observed: HashMap<_, _> = services
        .iter()
        .map(|status| (status.service.as_str(), status.lifecycle))
        .collect();
    let enabled_services = config
        .services
        .iter()
        .filter(|service| service.enabled)
        .count();
    let routable_services = config
        .services
        .iter()
        .filter(|service| service.enabled)
        .filter(|service| {
            observed
                .get(service.name.as_str())
                .is_some_and(|lifecycle| {
                    *lifecycle == Lifecycle::Running
                        || (service.automatic_start
                            && matches!(
                                lifecycle,
                                Lifecycle::Stopped
                                    | Lifecycle::Starting
                                    | Lifecycle::AsleepL1
                                    | Lifecycle::AsleepL2
                                    | Lifecycle::Waking
                            ))
                })
        })
        .count();
    RoutingReadiness {
        enabled_services,
        routable_services,
    }
}

fn readiness_response(config: &RouterConfig, services: &[ServiceStatus]) -> Response<Vec<u8>> {
    let readiness = routing_readiness(config, services);
    if readiness.is_ready() {
        return json_response(
            StatusCode::OK,
            json!({
                "status": "ready",
                "routable_services": readiness.routable_services
            }),
        );
    }
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "routing_not_ready",
        &format!(
            "no enabled service can currently accept requests ({} enabled services configured)",
            readiness.enabled_services
        ),
    )
}

fn api_error(status: StatusCode, code: &str, message: &str) -> Response<Vec<u8>> {
    json_response(status, json!({"error": {"code": code, "message": message}}))
}

fn json_response(status: StatusCode, value: impl serde::Serialize) -> Response<Vec<u8>> {
    match serde_json::to_vec(&value) {
        Ok(body) => Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, body.len())
            .body(body)
            .expect("valid static response"),
        Err(error) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header(header::CONTENT_TYPE, "application/json")
            .body(
                json!({"error": {"code": "serialization_failed", "message": error.to_string()}})
                    .to_string()
                    .into_bytes(),
            )
            .expect("valid static response"),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        sync::atomic::{AtomicBool, Ordering},
    };

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use tempfile::NamedTempFile;
    use tokio::sync::Notify;

    use super::*;
    use crate::{
        pool::{
            Pool, Runner, RunnerError, RunnerObservation,
            ServiceDefinition as PoolServiceDefinition,
        },
        state::{ProbeStatus, SystemdState, SystemdStatus},
    };

    #[derive(Clone)]
    struct ControlledRunner {
        running: Arc<AtomicBool>,
        start_started: Arc<Notify>,
        start_release: Arc<Notify>,
    }

    impl ControlledRunner {
        fn new(running: bool) -> Self {
            Self {
                running: Arc::new(AtomicBool::new(running)),
                start_started: Arc::new(Notify::new()),
                start_release: Arc::new(Notify::new()),
            }
        }
    }

    #[async_trait]
    impl Runner for ControlledRunner {
        async fn observe(
            &self,
            _service: &PoolServiceDefinition,
        ) -> Result<RunnerObservation, RunnerError> {
            let running = self.running.load(Ordering::Acquire);
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
                    healthy: running,
                    ready: running,
                    served_model_matches: running,
                    sleeping_level: None,
                    consecutive_successes: u32::from(running),
                    consecutive_failures: u32::from(!running),
                    age: Some(Duration::ZERO),
                },
                gpus: Vec::new(),
                placement_error: None,
            })
        }

        async fn start(&self, _service: &PoolServiceDefinition) -> Result<(), RunnerError> {
            self.start_started.notify_one();
            self.start_release.notified().await;
            self.running.store(true, Ordering::Release);
            Ok(())
        }

        async fn stop(
            &self,
            _service: &PoolServiceDefinition,
            _force: bool,
        ) -> Result<(), RunnerError> {
            self.running.store(false, Ordering::Release);
            Ok(())
        }

        async fn sleep(
            &self,
            _service: &PoolServiceDefinition,
            _level: SleepLevel,
        ) -> Result<(), RunnerError> {
            Ok(())
        }

        async fn wake(
            &self,
            _service: &PoolServiceDefinition,
            _level: SleepLevel,
        ) -> Result<(), RunnerError> {
            self.running.store(true, Ordering::Release);
            Ok(())
        }
    }

    fn test_auth() -> TokenAuth {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "secret").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
        TokenAuth::load(file.path()).unwrap()
    }

    fn response_json(response: &Response<Vec<u8>>) -> serde_json::Value {
        serde_json::from_slice(response.body()).unwrap()
    }

    fn config() -> RouterConfig {
        serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse")
    }

    fn observed(service: &str, lifecycle: Lifecycle) -> ServiceStatus {
        let mut status = ServiceStatus::new(service);
        status.lifecycle = lifecycle;
        status
    }

    fn configured_deadlines() -> DeadlineConfig {
        DeadlineConfig {
            startup_ms: 601_000,
            wake_ms: 62_000,
            drain_ms: 123_000,
            systemd_job_ms: 34_000,
        }
    }

    #[test]
    fn lifecycle_mutations_use_the_target_services_configured_deadlines() {
        let configured = configured_deadlines();
        for (action, expected_ms) in [
            ("start", configured.startup_ms),
            ("wake", configured.wake_ms),
            ("drain", configured.drain_ms),
            ("sleep", configured.drain_ms),
            ("stop", configured.drain_ms),
        ] {
            assert_eq!(
                mutation_deadline_ms(action, None, &configured),
                Some(expected_ms),
                "incorrect configured deadline for {action}"
            );
        }
        assert_eq!(mutation_deadline_ms("unknown", None, &configured), None);
    }

    #[test]
    fn requested_deadline_overrides_every_lifecycle_default() {
        let configured = configured_deadlines();
        for action in ["start", "wake", "drain", "sleep", "stop"] {
            assert_eq!(
                mutation_deadline_ms(action, Some(7_654), &configured),
                Some(7_654),
                "request override did not win for {action}"
            );
        }
        assert_eq!(
            mutation_deadline_ms("unknown", Some(7_654), &configured),
            None
        );
    }

    #[test]
    fn terminal_tracker_payload_is_bounded_and_retains_force_decision_metadata() {
        let operation_id = Uuid::new_v4();
        let (state, audit) = terminal_mutation_completion(
            "stop",
            "model-a",
            true,
            TransitionOutcome {
                operation_id,
                affected_requests: 8,
                result: Ok(observed("model-a", Lifecycle::Drained)),
            },
        );
        assert_eq!(state, OperationState::Succeeded);
        assert_eq!(
            audit,
            TerminalMutationAudit {
                operation_id,
                action: "stop".into(),
                service: "model-a".into(),
                force: true,
                result: "succeeded",
                affected_requests: 8,
            }
        );

        let (state, audit) = terminal_mutation_completion(
            "stop",
            "model-a",
            true,
            TransitionOutcome {
                operation_id,
                affected_requests: 8,
                result: Err(PoolError::Runner {
                    service: "model-a".into(),
                    message: "unbounded runner diagnostic stays out of the audit payload".into(),
                }),
            },
        );
        assert!(matches!(
            state,
            OperationState::Failed { ref code, .. } if code == "runner_failure"
        ));
        assert_eq!(audit.result, "failed");
        assert_eq!(audit.affected_requests, 8);
        assert_eq!(audit.operation_id, operation_id);
    }

    #[test]
    fn sleep_unsupported_begin_lifecycle_error_maps_to_bad_request() {
        let response = mutation_start_error(PoolError::SleepUnsupported {
            service: "model-a".into(),
            level: SleepLevel::L2,
        });

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(&response)["error"]["code"],
            "sleep_unsupported"
        );
    }

    #[test]
    fn readiness_accepts_running_and_automatic_convergence_states() {
        let config = config();
        for lifecycle in [
            Lifecycle::Running,
            Lifecycle::Stopped,
            Lifecycle::Starting,
            Lifecycle::AsleepL1,
            Lifecycle::AsleepL2,
            Lifecycle::Waking,
        ] {
            assert!(
                routing_readiness(&config, &[observed(&config.services[0].name, lifecycle)])
                    .is_ready(),
                "{lifecycle:?} should accept data-plane requests"
            );
        }
    }

    #[test]
    fn readiness_rejects_non_routable_state_sets() {
        let mut config = config();
        let service = config.services[0].name.clone();
        config.services[0].automatic_start = false;
        for lifecycle in [
            Lifecycle::Reconciling,
            Lifecycle::Stopped,
            Lifecycle::Draining,
            Lifecycle::Drained,
            Lifecycle::Stopping,
            Lifecycle::Failed,
            Lifecycle::Backoff,
            Lifecycle::Stale,
            Lifecycle::Unknown,
        ] {
            assert!(
                !routing_readiness(&config, &[observed(&service, lifecycle)]).is_ready(),
                "{lifecycle:?} should not be considered routable"
            );
        }
        assert!(!routing_readiness(&config, &[]).is_ready());
    }

    #[test]
    fn degraded_optional_service_does_not_fail_router_readiness() {
        let mut config = config();
        let primary = config.services[0].name.clone();
        let mut optional = config.services[0].clone();
        optional.name = "optional".into();
        config.services.push(optional);

        let readiness = routing_readiness(
            &config,
            &[
                observed(&primary, Lifecycle::Running),
                observed("optional", Lifecycle::Failed),
            ],
        );
        assert_eq!(
            readiness,
            RoutingReadiness {
                enabled_services: 2,
                routable_services: 1,
            }
        );
        let response = readiness_response(
            &config,
            &[
                observed(&primary, Lifecycle::Running),
                observed("optional", Lifecycle::Failed),
            ],
        );
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(body["status"], "ready");
        assert_eq!(body["routable_services"], 1);
    }

    #[test]
    fn readiness_response_is_non_success_and_router_shaped_when_no_route_exists() {
        let mut config = config();
        config.services[0].automatic_start = false;
        let response = readiness_response(
            &config,
            &[observed(&config.services[0].name, Lifecycle::Stopped)],
        );

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(body["error"]["code"], "routing_not_ready");
        assert!(body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn concurrent_mutations_share_pool_id_through_lookup_and_terminal_state() {
        let router_config = config();
        let service = router_config.services[0].name.clone();
        let config = Arc::new(ConfigStore::new(router_config).unwrap());
        let runner = ControlledRunner::new(false);
        let controls = runner.clone();
        let pool = Pool::new(
            vec![PoolServiceDefinition::new(
                service.clone(),
                service.clone(),
                vec!["gpu-control".into()],
            )],
            runner,
        )
        .unwrap();
        pool.reconcile(&service).await.unwrap();
        let facade: Arc<dyn PoolManager> = Arc::new(pool.clone());
        let control = ControlPlane::new(config, facade, test_auth(), 4);

        let (first, second) = tokio::join!(
            control.spawn_mutation(&service, "start", MutationOptions::default()),
            control.spawn_mutation(&service, "start", MutationOptions::default())
        );
        assert_eq!(first.status(), StatusCode::ACCEPTED);
        assert_eq!(second.status(), StatusCode::ACCEPTED);
        let first_id = Uuid::parse_str(
            response_json(&first)["operation_id"]
                .as_str()
                .expect("accepted response must contain an operation ID"),
        )
        .unwrap();
        let second_id = Uuid::parse_str(
            response_json(&second)["operation_id"]
                .as_str()
                .expect("joined response must contain an operation ID"),
        )
        .unwrap();
        assert_eq!(first_id, second_id);
        assert_eq!(control.operations.list().len(), 1);

        tokio::time::timeout(Duration::from_secs(1), controls.start_started.notified())
            .await
            .expect("pool start flight should reach the runner");
        let progress = control.operation_response(&first_id.to_string());
        assert_eq!(progress.status(), StatusCode::OK);
        assert_eq!(response_json(&progress)["id"], first_id.to_string());
        assert_eq!(response_json(&progress)["state"]["state"], "running");
        let state = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some(service.clone()),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        assert_eq!(state.lifecycle, Lifecycle::Starting);
        assert_eq!(state.operation_id, Some(first_id));

        let rejected = control
            .spawn_mutation(&service, "stop", MutationOptions::default())
            .await;
        assert_eq!(rejected.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(&rejected)["error"]["code"],
            "transition_in_progress"
        );
        assert_eq!(control.operations.list().len(), 1);

        controls.start_release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let terminal = control.operation_response(&first_id.to_string());
                if response_json(&terminal)["state"]["state"] == "succeeded" {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("operation tracker should publish terminal success");
        let terminal = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some(service),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        assert_eq!(terminal.lifecycle, Lifecycle::Running);
        assert_eq!(terminal.operation_id, Some(first_id));
    }

    #[tokio::test]
    async fn drain_force_option_is_plumbed_without_changing_non_force_safety() {
        let router_config = config();
        let service = router_config.services[0].name.clone();
        let config = Arc::new(ConfigStore::new(router_config).unwrap());
        let runner = ControlledRunner::new(true);
        let pool = Pool::new(
            vec![PoolServiceDefinition::new(
                service.clone(),
                service.clone(),
                vec!["gpu-control".into()],
            )],
            runner,
        )
        .unwrap();
        pool.reconcile(&service).await.unwrap();
        let active = PoolManager::acquire_active(&pool, service.clone())
            .await
            .unwrap();
        let facade: Arc<dyn PoolManager> = Arc::new(pool.clone());
        let control = ControlPlane::new(config, facade, test_auth(), 4);

        let non_force = control
            .spawn_mutation(
                &service,
                "drain",
                MutationOptions {
                    deadline_ms: Some(0),
                    force: Some(false),
                    ..MutationOptions::default()
                },
            )
            .await;
        let non_force_id =
            Uuid::parse_str(response_json(&non_force)["operation_id"].as_str().unwrap()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let operation = control.operation_response(&non_force_id.to_string());
                if response_json(&operation)["state"]["state"] == "failed" {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("non-force drain should publish its deadline failure");
        let still_running = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some(service.clone()),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        assert_eq!(still_running.lifecycle, Lifecycle::Running);

        let forced = control
            .spawn_mutation(
                &service,
                "drain",
                MutationOptions {
                    deadline_ms: Some(0),
                    force: Some(true),
                    ..MutationOptions::default()
                },
            )
            .await;
        let forced_id =
            Uuid::parse_str(response_json(&forced)["operation_id"].as_str().unwrap()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let operation = control.operation_response(&forced_id.to_string());
                if response_json(&operation)["state"]["state"] == "succeeded" {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("force drain should publish terminal success");
        let drained = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some(service),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        assert_eq!(drained.lifecycle, Lifecycle::Drained);
        assert_eq!(drained.active_requests, 1);
        drop(active);
    }

    #[tokio::test]
    async fn sleep_force_option_is_plumbed_to_safe_forced_stop_fallback() {
        let router_config = config();
        let service = router_config.services[0].name.clone();
        let config = Arc::new(ConfigStore::new(router_config).unwrap());
        let runner = ControlledRunner::new(true);
        let mut definition = PoolServiceDefinition::new(
            service.clone(),
            service.clone(),
            vec!["gpu-control".into()],
        );
        definition.sleep_levels = vec![SleepLevel::L1];
        let pool = Pool::new(vec![definition], runner).unwrap();
        pool.reconcile(&service).await.unwrap();
        let active = PoolManager::acquire_active(&pool, service.clone())
            .await
            .unwrap();
        let facade: Arc<dyn PoolManager> = Arc::new(pool.clone());
        let control = ControlPlane::new(config, facade, test_auth(), 4);

        let forced = control
            .spawn_mutation(
                &service,
                "sleep",
                MutationOptions {
                    deadline_ms: Some(0),
                    force: Some(true),
                    level: Some("l1".into()),
                    ..MutationOptions::default()
                },
            )
            .await;
        assert_eq!(forced.status(), StatusCode::ACCEPTED);
        let id = Uuid::parse_str(response_json(&forced)["operation_id"].as_str().unwrap()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let operation = control.operation_response(&id.to_string());
                if response_json(&operation)["state"]["state"] == "succeeded" {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("forced sleep fallback should publish terminal success");

        let status = PoolManager::status(
            &pool,
            StatusRequest {
                service: Some(service),
            },
        )
        .await
        .unwrap()
        .services
        .pop()
        .unwrap();
        assert_eq!(status.lifecycle, Lifecycle::Drained);
        assert_eq!(status.systemd.state, SystemdState::Inactive);
        assert_eq!(status.active_requests, 1);
        drop(active);
    }
}
