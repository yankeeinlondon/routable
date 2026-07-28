use std::{future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use http::{Method, Response, StatusCode, header};
use pingora::{apps::http_app::ServeHttp, protocols::http::ServerSession};
use serde::Deserialize;
use serde_json::json;
use tracing::info;
use uuid::Uuid;

use crate::{
    config::ConfigStore,
    pool::{
        DrainRequest, EnsureReadyRequest, PoolManager, SleepLevel, SleepRequest, StatusRequest,
        StopRequest,
    },
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
                Ok(_) => json_response(StatusCode::OK, json!({"status": "ready"})),
                Err(error) => json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"status": "not_ready", "code": error.code(), "message": error.to_string()}),
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
                return match Uuid::parse_str(id)
                    .ok()
                    .and_then(|id| self.operations.get(id))
                {
                    Some(operation) => json_response(StatusCode::OK, operation),
                    None => api_error(
                        StatusCode::NOT_FOUND,
                        "operation_not_found",
                        "operation does not exist or aged out",
                    ),
                };
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
                    return self.spawn_mutation(service, action, body);
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

    fn spawn_mutation(
        &self,
        service: &str,
        action: &str,
        options: MutationOptions,
    ) -> Response<Vec<u8>> {
        if self.config.snapshot().service(service).is_none() {
            return api_error(
                StatusCode::NOT_FOUND,
                "unknown_service",
                "service is not configured",
            );
        }
        let (kind, future): (OperationKind, MutationFuture) = match action {
            "start" | "wake" => {
                let pool = Arc::clone(&self.pool);
                let service = service.to_owned();
                let evict = options.evict.unwrap_or(false);
                let deadline = deadline(options.deadline_ms, 300_000);
                (
                    if action == "start" {
                        OperationKind::Start
                    } else {
                        OperationKind::Wake
                    },
                    Box::pin(async move {
                        pool.ensure_ready(EnsureReadyRequest {
                            service,
                            deadline,
                            evict,
                        })
                        .await
                        .map(|_| ())
                    }),
                )
            }
            "drain" => {
                let pool = Arc::clone(&self.pool);
                let service = service.to_owned();
                let deadline = deadline(options.deadline_ms, 120_000);
                (
                    OperationKind::Drain,
                    Box::pin(async move {
                        pool.drain(DrainRequest { service, deadline })
                            .await
                            .map(|_| ())
                    }),
                )
            }
            "sleep" => {
                let pool = Arc::clone(&self.pool);
                let service = service.to_owned();
                let deadline = deadline(options.deadline_ms, 120_000);
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
                    Box::pin(async move {
                        pool.sleep(SleepRequest {
                            service,
                            level,
                            deadline,
                        })
                        .await
                        .map(|_| ())
                    }),
                )
            }
            "stop" => {
                let pool = Arc::clone(&self.pool);
                let service = service.to_owned();
                let deadline = deadline(options.deadline_ms, 120_000);
                let force = options.force.unwrap_or(false);
                (
                    OperationKind::Stop,
                    Box::pin(async move {
                        pool.stop(StopRequest {
                            service,
                            deadline,
                            force,
                        })
                        .await
                        .map(|_| ())
                    }),
                )
            }
            _ => {
                return api_error(
                    StatusCode::NOT_FOUND,
                    "unknown_action",
                    "lifecycle action is not supported",
                );
            }
        };
        let operation = self.operations.create(kind, Some(service.to_owned()));
        self.operations
            .update(operation.id, OperationState::Running);
        info!(
            audit = true,
            operation_id = %operation.id,
            service,
            action,
            force = options.force.unwrap_or(false),
            evict = options.evict.unwrap_or(false),
            "pool mutation accepted"
        );
        let store = self.operations.clone();
        let id = operation.id;
        tokio::spawn(async move {
            let state = match future.await {
                Ok(()) => OperationState::Succeeded,
                Err(error) => OperationState::Failed {
                    code: error.code().into(),
                    message: error.to_string(),
                },
            };
            store.update(id, state);
        });
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

type MutationFuture =
    std::pin::Pin<Box<dyn Future<Output = Result<(), crate::pool::PoolError>> + Send + 'static>>;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MutationOptions {
    deadline_ms: Option<u64>,
    force: Option<bool>,
    evict: Option<bool>,
    level: Option<String>,
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

fn deadline(requested_ms: Option<u64>, default_ms: u64) -> std::time::Instant {
    std::time::Instant::now() + Duration::from_millis(requested_ms.unwrap_or(default_ms))
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
