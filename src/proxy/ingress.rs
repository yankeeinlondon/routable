use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use pingora::{
    apps::{HttpPersistentSettings, HttpServerApp, ReusedHttpStream},
    http::ResponseHeader,
    protocols::http::ServerSession,
    server::ShutdownWatch,
};
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HOST};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;
use uuid::Uuid;

use crate::{
    api::TokenAuth,
    config::{ConfigStore, RouterConfig, ServiceDefinition},
    metrics,
    pool::{EnsureReadyRequest, PoolError, PoolManager, StatusRequest},
    state::{ActiveGuard, Lifecycle, ServiceStatus},
};

use super::{
    ErrorDialect, RequestRoutingError, RouterErrorEnvelope, is_hop_by_hop, parse_model_route,
};

const MODEL_ENDPOINTS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/score",
    "/v1/rerank",
    "/v2/rerank",
    "/v2/embed",
    "/score",
    "/rerank",
    "/classify",
    "/tokenize",
    "/detokenize",
];
const MODELS_ENDPOINT: &str = "/v1/models";

#[derive(Debug, Serialize, PartialEq, Eq)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelDescription>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct ModelDescription {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
    root: String,
    parent: Option<String>,
    aliases: Vec<String>,
    max_model_len: Option<u64>,
}

struct RouterOwnedResponse {
    status: u16,
    body: Value,
}

#[derive(Debug, Serialize)]
struct StartupProgress {
    service: String,
    elapsed_seconds: u64,
    typical_duration_seconds: u64,
}

#[derive(Debug, Serialize)]
struct StartupProgressEnvelope {
    #[serde(flatten)]
    error: RouterErrorEnvelope,
    progress: StartupProgress,
}

struct ActiveStreamGuard {
    gauge: prometheus::IntGauge,
}

impl ActiveStreamGuard {
    fn for_response(service: &str, headers: &http::HeaderMap) -> Option<Self> {
        is_event_stream(headers).then(|| {
            let gauge = metrics::global()
                .active_streams
                .with_label_values(&[service]);
            gauge.inc();
            Self { gauge }
        })
    }
}

impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

/// Bounded custom ingress chosen by the Phase-0 body-replay spike.
///
/// Only the request body is buffered. Upstream response bytes are copied to
/// the downstream session as each chunk arrives, preserving SSE streaming.
pub struct DataPlane {
    config: Arc<ConfigStore>,
    pool: Arc<dyn PoolManager>,
    auth: TokenAuth,
    client: reqwest::Client,
    inter_chunk_timeout: Duration,
    global_parked: Arc<Semaphore>,
    model_parked: RwLock<HashMap<String, Arc<Semaphore>>>,
}

impl DataPlane {
    pub fn new(
        config: Arc<ConfigStore>,
        pool: Arc<dyn PoolManager>,
        auth: TokenAuth,
    ) -> Result<Self, reqwest::Error> {
        let snapshot = config.snapshot();
        let client = build_upstream_client(&snapshot)?;
        let mut model_parked = HashMap::new();
        for service in &snapshot.services {
            model_parked
                .entry(service.served_model.clone())
                .or_insert_with(|| {
                    Arc::new(Semaphore::new(
                        snapshot.policies.per_model_parked_request_limit,
                    ))
                });
        }
        let inter_chunk_timeout = Duration::from_millis(snapshot.policies.inter_chunk_timeout_ms);
        Ok(Self {
            config,
            pool,
            auth,
            client,
            inter_chunk_timeout,
            global_parked: Arc::new(Semaphore::new(
                snapshot.policies.global_parked_request_limit,
            )),
            model_parked: RwLock::new(model_parked),
        })
    }

    async fn process(&self, http: &mut ServerSession) -> Result<(), IngressError> {
        // First statement in the request path on purpose. Everything below it —
        // model discovery, the body read, parking, eviction, and cold starts —
        // spends router state or GPU capacity, so an unauthenticated caller must
        // not reach any of it. Liveness and readiness live on the management
        // listener and stay credential-free there.
        if !self.auth.authorize(
            http.get_header(AUTHORIZATION)
                .map(http::HeaderValue::as_bytes),
        ) {
            return self
                .write_error(
                    http,
                    401,
                    "unauthorized",
                    "a valid bearer token is required",
                    None,
                )
                .await;
        }

        let method = http.req_header().method.clone();
        let path = http.req_header().uri.path().to_owned();
        if let Some(response) =
            router_owned_response(&method, &path, self.config.snapshot().as_ref())
        {
            return self
                .write_json(http, response.status, &response.body, None)
                .await;
        }
        if path.starts_with("/sleep")
            || path.starts_with("/wake_up")
            || path.starts_with("/is_sleeping")
            || path.starts_with("/pool")
        {
            return self
                .write_error(
                    http,
                    404,
                    "endpoint_not_found",
                    "this endpoint is not exposed on the data plane",
                    None,
                )
                .await;
        }
        if !MODEL_ENDPOINTS.contains(&path.as_str()) {
            return self
                .write_error(
                    http,
                    404,
                    "endpoint_not_found",
                    "unsupported inference endpoint",
                    None,
                )
                .await;
        }
        if http.req_header().method != http::Method::POST {
            return self
                .write_error(
                    http,
                    405,
                    "method_not_allowed",
                    "model-bearing inference endpoints require POST",
                    None,
                )
                .await;
        }

        let config = self.config.snapshot();
        let max_body =
            usize::try_from(config.policies.maximum_request_body_bytes).unwrap_or(usize::MAX);
        if content_length(http).is_some_and(|length| length > max_body) {
            http.drain_request_body()
                .await
                .map_err(IngressError::Pingora)?;
            let error = RequestRoutingError::TooLarge { limit: max_body };
            return self.write_routing_error(http, &error).await;
        }
        let body = match read_bounded_body(http, max_body).await {
            Ok(body) => body,
            Err(IngressError::Routing(error)) => {
                return self.write_routing_error(http, &error).await;
            }
            Err(error) => return Err(error),
        };
        let aliases = alias_map(&config);
        let route = match parse_model_route(&body, max_body, &aliases) {
            Ok(route) => route,
            Err(error) => return self.write_routing_error(http, &error).await,
        };
        let request_id = request_id(http);
        let traceparent = traceparent(http);
        let started = Instant::now();

        let (mut service, active) = match self.admit(http, &config, &route.canonical, started).await
        {
            Ok(admission) => admission,
            Err(IngressError::ClientDisconnected) => return Err(IngressError::ClientDisconnected),
            Err(error) => {
                metrics::global()
                    .requests
                    .with_label_values(&[&route.canonical, "admission_error"])
                    .inc();
                return self.write_ingress_error(http, error).await;
            }
        };
        let mut active = Some(active);

        let body = body.freeze();
        let mut result = self
            .proxy_upstream(http, &service, body.clone(), &request_id, &traceparent)
            .await;
        if matches!(&result, Err(IngressError::ConnectFailure(_))) {
            drop(active.take());
            if let Some((alternative, guard)) = self
                .choose_running(&config, &route.canonical, Some(&service.name))
                .await?
            {
                service = alternative;
                active = Some(guard);
                result = self
                    .proxy_upstream(http, &service, body, &request_id, &traceparent)
                    .await;
            }
        }
        drop(active);
        metrics::global()
            .request_seconds
            .with_label_values(&[&route.canonical])
            .observe(started.elapsed().as_secs_f64());
        match result {
            Ok(status) => {
                metrics::global()
                    .requests
                    .with_label_values(&[&route.canonical, &status.to_string()])
                    .inc();
                Ok(())
            }
            // No error envelope can reach a socket the client already closed, and
            // writing one would only surface as a second, misleading failure.
            Err(IngressError::ClientDisconnected) => {
                metrics::global()
                    .requests
                    .with_label_values(&[&route.canonical, "client_disconnected"])
                    .inc();
                Err(IngressError::ClientDisconnected)
            }
            Err(error) => {
                metrics::global()
                    .requests
                    .with_label_values(&[&route.canonical, "proxy_error"])
                    .inc();
                if http.response_written().is_none() {
                    self.write_error(http, 502, "upstream_error", &error.to_string(), Some(1))
                        .await
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn admit(
        &self,
        http: &mut ServerSession,
        config: &RouterConfig,
        model: &str,
        started: Instant,
    ) -> Result<(ServiceDefinition, ActiveGuard), IngressError> {
        if let Some(admission) = self.choose_running(config, model, None).await? {
            return Ok(admission);
        }

        let status = self
            .pool
            .status(StatusRequest::default())
            .await
            .map_err(IngressError::Pool)?;
        let by_name: HashMap<_, _> = status
            .services
            .iter()
            .map(|status| (status.service.as_str(), status.lifecycle))
            .collect();
        let automatic: Vec<_> = config
            .services_for_model(model)
            .into_iter()
            .filter(|service| service.enabled && service.automatic_start)
            .collect();
        let all_drained = !automatic.is_empty()
            && automatic
                .iter()
                .all(|service| by_name.get(service.name.as_str()) == Some(&Lifecycle::Drained));
        let service = automatic
            .into_iter()
            .filter(|service| by_name.get(service.name.as_str()) != Some(&Lifecycle::Drained))
            .max_by_key(|service| service.priority)
            .cloned()
            .ok_or_else(|| {
                if all_drained {
                    IngressError::Unavailable {
                        code: "model_drained",
                        message: format!("model `{model}` is administratively drained"),
                        retry_after: None,
                    }
                } else {
                    IngressError::Unavailable {
                        code: "automatic_start_disabled",
                        message: format!("no enabled auto-start service serves `{model}`"),
                        retry_after: None,
                    }
                }
            })?;
        let parking_deadline = started + Duration::from_millis(service.parking.deadline_ms);
        let lifecycle_deadline = started
            + Duration::from_millis(service.deadlines.startup_ms.max(service.deadlines.wake_ms));
        let permits = self.acquire_parking(model, &service)?;
        let parked = self
            .pool
            .acquire_parked(service.name.clone())
            .map_err(IngressError::Pool)?;
        let parking_started = Instant::now();
        metrics::global().parked.with_label_values(&[model]).inc();
        metrics::global().global_parked.inc();

        let ensure = self.pool.ensure_ready(EnsureReadyRequest {
            service: service.name.clone(),
            deadline: lifecycle_deadline,
            evict: true,
            reactivate_drained: false,
        });
        tokio::pin!(ensure);
        let outcome = tokio::select! {
            result = tokio::time::timeout_at(parking_deadline.into(), &mut ensure) => {
                match result {
                    Ok(result) => result.map_err(IngressError::Pool),
                    Err(_) => Err(IngressError::ParkingExpired {
                        model: model.to_string(),
                        service: service.name.clone(),
                        elapsed_seconds: parking_started.elapsed().as_secs(),
                        typical_duration_seconds: service.parking.retry_after_seconds,
                        retry_after: service.parking.retry_after_seconds,
                    }),
                }
            },
            cancelled = downstream_cancelled(http) => Err(cancelled),
        };

        metrics::global()
            .parking_seconds
            .with_label_values(&[model])
            .observe(parking_started.elapsed().as_secs_f64());
        metrics::global().parked.with_label_values(&[model]).dec();
        metrics::global().global_parked.dec();
        drop(parked);
        drop(permits);
        outcome?;

        // Re-select across every now-ready replica, then atomically acquire its
        // active guard. A drain racing this step fails closed.
        self.choose_running(config, model, None)
            .await?
            .ok_or_else(|| IngressError::Unavailable {
                code: "model_not_ready",
                message: format!("model `{model}` did not become routable"),
                retry_after: Some(service.parking.retry_after_seconds),
            })
    }

    async fn choose_running(
        &self,
        config: &RouterConfig,
        model: &str,
        excluded_service: Option<&str>,
    ) -> Result<Option<(ServiceDefinition, ActiveGuard)>, IngressError> {
        let status = self
            .pool
            .status(StatusRequest::default())
            .await
            .map_err(IngressError::Pool)?;
        let by_name: HashMap<&str, &ServiceStatus> = status
            .services
            .iter()
            .map(|status| (status.service.as_str(), status))
            .collect();
        let mut candidates: Vec<_> = config
            .services_for_model(model)
            .into_iter()
            .filter(|service| {
                service.enabled
                    && excluded_service != Some(service.name.as_str())
                    && by_name
                        .get(service.name.as_str())
                        .is_some_and(|status| status.lifecycle == Lifecycle::Running)
            })
            .cloned()
            .collect();
        candidates.sort_by_key(|service| {
            by_name
                .get(service.name.as_str())
                .map_or(u64::MAX, |status| status.active_requests)
        });
        acquire_first_running(candidates, |service| self.pool.acquire_active(service))
            .await
            .map_err(IngressError::Pool)
    }

    fn acquire_parking(
        &self,
        model: &str,
        service: &ServiceDefinition,
    ) -> Result<ParkingPermits, IngressError> {
        let global = self
            .global_parked
            .clone()
            .try_acquire_owned()
            .map_err(|_| IngressError::Unavailable {
                code: "parking_capacity_exceeded",
                message: "global parked-request capacity is exhausted".into(),
                retry_after: Some(service.parking.retry_after_seconds),
            })?;
        // Registration publishes a fully validated config snapshot before demand can
        // reach this point. Lazily inserting the canonical model's semaphore makes
        // the data-plane registry additive without replacing permits held by existing
        // requests.
        let existing = {
            self.model_parked
                .read()
                .expect("model parking registry lock poisoned")
                .get(model)
                .cloned()
        };
        let model_semaphore = existing.unwrap_or_else(|| {
            let limit = self
                .config
                .snapshot()
                .policies
                .per_model_parked_request_limit;
            Arc::clone(
                self.model_parked
                    .write()
                    .expect("model parking registry lock poisoned")
                    .entry(model.to_string())
                    .or_insert_with(|| Arc::new(Semaphore::new(limit))),
            )
        });
        let model = model_semaphore
            .try_acquire_owned()
            .map_err(|_| IngressError::Unavailable {
                code: "parking_capacity_exceeded",
                message: format!("parked-request capacity for `{model}` is exhausted"),
                retry_after: Some(service.parking.retry_after_seconds),
            })?;
        Ok(ParkingPermits {
            _global: global,
            _model: model,
        })
    }

    async fn proxy_upstream(
        &self,
        http: &mut ServerSession,
        service: &ServiceDefinition,
        body: Bytes,
        request_id: &str,
        traceparent: &str,
    ) -> Result<u16, IngressError> {
        let path_and_query = http
            .req_header()
            .uri
            .path_and_query()
            .map_or("/", |value| value.as_str());
        let url = format!("http://{}{}", service.backend.socket_addr(), path_and_query);
        let mut request = self
            .client
            .request(http.req_header().method.clone(), url)
            .body(body);
        let connection_headers = connection_tokens(&http.req_header().headers);
        for (name, value) in &http.req_header().headers {
            // The inference token authenticates the client to the router and
            // nothing else. Backends are loopback-bound and hold no matching key,
            // so forwarding it would only widen where the credential is observable.
            if !is_hop_by_hop(name.as_str())
                && !connection_headers
                    .iter()
                    .any(|token| token.eq_ignore_ascii_case(name.as_str()))
                && name != HOST
                && name != CONTENT_LENGTH
                && name != AUTHORIZATION
                && name.as_str() != "x-request-id"
            {
                request = request.header(name, value);
            }
        }
        request = request
            .header(HOST, service.backend.socket_addr().to_string())
            .header("x-request-id", request_id);
        if http.get_header_bytes("traceparent").is_empty() {
            request = request.header("traceparent", traceparent);
        }
        // The window between forwarding and the first response byte is the longest
        // part of a cold generation, and nothing else in this function touches the
        // downstream socket until headers exist. Losing this race drops the
        // in-flight `send()` future, which closes the upstream connection and lets
        // the exact active guard go instead of holding it for an abandoned answer.
        let mut response = tokio::select! {
            result = request.send() => result.map_err(|error| {
                if error.is_connect() {
                    IngressError::ConnectFailure(error)
                } else {
                    classify_upstream_error(
                        error,
                        self.inter_chunk_timeout,
                        "waiting for response headers",
                    )
                }
            })?,
            cancelled = downstream_cancelled(http) => return Err(cancelled),
        };
        let status = response.status().as_u16();
        let _active_stream = ActiveStreamGuard::for_response(&service.name, response.headers());
        let mut downstream = ResponseHeader::build(status, Some(response.headers().len()))
            .map_err(IngressError::Pingora)?;
        let connection_headers = connection_tokens(response.headers());
        for (name, value) in response.headers() {
            if !is_hop_by_hop(name.as_str())
                && !connection_headers
                    .iter()
                    .any(|token| token.eq_ignore_ascii_case(name.as_str()))
            {
                downstream
                    .append_header(name.clone(), value.clone())
                    .map_err(IngressError::Pingora)?;
            }
        }
        downstream
            .insert_header("x-request-id", request_id)
            .map_err(IngressError::Pingora)?;
        // A missing length means chunked H1 or DATA frames on H2. Never add
        // compression, and never aggregate SSE chunks.
        if http.as_http1().is_some() && !response.headers().contains_key(CONTENT_LENGTH) {
            downstream
                .insert_header("transfer-encoding", "chunked")
                .map_err(IngressError::Pingora)?;
        }
        http.write_response_header(Box::new(downstream))
            .await
            .map_err(IngressError::Pingora)?;
        loop {
            // A stalled backend still yields the bounded idle-timeout classification
            // from the upstream arm; this select only adds the case where the client
            // leaves between chunks, which a write alone cannot reveal.
            let chunk = tokio::select! {
                chunk = response.chunk() => chunk.map_err(|error| {
                    classify_upstream_error(
                        error,
                        self.inter_chunk_timeout,
                        "reading the response body",
                    )
                })?,
                cancelled = downstream_cancelled(http) => return Err(cancelled),
            };
            let Some(chunk) = chunk else { break };
            http.write_response_body(chunk, false)
                .await
                .map_err(IngressError::Pingora)?;
        }
        http.write_response_body(Bytes::new(), true)
            .await
            .map_err(IngressError::Pingora)?;
        Ok(status)
    }

    async fn write_routing_error(
        &self,
        http: &mut ServerSession,
        error: &RequestRoutingError,
    ) -> Result<(), IngressError> {
        let dialect = ErrorDialect::for_path(http.req_header().uri.path());
        self.write_json(http, error.status(), &error.envelope(dialect), None)
            .await
    }

    async fn write_ingress_error(
        &self,
        http: &mut ServerSession,
        error: IngressError,
    ) -> Result<(), IngressError> {
        match error {
            IngressError::Pool(pool) => {
                let retry = matches!(
                    pool,
                    PoolError::Deadline { .. } | PoolError::PlacementUnavailable { .. }
                )
                .then_some(30);
                self.write_error(http, 503, pool.code(), &pool.to_string(), retry)
                    .await
            }
            IngressError::Unavailable {
                code,
                message,
                retry_after,
            } => {
                self.write_error(http, 503, code, &message, retry_after)
                    .await
            }
            IngressError::ParkingExpired {
                model,
                service,
                elapsed_seconds,
                typical_duration_seconds,
                retry_after,
            } => {
                let message = format!(
                    "model `{model}` is starting on service `{service}`: {elapsed_seconds} s elapsed, typically ~{typical_duration_seconds} s"
                );
                let error = RouterErrorEnvelope::new(
                    ErrorDialect::for_path(http.req_header().uri.path()),
                    503,
                    "model_starting",
                    message,
                    "router_error",
                );
                self.write_json(
                    http,
                    503,
                    &StartupProgressEnvelope {
                        error,
                        progress: StartupProgress {
                            service,
                            elapsed_seconds,
                            typical_duration_seconds,
                        },
                    },
                    Some(retry_after),
                )
                .await
            }
            other => Err(other),
        }
    }

    async fn write_error(
        &self,
        http: &mut ServerSession,
        status: u16,
        code: &str,
        message: &str,
        retry_after: Option<u64>,
    ) -> Result<(), IngressError> {
        let error = RouterErrorEnvelope::new(
            ErrorDialect::for_path(http.req_header().uri.path()),
            status,
            code,
            message,
            "router_error",
        );
        self.write_json(http, status, &error, retry_after).await
    }

    async fn write_json(
        &self,
        http: &mut ServerSession,
        status: u16,
        value: &impl serde::Serialize,
        retry_after: Option<u64>,
    ) -> Result<(), IngressError> {
        let body = serde_json::to_vec(value).map_err(IngressError::Json)?;
        let mut response = ResponseHeader::build(status, Some(4)).map_err(IngressError::Pingora)?;
        response
            .insert_header("content-type", "application/json")
            .map_err(IngressError::Pingora)?;
        response
            .insert_header("content-length", body.len().to_string())
            .map_err(IngressError::Pingora)?;
        if let Some(seconds) = retry_after {
            response
                .insert_header("retry-after", seconds.to_string())
                .map_err(IngressError::Pingora)?;
        }
        http.write_response_header(Box::new(response))
            .await
            .map_err(IngressError::Pingora)?;
        http.write_response_body(Bytes::from(body), true)
            .await
            .map_err(IngressError::Pingora)
    }
}

fn build_upstream_client(config: &RouterConfig) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(config.policies.connect_timeout_ms))
        // Reqwest resets this timer whenever bytes arrive, so this is an
        // inter-chunk idle deadline rather than a total generation limit.
        .read_timeout(Duration::from_millis(
            config.policies.inter_chunk_timeout_ms,
        ))
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(30))
        // A reverse proxy relays 3xx responses instead of turning them into a
        // second upstream request.
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Resolves only once the downstream connection can no longer carry this
/// response, so it is safe to race against any upstream progress.
///
/// The request body is fully buffered before admission, so `no_body_expected`
/// is unconditionally true here and this read never steals bytes the router
/// still needs. In that mode every outcome is terminal for this exchange — a
/// FIN, a read error, or bytes arriving after a complete body, which pingora
/// has already consumed by the time it reports them. Detection is purely
/// event-driven: a backend that takes minutes to emit its first token leaves
/// this future pending, so only the inter-chunk timeout bounds a slow answer.
async fn downstream_cancelled(http: &mut ServerSession) -> IngressError {
    let _ = http.read_body_or_idle(true).await;
    IngressError::ClientDisconnected
}

fn classify_upstream_error(
    error: reqwest::Error,
    idle_timeout: Duration,
    phase: &'static str,
) -> IngressError {
    if error.is_timeout() {
        IngressError::UpstreamTimeout {
            idle_timeout,
            phase,
            source: error,
        }
    } else {
        IngressError::Upstream(error)
    }
}

async fn acquire_first_running<F, Fut>(
    candidates: Vec<ServiceDefinition>,
    mut acquire: F,
) -> Result<Option<(ServiceDefinition, ActiveGuard)>, PoolError>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<ActiveGuard, PoolError>>,
{
    for service in candidates {
        match acquire(service.name.clone()).await {
            Ok(guard) => return Ok(Some((service, guard))),
            Err(error) if is_replica_local_acquire_error(&error) => {
                warn!(
                    service = %service.name,
                    error = %error,
                    "ready replica became unavailable during admission; trying the next replica"
                );
            }
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

fn is_replica_local_acquire_error(error: &PoolError) -> bool {
    matches!(
        error,
        PoolError::AutomaticStartDisabled(_)
            | PoolError::ServiceDrained(_)
            | PoolError::TransitionInProgress { .. }
            | PoolError::PlacementUnavailable { .. }
            | PoolError::Runner { .. }
            | PoolError::SleepUnsupported { .. }
            | PoolError::ParkingFull { .. }
    )
}

#[async_trait]
impl HttpServerApp for DataPlane {
    async fn process_new_http(
        self: &Arc<Self>,
        mut http: ServerSession,
        shutdown: &ShutdownWatch,
    ) -> Option<ReusedHttpStream> {
        if !http.read_request().await.ok()? {
            return None;
        }
        if *shutdown.borrow() {
            http.set_keepalive(None);
        } else {
            http.set_keepalive(Some(60));
        }
        let summary = http.request_summary();
        if let Err(error) = self.process(&mut http).await {
            if !matches!(error, IngressError::ClientDisconnected) {
                warn!(error = %error, request = %summary, "data-plane request failed");
            }
            return None;
        }
        let persistent = HttpPersistentSettings::for_session(&http);
        match http.finish().await {
            Ok(Some(stream)) => Some(ReusedHttpStream::new(stream, Some(persistent))),
            Ok(None) => None,
            Err(error) => {
                warn!(error = %error, request = %summary, "downstream finish failed");
                None
            }
        }
    }
}

struct ParkingPermits {
    _global: OwnedSemaphorePermit,
    _model: OwnedSemaphorePermit,
}

#[derive(Debug, thiserror::Error)]
enum IngressError {
    #[error("downstream HTTP error: {0}")]
    Pingora(#[source] Box<pingora::Error>),
    #[error(transparent)]
    Routing(#[from] RequestRoutingError),
    #[error(transparent)]
    Pool(#[from] PoolError),
    #[error("upstream HTTP error: {0}")]
    Upstream(#[source] reqwest::Error),
    #[error("upstream response idle timeout after {idle_timeout:?} while {phase}")]
    UpstreamTimeout {
        idle_timeout: Duration,
        phase: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("upstream connection failed before request transmission: {0}")]
    ConnectFailure(#[source] reqwest::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[source] serde_json::Error),
    #[error("{message}")]
    Unavailable {
        code: &'static str,
        message: String,
        retry_after: Option<u64>,
    },
    #[error(
        "model `{model}` is starting on service `{service}`: {elapsed_seconds} s elapsed, typically ~{typical_duration_seconds} s"
    )]
    ParkingExpired {
        model: String,
        service: String,
        elapsed_seconds: u64,
        typical_duration_seconds: u64,
        retry_after: u64,
    },
    #[error("downstream disconnected before the response completed")]
    ClientDisconnected,
}

fn alias_map(config: &RouterConfig) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for service in &config.services {
        aliases.insert(service.served_model.clone(), service.served_model.clone());
        for alias in &service.aliases {
            aliases.insert(alias.clone(), service.served_model.clone());
        }
    }
    aliases
}

fn router_owned_response(
    method: &http::Method,
    path: &str,
    config: &RouterConfig,
) -> Option<RouterOwnedResponse> {
    if path != MODELS_ENDPOINT {
        return None;
    }
    if method != http::Method::GET {
        return Some(RouterOwnedResponse {
            status: 405,
            body: json!({
                "error": {
                    "message": "model discovery requires GET",
                    "type": "router_error",
                    "code": "method_not_allowed"
                }
            }),
        });
    }

    Some(RouterOwnedResponse {
        status: 200,
        body: serde_json::to_value(configured_models(config))
            .expect("model discovery response contains only serializable fields"),
    })
}

fn configured_models(config: &RouterConfig) -> ModelsResponse {
    let mut canonical_indices = HashMap::new();
    let mut data: Vec<ModelDescription> = Vec::new();

    for service in &config.services {
        let index = *canonical_indices
            .entry(service.served_model.clone())
            .or_insert_with(|| {
                let index = data.len();
                data.push(ModelDescription {
                    id: service.served_model.clone(),
                    object: "model",
                    created: 0,
                    owned_by: "vllm-router",
                    root: service.served_model.clone(),
                    parent: None,
                    aliases: Vec::new(),
                    max_model_len: service.compatibility.maximum_model_length,
                });
                index
            });

        for alias in &service.aliases {
            if !data[index].aliases.contains(alias) {
                data[index].aliases.push(alias.clone());
            }
        }
    }

    ModelsResponse {
        object: "list",
        data,
    }
}

fn connection_tokens(headers: &http::HeaderMap) -> Vec<String> {
    headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn is_event_stream(headers: &http::HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("text/event-stream"))
}

async fn read_bounded_body(
    http: &mut ServerSession,
    limit: usize,
) -> Result<BytesMut, IngressError> {
    let mut body = BytesMut::new();
    while let Some(chunk) = http
        .read_request_body()
        .await
        .map_err(IngressError::Pingora)?
    {
        if body.len().saturating_add(chunk.len()) > limit {
            http.drain_request_body()
                .await
                .map_err(IngressError::Pingora)?;
            return Err(RequestRoutingError::TooLarge { limit }.into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn content_length(http: &ServerSession) -> Option<usize> {
    std::str::from_utf8(http.get_header_bytes(CONTENT_LENGTH))
        .ok()?
        .parse()
        .ok()
}

fn request_id(http: &ServerSession) -> String {
    if http.get_header_bytes("x-request-id").is_empty() {
        Uuid::new_v4().to_string()
    } else {
        String::from_utf8_lossy(http.get_header_bytes("x-request-id")).into_owned()
    }
}

fn traceparent(http: &ServerSession) -> String {
    if !http.get_header_bytes("traceparent").is_empty() {
        return String::from_utf8_lossy(http.get_header_bytes("traceparent")).into_owned();
    }
    let trace_id = Uuid::new_v4().simple().to_string();
    let span_id = &Uuid::new_v4().simple().to_string()[..16];
    format!("00-{trace_id}-{span_id}-01")
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        net::SocketAddr,
        sync::{
            Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use pingora::protocols::l4::stream::Stream as PingoraStream;
    use tempfile::NamedTempFile;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::{Notify, watch},
        task::JoinHandle,
    };

    use super::*;
    use crate::{
        pool::{
            DrainRequest, LifecycleRequest, SleepRequest, StatusResponse, StopRequest,
            TransitionHandle, TransitionResponse,
        },
        state::{Activity, ObservedState, ParkedGuard},
    };

    #[derive(Clone)]
    enum EnsureBehavior {
        Ready,
        Pending,
        SlowReady(Duration),
        Error(PoolError),
    }

    struct FakePool {
        service_names: Vec<String>,
        lifecycles: Arc<Mutex<HashMap<String, Lifecycle>>>,
        activities: HashMap<String, Activity>,
        ensure_behavior: EnsureBehavior,
        ensure_calls: AtomicUsize,
        ensure_started: Notify,
        ensure_cancelled: Arc<AtomicBool>,
        active_attempts: Mutex<Vec<String>>,
        sleep_on_next_acquire: AtomicBool,
    }

    struct CancellationFlag(Arc<AtomicBool>);

    impl Drop for CancellationFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    impl FakePool {
        fn new(config: &RouterConfig, lifecycle: Lifecycle, ensure: EnsureBehavior) -> Self {
            let service_names: Vec<_> = config
                .services
                .iter()
                .map(|service| service.name.clone())
                .collect();
            Self {
                lifecycles: Arc::new(Mutex::new(
                    service_names
                        .iter()
                        .cloned()
                        .map(|name| (name, lifecycle))
                        .collect(),
                )),
                activities: service_names
                    .iter()
                    .cloned()
                    .map(|name| (name, Activity::new()))
                    .collect(),
                service_names,
                ensure_behavior: ensure,
                ensure_calls: AtomicUsize::new(0),
                ensure_started: Notify::new(),
                ensure_cancelled: Arc::new(AtomicBool::new(false)),
                active_attempts: Mutex::new(Vec::new()),
                sleep_on_next_acquire: AtomicBool::new(false),
            }
        }

        fn status_for(&self, service: &str) -> Result<ServiceStatus, PoolError> {
            let lifecycle = *self
                .lifecycles
                .lock()
                .unwrap()
                .get(service)
                .ok_or_else(|| PoolError::UnknownService(service.to_string()))?;
            let activity = self
                .activities
                .get(service)
                .ok_or_else(|| PoolError::UnknownService(service.to_string()))?
                .snapshot();
            let mut status = ObservedState::new(service);
            status.lifecycle = lifecycle;
            status.active_requests = activity.active;
            status.parked_requests = activity.parked;
            status.idle_for = activity.idle_for;
            Ok(status)
        }

        fn activity(&self, service: &str) -> &Activity {
            &self.activities[service]
        }

        async fn wait_for_ensure(&self) {
            loop {
                let notified = self.ensure_started.notified();
                if self.ensure_calls.load(Ordering::Acquire) != 0 {
                    return;
                }
                notified.await;
            }
        }

        async fn wait_for_cancellation(&self) {
            tokio::time::timeout(Duration::from_secs(1), async {
                while !self.ensure_cancelled.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("parked readiness future was not cancelled");
        }

        fn unsupported(service: String) -> PoolError {
            PoolError::UnknownService(service)
        }
    }

    #[async_trait]
    impl PoolManager for FakePool {
        async fn begin_lifecycle(
            &self,
            request: LifecycleRequest,
        ) -> Result<TransitionHandle, PoolError> {
            let service = match request {
                LifecycleRequest::EnsureReady(request) => request.service,
                LifecycleRequest::Drain(request) => request.service,
                LifecycleRequest::Sleep(request) => request.service,
                LifecycleRequest::Stop(request) => request.service,
            };
            Err(Self::unsupported(service))
        }

        async fn ensure_ready(
            &self,
            request: EnsureReadyRequest,
        ) -> Result<TransitionResponse, PoolError> {
            self.ensure_calls.fetch_add(1, Ordering::AcqRel);
            self.ensure_started.notify_waiters();
            match &self.ensure_behavior {
                EnsureBehavior::Ready => {
                    *self
                        .lifecycles
                        .lock()
                        .unwrap()
                        .get_mut(&request.service)
                        .ok_or_else(|| PoolError::UnknownService(request.service.clone()))? =
                        Lifecycle::Running;
                    Ok(TransitionResponse {
                        operation_id: None,
                        affected_requests: 0,
                        status: self.status_for(&request.service)?,
                    })
                }
                EnsureBehavior::Error(error) => Err(error.clone()),
                EnsureBehavior::Pending => {
                    let _cancelled = CancellationFlag(self.ensure_cancelled.clone());
                    std::future::pending::<Result<TransitionResponse, PoolError>>().await
                }
                EnsureBehavior::SlowReady(delay) => {
                    let lifecycles = self.lifecycles.clone();
                    let service = request.service.clone();
                    let delay = *delay;
                    tokio::spawn(async move {
                        tokio::time::sleep(delay).await;
                        *lifecycles
                            .lock()
                            .unwrap()
                            .get_mut(&service)
                            .expect("slow-start service should remain configured") =
                            Lifecycle::Running;
                    });
                    loop {
                        if self.status_for(&request.service)?.lifecycle == Lifecycle::Running {
                            return Ok(TransitionResponse {
                                operation_id: None,
                                affected_requests: 0,
                                status: self.status_for(&request.service)?,
                            });
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                }
            }
        }

        async fn drain(&self, request: DrainRequest) -> Result<TransitionResponse, PoolError> {
            Err(Self::unsupported(request.service))
        }

        async fn sleep(&self, request: SleepRequest) -> Result<TransitionResponse, PoolError> {
            Err(Self::unsupported(request.service))
        }

        async fn stop(&self, request: StopRequest) -> Result<TransitionResponse, PoolError> {
            Err(Self::unsupported(request.service))
        }

        async fn status(&self, request: StatusRequest) -> Result<StatusResponse, PoolError> {
            let names: Vec<_> = request
                .service
                .map_or_else(|| self.service_names.clone(), |service| vec![service]);
            names
                .iter()
                .map(|service| self.status_for(service))
                .collect::<Result<Vec<_>, _>>()
                .map(|services| StatusResponse { services })
        }

        async fn reconcile(&self, service: Option<String>) -> Result<StatusResponse, PoolError> {
            self.status(StatusRequest { service }).await
        }

        fn subscribe(&self, service: String) -> Result<watch::Receiver<ServiceStatus>, PoolError> {
            let (_, receiver) = watch::channel(self.status_for(&service)?);
            Ok(receiver)
        }

        async fn acquire_active(&self, service: String) -> Result<ActiveGuard, PoolError> {
            self.active_attempts.lock().unwrap().push(service.clone());
            if self.sleep_on_next_acquire.swap(false, Ordering::AcqRel) {
                *self
                    .lifecycles
                    .lock()
                    .unwrap()
                    .get_mut(&service)
                    .ok_or_else(|| PoolError::UnknownService(service.clone()))? =
                    Lifecycle::AsleepL1;
                return Err(PoolError::PlacementUnavailable {
                    service,
                    reason: "sleep was observed during active admission".into(),
                });
            }
            if self.status_for(&service)?.lifecycle != Lifecycle::Running {
                return Err(PoolError::PlacementUnavailable {
                    service,
                    reason: "test service is not running".into(),
                });
            }
            Ok(self.activity(&service).acquire_active())
        }

        fn acquire_parked(&self, service: String) -> Result<ParkedGuard, PoolError> {
            self.activities
                .get(&service)
                .ok_or_else(|| PoolError::UnknownService(service.clone()))?
                .try_acquire_parked(u64::MAX)
                .ok_or(PoolError::ParkingFull { service })
        }
    }

    const TEST_TOKEN: &str = "inference-secret";

    fn test_auth() -> TokenAuth {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), format!("{TEST_TOKEN}\n")).unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(0o600)).unwrap();
        TokenAuth::load(file.path()).unwrap()
    }

    fn test_config(backends: &[SocketAddr]) -> Arc<ConfigStore> {
        test_config_named(backends, "replica")
    }

    fn test_config_named(backends: &[SocketAddr], service_prefix: &str) -> Arc<ConfigStore> {
        Arc::new(
            ConfigStore::new(test_router_config_named(backends, service_prefix))
                .expect("test config should validate"),
        )
    }

    fn test_router_config_named(backends: &[SocketAddr], service_prefix: &str) -> RouterConfig {
        let mut config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        config.policies.connect_timeout_ms = 100;
        config.policies.inter_chunk_timeout_ms = 1_000;
        let template = config.services[0].clone();
        config.services = backends
            .iter()
            .enumerate()
            .map(|(index, backend)| {
                let mut service = template.clone();
                service.name = format!("{service_prefix}-{index}");
                service.aliases.clear();
                service.backend.address = backend.ip();
                service.backend.port = backend.port();
                service.systemd.unit = format!("vllm@{service_prefix}-{index}.service");
                service.systemd.environment_file =
                    format!("/etc/vllm/{service_prefix}-{index}.env").into();
                service.parking.deadline_ms = 1_000;
                service
            })
            .collect();
        config
    }

    async fn ingress_connection(plane: Arc<DataPlane>) -> (tokio::net::TcpStream, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = tokio::net::TcpStream::connect(address).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let (shutdown_tx, shutdown) = watch::channel(false);
        let task = tokio::spawn(async move {
            let _shutdown_tx = shutdown_tx;
            let stream = PingoraStream::from(server);
            let session = ServerSession::new_http1(Box::new(stream));
            let _ = plane.process_new_http(session, &shutdown).await;
        });
        (client, task)
    }

    async fn write_inference_request(client: &mut tokio::net::TcpStream, body: &[u8]) {
        write_inference_request_to(client, "/v1/chat/completions", body).await;
    }

    async fn write_inference_request_to(
        client: &mut tokio::net::TcpStream,
        path: &str,
        body: &[u8],
    ) {
        write_credentialed_request(client, path, Some(&format!("Bearer {TEST_TOKEN}")), body).await;
    }

    async fn write_credentialed_request(
        client: &mut tokio::net::TcpStream,
        path: &str,
        authorization: Option<&str>,
        body: &[u8],
    ) {
        let authorization = authorization
            .map(|value| format!("Authorization: {value}\r\n"))
            .unwrap_or_default();
        let header = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: router.test\r\n\
             {authorization}\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n",
            body.len()
        );
        client.write_all(header.as_bytes()).await.unwrap();
        client.write_all(body).await.unwrap();
    }

    async fn read_response(mut client: tokio::net::TcpStream) -> Vec<u8> {
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), client.read_to_end(&mut response))
            .await
            .expect("ingress response timed out")
            .unwrap();
        response
    }

    async fn read_upstream_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        loop {
            let mut chunk = [0; 1024];
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read != 0, "upstream request ended before its body");
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
                continue;
            };
            let header_end = header_end + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= header_end + content_length {
                return request;
            }
        }
    }

    async fn closed_loopback_address() -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        address
    }

    async fn read_until(client: &mut tokio::net::TcpStream, needle: &[u8]) -> Vec<u8> {
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let mut chunk = [0; 256];
                let read = client.read(&mut chunk).await.unwrap();
                assert!(read != 0, "response ended before expected bytes");
                response.extend_from_slice(&chunk[..read]);
                if response.windows(needle.len()).any(|bytes| bytes == needle) {
                    return;
                }
            }
        })
        .await
        .expect("incremental response bytes were not received");
        response
    }

    #[test]
    fn public_endpoint_allowlist_matches_model_bearing_vllm_routes() {
        for path in [
            "/tokenize",
            "/detokenize",
            "/classify",
            "/v2/rerank",
            "/v2/embed",
            "/v1/messages/count_tokens",
        ] {
            assert!(
                MODEL_ENDPOINTS.contains(&path),
                "real vLLM endpoint {path} should be exposed"
            );
        }

        for path in [
            "/v1/tokenize",
            "/v1/detokenize",
            "/v1/classify",
            "/sleep",
            "/is_sleeping",
            MODELS_ENDPOINT,
        ] {
            assert!(
                !MODEL_ENDPOINTS.contains(&path),
                "unsupported or router-owned endpoint {path} should not be proxied"
            );
        }
    }

    #[test]
    fn event_stream_detection_accepts_parameters_but_not_json() {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            http::HeaderValue::from_static("Text/Event-Stream; charset=utf-8"),
        );
        assert!(is_event_stream(&headers));

        headers.insert(
            CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        assert!(!is_event_stream(&headers));
    }

    #[tokio::test]
    async fn stalled_response_body_is_classified_as_bounded_inter_chunk_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = listener.local_addr().unwrap();
        let upstream = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Transfer-Encoding: chunked\r\n\
                      Connection: close\r\n\r\n\
                      D\r\n\
                      data: first\n\n\r\n",
                )
                .await
                .unwrap();
            stream.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(200)).await;
        });

        let idle_timeout = Duration::from_millis(30);
        let client = reqwest::Client::builder()
            .read_timeout(idle_timeout)
            .build()
            .unwrap();
        let mut response = client
            .post(format!("http://{backend}/v1/chat/completions"))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.chunk().await.unwrap().unwrap(),
            Bytes::from_static(b"data: first\n\n")
        );

        let error = response.chunk().await.unwrap_err();
        assert!(error.is_timeout());
        let classified = classify_upstream_error(error, idle_timeout, "reading the response body");
        match &classified {
            IngressError::UpstreamTimeout {
                idle_timeout,
                phase,
                ..
            } => {
                assert_eq!(*idle_timeout, Duration::from_millis(30));
                assert_eq!(*phase, "reading the response body");
            }
            other => panic!("expected timeout classification, got {other}"),
        }
        let message = classified.to_string();
        assert_eq!(
            message,
            "upstream response idle timeout after 30ms while reading the response body"
        );
        assert!(!message.contains("decoding response body"));
        assert!(message.len() < 128);
        upstream.await.unwrap();
    }

    #[test]
    fn model_discovery_is_an_explicit_get_only_router_owned_route() {
        let config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");

        assert!(
            router_owned_response(&http::Method::GET, "/v1/chat/completions", &config).is_none()
        );
        let method_error =
            router_owned_response(&http::Method::POST, MODELS_ENDPOINT, &config).unwrap();
        assert_eq!(method_error.status, 405);
        assert_eq!(
            method_error.body["error"]["code"],
            Value::String("method_not_allowed".into())
        );
    }

    #[test]
    fn model_discovery_lists_two_unique_models_without_alias_ids() {
        let mut config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        config.services[0].aliases = vec!["alias-b".into(), "alias-a".into()];
        let mut distinct_model = config.services[0].clone();
        distinct_model.name = "distinct-model".into();
        distinct_model.served_model = "example/model-70b".into();
        distinct_model.aliases = vec!["large".into()];
        distinct_model.compatibility.maximum_model_length = Some(65_536);
        config.services.push(distinct_model);

        let response = router_owned_response(&http::Method::GET, MODELS_ENDPOINT, &config).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body["object"], "list");
        let data = response.body["data"].as_array().unwrap();
        let ids: Vec<_> = data
            .iter()
            .map(|model| model["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["example/model-30b", "example/model-70b"]);

        assert_eq!(
            data[0],
            json!({
                "id": "example/model-30b",
                "object": "model",
                "created": 0,
                "owned_by": "vllm-router",
                "root": "example/model-30b",
                "parent": null,
                "aliases": ["alias-b", "alias-a"],
                "max_model_len": 32768
            })
        );
        assert_eq!(
            data[1],
            json!({
                "id": "example/model-70b",
                "object": "model",
                "created": 0,
                "owned_by": "vllm-router",
                "root": "example/model-70b",
                "parent": null,
                "aliases": ["large"],
                "max_model_len": 65536
            })
        );
        assert!(data.iter().all(|model| {
            !["alias-a", "alias-b", "large"].contains(&model["id"].as_str().unwrap())
        }));
    }

    #[test]
    fn model_discovery_collapses_replicas_and_unions_aliases_in_declaration_order() {
        let mut config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        config.services[0].aliases = vec!["alias-b".into(), "alias-a".into()];
        let mut replica = config.services[0].clone();
        replica.name = "replica".into();
        replica.aliases = vec!["alias-a".into(), "alias-c".into()];
        config.services.push(replica);

        let response = router_owned_response(&http::Method::GET, MODELS_ENDPOINT, &config).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(
            response.body["data"],
            json!([{
                "id": "example/model-30b",
                "object": "model",
                "created": 0,
                "owned_by": "vllm-router",
                "root": "example/model-30b",
                "parent": null,
                "aliases": ["alias-b", "alias-a", "alias-c"],
                "max_model_len": 32768
            }])
        );
    }

    #[tokio::test]
    async fn additive_config_publication_gains_model_discovery_and_parking_capacity() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Ready,
        ));
        let plane = DataPlane::new(config.clone(), pool, test_auth()).unwrap();
        let mut candidate = config.snapshot().services[0].clone();
        candidate.name = "registered".into();
        candidate.served_model = "example/registered".into();
        candidate.aliases = vec!["registered-alias".into()];
        candidate.backend.port = candidate.backend.port.saturating_add(1);
        candidate.systemd.unit = "vllm@registered.service".into();
        candidate.systemd.environment_file = "/etc/vllm/registered.env".into();

        assert!(
            !plane
                .model_parked
                .read()
                .unwrap()
                .contains_key("example/registered")
        );
        config.register_service(candidate.clone()).unwrap();

        let response = router_owned_response(
            &http::Method::GET,
            MODELS_ENDPOINT,
            config.snapshot().as_ref(),
        )
        .unwrap();
        assert_eq!(response.body["data"][1]["id"], "example/registered");
        let _permits = plane
            .acquire_parking("example/registered", &candidate)
            .unwrap();
        assert!(
            plane
                .model_parked
                .read()
                .unwrap()
                .contains_key("example/registered")
        );
    }

    #[tokio::test]
    async fn model_discovery_get_is_serialized_and_post_remains_method_not_allowed() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());

        let (mut get_client, get_task) = ingress_connection(Arc::clone(&plane)).await;
        get_client
            .write_all(
                format!(
                    "GET /v1/models HTTP/1.1\r\n\
                     Host: router.test\r\n\
                     Authorization: Bearer {TEST_TOKEN}\r\n\
                     Connection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let get_response = read_response(get_client).await;
        get_task.await.unwrap();
        assert!(
            get_response.starts_with(b"HTTP/1.1 200"),
            "unexpected GET response: {}",
            String::from_utf8_lossy(&get_response)
        );
        let get_body_start = get_response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        let get_body: Value = serde_json::from_slice(&get_response[get_body_start..]).unwrap();
        assert_eq!(
            get_body,
            json!({
                "object": "list",
                "data": [{
                    "id": "example/model-30b",
                    "object": "model",
                    "created": 0,
                    "owned_by": "vllm-router",
                    "root": "example/model-30b",
                    "parent": null,
                    "aliases": [],
                    "max_model_len": 32768
                }]
            })
        );

        let (mut post_client, post_task) = ingress_connection(plane).await;
        write_inference_request_to(&mut post_client, MODELS_ENDPOINT, b"{}").await;
        let post_response = read_response(post_client).await;
        post_task.await.unwrap();
        assert!(
            post_response.starts_with(b"HTTP/1.1 405"),
            "unexpected POST response: {}",
            String::from_utf8_lossy(&post_response)
        );
        let post_body_start = post_response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        let post_body: Value = serde_json::from_slice(&post_response[post_body_start..]).unwrap();
        assert_eq!(post_body["error"]["code"], "method_not_allowed");
    }

    #[tokio::test]
    async fn unauthenticated_valid_model_request_does_no_lifecycle_work() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_credentialed_request(
            &mut client,
            "/v1/chat/completions",
            None,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;

        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 401"),
            "unexpected response: {}",
            String::from_utf8_lossy(&response)
        );
        let body_start = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        let body: Value = serde_json::from_slice(&response[body_start..]).unwrap();
        assert_eq!(body["error"]["code"], "unauthorized");
        assert_eq!(body["error"]["type"], "router_error");

        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 0);
        assert!(pool.active_attempts.lock().unwrap().is_empty());
        assert_eq!(pool.activity("replica-0").parked(), 0);
        assert_eq!(pool.activity("replica-0").active(), 0);
        assert_eq!(
            pool.status_for("replica-0").unwrap().lifecycle,
            Lifecycle::Stopped
        );
    }

    #[tokio::test]
    async fn unauthorized_response_is_written_without_reading_the_request_body() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        // A promised body that is never sent: any read of it would block until the
        // test's own timeout, so an answered request proves the rejection precedes it.
        client
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\n\
                  Host: router.test\r\n\
                  Content-Type: application/json\r\n\
                  Content-Length: 1048576\r\n\r\n",
            )
            .await
            .unwrap();

        let response = read_until(&mut client, b"\r\n\r\n").await;
        assert!(
            response.starts_with(b"HTTP/1.1 401"),
            "unexpected response: {}",
            String::from_utf8_lossy(&response)
        );
        drop(client);
        ingress_task.await.unwrap();
        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 0);
        assert!(pool.active_attempts.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unauthenticated_model_discovery_is_refused() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        client
            .write_all(
                b"GET /v1/models HTTP/1.1\r\n\
                  Host: router.test\r\n\
                  Connection: close\r\n\r\n",
            )
            .await
            .unwrap();

        let response = read_response(client).await;
        ingress_task.await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 401"), "{response}");
        assert!(!response.contains("example/model-30b"));
    }

    #[tokio::test]
    async fn wrong_token_is_refused_and_the_accepted_credential_never_reaches_the_backend() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let request = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Length: 2\r\n\
                      Connection: close\r\n\r\n\
                      ok",
                )
                .await
                .unwrap();
            request
        });

        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let body = br#"{"model":"example/model-30b","messages":[]}"#;

        let (mut wrong_client, wrong_ingress) = ingress_connection(Arc::clone(&plane)).await;
        write_credentialed_request(
            &mut wrong_client,
            "/v1/chat/completions",
            Some("Bearer inference-secre"),
            body,
        )
        .await;
        let wrong_response = read_response(wrong_client).await;
        wrong_ingress.await.unwrap();
        assert!(
            wrong_response.starts_with(b"HTTP/1.1 401"),
            "unexpected response: {}",
            String::from_utf8_lossy(&wrong_response)
        );
        assert!(pool.active_attempts.lock().unwrap().is_empty());

        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(&mut client, body).await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        let upstream_request = upstream_task.await.unwrap();

        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "unexpected response: {}",
            String::from_utf8_lossy(&response)
        );
        assert_eq!(*pool.active_attempts.lock().unwrap(), vec!["replica-0"]);
        let forwarded = String::from_utf8_lossy(&upstream_request).to_ascii_lowercase();
        assert!(!forwarded.contains("authorization"), "{forwarded}");
        assert!(!forwarded.contains(TEST_TOKEN), "{forwarded}");
    }

    #[tokio::test]
    async fn anthropic_endpoints_shape_the_unauthorized_error_in_their_own_dialect() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_credentialed_request(
            &mut client,
            "/v1/messages",
            None,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;

        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 401, "authentication_error", "unauthorized");
        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn replica_local_acquire_failure_falls_back_to_the_next_ready_replica() {
        let config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        let first = config.services[0].clone();
        let mut second = first.clone();
        second.name = "second-replica".into();
        let activity = Activity::new();
        let attempts = Arc::new(Mutex::new(Vec::new()));

        let selected = acquire_first_running(vec![first, second], {
            let attempts = Arc::clone(&attempts);
            move |service| {
                let activity = activity.clone();
                let attempts = Arc::clone(&attempts);
                async move {
                    attempts.lock().unwrap().push(service.clone());
                    if service == "example-30b" {
                        Err(PoolError::PlacementUnavailable {
                            service,
                            reason: "readiness changed during admission".into(),
                        })
                    } else {
                        Ok(activity.acquire_active())
                    }
                }
            }
        })
        .await
        .unwrap()
        .expect("second replica should be selected");

        assert_eq!(selected.0.name, "second-replica");
        assert_eq!(
            *attempts.lock().unwrap(),
            vec!["example-30b", "second-replica"]
        );
    }

    #[tokio::test]
    async fn sleep_probe_failure_falls_back_to_the_next_ready_replica() {
        let config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        let first = config.services[0].clone();
        let mut second = first.clone();
        second.name = "second-replica".into();
        let activity = Activity::new();
        let attempts = Arc::new(Mutex::new(Vec::new()));

        let selected = acquire_first_running(vec![first, second], {
            let attempts = Arc::clone(&attempts);
            move |service| {
                let activity = activity.clone();
                let attempts = Arc::clone(&attempts);
                async move {
                    attempts.lock().unwrap().push(service.clone());
                    if service == "example-30b" {
                        Err(PoolError::Runner {
                            service,
                            message: "sleep observation timed out".into(),
                        })
                    } else {
                        Ok(activity.acquire_active())
                    }
                }
            }
        })
        .await
        .unwrap()
        .expect("bounded sleep-probe failure should fail over");

        assert_eq!(selected.0.name, "second-replica");
        assert_eq!(
            *attempts.lock().unwrap(),
            vec!["example-30b", "second-replica"]
        );
    }

    #[tokio::test]
    async fn sleep_racing_active_admission_never_reaches_the_backend() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let upstream_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_millis(250), upstream.accept()).await
        });
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Error(PoolError::Deadline {
                service: "replica-0".into(),
                operation: "wake",
            }),
        ));
        pool.sleep_on_next_acquire.store(true, Ordering::Release);
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;

        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 503"),
            "sleep race must fail before forwarding: {}",
            String::from_utf8_lossy(&response)
        );
        assert_eq!(pool.activity("replica-0").active(), 0);
        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 1);
        assert!(
            upstream_task.await.unwrap().is_err(),
            "a sleeping backend must not receive inference bytes"
        );
    }

    #[tokio::test]
    async fn terminal_acquire_failure_is_not_hidden_by_replica_fallback() {
        let config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        let candidate = config.services[0].clone();

        let result = acquire_first_running(vec![candidate], |service| async move {
            Err(PoolError::TerminalFailure {
                service,
                message: "permanent failure".into(),
            })
        })
        .await;

        assert!(matches!(result, Err(PoolError::TerminalFailure { .. })));
    }

    #[tokio::test]
    async fn parked_admission_holds_active_guard_and_streams_before_upstream_completion() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let (first_chunk_tx, first_chunk_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let request = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Transfer-Encoding: chunked\r\n\
                      Connection: close\r\n\r\n\
                      D\r\n\
                      data: first\n\n\r\n",
                )
                .await
                .unwrap();
            stream.flush().await.unwrap();
            let _ = first_chunk_tx.send(());
            release_rx.await.unwrap();
            stream
                .write_all(b"E\r\ndata: second\n\n\r\n0\r\n\r\n")
                .await
                .unwrap();
            stream.flush().await.unwrap();
            request
        });

        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        let body = br#"{"model":"example/model-30b","stream":true}"#;
        write_inference_request(&mut client, body).await;

        first_chunk_rx.await.unwrap();
        let mut response = read_until(&mut client, b"data: first").await;

        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 1);
        assert_eq!(pool.activity("replica-0").parked(), 0);
        assert_eq!(pool.activity("replica-0").active(), 1);
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&["replica-0"])
                .get(),
            1
        );

        release_tx.send(()).unwrap();
        client.read_to_end(&mut response).await.unwrap();
        ingress_task.await.unwrap();
        let upstream_request = upstream_task.await.unwrap();

        assert!(
            response
                .windows(b"data: second".len())
                .any(|bytes| { bytes == b"data: second" })
        );
        assert!(upstream_request.ends_with(body));
        assert_eq!(pool.activity("replica-0").active(), 0);
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&["replica-0"])
                .get(),
            0
        );
    }

    #[tokio::test]
    async fn short_park_expiry_reports_progress_while_start_continues_and_retry_is_warm() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let _ = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Length: 4\r\n\
                      Connection: close\r\n\r\n\
                      warm",
                )
                .await
                .unwrap();
        });

        let mut config = test_router_config_named(&[backend], "slow-start");
        config.services[0].parking.deadline_ms = 30;
        config.services[0].parking.retry_after_seconds = 7;
        let pool = Arc::new(FakePool::new(
            &config,
            Lifecycle::Stopped,
            EnsureBehavior::SlowReady(Duration::from_millis(80)),
        ));
        let config = Arc::new(ConfigStore::new(config).unwrap());
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());

        let (mut first_client, first_ingress) = ingress_connection(plane.clone()).await;
        write_inference_request(
            &mut first_client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;
        let first_response = read_response(first_client).await;
        first_ingress.await.unwrap();
        let first_response = String::from_utf8(first_response).unwrap();
        assert!(first_response.starts_with("HTTP/1.1 503"));
        assert!(
            first_response
                .to_ascii_lowercase()
                .contains("retry-after: 7")
        );
        assert!(first_response.contains(r#""code":"model_starting""#));
        assert!(first_response.contains(r#""service":"slow-start-0""#));
        assert!(first_response.contains(r#""elapsed_seconds":"#));
        assert!(first_response.contains(r#""typical_duration_seconds":7"#));
        assert!(first_response.contains("is starting"));

        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            pool.status_for("slow-start-0").unwrap().lifecycle,
            Lifecycle::Running
        );

        let (mut retry_client, retry_ingress) = ingress_connection(plane).await;
        write_inference_request(
            &mut retry_client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;
        let retry_response = read_response(retry_client).await;
        retry_ingress.await.unwrap();
        upstream_task.await.unwrap();
        assert!(retry_response.starts_with(b"HTTP/1.1 200"));
        assert!(retry_response.ends_with(b"warm"));
        assert_eq!(pool.ensure_calls.load(Ordering::Acquire), 1);
    }

    #[tokio::test]
    async fn active_stream_gauge_releases_on_upstream_body_error() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let (first_chunk_tx, first_chunk_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let _ = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Transfer-Encoding: chunked\r\n\
                      Connection: close\r\n\r\n\
                      D\r\n\
                      data: first\n\n\r\n",
                )
                .await
                .unwrap();
            stream.flush().await.unwrap();
            let _ = first_chunk_tx.send(());
            release_rx.await.unwrap();
            stream.write_all(b"20\r\nincomplete").await.unwrap();
        });

        let service = "stream-upstream-error-0";
        let config = test_config_named(&[backend], "stream-upstream-error");
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","stream":true}"#,
        )
        .await;

        first_chunk_rx.await.unwrap();
        let mut response = read_until(&mut client, b"data: first").await;
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&[service])
                .get(),
            1
        );

        release_tx.send(()).unwrap();
        client.read_to_end(&mut response).await.unwrap();
        ingress_task.await.unwrap();
        upstream_task.await.unwrap();
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&[service])
                .get(),
            0
        );
        assert_eq!(pool.activity(service).active(), 0);
    }

    #[tokio::test]
    async fn active_stream_gauge_releases_on_downstream_disconnect() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let (first_chunk_tx, first_chunk_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let _ = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\r\n\
                      data: first\n\n",
                )
                .await
                .unwrap();
            stream.flush().await.unwrap();
            let _ = first_chunk_tx.send(());
            release_rx.await.unwrap();
            let payload = vec![b'x'; 64 * 1024];
            for _ in 0..64 {
                if stream.write_all(&payload).await.is_err() {
                    break;
                }
                if stream.flush().await.is_err() {
                    break;
                }
            }
        });

        let service = "stream-downstream-error-0";
        let config = test_config_named(&[backend], "stream-downstream-error");
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","stream":true}"#,
        )
        .await;

        first_chunk_rx.await.unwrap();
        let _response = read_until(&mut client, b"data: first").await;
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&[service])
                .get(),
            1
        );

        drop(client);
        tokio::time::sleep(Duration::from_millis(20)).await;
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), ingress_task)
            .await
            .expect("ingress did not observe downstream disconnect")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), upstream_task)
            .await
            .expect("upstream mock did not stop after disconnect")
            .unwrap();
        assert_eq!(
            metrics::global()
                .active_streams
                .with_label_values(&[service])
                .get(),
            0
        );
        assert_eq!(pool.activity(service).active(), 0);
    }

    #[tokio::test]
    async fn downstream_disconnect_before_upstream_headers_cancels_the_inference() {
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = upstream.local_addr().unwrap();
        let (forwarded_tx, forwarded_rx) = tokio::sync::oneshot::channel();
        // Accepts and reads the whole request but never writes a status line, the
        // shape of a backend still assembling its first token.
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let _ = read_upstream_request(&mut stream).await;
            forwarded_tx.send(()).unwrap();
            let mut trailing = [0; 64];
            stream.read(&mut trailing).await.unwrap()
        });

        let service = "headerless-upstream-0";
        let mut config = test_router_config_named(&[backend], "headerless-upstream");
        // Far longer than anything this test waits for, so a release can only come
        // from observing the client leave. Production runs 120 s here precisely
        // because a cold first token is legitimately slow.
        config.policies.inter_chunk_timeout_ms = 30_000;
        let pool = Arc::new(FakePool::new(
            &config,
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let config = Arc::new(ConfigStore::new(config).expect("test config should validate"));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;

        forwarded_rx.await.unwrap();
        assert_eq!(pool.activity(service).active(), 1);
        drop(client);

        tokio::time::timeout(Duration::from_secs(2), ingress_task)
            .await
            .expect("ingress kept waiting on headers after the client left")
            .unwrap();
        let trailing = tokio::time::timeout(Duration::from_secs(2), upstream_task)
            .await
            .expect("upstream request was not cancelled")
            .unwrap();
        assert_eq!(
            trailing, 0,
            "upstream connection should be closed, not idle"
        );
        assert_eq!(pool.activity(service).active(), 0);
    }

    #[tokio::test]
    async fn downstream_disconnect_cancels_parking_and_releases_every_guard() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Pending,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;

        pool.wait_for_ensure().await;
        assert_eq!(pool.activity("replica-0").parked(), 1);
        drop(client);

        pool.wait_for_cancellation().await;
        tokio::time::timeout(Duration::from_secs(1), ingress_task)
            .await
            .expect("ingress did not observe the downstream disconnect")
            .unwrap();
        assert_eq!(pool.activity("replica-0").parked(), 0);
        assert_eq!(pool.activity("replica-0").active(), 0);
    }

    #[tokio::test]
    async fn connect_failure_replays_exact_body_to_next_running_replica() {
        let failed_backend = closed_loopback_address().await;
        let upstream = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let fallback_backend = upstream.local_addr().unwrap();
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream.accept().await.unwrap();
            let request = read_upstream_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Length: 8\r\n\
                      Connection: close\r\n\r\n\
                      fallback",
                )
                .await
                .unwrap();
            request
        });

        let config = test_config(&[failed_backend, fallback_backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool.clone(), test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        let body = br#"{"model":"example/model-30b","messages":[{"role":"user","content":"hi"}]}"#;
        write_inference_request(&mut client, body).await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        let upstream_request = upstream_task.await.unwrap();

        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"fallback"));
        assert!(upstream_request.ends_with(body));
        assert_eq!(
            *pool.active_attempts.lock().unwrap(),
            vec!["replica-0", "replica-1"]
        );
        assert_eq!(pool.activity("replica-0").active(), 0);
        assert_eq!(pool.activity("replica-1").active(), 0);
    }

    #[tokio::test]
    async fn messages_router_errors_use_anthropic_envelopes_for_400_404_413_and_503() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request_to(&mut client, "/v1/messages", b"not JSON").await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 400, "invalid_request_error", "invalid_json");

        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request_to(&mut client, "/v1/messages/count_tokens", b"not JSON").await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 400, "invalid_request_error", "invalid_json");

        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request_to(
            &mut client,
            "/v1/messages",
            br#"{"model":"not-configured","messages":[]}"#,
        )
        .await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 404, "not_found_error", "model_not_found");

        let backend = closed_loopback_address().await;
        let mut config = test_router_config_named(&[backend], "replica");
        config.policies.maximum_request_body_bytes = 8;
        let pool = Arc::new(FakePool::new(
            &config,
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let config =
            Arc::new(ConfigStore::new(config).expect("small test body limit should validate"));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request_to(
            &mut client,
            "/v1/messages",
            br#"{"model":"example/model-30b"}"#,
        )
        .await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 413, "request_too_large", "request_too_large");

        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Error(PoolError::Deadline {
                service: "replica-0".into(),
                operation: "readiness",
            }),
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request_to(
            &mut client,
            "/v1/messages",
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        assert_anthropic_error(&response, 503, "overloaded_error", "deadline_exceeded");
        assert!(
            String::from_utf8_lossy(&response)
                .to_ascii_lowercase()
                .contains("retry-after: 30")
        );
    }

    #[tokio::test]
    async fn pool_and_connect_errors_map_to_openai_shaped_http_responses() {
        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Stopped,
            EnsureBehavior::Error(PoolError::Deadline {
                service: "replica-0".into(),
                operation: "readiness",
            }),
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"));
        assert!(response.to_ascii_lowercase().contains("retry-after: 30"));
        assert!(response.contains(r#""code":"deadline_exceeded""#));

        let backend = closed_loopback_address().await;
        let config = test_config(&[backend]);
        let pool = Arc::new(FakePool::new(
            config.snapshot().as_ref(),
            Lifecycle::Running,
            EnsureBehavior::Ready,
        ));
        let plane = Arc::new(DataPlane::new(config, pool, test_auth()).unwrap());
        let (mut client, ingress_task) = ingress_connection(plane).await;
        write_inference_request(
            &mut client,
            br#"{"model":"example/model-30b","messages":[]}"#,
        )
        .await;
        let response = read_response(client).await;
        ingress_task.await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 502"));
        assert!(response.contains(r#""code":"upstream_error""#));
    }

    fn assert_anthropic_error(
        response: &[u8],
        expected_status: u16,
        expected_type: &str,
        expected_code: &str,
    ) {
        let split = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .expect("response should contain a header terminator");
        let headers = String::from_utf8_lossy(&response[..split]);
        assert!(
            headers.starts_with(&format!("HTTP/1.1 {expected_status}")),
            "{headers}"
        );
        let body: Value =
            serde_json::from_slice(&response[split + 4..]).expect("response body should be JSON");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], expected_type);
        assert!(body["error"].get("code").is_none(), "{expected_code}");
        assert!(body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn upstream_client_returns_redirect_without_following_it() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            let count = stream.read(&mut request).await.unwrap();
            assert!(count > 0);
            stream
                .write_all(
                    b"HTTP/1.1 307 Temporary Redirect\r\n\
                      Location: /must-not-be-followed\r\n\
                      Content-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });
        let config: RouterConfig = serde_yaml::from_str(include_str!("../config/example.yaml"))
            .expect("example config should parse");
        let client = build_upstream_client(&config).unwrap();

        let response = client
            .post(format!("http://{address}/original"))
            .body("{}")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            response.headers().get("location").unwrap(),
            "/must-not-be-followed"
        );
        server.await.unwrap();
    }
}
