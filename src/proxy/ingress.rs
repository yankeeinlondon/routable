use std::{
    collections::HashMap,
    sync::Arc,
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
use reqwest::header::{CONTENT_LENGTH, HOST};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::warn;
use uuid::Uuid;

use crate::{
    config::{ConfigStore, RouterConfig, ServiceDefinition},
    metrics,
    pool::{EnsureReadyRequest, PoolError, PoolManager, StatusRequest},
    state::{ActiveGuard, Lifecycle, ServiceStatus},
};

use super::{OpenAiError, RequestRoutingError, is_hop_by_hop, parse_model_route};

const MODEL_ENDPOINTS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/messages",
    "/v1/score",
    "/v1/rerank",
    "/score",
    "/rerank",
    "/v1/classify",
    "/v1/tokenize",
    "/v1/detokenize",
];

/// Bounded custom ingress chosen by the Phase-0 body-replay spike.
///
/// Only the request body is buffered. Upstream response bytes are copied to
/// the downstream session as each chunk arrives, preserving SSE streaming.
pub struct DataPlane {
    config: Arc<ConfigStore>,
    pool: Arc<dyn PoolManager>,
    client: reqwest::Client,
    global_parked: Arc<Semaphore>,
    model_parked: HashMap<String, Arc<Semaphore>>,
}

impl DataPlane {
    pub fn new(
        config: Arc<ConfigStore>,
        pool: Arc<dyn PoolManager>,
    ) -> Result<Self, reqwest::Error> {
        let snapshot = config.snapshot();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(snapshot.policies.connect_timeout_ms))
            // Reqwest resets this timer whenever bytes arrive, so this is an
            // inter-chunk idle deadline rather than a total generation limit.
            .read_timeout(Duration::from_millis(
                snapshot.policies.inter_chunk_timeout_ms,
            ))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30))
            .build()?;
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
        Ok(Self {
            config,
            pool,
            client,
            global_parked: Arc::new(Semaphore::new(
                snapshot.policies.global_parked_request_limit,
            )),
            model_parked,
        })
    }

    async fn process(&self, http: &mut ServerSession) -> Result<(), IngressError> {
        let path = http.req_header().uri.path().to_owned();
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
        metrics::global()
            .active
            .with_label_values(&[&service.name])
            .inc();

        let body = body.freeze();
        let mut result = self
            .proxy_upstream(http, &service, body.clone(), &request_id, &traceparent)
            .await;
        if matches!(&result, Err(IngressError::ConnectFailure(_))) {
            drop(active.take());
            metrics::global()
                .active
                .with_label_values(&[&service.name])
                .dec();
            if let Some((alternative, guard)) = self
                .choose_running(&config, &route.canonical, Some(&service.name))
                .await?
            {
                service = alternative;
                active = Some(guard);
                metrics::global()
                    .active
                    .with_label_values(&[&service.name])
                    .inc();
                result = self
                    .proxy_upstream(http, &service, body, &request_id, &traceparent)
                    .await;
            }
        }
        let held_active = active.take().is_some();
        drop(active);
        if held_active {
            metrics::global()
                .active
                .with_label_values(&[&service.name])
                .dec();
        }
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

        let service = config
            .services_for_model(model)
            .into_iter()
            .filter(|service| service.enabled && service.automatic_start)
            .max_by_key(|service| service.priority)
            .cloned()
            .ok_or_else(|| IngressError::Unavailable {
                code: "automatic_start_disabled",
                message: format!("no enabled auto-start service serves `{model}`"),
                retry_after: None,
            })?;
        let parking_deadline = started + Duration::from_millis(service.parking.deadline_ms);
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
            deadline: parking_deadline,
            evict: true,
        });
        tokio::pin!(ensure);
        let outcome = tokio::select! {
            result = &mut ensure => result.map_err(IngressError::Pool),
            disconnected = http.read_body_or_idle(true) => {
                match disconnected {
                    Err(_) | Ok(None) => Err(IngressError::ClientDisconnected),
                    Ok(Some(_)) => Err(IngressError::ClientDisconnected),
                }
            }
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
        for service in candidates {
            match self.pool.acquire_active(service.name.clone()).await {
                Ok(guard) => return Ok(Some((service, guard))),
                Err(PoolError::TransitionInProgress { .. }) => continue,
                Err(error) => return Err(IngressError::Pool(error)),
            }
        }
        Ok(None)
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
        let model_semaphore =
            self.model_parked
                .get(model)
                .ok_or_else(|| IngressError::Unavailable {
                    code: "model_not_found",
                    message: format!("model `{model}` is not configured"),
                    retry_after: None,
                })?;
        let model =
            model_semaphore
                .clone()
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
            if !is_hop_by_hop(name.as_str())
                && !connection_headers
                    .iter()
                    .any(|token| token.eq_ignore_ascii_case(name.as_str()))
                && name != HOST
                && name != CONTENT_LENGTH
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
        let mut response = request.send().await.map_err(|error| {
            if error.is_connect() {
                IngressError::ConnectFailure(error)
            } else {
                IngressError::Upstream(error)
            }
        })?;
        let status = response.status().as_u16();
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
        while let Some(chunk) = response.chunk().await.map_err(IngressError::Upstream)? {
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
        self.write_json(http, error.status(), &error.openai(), None)
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
        let error = OpenAiError {
            error: super::routing::OpenAiErrorBody {
                message: message.into(),
                error_type: "router_error".into(),
                code: code.into(),
            },
        };
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
    #[error("downstream disconnected while parked")]
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
    use super::*;

    #[test]
    fn public_endpoint_allowlist_excludes_vllm_admin() {
        assert!(MODEL_ENDPOINTS.contains(&"/v1/chat/completions"));
        assert!(!MODEL_ENDPOINTS.contains(&"/sleep"));
        assert!(!MODEL_ENDPOINTS.contains(&"/is_sleeping"));
        assert!(!MODEL_ENDPOINTS.contains(&"/v1/models"));
    }
}
