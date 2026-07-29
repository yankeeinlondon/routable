use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, Method, Response, Url};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Runner, RunnerError, SleepLevel, WarmupRequest};

const ERROR_BODY_LIMIT_BYTES: usize = 8 * 1024;

#[derive(Clone, Debug)]
pub struct VllmRunnerConfig {
    pub base_url: String,
    pub timeouts: VllmTimeouts,
    pub api_key: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VllmTimeouts {
    pub probe: Duration,
    pub sleep: Duration,
    pub wake: Duration,
    pub warmup: Duration,
}

impl VllmTimeouts {
    pub const fn uniform(timeout: Duration) -> Self {
        Self {
            probe: timeout,
            sleep: timeout,
            wake: timeout,
            warmup: timeout,
        }
    }
}

impl VllmRunnerConfig {
    pub fn loopback(port: u16, request_timeout: Duration) -> Self {
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            timeouts: VllmTimeouts::uniform(request_timeout),
            api_key: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VllmRunner {
    client: Client,
    base_url: Url,
    timeouts: VllmTimeouts,
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
}

#[derive(Deserialize)]
struct SleepingResponse {
    is_sleeping: bool,
}

impl VllmRunner {
    pub fn new(config: VllmRunnerConfig) -> Result<Self, RunnerError> {
        let client = Client::builder()
            .connect_timeout(config.timeouts.probe)
            .build()
            .map_err(|source| RunnerError::Transport {
                operation: "build HTTP client",
                source,
            })?;
        let mut base_url = Url::parse(&config.base_url)
            .map_err(|error| RunnerError::InvalidBaseUrl(error.to_string()))?;
        if !base_url.path().ends_with('/') {
            let path = format!("{}/", base_url.path());
            base_url.set_path(&path);
        }
        Ok(Self {
            client,
            base_url,
            timeouts: config.timeouts,
            api_key: config.api_key,
        })
    }

    fn endpoint(&self, path: &str) -> Result<Url, RunnerError> {
        self.base_url
            .join(path.trim_start_matches('/'))
            .map_err(|error| RunnerError::InvalidBaseUrl(error.to_string()))
    }

    async fn send(
        &self,
        operation: &'static str,
        method: Method,
        path: &str,
        body: Option<Value>,
        timeout: Duration,
    ) -> Result<Response, RunnerError> {
        let url = self.endpoint(path)?;
        let mut request = self.client.request(method, url).timeout(timeout);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = tokio::time::timeout(timeout, request.send())
            .await
            .map_err(|_| RunnerError::Timeout { operation, timeout })?
            .map_err(|source| RunnerError::Transport { operation, source })?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status();
        let body = tokio::time::timeout(timeout, read_error_body(response))
            .await
            .map_err(|_| RunnerError::Timeout { operation, timeout })?
            .unwrap_or_else(|error| format!("<body read failed: {error}>"));
        Err(RunnerError::Http {
            operation,
            status,
            body,
        })
    }

    async fn post(
        &self,
        operation: &'static str,
        path: &str,
        timeout: Duration,
    ) -> Result<(), RunnerError> {
        self.send(operation, Method::POST, path, None, timeout)
            .await?;
        Ok(())
    }

    pub(crate) async fn sleep_from(
        &self,
        current: Option<SleepLevel>,
        observed_sleeping: bool,
        target: SleepLevel,
    ) -> Result<(), RunnerError> {
        if current.is_some_and(|current| current != target)
            || (current.is_none() && observed_sleeping)
        {
            // vLLM 0.26 ignores sleep commands while already sleeping, so an
            // existing or unknown-level sleeper must be made resident before
            // changing tiers.
            self.wake_from_with_timeout(current, self.timeouts.sleep)
                .await?;
        }
        self.post(
            "sleep",
            &format!("/sleep?level={}", target.as_u8()),
            self.timeouts.sleep,
        )
        .await
    }

    pub(crate) async fn wake_from(&self, current: Option<SleepLevel>) -> Result<(), RunnerError> {
        self.wake_from_with_timeout(current, self.timeouts.wake)
            .await
    }

    async fn wake_from_with_timeout(
        &self,
        current: Option<SleepLevel>,
        timeout: Duration,
    ) -> Result<(), RunnerError> {
        match current.unwrap_or(SleepLevel::L2) {
            SleepLevel::L1 => self.post("wake", "/wake_up", timeout).await,
            SleepLevel::L2 => {
                // Unknown sleepers are treated as L2: restoring/reloading
                // discarded weights is safe for an L1 sleeper as well.
                self.post("wake weights", "/wake_up?tags=weights", timeout)
                    .await?;
                self.send(
                    "reload weights",
                    Method::POST,
                    "/collective_rpc",
                    Some(json!({"method": "reload_weights"})),
                    timeout,
                )
                .await?;
                self.post("wake KV cache", "/wake_up?tags=kv_cache", timeout)
                    .await
            }
        }
    }
}

async fn read_error_body(mut response: Response) -> Result<String, reqwest::Error> {
    let mut body = Vec::with_capacity(ERROR_BODY_LIMIT_BYTES);
    let mut truncated = false;

    while let Some(chunk) = response.chunk().await? {
        let remaining = ERROR_BODY_LIMIT_BYTES.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }

    let mut diagnostic = String::from_utf8_lossy(&body).into_owned();
    if truncated {
        diagnostic.push_str(&format!(
            " <truncated after {ERROR_BODY_LIMIT_BYTES} bytes>"
        ));
    }
    Ok(diagnostic)
}

#[async_trait]
impl Runner for VllmRunner {
    async fn health(&self) -> Result<(), RunnerError> {
        self.send("health", Method::GET, "/health", None, self.timeouts.probe)
            .await?;
        Ok(())
    }

    async fn models(&self) -> Result<Vec<String>, RunnerError> {
        let response = self
            .send(
                "model discovery",
                Method::GET,
                "/v1/models",
                None,
                self.timeouts.probe,
            )
            .await?;
        let response = tokio::time::timeout(self.timeouts.probe, response.json::<ModelsResponse>())
            .await
            .map_err(|_| RunnerError::Timeout {
                operation: "model discovery",
                timeout: self.timeouts.probe,
            })?
            .map_err(|source| RunnerError::InvalidResponse {
                operation: "model discovery",
                source,
            })?;
        Ok(response.data.into_iter().map(|model| model.id).collect())
    }

    async fn is_sleeping(&self) -> Result<bool, RunnerError> {
        let response = self
            .send(
                "sleep observation",
                Method::GET,
                "/is_sleeping",
                None,
                self.timeouts.probe,
            )
            .await?;
        let response =
            tokio::time::timeout(self.timeouts.probe, response.json::<SleepingResponse>())
                .await
                .map_err(|_| RunnerError::Timeout {
                    operation: "sleep observation",
                    timeout: self.timeouts.probe,
                })?
                .map_err(|source| RunnerError::InvalidResponse {
                    operation: "sleep observation",
                    source,
                })?;
        Ok(response.is_sleeping)
    }

    async fn warmup(&self, request: &WarmupRequest) -> Result<(), RunnerError> {
        let mut payload = request.body.clone();
        payload.insert("model".into(), Value::String(request.model.clone()));
        payload.insert("stream".into(), Value::Bool(false));
        self.send(
            "warmup",
            Method::POST,
            &request.path,
            Some(Value::Object(payload)),
            self.timeouts.warmup,
        )
        .await?;
        Ok(())
    }

    async fn sleep(&self, level: SleepLevel) -> Result<(), RunnerError> {
        self.sleep_from(None, false, level).await
    }

    async fn wake(&self, level: SleepLevel) -> Result<(), RunnerError> {
        self.wake_from(Some(level)).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;

    async fn mock_server(
        responses: Vec<&'static str>,
    ) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        let handle = tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = vec![0; 8192];
                let count = stream.read(&mut bytes).await.unwrap();
                seen.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&bytes[..count]).into_owned());
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (format!("http://{address}"), requests, handle)
    }

    async fn delayed_mock_server(
        delays: Vec<Duration>,
        response: &'static str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            for delay in delays {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = vec![0; 8192];
                let count = stream.read(&mut bytes).await.unwrap();
                assert!(count > 0, "mock server received an empty request");
                tokio::time::sleep(delay).await;
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{address}"), handle)
    }

    const EMPTY_OK: &str = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    #[tokio::test]
    async fn unknown_sleep_level_after_restart_uses_l2_reload_sequence() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK, EMPTY_OK, EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner.wake_from(None).await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].starts_with("POST /wake_up?tags=weights "));
        assert!(requests[1].starts_with("POST /collective_rpc "));
        assert!(requests[1].contains(r#""method":"reload_weights""#));
        assert!(requests[2].starts_with("POST /wake_up?tags=kv_cache "));
    }

    #[tokio::test]
    async fn level_two_wake_uses_vllm_three_step_sequence() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK, EMPTY_OK, EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner.wake(SleepLevel::L2).await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert!(requests[0].starts_with("POST /wake_up?tags=weights "));
        assert!(requests[1].starts_with("POST /collective_rpc "));
        assert!(requests[1].contains(r#""method":"reload_weights""#));
        assert!(requests[2].starts_with("POST /wake_up?tags=kv_cache "));
    }

    #[tokio::test]
    async fn unknown_l1_sleeper_promotes_to_l2_through_safe_reload() {
        let (base_url, requests, server) =
            mock_server(vec![EMPTY_OK, EMPTY_OK, EMPTY_OK, EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner.sleep_from(None, true, SleepLevel::L2).await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests[0].starts_with("POST /wake_up?tags=weights "));
        assert!(requests[1].starts_with("POST /collective_rpc "));
        assert!(requests[1].contains(r#""method":"reload_weights""#));
        assert!(requests[2].starts_with("POST /wake_up?tags=kv_cache "));
        assert!(requests[3].starts_with("POST /sleep?level=2 "));
    }

    #[tokio::test]
    async fn l1_to_l2_promotion_wakes_once_before_level_two_sleep() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK, EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner
            .sleep_from(Some(SleepLevel::L1), true, SleepLevel::L2)
            .await
            .unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /wake_up "));
        assert!(requests[1].starts_with("POST /sleep?level=2 "));
    }

    #[tokio::test]
    async fn direct_level_two_sleep_does_not_wake_first() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner.sleep(SleepLevel::L2).await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /sleep?level=2 "));
    }

    #[tokio::test]
    async fn readiness_checks_health_and_served_model() {
        let models = r#"{"object":"list","data":[{"id":"served-name","object":"model"}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            models.len(),
            models
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let (base_url, _, server) = mock_server(vec![EMPTY_OK, response]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        runner.ready("served-name").await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn warmup_preserves_fields_and_forces_model_and_non_streaming() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();
        let configured = json!({
            "model": "operator-supplied-name",
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]
            }],
            "max_tokens": 7,
            "guided_json": {"type": "object", "properties": {"answer": {"type": "string"}}},
            "extra_body": {"priority": 3}
        });
        runner
            .warmup(&WarmupRequest {
                path: "/v1/chat/completions".into(),
                model: "canonical-model".into(),
                body: configured.as_object().unwrap().clone(),
            })
            .await
            .unwrap();
        server.await.unwrap();

        let request = &requests.lock().unwrap()[0];
        assert!(request.starts_with("POST /v1/chat/completions "));
        let (_, body) = request
            .split_once("\r\n\r\n")
            .expect("mock request should contain a JSON body");
        let actual: Value = serde_json::from_str(body).unwrap();
        let mut expected = configured;
        expected["model"] = json!("canonical-model");
        expected["stream"] = json!(false);
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn calls_are_bounded() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url: format!("http://{address}"),
            timeouts: VllmTimeouts::uniform(Duration::from_millis(30)),
            api_key: None,
        })
        .unwrap();
        assert!(runner.health().await.is_err());
        server.abort();
    }

    #[tokio::test]
    async fn lifecycle_responses_can_exceed_probe_timeout() {
        let (base_url, server) = delayed_mock_server(
            vec![
                Duration::from_millis(80),
                Duration::from_millis(80),
                Duration::from_millis(80),
            ],
            EMPTY_OK,
        )
        .await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts {
                probe: Duration::from_millis(20),
                sleep: Duration::from_millis(250),
                wake: Duration::from_millis(250),
                warmup: Duration::from_millis(250),
            },
            api_key: None,
        })
        .unwrap();

        runner.sleep(SleepLevel::L1).await.unwrap();
        runner.wake(SleepLevel::L1).await.unwrap();
        runner
            .warmup(&WarmupRequest {
                path: "/v1/completions".into(),
                model: "m".into(),
                body: json!({"prompt": "warm me", "max_tokens": 1})
                    .as_object()
                    .unwrap()
                    .clone(),
            })
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn probe_still_times_out_promptly() {
        let (base_url, server) =
            delayed_mock_server(vec![Duration::from_millis(250)], EMPTY_OK).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts {
                probe: Duration::from_millis(30),
                sleep: Duration::from_millis(300),
                wake: Duration::from_millis(300),
                warmup: Duration::from_millis(300),
            },
            api_key: None,
        })
        .unwrap();
        let started = tokio::time::Instant::now();

        assert!(runner.health().await.is_err());
        assert!(started.elapsed() < Duration::from_millis(150));
        server.abort();
    }

    #[tokio::test]
    async fn sleep_observation_still_times_out_promptly() {
        let sleeping = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"is_sleeping\":true}";
        let (base_url, server) =
            delayed_mock_server(vec![Duration::from_millis(250)], sleeping).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts {
                probe: Duration::from_millis(30),
                sleep: Duration::from_millis(300),
                wake: Duration::from_millis(300),
                warmup: Duration::from_millis(300),
            },
            api_key: None,
        })
        .unwrap();
        let started = tokio::time::Instant::now();

        let error = runner.is_sleeping().await.unwrap_err();
        match error {
            RunnerError::Timeout { operation, .. } | RunnerError::Transport { operation, .. } => {
                assert_eq!(operation, "sleep observation")
            }
            error => panic!("unexpected bounded sleep-observation error: {error}"),
        }
        assert!(started.elapsed() < Duration::from_millis(150));
        server.abort();
    }

    #[tokio::test]
    async fn non_success_response_body_is_capped_and_marked() {
        let diagnostic = format!("vLLM failure: {}", "x".repeat(ERROR_BODY_LIMIT_BYTES));
        let response = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            diagnostic.len(),
            diagnostic
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let (base_url, _, server) = mock_server(vec![response]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        let error = runner.health().await.unwrap_err();
        server.await.unwrap();

        let RunnerError::Http { status, body, .. } = error else {
            panic!("expected HTTP error, got {error}");
        };
        assert_eq!(status.as_u16(), 500);
        assert!(body.starts_with("vLLM failure: "));
        assert!(body.ends_with(&format!("<truncated after {ERROR_BODY_LIMIT_BYTES} bytes>")));
        assert_eq!(
            body.len(),
            ERROR_BODY_LIMIT_BYTES
                + format!(" <truncated after {ERROR_BODY_LIMIT_BYTES} bytes>").len()
        );
    }

    #[tokio::test]
    async fn non_success_response_preserves_small_diagnostic() {
        let diagnostic = r#"{"error":"broken"}"#;
        let response = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            diagnostic.len(),
            diagnostic
        );
        let response: &'static str = Box::leak(response.into_boxed_str());
        let (base_url, _, server) = mock_server(vec![response]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            timeouts: VllmTimeouts::uniform(Duration::from_secs(1)),
            api_key: None,
        })
        .unwrap();

        let error = runner.health().await.unwrap_err();
        server.await.unwrap();

        let RunnerError::Http { body, .. } = error else {
            panic!("expected HTTP error, got {error}");
        };
        assert_eq!(body, diagnostic);
    }
}
