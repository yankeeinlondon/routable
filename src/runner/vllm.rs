use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, Method, Response, Url};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Runner, RunnerError, SleepLevel, WarmupRequest};

#[derive(Clone, Debug)]
pub struct VllmRunnerConfig {
    pub base_url: String,
    pub request_timeout: Duration,
    pub api_key: Option<String>,
}

impl VllmRunnerConfig {
    pub fn loopback(port: u16, request_timeout: Duration) -> Self {
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
            request_timeout,
            api_key: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VllmRunner {
    client: Client,
    base_url: Url,
    timeout: Duration,
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
        let timeout = config.request_timeout;
        let client = Client::builder()
            .connect_timeout(timeout)
            .timeout(timeout)
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
            timeout,
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
    ) -> Result<Response, RunnerError> {
        let url = self.endpoint(path)?;
        let mut request = self.client.request(method, url).timeout(self.timeout);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = tokio::time::timeout(self.timeout, request.send())
            .await
            .map_err(|_| RunnerError::Timeout {
                operation,
                timeout: self.timeout,
            })?
            .map_err(|source| RunnerError::Transport { operation, source })?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status();
        let body = tokio::time::timeout(self.timeout, response.text())
            .await
            .map_err(|_| RunnerError::Timeout {
                operation,
                timeout: self.timeout,
            })?
            .unwrap_or_else(|error| format!("<body read failed: {error}>"));
        Err(RunnerError::Http {
            operation,
            status,
            body,
        })
    }

    async fn post(&self, operation: &'static str, path: &str) -> Result<(), RunnerError> {
        self.send(operation, Method::POST, path, None).await?;
        Ok(())
    }
}

#[async_trait]
impl Runner for VllmRunner {
    async fn health(&self) -> Result<(), RunnerError> {
        self.send("health", Method::GET, "/health", None).await?;
        Ok(())
    }

    async fn models(&self) -> Result<Vec<String>, RunnerError> {
        let response = self
            .send("model discovery", Method::GET, "/v1/models", None)
            .await?;
        let response = tokio::time::timeout(self.timeout, response.json::<ModelsResponse>())
            .await
            .map_err(|_| RunnerError::Timeout {
                operation: "model discovery",
                timeout: self.timeout,
            })?
            .map_err(|source| RunnerError::InvalidResponse {
                operation: "model discovery",
                source,
            })?;
        Ok(response.data.into_iter().map(|model| model.id).collect())
    }

    async fn is_sleeping(&self) -> Result<bool, RunnerError> {
        let response = self
            .send("sleep observation", Method::GET, "/is_sleeping", None)
            .await?;
        let response = tokio::time::timeout(self.timeout, response.json::<SleepingResponse>())
            .await
            .map_err(|_| RunnerError::Timeout {
                operation: "sleep observation",
                timeout: self.timeout,
            })?
            .map_err(|source| RunnerError::InvalidResponse {
                operation: "sleep observation",
                source,
            })?;
        Ok(response.is_sleeping)
    }

    async fn warmup(&self, request: &WarmupRequest) -> Result<(), RunnerError> {
        let payload = json!({
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": false
        });
        self.send("warmup", Method::POST, "/v1/completions", Some(payload))
            .await?;
        Ok(())
    }

    async fn sleep(&self, level: SleepLevel) -> Result<(), RunnerError> {
        self.post("sleep", &format!("/sleep?level={}", level.as_u8()))
            .await
    }

    async fn wake(&self, level: SleepLevel) -> Result<(), RunnerError> {
        match level {
            SleepLevel::L1 => self.post("wake", "/wake_up").await,
            SleepLevel::L2 => {
                // vLLM 0.26 deep sleep discards weights. Restore allocation,
                // reload weights in place, then restore KV cache/graphs.
                self.post("wake weights", "/wake_up?tags=weights").await?;
                self.send(
                    "reload weights",
                    Method::POST,
                    "/collective_rpc",
                    Some(json!({"method": "reload_weights"})),
                )
                .await?;
                self.post("wake KV cache", "/wake_up?tags=kv_cache").await
            }
        }
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

    const EMPTY_OK: &str = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

    #[tokio::test]
    async fn level_two_wake_uses_vllm_three_step_sequence() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK, EMPTY_OK, EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            request_timeout: Duration::from_secs(1),
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
            request_timeout: Duration::from_secs(1),
            api_key: None,
        })
        .unwrap();

        runner.ready("served-name").await.unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn warmup_is_openai_shaped_and_non_streaming() {
        let (base_url, requests, server) = mock_server(vec![EMPTY_OK]).await;
        let runner = VllmRunner::new(VllmRunnerConfig {
            base_url,
            request_timeout: Duration::from_secs(1),
            api_key: None,
        })
        .unwrap();
        runner
            .warmup(&WarmupRequest {
                model: "m".into(),
                prompt: "warm me".into(),
                max_tokens: 2,
                temperature: 0.0,
            })
            .await
            .unwrap();
        server.await.unwrap();

        let request = &requests.lock().unwrap()[0];
        assert!(request.starts_with("POST /v1/completions "));
        assert!(request.contains(r#""model":"m""#));
        assert!(request.contains(r#""stream":false"#));
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
            request_timeout: Duration::from_millis(30),
            api_key: None,
        })
        .unwrap();
        assert!(runner.health().await.is_err());
        server.abort();
    }
}
