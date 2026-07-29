use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod vllm;

pub use vllm::{VllmRunner, VllmRunnerConfig, VllmTimeouts};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SleepLevel {
    L1,
    L2,
}

impl SleepLevel {
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::L1 => 1,
            Self::L2 => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarmupRequest {
    pub path: String,
    pub model: String,
    pub body: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{operation} timed out after {timeout:?}")]
    Timeout {
        operation: &'static str,
        timeout: Duration,
    },
    #[error("{operation} request failed: {source}")]
    Transport {
        operation: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("{operation} returned HTTP {status}: {body}")]
    Http {
        operation: &'static str,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("{operation} returned an invalid response: {source}")]
    InvalidResponse {
        operation: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("configured base URL is invalid: {0}")]
    InvalidBaseUrl(String),
    #[error("vLLM did not advertise expected model {expected:?}; advertised: {advertised:?}")]
    ModelNotServed {
        expected: String,
        advertised: Vec<String>,
    },
}

/// Runner-specific lifecycle and observation seam owned by the pool manager.
///
/// Implementations must bound every network operation. `wake(L2)` is responsible
/// for any runner-specific multi-step restore sequence.
#[async_trait]
pub trait Runner: Send + Sync {
    async fn health(&self) -> Result<(), RunnerError>;
    async fn models(&self) -> Result<Vec<String>, RunnerError>;
    async fn is_sleeping(&self) -> Result<bool, RunnerError>;
    async fn warmup(&self, request: &WarmupRequest) -> Result<(), RunnerError>;
    async fn sleep(&self, level: SleepLevel) -> Result<(), RunnerError>;
    async fn wake(&self, level: SleepLevel) -> Result<(), RunnerError>;

    async fn ready(&self, expected_model: &str) -> Result<(), RunnerError> {
        self.health().await?;
        let advertised = self.models().await?;
        if advertised.iter().any(|model| model == expected_model) {
            Ok(())
        } else {
            Err(RunnerError::ModelNotServed {
                expected: expected_model.to_owned(),
                advertised,
            })
        }
    }
}
