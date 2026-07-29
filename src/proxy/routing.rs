use std::{borrow::Cow, collections::HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoute {
    pub requested: String,
    pub canonical: String,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RequestRoutingError {
    #[error("request body exceeds the configured {limit}-byte limit")]
    TooLarge { limit: usize },
    #[error("request body is not valid JSON: {0}")]
    InvalidJson(String),
    #[error("request body must contain a string field named `model`")]
    MissingModel,
    #[error("model `{0}` is not configured")]
    UnknownModel(String),
}

impl RequestRoutingError {
    pub fn status(&self) -> u16 {
        match self {
            Self::TooLarge { .. } => 413,
            Self::UnknownModel(_) => 404,
            Self::InvalidJson(_) | Self::MissingModel => 400,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::TooLarge { .. } => "request_too_large",
            Self::InvalidJson(_) => "invalid_json",
            Self::MissingModel => "missing_model",
            Self::UnknownModel(_) => "model_not_found",
        }
    }

    pub fn envelope(&self, dialect: ErrorDialect) -> RouterErrorEnvelope {
        RouterErrorEnvelope::new(
            dialect,
            self.status(),
            self.code(),
            self.to_string(),
            "invalid_request_error",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorDialect {
    OpenAi,
    Anthropic,
}

impl ErrorDialect {
    pub fn for_path(path: &str) -> Self {
        match path {
            "/v1/messages" | "/v1/messages/count_tokens" => Self::Anthropic,
            _ => Self::OpenAi,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum RouterErrorEnvelope {
    OpenAi(OpenAiError),
    Anthropic(AnthropicError),
}

impl RouterErrorEnvelope {
    pub fn new(
        dialect: ErrorDialect,
        status: u16,
        code: impl Into<String>,
        message: impl Into<String>,
        openai_type: impl Into<String>,
    ) -> Self {
        let code = code.into();
        let message = message.into();
        match dialect {
            ErrorDialect::OpenAi => Self::OpenAi(OpenAiError {
                error: OpenAiErrorBody {
                    message,
                    error_type: openai_type.into(),
                    code,
                },
            }),
            ErrorDialect::Anthropic => Self::Anthropic(AnthropicError {
                response_type: "error".into(),
                error: AnthropicErrorBody {
                    message,
                    error_type: anthropic_error_type(status).into(),
                },
            }),
        }
    }
}

fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiError {
    pub error: OpenAiErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

#[derive(Deserialize)]
struct ModelBody<'a> {
    #[serde(borrow)]
    model: Option<Cow<'a, str>>,
}

/// Parse the authoritative model field without modifying the original bytes.
pub fn parse_model_route(
    body: &[u8],
    max_body_bytes: usize,
    aliases: &HashMap<String, String>,
) -> Result<ModelRoute, RequestRoutingError> {
    if body.len() > max_body_bytes {
        return Err(RequestRoutingError::TooLarge {
            limit: max_body_bytes,
        });
    }
    let parsed: ModelBody<'_> = serde_json::from_slice(body)
        .map_err(|error| RequestRoutingError::InvalidJson(error.to_string()))?;
    let requested = parsed
        .model
        .filter(|model| !model.is_empty())
        .ok_or(RequestRoutingError::MissingModel)?;
    let canonical = aliases
        .get(requested.as_ref())
        .cloned()
        .ok_or_else(|| RequestRoutingError::UnknownModel(requested.to_string()))?;
    Ok(ModelRoute {
        requested: requested.into_owned(),
        canonical,
    })
}

/// RFC 9110 hop-by-hop fields plus the legacy proxy connection field.
pub fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aliases() -> HashMap<String, String> {
        HashMap::from([
            ("chat".into(), "chat".into()),
            ("friendly".into(), "chat".into()),
        ])
    }

    #[test]
    fn parses_alias_without_touching_body() {
        let body = br#"{"model":"friendly","messages":[]}"#;
        let before = body.to_vec();
        let route = parse_model_route(body, 1024, &aliases()).unwrap();
        assert_eq!(route.canonical, "chat");
        assert_eq!(body, before.as_slice());
    }

    #[test]
    fn parses_escaped_model_names_without_touching_body() {
        let mut aliases = aliases();
        aliases.insert("quote\"model".into(), "chat".into());
        let cases = [
            (
                br#"{"model":"friend\u006cy","messages":[]}"#.as_slice(),
                "friendly",
            ),
            (
                br#"{"model":"quote\"model","messages":[]}"#.as_slice(),
                "quote\"model",
            ),
        ];

        for (body, requested) in cases {
            let before = body.to_vec();
            let route = parse_model_route(body, 1024, &aliases).unwrap();
            assert_eq!(route.requested, requested);
            assert_eq!(route.canonical, "chat");
            assert_eq!(body, before.as_slice());
        }
    }

    #[test]
    fn rejects_bad_requests_before_lifecycle_work() {
        let cases = [
            (b"not json".as_slice(), 400),
            (br#"{"model":3}"#.as_slice(), 400),
            (br#"{"messages":[]}"#.as_slice(), 400),
            (br#"{"model":"missing"}"#.as_slice(), 404),
        ];
        for (body, status) in cases {
            assert_eq!(
                parse_model_route(body, 1024, &aliases())
                    .unwrap_err()
                    .status(),
                status
            );
        }
        assert_eq!(
            parse_model_route(br#"{"model":"chat"}"#, 2, &aliases())
                .unwrap_err()
                .status(),
            413
        );
    }

    #[test]
    fn selects_anthropic_errors_only_for_messages_endpoints() {
        assert_eq!(
            ErrorDialect::for_path("/v1/messages"),
            ErrorDialect::Anthropic
        );
        assert_eq!(
            ErrorDialect::for_path("/v1/messages/count_tokens"),
            ErrorDialect::Anthropic
        );
        assert_eq!(
            ErrorDialect::for_path("/v1/chat/completions"),
            ErrorDialect::OpenAi
        );
        assert_eq!(ErrorDialect::for_path("/v1/rerank"), ErrorDialect::OpenAi);
    }

    #[test]
    fn anthropic_envelopes_cover_router_error_statuses() {
        let cases = [
            (400, "invalid_request_error"),
            (404, "not_found_error"),
            (405, "invalid_request_error"),
            (413, "request_too_large"),
            (429, "rate_limit_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
        ];
        for (status, expected_type) in cases {
            let value = serde_json::to_value(RouterErrorEnvelope::new(
                ErrorDialect::Anthropic,
                status,
                "stable_code",
                "human-readable message",
                "router_error",
            ))
            .unwrap();
            assert_eq!(value["type"], "error", "status {status}");
            assert_eq!(value["error"]["type"], expected_type, "status {status}");
            assert!(value["error"].get("code").is_none(), "status {status}");
            assert_eq!(
                value["error"]["message"], "human-readable message",
                "status {status}"
            );
        }
    }

    #[test]
    fn openai_and_cohere_paths_keep_the_existing_error_shape() {
        for path in ["/v1/chat/completions", "/v1/rerank", "/v2/embed"] {
            let value = serde_json::to_value(RouterErrorEnvelope::new(
                ErrorDialect::for_path(path),
                503,
                "deadline_exceeded",
                "timed out",
                "router_error",
            ))
            .unwrap();
            assert!(value.get("type").is_none(), "path {path}");
            assert_eq!(value["error"]["type"], "router_error", "path {path}");
            assert_eq!(value["error"]["code"], "deadline_exceeded", "path {path}");
        }
    }

    #[test]
    fn recognizes_hop_headers_case_insensitively() {
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("transfer-encoding"));
        assert!(!is_hop_by_hop("authorization"));
    }
}
