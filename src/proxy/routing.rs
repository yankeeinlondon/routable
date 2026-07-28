use std::collections::HashMap;

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

    pub fn openai(&self) -> OpenAiError {
        OpenAiError {
            error: OpenAiErrorBody {
                message: self.to_string(),
                error_type: "invalid_request_error".into(),
                code: self.code().into(),
            },
        }
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

#[derive(Deserialize)]
struct ModelBody<'a> {
    #[serde(borrow)]
    model: Option<&'a str>,
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
        .get(requested)
        .cloned()
        .ok_or_else(|| RequestRoutingError::UnknownModel(requested.to_owned()))?;
    Ok(ModelRoute {
        requested: requested.to_owned(),
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
    fn recognizes_hop_headers_case_insensitively() {
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("transfer-encoding"));
        assert!(!is_hop_by_hop("authorization"));
    }
}
