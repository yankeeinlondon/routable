use std::{fs, path::Path};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("cannot read bearer-token file {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("bearer-token file must be owned by root")]
    NotRootOwned,
    #[error("bearer-token file must not be readable or writable by group/other")]
    InsecurePermissions,
    #[error("bearer-token file is empty")]
    Empty,
}

#[derive(Clone)]
pub struct TokenAuth {
    token: Vec<u8>,
}

impl TokenAuth {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AuthError> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|source| AuthError::Read {
            path: path.display().to_string(),
            source,
        })?;
        #[cfg(unix)]
        {
            if metadata.uid() != 0 {
                return Err(AuthError::NotRootOwned);
            }
            if metadata.mode() & 0o077 != 0 {
                return Err(AuthError::InsecurePermissions);
            }
        }
        let raw = fs::read(path).map_err(|source| AuthError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let token = trim_ascii(&raw);
        if token.is_empty() {
            return Err(AuthError::Empty);
        }
        Ok(Self {
            token: token.to_vec(),
        })
    }

    pub fn authorize(&self, authorization: Option<&[u8]>) -> bool {
        let Some(value) = authorization else {
            return false;
        };
        let Some(candidate) = value.strip_prefix(b"Bearer ") else {
            return false;
        };
        constant_time_eq(candidate, &self.token)
    }
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let width = left.len().max(right.len());
    for index in 0..width {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn requires_bearer_scheme_and_exact_token() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "secret").unwrap();
        #[cfg(unix)]
        fs::set_permissions(file.path(), fs::Permissions::from_mode(0o600)).unwrap();
        let auth = TokenAuth::load(file.path()).unwrap();
        assert!(auth.authorize(Some(b"Bearer secret")));
        assert!(!auth.authorize(Some(b"bearer secret")));
        assert!(!auth.authorize(Some(b"Bearer nope")));
        assert!(!auth.authorize(None));
    }

    #[test]
    fn rejects_insecure_mode() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "secret").unwrap();
        #[cfg(unix)]
        {
            fs::set_permissions(file.path(), fs::Permissions::from_mode(0o644)).unwrap();
            assert!(matches!(
                TokenAuth::load(file.path()),
                Err(AuthError::InsecurePermissions)
            ));
        }
    }
}
