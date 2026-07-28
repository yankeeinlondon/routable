# Pingora crate ecosystem: inventory, features, versions, landscape

Verified 2026-07-28 against crates.io and the cloudflare/pingora `main` branch.

## Versions

Latest: **0.8.1 (2026-06-04)**. History: 0.8.0 (2026-03, mTLS client-cert verify, pipe
subrequests, proxy service builder, Windows fixes, WebSocket upgrade fixes), 0.7.0 (2026-01,
ConnectionFilter, s2n backend, background subrequests), 0.6.0 (2025-08, h2 "MadeYouReset"
mitigation, H2Options, multipart ranges), 0.5.0 (2025-05, TCP hooks, retry limits, custom
error responses), 0.4.0 (2024-11), 0.1.0 (2024-02, open-sourcing). 0.8.1 is a security patch
(bounded default HTTP/2 server limits, rustls advisory bumps).

MSRV **1.85**, rolling ~6-month policy. Build deps: clang (BoringSSL), perl (OpenSSL).
Platform: Linux x86_64/aarch64 tier 1; macOS best-effort; Windows preliminary (community).

## The facade crate and its features

`pingora` re-exports everything. **default = [] — you must pick a TLS backend** (mutually
exclusive): `openssl` | `boringssl` | `s2n` | `rustls` (experimental). Functional features:

| Feature | Pulls in |
|---|---|
| `proxy` | pingora-proxy (ProxyHttp, http_proxy_service) |
| `lb` | pingora-load-balancing (+ implies proxy) |
| `cache` | pingora-cache |
| `time` | pingora-timeout re-export |
| `sentry` | panic capture to Sentry |
| `connection_filter` | pre-TLS TCP connection filtering (0.7+) |
| `patched_http1` | lenient/modified HTTP/1 parsing |
| `dial9`, `dial9-worker-s3`, `dial9-cpu-profiling` | tokio runtime telemetry (needs `--cfg tokio_unstable`) |

Modules re-exported: `apps`, `connectors`, `http`, `listeners`, `modules`, `protocols`,
`server`, `services`, `upstreams`, `utils`, `prelude` (+ optional `proxy`, `lb`, `cache`).
`use pingora::prelude::*` gives Server, Opt, HttpPeer, ProxyHttp, Session, Result,
LoadBalancer, RoundRobin, TcpHealthCheck, background_service, http_proxy_service.

## Workspace members (20)

| Crate | Role |
|---|---|
| **pingora** | facade / feature-gated re-exports |
| **pingora-core** | Server, Service, listeners (TCP/UDS/TLS), connectors, protocols, daemonization, graceful upgrade (`Fds` passing), `ServerConf`/`Opt`, `ShutdownWatch`, apps (`ServerApp`/`HttpServerApp`) |
| **pingora-proxy** | `ProxyHttp` phase trait, `Session`, `http_proxy_service()`; HTTP/1+2, pooling, mTLS, retries, ranges |
| **pingora-http** | case-preserving `RequestHeader`/`ResponseHeader` over the `http` crate |
| **pingora-load-balancing** | `LoadBalancer`, `Backend(s)`, selection (RoundRobin/Random/FNV/Ketama), `ServiceDiscovery`, TCP/HTTP health checks |
| **pingora-cache** | HTTP cache state machine: `Storage` trait, `CacheKey/Meta`, Vary, cache lock, eviction, predictor (experimental) |
| **pingora-memory-cache** | `MemoryCache`/`RTCache` — TTL cache w/ stampede protection (TinyUFO-backed) |
| **tinyufo** | lock-free TinyLFU + S3-FIFO cache |
| **pingora-lru** | sharded LRU (used by cache eviction) |
| **pingora-ketama** | nginx-compatible consistent hashing (`Continuum`, `Bucket`, 160 pts/weight) |
| **pingora-limits** | `Estimator` (CM-sketch), `Rate`, `Inflight` |
| **pingora-timeout** | fast timeout/sleep (~25x cheaper than tokio's) |
| **pingora-error** | `Error`/`BError`/`Result`, `ErrorType`, `ErrorSource`, `RetryType` |
| **pingora-pool** | upstream connection pooling (`ConnectionPool`, lock-free hot path) |
| **pingora-runtime** | `Steal` (tokio multithread) vs `NoSteal` (per-core single-thread pool) |
| **pingora-header-serde** | zstd + trained-dict compression of cached headers |
| **pingora-openssl / -boringssl** | async TLS glue + FFI extensions per backend |
| **pingora-rustls** | rustls backend (experimental) |
| **pingora-s2n** | AWS s2n-tls backend (new in 0.7.0) |
| **pingora-prometheus** | prometheus HTTP service (repo main; not yet on crates.io — use `pingora_core::services::listening::Service::prometheus_http_service()` on published versions) |

## Ecosystem / prior art worth reading

- **Cloudflare blogs**: "How we built Pingora" (2022) and "Open-sourcing Pingora" (2024) —
  architecture rationale (multi-threaded vs nginx multi-process, connection-reuse wins).
- **River** (github.com/memorysafety/river) — ISRG's batteries-included reverse proxy on
  Pingora, KDL config. Development paused since v0.5.0 (2024-08) but good reference code.
- **Pingap** (github.com/vicanso/pingap) — most active community derivative: nginx-like
  proxy, TOML config, hot reload, web admin.
- **Sentinel** (sentinel.raskell.io) — pingora-based proxy inspired by River.

## Known limitations

- **No HTTP/3/QUIC** (issue #95, "long term goal").
- Pre-1.0: breaking changes between minors are routine.
- Cache layer experimental; no bundled disk storage backend.
- rustls backend experimental; Windows preliminary.
- WebSocket and gRPC (h2 e2e) work; gRPC-web via bundled module.
