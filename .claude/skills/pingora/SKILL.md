---
name: pingora
description: Build proxies, gateways, and dynamic routers with Cloudflare's Pingora Rust crate ecosystem (pingora-core, pingora-proxy, pingora-load-balancing, pingora-cache, pingora-limits, etc). Covers the ProxyHttp phase lifecycle, HttpPeer/upstream selection, load balancing, health checks, dynamic service discovery, rate limiting, caching, streaming/SSE proxying, server lifecycle and zero-downtime upgrades. Trigger on "pingora", "reverse proxy in Rust", "ProxyHttp", "upstream_peer", "HttpPeer", "LoadBalancer", "service discovery", "health check", "dynamic router", "LLM gateway/router", "graceful upgrade", "pingora-cache", "rate limiting in pingora".
---

# Pingora

Pingora is Cloudflare's Rust framework for building fast, programmable network services (it replaced nginx for Cloudflare's origin-facing traffic; >40M req/s in production). It is a **library, not a binary**: you write a Rust program that implements trait callbacks (phases) and Pingora provides the server runtime, connection pooling, TLS, HTTP/1+2, load balancing, and zero-downtime restarts.

**Current release: `pingora` 0.8.1 (2026-06-04). MSRV 1.85 (rolling ~6mo). Pre-1.0: APIs shift between minors — always check docs.rs for your pinned version.** Linux x86_64/aarch64 is tier 1; macOS best-effort dev; Windows preliminary/community. HTTP/3 is NOT supported (long-term goal). WebSockets and gRPC (end-to-end h2) ARE supported.

## Cargo setup

```toml
[dependencies]
# TLS backend is REQUIRED and mutually exclusive: openssl | boringssl | rustls (experimental) | s2n
pingora = { version = "0.8", features = ["openssl", "proxy", "lb"] }
async-trait = "0.1"
```

Feature flags: `proxy` (ProxyHttp), `lb` (load balancing, implies proxy), `cache`, `time`, `sentry`, `connection_filter`. No features are on by default. BoringSSL needs clang; OpenSSL needs perl.

## The 60-second mental model

```
Server (process, signals, upgrade)            pingora-core
 └── Service (listeners + threads)
      ├── http_proxy_service(conf, MyProxy)   ← your ProxyHttp impl (pingora-proxy)
      └── background_service("hc", lb)        ← health checks / discovery (pingora-load-balancing)

Request lifecycle (phases you override on ProxyHttp):
 early_request_filter → request_filter → upstream_peer → connect
   → upstream_request_filter → (stream body up) → upstream_response_filter
   → response_filter → *_body_filter (per chunk, streaming) → logging
 fail_to_connect / error_while_proxy → retry loops back to upstream_peer
```

Only `upstream_peer()` (pick a `Box<HttpPeer>`) and `type CTX`/`new_ctx()` (per-request state) are required. `request_filter` returning `Ok(true)` means "I already wrote the response" (auth denials, rate limits, synthetic responses). Proxying is **streaming by default** — body filters see chunks as they arrive; nothing buffers the whole response.

## Minimal working proxy (verbatim-style skeleton)

```rust
use async_trait::async_trait;
use pingora::prelude::*;
use std::sync::Arc;

pub struct LB(Arc<LoadBalancer<RoundRobin>>);

#[async_trait]
impl ProxyHttp for LB {
    type CTX = ();
    fn new_ctx(&self) -> Self::CTX {}

    async fn upstream_peer(&self, _s: &mut Session, _ctx: &mut ()) -> Result<Box<HttpPeer>> {
        let upstream = self.0.select(b"", 256).unwrap(); // key only matters for consistent hashing
        // (addr, tls: bool, sni). Plain HTTP upstream => (addr, false, String::new())
        Ok(Box::new(HttpPeer::new(upstream, false, String::new())))
    }
}

fn main() {
    let mut server = Server::new(Some(Opt::parse_args())).unwrap();
    server.bootstrap();

    let mut upstreams = LoadBalancer::try_from_iter(["10.0.0.1:8000", "10.0.0.2:8000"]).unwrap();
    upstreams.set_health_check(TcpHealthCheck::new());
    upstreams.health_check_frequency = Some(std::time::Duration::from_secs(1));

    let background = background_service("health check", upstreams); // runs checks + discovery
    let upstreams = background.task();                              // Arc<LoadBalancer<_>> shared handle

    let mut proxy = http_proxy_service(&server.configuration, LB(upstreams));
    proxy.add_tcp("0.0.0.0:6188");

    server.add_service(proxy);
    server.add_service(background);
    server.run_forever();
}
```

Run: `RUST_LOG=INFO cargo run -- -c conf.yaml` (flags: `-d` daemon, `-t` test config, `-u` upgrade).

## Which reference to read

| Task | Read |
|---|---|
| Phase callbacks, CTX, HttpPeer/PeerOptions/timeouts, short-circuit responses, header/body rewriting, retries/failover, SSE/LLM token streaming | [references/proxy-lifecycle.md](references/proxy-lifecycle.md) |
| LoadBalancer, selection algorithms, health checks, **dynamic backends / ServiceDiscovery**, background services, multi-pool routing, and coordinated dynamic route/service registration | [references/load-balancing.md](references/load-balancing.md) |
| Server/Service/runtime model, YAML config + Opt/ServerConf, daemonization, signals, **zero-downtime graceful upgrade**, prometheus, sentry | [references/server-lifecycle.md](references/server-lifecycle.md) |
| Rate limiting (Rate/Inflight/Estimator), in-memory caches (MemoryCache/TinyUFO), HTTP cache layer (pingora-cache), fast timeouts | [references/limits-and-cache.md](references/limits-and-cache.md) |
| Full crate inventory (all 20 workspace members), feature-flag matrix, version history, ecosystem projects (River, Pingap), limitations | [references/crate-ecosystem.md](references/crate-ecosystem.md) |

## Gotchas that bite everyone

- **TLS mismatch = 502**: `HttpPeer::new(addr, true, sni)` against a plain-HTTP upstream fails. For non-TLS upstreams pass `(addr, false, String::new())`.
- **Forgetting the background service**: health checks and discovery only run if you `server.add_service(background_service(...))` — the LoadBalancer alone does nothing periodically.
- **Rewriting bodies**: remove `Content-Length` (and set `Transfer-Encoding: chunked`) in `response_filter` before mutating chunks in `response_body_filter`, or clients hang.
- **Body-based upstream choice is not ordinary `ProxyHttp` routing**: request-body filters run after `upstream_peer`, and replay is capped at 64 KiB. For bounded multimodal/LLM JSON bodies, use a custom `HttpServerApp` plus your own client, or route on a trusted header when clients are controlled.
- **`peer.options.read_timeout` resets on every read** — it's an inter-chunk idle timeout, which is exactly right for LLM token streams (won't kill an active long stream).
- **Retry safety**: `fail_to_connect` guarantees nothing was sent upstream (safe to retry even POSTs, via `e.set_retry(true)`); `error_while_proxy` does not — default only retries reused connections.
- **`Backend` identity = (addr, weight)**: changing a weight makes it a "new" backend (health state resets). `ext` is ignored for equality.
- **`ServiceDiscovery` changes members of one existing pool**: adding a new model/route/service is an application transaction across every registry and must publish its routing snapshot last.
- **Approximate inflight is not drain safety**: `pingora_limits::Inflight` is suitable for selection and soft caps; use exact RAII counters for lifecycle drain/stop decisions.
- **Version drift**: repo `main` docs/examples may be ahead of your pinned crates.io release (e.g. `fail_to_proxy` returns `FailToProxy` struct on main, `u16` on ≤0.5). Verify signatures on docs.rs for your version.
- **`run_forever()` must be called from a synchronous `main`** — it does a `block_on` internally and panics ("Cannot start a runtime from within a runtime") under `#[tokio::main]`. Async setup before it needs a dedicated runtime you keep alive. See server-lifecycle.md § Runtime rules.
- **Detached `tokio::spawn` tasks on a service's runtime block graceful shutdown** — pingora waits for each service runtime to quiesce; a lingering task hangs SIGTERM until the (long) timeout. Spawn long-lived work on your own runtime or abort it on `ShutdownWatch`.
- **A busy listen port is retried every second forever**, not failed fast — guard for "already running" yourself if you want a clean error.
- **`-t` exits inside `bootstrap()`** — code after `bootstrap()` never runs in test mode, so do your own config validation *before* it if `-t` should cover it.
