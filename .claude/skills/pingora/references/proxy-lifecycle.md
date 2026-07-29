# pingora-proxy: the ProxyHttp lifecycle

Source of truth: `pingora-proxy/src/proxy_trait.rs` and `docs/user_guide/phase.md` in
https://github.com/cloudflare/pingora. Signatures below verified against `main` (0.8.x era);
older releases differ where noted.

## Phase order

```
new request → early_request_filter → request_filter → [cache phases] → proxy_upstream_filter
  → upstream_peer → connect
     ├─ ok  → connected_to_upstream → upstream_request_filter → request_body_filter (per chunk)
     │        → send req → read resp → upstream_response_filter → response_filter
     │        → upstream_response_body_filter → response_body_filter (per chunk) → logging
     └─ fail → fail_to_connect ──can retry──→ upstream_peer (again)
                              └─can't──→ fail_to_proxy (writes error response) → logging
request_filter returning Ok(true) ──response already sent──→ logging   (short-circuit)
mid-stream errors → error_while_proxy ──can retry──→ upstream_peer, else fail_to_proxy
```

If any filter returns `Err`, the request fails through `fail_to_proxy` and the error is logged.
`logging()` ALWAYS runs last, success or failure — put metrics/access logs there.

## The trait, in lifecycle order

Only `type CTX` + `new_ctx()` + `upstream_peer()` are required. All methods take `&self`
(the struct is shared across all requests — use `Arc`/`Mutex`/atomics for cross-request state)
and `&mut Self::CTX` (per-request state, created by `new_ctx()` before phases, dropped at end).

```rust
#[async_trait]
pub trait ProxyHttp {
    type CTX;
    fn new_ctx(&self) -> Self::CTX;

    // REQUIRED: pick the upstream
    async fn upstream_peer(&self, session: &mut Session, ctx: &mut Self::CTX)
        -> Result<Box<HttpPeer>>;

    // Setup (once, before server start): configure downstream HttpModules.
    // Default registers compression DISABLED: ResponseCompressionBuilder::enable(0)
    fn init_downstream_modules(&self, modules: &mut HttpModules) { ... }

    // First per-request hook, before any module logic
    async fn early_request_filter(&self, _: &mut Session, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Auth / rate limit / routing guard. Ok(true) = "response already written, stop here".
    async fn request_filter(&self, _: &mut Session, _: &mut Self::CTX) -> Result<bool> { Ok(false) }

    // Per-chunk request body (async — can throttle uploads / run WAF)
    async fn request_body_filter(&self, _: &mut Session, body: &mut Option<Bytes>,
        end_of_stream: bool, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Gate before upstream (after cache miss). Ok(false) = you wrote the response (else 502).
    async fn proxy_upstream_filter(&self, _: &mut Session, _: &mut Self::CTX) -> Result<bool> { Ok(true) }

    // After connect (new or reused). fd param is unix-only.
    async fn connected_to_upstream(&self, _: &mut Session, reused: bool, peer: &HttpPeer,
        fd: RawFd, digest: Option<&Digest>, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Mutate the request going upstream (Host header, auth injection, path rewrite)
    async fn upstream_request_filter(&self, _: &mut Session,
        upstream_request: &mut RequestHeader, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Response header from upstream, BEFORE cache (mutations get cached). Skipped on cache hit.
    async fn upstream_response_filter(&self, _: &mut Session,
        upstream_response: &mut ResponseHeader, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Response header to downstream, AFTER cache. Runs for ALL responses incl. cache hits.
    async fn response_filter(&self, _: &mut Session,
        upstream_response: &mut ResponseHeader, _: &mut Self::CTX) -> Result<()> { Ok(()) }

    // Body chunk filters — NOTE: sync fns, not async.
    // Returning Ok(Some(Duration)) sleeps that long before forwarding the chunk (throttle hook).
    fn upstream_response_body_filter(&self, _: &mut Session, body: &mut Option<Bytes>,
        end_of_stream: bool, _: &mut Self::CTX) -> Result<Option<Duration>> { Ok(None) } // pre-cache
    fn response_body_filter(&self, _: &mut Session, body: &mut Option<Bytes>,
        end_of_stream: bool, _: &mut Self::CTX) -> Result<Option<Duration>> { Ok(None) } // to client

    // Trailers (h2/gRPC)
    fn upstream_response_trailer_filter(&self, ...) -> Result<()> { Ok(()) }
    async fn response_trailer_filter(&self, ...) -> Result<Option<Bytes>> { Ok(None) }

    // --- errors ---
    // Connect-phase failure. GUARANTEE: nothing was sent upstream → always safe to retry.
    fn fail_to_connect(&self, _: &mut Session, peer: &HttpPeer, _: &mut Self::CTX,
        e: Box<Error>) -> Box<Error> { e }
    // Mid-proxy failure (connection was established). Default retries only reused connections.
    fn error_while_proxy(&self, peer: &HttpPeer, session: &mut Session, e: Box<Error>,
        _: &mut Self::CTX, client_reused: bool) -> Box<Error> { ... }
    // Terminal: writes error response. Upstream errors → 502, Downstream → 400, Internal → 500.
    // main branch: returns FailToProxy { error_code: u16, can_reuse_downstream: bool }
    // (≤0.5 returned plain u16)
    async fn fail_to_proxy(&self, session: &mut Session, e: &Error, _: &mut Self::CTX) -> FailToProxy;

    // Always last. Metrics + access logging.
    async fn logging(&self, _: &mut Session, e: Option<&Error>, _: &mut Self::CTX) {}

    // Misc: request_summary(), suppress_error_log(), and (main branch) HTTP/1 keepalive
    // cross-request state via persist_connection_context()/on_connection_reuse().
    // Cache phases (request_cache_filter, cache_key_callback, cache_hit_filter,
    // response_cache_filter, cache_vary_filter, ...) — see limits-and-cache.md.
}
```

Most async methods carry `where Self::CTX: Send + Sync` bounds; examples include or omit the
where-clause freely when overriding.

## CTX patterns

```rust
// No state:
type CTX = ();
fn new_ctx(&self) -> Self::CTX {}

// Per-request state, filled early and consumed later:
pub struct MyCtx { beta_user: bool, tries: usize }
// Cross-request state lives on the ProxyHttp struct (&self) or in statics:
pub struct MyProxy { beta_counter: Mutex<usize> }   // AtomicUsize works too
static REQ_COUNTER: Mutex<usize> = Mutex::new(0);
```

Verbatim from `examples/ctx.rs` — route by a flag detected in `request_filter`:

```rust
async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
    ctx.beta_user = check_beta_user(session.req_header());
    Ok(false)
}
async fn upstream_peer(&self, _session: &mut Session, ctx: &mut Self::CTX) -> Result<Box<HttpPeer>> {
    let addr = if ctx.beta_user { ("1.0.0.1", 443) } else { ("1.1.1.1", 443) };
    Ok(Box::new(HttpPeer::new(addr, true, "one.one.one.one".to_string())))
}
```

## HttpPeer and PeerOptions

```rust
// (address, tls, sni). address: ("host", port), SocketAddr, a Backend from lb.select(), ...
HttpPeer::new(("10.0.0.1", 8000), false, String::new())        // plain HTTP upstream
HttpPeer::new(backend, true, "api.internal".to_string())        // TLS + SNI
HttpPeer::new_uds("/run/app.sock", false, String::new())?       // unix socket (unix only)
```

Tune per-request via `peer.options.<field> = ...`:

| Option | Meaning |
|---|---|
| `connection_timeout` | TCP connect only |
| `total_connection_timeout` | connect incl. TLS handshake |
| `read_timeout` | per individual read; **timer resets after each read** → inter-chunk idle timeout, ideal for token streams |
| `write_timeout` | per write to upstream |
| `idle_timeout` | how long the connection may sit idle in the reuse pool |
| `alpn` | `ALPN::H1` / h2 / both |
| `verify_cert`, `verify_hostname`, `alternative_cn`, `ca` | upstream TLS validation |
| `tcp_keepalive`, `tcp_fast_open`, `bind_to`, `dscp` | socket tuning |
| `max_h2_streams`, `h2_ping_interval`, `h2_*_window_size` | h2 upstream tuning |
| `extra_proxy_headers` | headers added when going via CONNECT proxy |

```rust
let mut peer = Box::new(HttpPeer::new(addr, true, "one.one.one.one".to_string()));
peer.options.connection_timeout = Some(Duration::from_millis(100));
```

Upstream connection reuse (keepalive pooling) is on by default; pool size is the server conf
`upstream_keepalive_pool_size`. mTLS: set `peer.client_cert_key: Option<Arc<CertKey>>`.

## Short-circuit responses (auth, rate limit, synthetic)

From `examples/gateway.rs` — deny in `request_filter`, return `Ok(true)`:

```rust
async fn request_filter(&self, session: &mut Session, _ctx: &mut Self::CTX) -> Result<bool> {
    if session.req_header().uri.path().starts_with("/login")
        && !check_login(session.req_header())
    {
        let _ = session
            .respond_error_with_body(403, Bytes::from_static(b"no way!"))
            .await;
        return Ok(true); // response already written, proxy exits to logging
    }
    Ok(false)
}
```

Full synthetic response (from `examples/rate_limiter.rs`):

```rust
let mut header = ResponseHeader::build(429, None).unwrap();
header.insert_header("X-Rate-Limit-Limit", MAX_REQ_PER_SEC.to_string()).unwrap();
session.set_keepalive(None);                                  // close after this response
session.write_response_header(Box::new(header), true).await?; // true = end of stream
return Ok(true);
```

Useful `Session` methods: `req_header()` / `req_header_mut()` (`.uri.path()`, `.headers`,
`.insert_header()`, `.remove_header()`, `.append_header()`), `respond_error(code)`,
`respond_error_with_body(code, bytes)`, `write_response_header(Box<ResponseHeader>, eos)`,
`write_response_body(Option<Bytes>, eos)`, `set_keepalive(None)`,
`response_written() -> Option<&ResponseHeader>` (for logging).

## Header rewriting

```rust
// Request going upstream (examples/load_balancer.rs):
async fn upstream_request_filter(&self, _s: &mut Session,
    upstream_request: &mut RequestHeader, _ctx: &mut Self::CTX) -> Result<()> {
    upstream_request.insert_header("Host", "one.one.one.one").unwrap();
    Ok(())
}
// Response to client (examples/gateway.rs):
async fn response_filter(&self, _s: &mut Session,
    upstream_response: &mut ResponseHeader, _ctx: &mut Self::CTX) -> Result<()> {
    upstream_response.insert_header("Server", "MyGateway").unwrap();
    upstream_response.remove_header("alt-svc"); // we don't do h3
    Ok(())
}
```

## Body rewriting (buffer-then-transform, from examples/modify_response.rs)

Body arrives per chunk as `&mut Option<Bytes>` + `end_of_stream`. To transform a whole body:
buffer chunks into CTX, clear each, emit the transformed body on the last chunk — and remove
`Content-Length` in the header filter first:

```rust
async fn response_filter(&self, _s: &mut Session, resp: &mut ResponseHeader,
    _ctx: &mut Self::CTX) -> Result<()> {
    resp.remove_header("Content-Length");                    // size will change
    resp.insert_header("Transfer-Encoding", "Chunked").unwrap();
    Ok(())
}
fn response_body_filter(&self, _s: &mut Session, body: &mut Option<Bytes>,
    end_of_stream: bool, ctx: &mut Self::CTX) -> Result<Option<Duration>> {
    if let Some(b) = body { ctx.buffer.extend(&b[..]); b.clear(); }
    if end_of_stream {
        let transformed = transform(&ctx.buffer);
        *body = Some(Bytes::from(transformed));
    }
    Ok(None)
}
```

To drop a chunk: `b.clear()` or `*body = None`. To throttle: return `Ok(Some(duration))`.

## Retries and failover

`Error` carries `etype: ErrorType`, `esource: ErrorSource {Upstream, Downstream, Internal}`,
`retry: RetryType {Decided(bool), ReusedOnly}`. Key APIs: `e.set_retry(true)`, `e.retry()`.

- **`fail_to_connect`**: nothing was sent upstream → retrying is always safe (even POST).
  Set `e.set_retry(true)` and `upstream_peer()` is called again; use CTX to pick a different peer.
- **`error_while_proxy`**: connection was live; default marks retry only for reused connections
  with an intact retry buffer. Only loosen for known-idempotent traffic.
- Once response headers reached the client, no recovery — only logging.

Verbatim failover (docs/user_guide/failover.md): try a primary with a tight timeout, fall back once:

```rust
pub struct MyCtx { tries: usize }

fn fail_to_connect(&self, _s: &mut Session, _peer: &HttpPeer, ctx: &mut Self::CTX,
    mut e: Box<Error>) -> Box<Error> {
    if ctx.tries > 0 { return e; }   // already failed over once — give up
    ctx.tries += 1;
    e.set_retry(true);
    e
}
async fn upstream_peer(&self, _s: &mut Session, ctx: &mut Self::CTX) -> Result<Box<HttpPeer>> {
    let addr = if ctx.tries < 1 { ("192.0.2.1", 443) } else { ("1.1.1.1", 443) };
    let mut peer = Box::new(HttpPeer::new(addr, true, "one.one.one.one".to_string()));
    peer.options.connection_timeout = Some(Duration::from_millis(100));
    Ok(peer)
}
```

## Streaming / SSE / LLM token streams

Pingora proxies **streaming by default** — after the request header is sent it enters duplex
mode, simultaneously forwarding the upstream response and downstream request body. There is no
whole-response buffering to disable; the internal body channel is bounded (`TASK_BUFFER_SIZE = 4`
tasks). Each body chunk flows range-filter → `response_body_filter` → written downstream
immediately. There is no explicit flush API and none is needed — chunks are not coalesced.

For LLM/SSE proxying:
- `peer.options.read_timeout` is the knob you want: it resets on every read, so it bounds
  *silence between tokens* without killing a long active stream.
- Chunk boundaries follow upstream network reads, NOT SSE event boundaries. If you need to
  parse/rewrite events, reassemble in CTX and emit incrementally (don't hold to end_of_stream).
- Don't enable response compression for SSE (default module is registered at level 0 = off).
- If rewriting a streamed body, strip `Content-Length` in `response_filter`.
- HTTP/1 upstream + request body with no framing headers → Pingora auto-adds
  `Transfer-Encoding: chunked`.

## Downstream modules

`init_downstream_modules` runs once at startup. Example (gRPC-web bridge, from
`examples/grpc_web_module.rs`): add the module there, then grab its per-request ctx in
`early_request_filter` via `session.downstream_modules_ctx.get_mut::<GrpcWebBridge>()`.
Enable compression by registering `ResponseCompressionBuilder::enable(level)` with level > 0.

## Reading the request body before `upstream_peer` (0.8.1 limitation)

Routing on a body field (e.g. an LLM gateway routing on JSON `model`) wants
read-body → pick peer → replay body. At 0.8.1 this only works for small bodies: the
only replay facility is the retry buffer, hard-capped at `BODY_BUF_LIMIT` = 64 KiB
(H1 and H2), and there is no public API to inject an arbitrary consumed body back
into the downstream session. Beyond 64 KiB the choice is: route on a header instead,
or skip `ProxyHttp` for that listener and implement a custom `HttpServerApp` that
buffers the (bounded) body itself and speaks to upstreams with its own client —
losing the proxy phase machinery but keeping pingora's server/runtime layer. Verified
in production by a router that took the second path for 32 MiB multimodal payloads.
