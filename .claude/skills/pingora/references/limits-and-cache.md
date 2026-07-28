# pingora-limits, in-memory caches, pingora-cache, pingora-timeout

## pingora-limits — rate limiting & concurrency accounting

All structures are Count-Min-Sketch based: O(1) memory regardless of key cardinality,
lock-free, slight over-estimation possible. No config/persistence — plain in-process.

### `Rate` — sliding-interval rate estimation

```rust
use pingora_limits::rate::Rate;
static RATE_LIMITER: Lazy<Rate> = Lazy::new(|| Rate::new(Duration::from_secs(1)));

let seen_so_far = RATE_LIMITER.observe(&key, 1);  // count 1 event; returns count in current interval
let per_sec     = RATE_LIMITER.rate(&key);        // avg rate of the LAST completed interval
```

Verbatim rate-limit phase (from `pingora-proxy/examples/rate_limiter.rs`; key = client header):

```rust
async fn request_filter(&self, session: &mut Session, _ctx: &mut Self::CTX) -> Result<bool> {
    let appid = match self.get_request_appid(session) {
        None => return Ok(false),          // no id → skip limiting
        Some(a) => a,
    };
    let curr_window_requests = RATE_LIMITER.observe(&appid, 1);
    if curr_window_requests > MAX_REQ_PER_SEC {
        let mut header = ResponseHeader::build(429, None).unwrap();
        header.insert_header("X-Rate-Limit-Limit", MAX_REQ_PER_SEC.to_string()).unwrap();
        header.insert_header("X-Rate-Limit-Remaining", "0").unwrap();
        header.insert_header("X-Rate-Limit-Reset", "1").unwrap();
        session.set_keepalive(None);
        session.write_response_header(Box::new(header), true).await?;
        return Ok(true);                   // response sent — short-circuit
    }
    Ok(false)
}
```

Smoother limiting: `rate_with(&key, PROPORTIONAL_RATE_ESTIMATE_CALC_FN)` interpolates between
the previous and current interval (Cloudflare "counting things" algorithm).

### `Inflight` — concurrent request counting with RAII guards

```rust
use pingora_limits::inflight::Inflight;
let inflight = Inflight::new();
let (guard, count) = inflight.incr(&backend_addr, 1);  // count = current estimate
// hold `guard` in ProxyHttp::CTX → auto-decrement when the request (and CTX) drops
```

This is the building block for least-loaded upstream selection and per-backend concurrency
caps (e.g. cap in-flight requests per inference server; reject or requeue above a threshold).

### `Estimator` — raw CM-sketch

`Estimator::new(hashes, slots)`, `incr(key, n)`, `decr`, `get(key)`, `reset()`.

## pingora-memory-cache & TinyUFO — standalone in-memory caches

- `MemoryCache<K: Hash, T: Clone>::new(size)`: per-entry TTL, backed by TinyUFO.
  `get(&k) -> (Option<T>, CacheStatus)` (`Hit|Miss|Expired|LockHit|Stale`),
  `put(&k, v, Some(ttl))`, `get_stale`, `multi_get`.
- `RTCache` (read-through wrapper): supply a `Lookup` impl; concurrent misses on the same key
  are coalesced (one fetch, others wait) — cache-stampede protection.
- `tinyufo::TinyUfo`: lock-free cache with TinyLFU admission + S3-FIFO eviction; better hit
  ratio than LRU at high concurrency. `TinyUfo::new(total_weight_limit, estimated_count)`,
  `get`, `put(k, v, weight) -> Vec<KV>` (returns evicted), `remove`, `force_put`.

Good for: route tables, auth-token validation results, model→pool lookups, response memoization
outside the full HTTP-cache machinery.

## pingora-cache — the HTTP caching layer (experimental)

Integrates with pingora-proxy via the cache phases on `ProxyHttp` (`request_cache_filter` to
enable per request, `cache_key_callback` — panics by default, must override,
`response_cache_filter` — decides cacheability, `cache_hit_filter`, `cache_vary_filter`).
Key pieces:

- `Storage` trait — bring your own backend (`lookup`, `get_miss_handler`, `purge`,
  `update_meta`, streaming partial writes). The bundled `MemCache` is test-only.
- `eviction` — `EvictionManager` impls: `simple_lru`, sharded `lru` (pingora-lru), TinyUFO.
- `lock` — `CacheKeyLock`: only one request fetches upstream per key (thundering herd).
- `predictor` — remembers uncacheable assets to bypass cache/lock early.
- `CacheKey`, `CacheMeta`, `VarianceBuilder` (Vary), cache_control parsing, purge
  (invalidation vs eviction), stale-while-revalidate/stale-if-error via `should_serve_stale`.
- `pingora-header-serde` — zstd-with-trained-dictionary compression of cached response headers.

Status: marked experimental upstream; no bundled disk storage. For an LLM router you rarely
want HTTP-layer caching of completions — prefer `MemoryCache`/`TinyUfo` for metadata.

## pingora-timeout — cheap timeouts

Drop-in `tokio::time::timeout` replacement: `pingora_timeout::timeout(dur, fut)` /
`fast_sleep`. Lazy timer arming, no global lock, deadlines pooled at 10ms granularity —
~4ns vs ~107ns per create/cancel. Use in hot per-request paths (facade feature `time`).
