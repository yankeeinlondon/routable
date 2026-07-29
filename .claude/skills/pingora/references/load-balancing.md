# pingora-load-balancing: pools, health checks, dynamic discovery

Source of truth: `pingora-load-balancing/src/{lib,discovery,health_check,background}.rs` and
`pingora-proxy/examples/{load_balancer,multi_lb,rate_limiter}.rs` in cloudflare/pingora.

## Core types

```rust
pub struct Backend {
    pub addr: SocketAddr,   // pingora_core l4 SocketAddr: Inet or Uds
    pub weight: usize,      // relative weight
    pub ext: Extensions,    // arbitrary metadata — IGNORED for Hash/Eq/Ord
}
Backend::new("10.0.0.1:8000")?                     // weight 1
Backend::new_with_weight("10.0.0.1:8000", 3)?
// Identity = (addr, weight). Changing weight => "new" backend => health state resets.
```

```rust
pub struct LoadBalancer<S: BackendSelection> {
    pub health_check_frequency: Option<Duration>,  // None => checks run once at startup only
    pub update_frequency: Option<Duration>,        // None => discovery runs once at startup only
    pub parallel_health_check: bool,               // default false
}
LoadBalancer::try_from_iter(["10.0.0.1:8000", "10.0.0.2:8000"])?  // static discovery
LoadBalancer::from_backends(backends)                              // custom discovery
lb.select(key: &[u8], max_iterations: usize) -> Option<Backend>
lb.select_with(key, max_iter, |backend, healthy| ...) -> Option<Backend>  // custom accept
lb.set_health_check(hc);          // Box<dyn HealthCheck + Send + Sync>
lb.backends() -> &Backends
lb.update().await                 // run discovery + rebuild selector NOW (push-style)
```

`select()` walks up to `max_iterations` candidates (examples use 256), dedups, returns the
first backend that is `ready()` (healthy && enabled). The `key` matters only for hash-based
selection — RoundRobin/Random ignore it, so `b""` is fine.

## Selection algorithms

| Type alias | Behavior |
|---|---|
| `RoundRobin` | atomic counter, weight-aware (weighted index expansion) |
| `Random` | uniform random over weighted index |
| `FNVHash` | `Weighted<fnv::FnvHasher>` — stable hash on your key |
| `Consistent` | ketama consistent hashing (`pingora-ketama` `Continuum`, nginx-compatible, 160 points/weight); fallback iterations walk the ring |

Consistent-hash on any request attribute: `lb.select(client_ip.as_bytes(), 256)` or a session
ID, tenant, model name, etc. — same key → same backend while the pool is stable (good for
KV-cache / prefix-cache affinity on inference servers).

## Health checks

```rust
#[async_trait]
pub trait HealthCheck {
    async fn check(&self, target: &Backend) -> Result<()>;      // Ok = pass
    async fn health_status_change(&self, target: &Backend, healthy: bool) {}
    fn health_threshold(&self, success: bool) -> usize;         // consecutive flips needed
}
```

**TcpHealthCheck** — connect test. `TcpHealthCheck::new() -> Box<Self>`,
`new_tls(sni) -> Box<Self>`. Fields: `consecutive_success`/`consecutive_failure` (both default
1), `peer_template` (connect timeout default 1s).

**HttpHealthCheck** — real HTTP probe. `HttpHealthCheck::new(host: &str, tls: bool)` (box it
yourself). Defaults: `GET /`, Host = host, 1s connect/read timeouts, only 200 passes,
`reuse_connection: false` (fresh handshake each probe — catches L4 issues; set true for speed).
Customize:

```rust
let mut hc = HttpHealthCheck::new("10.0.0.1", false);
hc.req.set_uri("/health".try_into().unwrap());        // hc.req is a RequestHeader — mutate freely
hc.port_override = Some(9000);                        // probe a different port
hc.validator = Some(Box::new(|resp| { /* accept 200 or 301, inspect body headers, ... */ Ok(()) }));
hc.consecutive_failure = 2;                           // require 2 fails to mark down
lb.set_health_check(Box::new(hc));
lb.health_check_frequency = Some(Duration::from_secs(1));
```

Flip mechanics: each backend starts **healthy**; opposing observations increment a counter;
at `consecutive_{success,failure}` the state flips (and `health_status_change` fires). When a
health check is configured, backends unknown to it fail closed. Manual drain without touching
health: `lb.backends().set_enable(&backend, false)`.

**Health checks only run if the LoadBalancer is registered as a background service** (below).

## Background services — how checks/discovery actually run

```rust
// pingora_core::services::background
pub trait BackgroundService {
    async fn start(&self, shutdown: ShutdownWatch) {}   // ShutdownWatch = watch::Receiver<bool>
}
pub fn background_service<SV>(name: &str, task: SV) -> GenBackgroundService<SV>
// GenBackgroundService::task() -> Arc<SV>   ← THE state-sharing hook
```

`LoadBalancer` implements `BackgroundService`: its loop wakes on the earlier of
`update_frequency` / `health_check_frequency`, runs discovery and/or checks, and exits on
shutdown. The universal wiring pattern:

```rust
let background = background_service("health check", upstreams);
let upstreams: Arc<LoadBalancer<RoundRobin>> = background.task(); // clone into your ProxyHttp
server.add_service(background);                                   // ← without this, nothing runs
```

Custom background services work the same way — implement `BackgroundService`, share state via
`Arc` between `task()` and your proxy struct.

## Dynamic backends: the ServiceDiscovery trait

```rust
#[async_trait]
pub trait ServiceDiscovery {
    /// Returns (all backends, per-backend enablement overrides keyed by backend.hash_key()).
    /// Backends absent from the map are enabled.
    async fn discover(&self) -> Result<(BTreeSet<Backend>, HashMap<u64, bool>)>;
}
```

The built-in `discovery::Static` cannot be mutated from outside (upstream closed that request —
issue #291). For dynamic pools, implement the trait over shared state:

```rust
struct MyDiscovery {
    backends: Arc<RwLock<BTreeSet<Backend>>>,  // mutated by your API/watcher/DNS refresher
}

#[async_trait]
impl ServiceDiscovery for MyDiscovery {
    async fn discover(&self) -> Result<(BTreeSet<Backend>, HashMap<u64, bool>)> {
        Ok((self.backends.read().unwrap().clone(), HashMap::new()))
    }
}

let shared = Arc::new(RwLock::new(initial_backends));
let mut backends = Backends::new(Box::new(MyDiscovery { backends: shared.clone() }));
backends.set_health_check(TcpHealthCheck::new());
let mut lb: LoadBalancer<RoundRobin> = LoadBalancer::from_backends(backends);
lb.update_frequency = Some(Duration::from_secs(5));       // poll discovery every 5s
lb.health_check_frequency = Some(Duration::from_secs(1));
let bg = background_service("dynamic lb", lb);
let lb = bg.task();
server.add_service(bg);
// elsewhere: shared.write().unwrap().insert(Backend::new("10.0.0.9:8000")?);
// picked up at the next update tick — or push immediately with lb.update().await
```

`Backends::update()` semantics: runs `discover()`; if the set changed, existing health states
are preserved by hash_key (new backends start healthy+enabled), the selector is rebuilt, and
the swap is done via ArcSwap (lock-free reads on the hot path). Push-style: skip
`update_frequency` and call `lb.update().await` from your own BackgroundService when a watch
event (k8s, etcd, DNS TTL, config file) fires. Community reference impls exist for DNS
(hickory resolver) and k8s EndpointSlices (kube-rs reflector) — see
https://gist.github.com/Object905/6cafd5e8e56dd60670149296411a407f.

## Multi-pool routing (by path / header / model name)

From `pingora-proxy/examples/multi_lb.rs` — hold several LoadBalancers, pick in `upstream_peer`:

```rust
struct Router {
    cluster_one: Arc<LoadBalancer<RoundRobin>>,
    cluster_two: Arc<LoadBalancer<RoundRobin>>,
}

#[async_trait]
impl ProxyHttp for Router {
    type CTX = ();
    fn new_ctx(&self) {}
    async fn upstream_peer(&self, session: &mut Session, _ctx: &mut ()) -> Result<Box<HttpPeer>> {
        let cluster = if session.req_header().uri.path().starts_with("/one/") {
            &self.cluster_one
        } else {
            &self.cluster_two
        };
        let upstream = cluster.select(b"", 256).unwrap();
        Ok(Box::new(HttpPeer::new(upstream, true, "one.one.one.one".to_string())))
    }
}

fn build_cluster_service<S>(upstreams: &[&str]) -> GenBackgroundService<LoadBalancer<S>>
where S: BackendSelection + 'static, S::Iter: BackendIter {
    let mut cluster = LoadBalancer::try_from_iter(upstreams).unwrap();
    cluster.set_health_check(TcpHealthCheck::new());
    cluster.health_check_frequency = Some(Duration::from_secs(1));
    background_service("cluster health check", cluster)
}
// main(): build each cluster, Router { cluster_one: c1.task(), cluster_two: c2.task() },
// server.add_service for the proxy AND every cluster's background service.
```

Scaling this up (e.g. an LLM router keyed by model): keep an immutable snapshot
of `model -> pool` state and select from it per request. Path/header routing can
still use `ProxyHttp`. OpenAI-style routing on a JSON `model` field cannot
generally choose in `request_body_filter`: `upstream_peer` has already run, and
Pingora's only replay buffer is capped at 64 KiB. For larger bounded bodies,
use a custom `HttpServerApp` that reads the body, resolves the route, and sends
the exact bytes with its own upstream client. A trusted routing header avoids
body buffering when controlled clients can provide one, but the body remains
authoritative unless the trust contract explicitly says otherwise. See
`proxy-lifecycle.md` § "Reading the request body before upstream_peer".

## Dynamic backends are not dynamic services

`ServiceDiscovery` changes backend membership inside one already-constructed
`LoadBalancer`. It does not add a new route key, parking/admission capacity,
lifecycle state, runner adapter, generated unit, or background participant.

For additive application registration, use one serialized prepare/commit
transaction:

1. Snapshot the latest configuration, merge the candidate, and run the complete
   validator.
2. Prepare every fallible adapter, route entry, generated file, and state object.
3. Install external inputs and complete required reloads before routing can see
   the candidate.
4. Commit runner/pool/state entries without holding a registry lock across
   `.await`.
5. Publish the immutable routing/config snapshot last; old request snapshots
   remain valid until their requests finish.

Preserve existing `Arc<Semaphore>` or guard-owning objects when adding entries;
replacing one strands permits held by in-flight requests. Recompute or constrain
boot-derived aggregates such as background cadence and operation timeouts when
new entries can request stricter values.

If routing resolves aliases but forwards the original body unchanged, publish a
new alias only when every selectable backend already advertises it. Otherwise
canonicalize the body deliberately or reject alias expansion. Also validate
endpoint capabilities before any cold-start/lifecycle action: a global path
allowlist must not start a backend that cannot serve that path.

### Selection-key tips for inference routing

- Round-robin / Random: throughput spreading, no affinity.
- `Consistent` with the model name or session/user ID as key: keeps a given key pinned to a
  backend (prefix-cache/KV-cache hits) while ring rebalancing on membership change stays minimal.
- `select_with(key, 256, |b, healthy| healthy && custom_pred(b))`: layer custom logic
  (e.g. queue depth read from shared state / `ext` metadata) on top of health.
- Live load signals: pair with `pingora_limits::Inflight` guards held in CTX to track
  per-backend in-flight counts and pick the least-loaded backend in `upstream_peer`.
