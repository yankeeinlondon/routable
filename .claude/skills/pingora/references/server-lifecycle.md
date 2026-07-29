# pingora-core: Server, Services, config, and zero-downtime operation

Source of truth: `pingora_core::server` docs and `docs/user_guide/{conf,start_stop,graceful,
daemon,systemd,panic,prom}.md` in cloudflare/pingora.

## Assembly pattern

```rust
let opt = Opt::parse_args();                          // CLI: -c/--conf, -d/--daemon, -t/--test, -u/--upgrade
let mut server = Server::new(Some(opt)).unwrap();     // or Server::new(None) to skip CLI parsing
server.bootstrap();                                   // loads conf; with -t validates and exits

let mut proxy = pingora_proxy::http_proxy_service(&server.configuration, MyProxy { .. });
proxy.add_tcp("0.0.0.0:6188");                        // plain listener; UDS also supported

let mut tls = pingora_core::listeners::tls::TlsSettings::intermediate(&cert_path, &key_path).unwrap();
tls.enable_h2();
proxy.add_tls_with_settings("0.0.0.0:6189", None, tls);

server.add_service(proxy);
server.add_service(background);                       // health checks, discovery, custom loops
server.run_forever();                                 // spawns runtimes, daemonizes if configured, blocks
```

A **Service** = an application (ProxyHttp wrapper, or a raw `ServerApp`/`HttpServerApp` impl
from `pingora_core::apps` for non-proxy TCP/HTTP services) + its listeners + its thread count.
One Server hosts many Services, each with independent listeners and runtime options.

## Runtimes and threads

- Each service gets `threads` workers (conf `threads`, default = 1; set to cores for busy proxies).
- `work_stealing: true` (default) = stock tokio multithread runtime. `false` = pingora-runtime
  "NoSteal" mode: a pool of single-threaded tokio runtimes (lower scheduler overhead, per-core
  model like nginx).
- Optional dedicated offload pools: `upstream_connect_offload_threadpools`,
  `downstream_tls_offload_threadpools` (+ per-pool thread counts).

## Configuration (YAML, via `-c conf.yaml`)

```yaml
---
version: 1
threads: 8
work_stealing: true
error_log: /var/log/proxy/error.log      # stderr if unset
pid_file: /run/proxy.pid
upgrade_sock: /run/proxy_upgrade.sock    # must match between old and new binary
user: nobody                             # drop privileges after daemonize
group: webusers
daemon: false
ca_file: /etc/ssl/certs/ca-certificates.crt
upstream_keepalive_pool_size: 128
```

Unknown keys are silently ignored — you can piggyback your own app config in the same file
(parse it separately) or keep it apart. Other `ServerConf` fields:
`client_bind_to_ipv4`/`client_bind_to_ipv6`, `working_directory`,
`daemon_wait_for_ready` / `daemon_ready_timeout_seconds` (default 600, parent waits for
readiness), `grace_period_seconds`, `graceful_shutdown_timeout_seconds`,
`s2n_config_cache_size`.

`Opt` fields (CLI): `conf` (-c), `daemon` (-d), `test` (-t validate-and-exit), `upgrade` (-u).

## Signals and shutdown

| Signal | Effect |
|---|---|
| SIGINT | fast shutdown |
| SIGTERM | graceful shutdown: stop accepting, drain within grace period |
| SIGQUIT | graceful **upgrade** handoff (see below) |

## Zero-downtime upgrade (Linux only)

Mechanism is **listening-fd passing over a unix socket** (`upgrade_sock`), not SO_REUSEPORT:

1. Start the new binary with `-u` (and the same `upgrade_sock` path). It waits for fds.
2. Send SIGQUIT to the old process. It ships all listening sockets to the new process
   (SCM_RIGHTS), which starts accepting immediately.
3. Old process stops accepting, drains in-flight requests within the grace period, exits.

Guarantees: no connection refused during handoff; requests completing within the grace period
are never terminated. Works cleanly under systemd with `ExecReload` running the new binary
with `-u -d` then `kill -QUIT $MAINPID` style orchestration (see docs/user_guide/systemd.md).

## Daemonization

`-d` / `daemon: true`: forks to background inside `run_forever()`, writes `pid_file`, drops to
`user`/`group`, chdirs to `working_directory`. Panics in request handlers do NOT kill the
server (tokio task isolation); set `server.sentry = Some(sentry::ClientOptions{...})` (feature
`sentry`) to capture them.

## Observability

- Prometheus sidecar service (gateway.rs example):
  ```rust
  // published crates: pingora_core::services::listening::Service::prometheus_http_service()
  // repo main: pingora_prometheus::prometheus_http_service()
  let mut prom = Service::prometheus_http_service();
  prom.add_tcp("127.0.0.1:6192");
  server.add_service(prom);
  ```
  Any `prometheus::register_int_counter!`-style metric in your ProxyHttp impl is exposed there.
- Access/metrics logging belongs in `ProxyHttp::logging()` (always runs, has the final
  status via `session.response_written()` and the `Option<&Error>`).
- Error log path: conf `error_log`; `RUST_LOG=INFO` controls `env_logger` verbosity in examples.

## Runtime rules learned in production (verified on 0.8.1, 2026-07-29)

- **`run_forever()` calls `block_on` internally** (`server/mod.rs:756`): calling it
  from inside any tokio runtime — most commonly `#[tokio::main] async fn main` —
  panics deterministically with "Cannot start a runtime from within a runtime".
  Nothing catches this before the first real launch: `-t` exits earlier and unit
  tests don't serve. Pattern: `fn main()` synchronous; if construction needs async
  (e.g. a zbus connection whose I/O tasks live on the runtime that created them),
  build a dedicated `tokio::runtime::Runtime`, `block_on` the construction, and keep
  that runtime alive for the whole process (it must never be dropped — its tasks die
  with it; `run_forever` never returning makes an owning local sufficient). Awaiting
  those objects later from pingora's worker runtimes is fine.
- **Graceful shutdown waits for every service runtime to fully quiesce.** Any
  detached `tokio::spawn` on a service's runtime (including a background service's)
  that outlives the shutdown signal keeps that runtime alive — SIGTERM then hangs at
  "waiting for service runtime … to exit" until the graceful-shutdown timeout (which
  you may have set to many minutes). Spawn long-lived detached work on your own
  process-lifetime runtime instead, or track and abort it on `ShutdownWatch`.
  Regression-test with a spawned binary: SIGTERM must exit within the grace period.
- **Busy listeners retry, not fail**: a port already in use logs
  `"<addr> is in use, will try again"` every second indefinitely. If "another
  instance is already running" should be an error, detect it before `run_forever`.
- **Test mode exits early**: `bootstrap()` terminates the process under `-t`, so any
  validation you want `-t` to cover must run before `bootstrap()`; everything after
  it is unreachable in test mode (also useful: expensive construction after
  `bootstrap()` is automatically skipped during config checks).
