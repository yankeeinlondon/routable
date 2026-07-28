# Implementation Plan — vLLM Pooling Router

Builds the spec in `spec.md` (finalized 2026-07-28). Ordering principle: **retire the
riskiest assumptions first**, so every later phase builds on proven mechanics rather than
hope. Pins: `pingora = "=0.8.1"`, vLLM 0.26.0, Rust ≥ 1.85, target host CT 107.

Phases are sequential; each has explicit exit criteria. A phase is done when its criteria
pass on this host, not when its code compiles.

## Phase 0 — De-risking spikes (gates everything)

Three throwaway-quality spikes, each answering one question the design depends on. Findings
get written down (feature dir for project facts, the `vllm`/`pingora` skills for durable
host/framework facts, `docs/{model-id}.md` for any model brought up per our convention).

**S1 — Pingora body read/replay.** Minimal `ProxyHttp` against a running vLLM: read the
downstream JSON body before `upstream_peer` (bounded), parse `model`, forward with exact body
bytes, confirm streaming SSE passes through chunk-by-chunk. Also verify parking mechanics: an
`upstream_peer` that awaits a `tokio::sync::watch` for 60+ s, then proxies normally, with
client disconnect during parking handled cleanly.
*Exit:* body-bearing POST routed by `model` with byte-identical upstream body; SSE stream
observed incremental at the client; parked request completes after delayed readiness.
*If it fails:* per spec, fall back to a custom ingress `HttpServerApp` in the same binary —
decide before Phase 4 is designed, which is why this runs first.

**S2 — Sleep-mode validation matrix (sm_120).** For each cell of {TP=1, TP=2} × {FP8, NVFP4}
× {sleep L1, L2}: enable sleep mode, run warmup, fixed-prompt generation before/after 5+
sleep/wake cycles (output sanity), measure wake latency, residual VRAM (nvidia-smi, not torch)
and pinned host RAM, verify 503-while-asleep, verify drain-then-sleep vs sleep-mid-request,
and L2's 3-step wake sequence. Known risk cells: TP=2 init crash (#21336), FP8-KV wake crash
(#39078).
*Exit:* a per-configuration pass/fail table; sleep enabled in service definitions only for
passing cells; results appended to the vllm skill's Blackwell reference and the spec's gate
satisfied (or sleep deferred and stop/start declared the v1 fast path — the plan does not
change shape either way).

**S3 — systemd control from the unprivileged LXC via `zbus`.** Generate a `vllm@.service`
template + drop-in, then from Rust: StartUnit, StopUnit, subscribe to unit state changes,
read active state after a `kill -9` of the vLLM process.
*Exit:* start/stop/observe round-trip works over D-Bus as root in CT 107; crash produces an
observable state change. *Fallback:* `systemctl` subprocess for commands (spec allows it
outside steady-state) with D-Bus reserved for subscriptions.

## Phase 1 — Skeleton, config, CI

- Single crate, modules laid out on the spec's seam (see module map below).
- Dependencies: `pingora =0.8.1` (features `openssl`, `proxy`, `lb`), `tokio`, `async-trait`,
  `serde`/`serde_yaml`, `zbus`, `nvml-wrapper`, `prometheus`, `thiserror`, `tracing` +
  `tracing-subscriber` (JSON to stderr → journald).
- Config: two files — pingora server conf (its native YAML) and router config (services,
  GPUs, policies). Full validation per spec §Service definition: duplicate names/ports,
  unknown GPU UUIDs, TP topology, RAM/VRAM budgets under every awake/asleep combination,
  allowlisted unit-file fields. Table-driven tests for every rejection rule.
- `cargo fmt` + `clippy -D warnings` + `cargo test` wired into a pre-commit or CI script.

*Exit:* binary boots under pingora with data-plane, management, and metrics listeners bound;
`/livez` answers; invalid configs are rejected with actionable errors; `-t` config-check mode
works.

## Phase 2 — Observation layer (read-only truth)

- `gpu`: NVML inventory keyed by GPU UUID; per-GPU used-memory and process list for drift
  detection.
- `sysd`: zbus client (per S3) — unit status queries, state-change subscription.
- `probe`: HTTP readiness (`/health`, then `/v1/models` served-name check), `/is_sleeping`,
  with consecutive-success/failure thresholds and probe-age tracking.
- `state`: per-service observed-state store with generation/operation IDs; startup
  `reconciling` pass that reconstructs state purely from systemd + probes + NVML.
- One Pingora background service driving all periodic observation on bounded intervals.

*Exit:* with services started/stopped/killed manually behind the router's back, `GET
/pool/v1/services` (read-only skeleton) converges to correct state within one probe cycle;
no observation path can block another service's probes (verified with an injected hang).

## Phase 3 — Lifecycle engine (the PoolManager seam)

- `PoolManager` trait exactly as spec'd (`ensure_ready` / `drain` / `sleep` / `stop` /
  `status` / `subscribe`); the only interface the data plane and API layer may use.
- Transition executor: serialized per service, single-flight `ensure_ready`, operation IDs,
  exponential backoff with jitter, deadlines on every transition, `failed` terminal fanout to
  waiters.
- Sleep/wake sequences incl. L2's 3-step wake and post-start/wake warmup request (per S2
  findings).
- Scheduler: slot-based placement (one awake owner per GPU), FIFO cold-start contention,
  eviction eligibility (evictable + zero active + zero parked + no transition) ordered
  priority-then-LRU, `evict=true` administrative override.
- Idle reaper: t1 → sleep-L1, t2 → L2, t3 → stop, driven off exact activity counters.

*Exit:* integration tests against a **mock runner** (small HTTP binary imitating vLLM:
readiness, SSE, 503-when-asleep, sleep/wake endpoints, crash-on-command) prove: single-flight
under 50 concurrent `ensure_ready`; drain waits and deadlines; no eviction of active/parked
services under randomized load; reconciliation after simulated router restart. Smoke test of
the same flows against one real vLLM service.

## Phase 4 — Data plane

- `ProxyHttp` impl per spec §Request routing: bounded body read + `model` parse (S1
  mechanics), alias resolution, 400/404/413 OpenAI-shaped errors, request/trace IDs,
  hop-by-hop stripping, deliberate `Host`.
- Parking: per-service `tokio::sync::watch` of lifecycle state; global + per-model caps;
  parking deadline → 503 + `Retry-After`; disconnect releases the slot without cancelling the
  shared start.
- Selection: ready/enabled replicas for the served model, least-inflight by exact per-service
  atomic counters, optional stable affinity key; RAII active-request guard in CTX released in
  `logging` on every outcome.
- Peer options: `read_timeout` as inter-token idle timeout, connect timeouts, h1 keepalive to
  loopback backends; retry policy = connect-failure only (`fail_to_connect`), never after
  bytes may have reached an upstream.
- Cancellation propagation; no buffering/transform once streaming begins; compression off.

*Exit:* acceptance-style tests: concurrent stampede on a stopped model → one start, all park
or bounded-fail; mid-stream inter-token stall kills the stream while a 10-minute active
stream survives; upstream connect-refused fails over; kill-during-stream surfaces a clean
error and releases counters (verified via metrics).

## Phase 5 — Control API, health, metrics

- `/pool/v1` as a dedicated Pingora `HttpServerApp` on the management listener: services
  list, start/wake/drain/sleep/stop with deadlines + force policy, operations store
  (bounded in-memory ring), reconcile trigger. Bearer-token auth from root-owned token file;
  audit line per mutation.
- `/livez` (process), `/readyz` (can accept + route), pool degradation only in status/metrics.
- Prometheus service exposing the spec's metric set (requests/latency by model, parked count
  + wait, transitions, probe failures, active streams, per-service state gauge).

*Exit:* every endpoint exercised by tests incl. auth-failure paths; destructive ops refused
on GET; metrics observed changing under a scripted load + lifecycle scenario.

## Phase 6 — Hardening, acceptance, deployment

- Acceptance matrix: one test (or scripted drill) per spec §Acceptance criteria bullet,
  results recorded in the feature dir.
- Chaos drills on-box: `kill -9` vLLM mid-stream, hung start (SIGSTOP), NVML lies (stale
  data), router restart mid-traffic, pingora graceful upgrade (`-u` + SIGQUIT) under an
  active SSE stream, journald-visible audit trail.
- Deployment: router's own systemd unit (+ socket ordering, `daemon_wait_for_ready`),
  install/runbook doc, example service definitions for the real models on this box
  (with `docs/{model-id}.md` per convention).

*Exit:* all acceptance criteria pass on CT 107; upgrade drill preserves running models and
in-flight streams per spec.

## Module map (single crate; the seam is a compile-time boundary)

```
src/
  main.rs          server assembly: pingora Server, listeners, background services
  config/          schema, validation, atomic reload
  proxy/           data plane: ProxyHttp impl, parking, selection, error mapping
  api/             /pool/v1 + livez/readyz HttpServerApp, auth, audit
  metrics/         prometheus registry + service
  pool/            PoolManager trait + v1 impl: transitions, scheduler, reaper
  runner/          Runner trait + vllm adapter (probes, sleep/wake, unit template)
  sysd/            zbus systemd client
  gpu/             NVML inventory + drift detection
  probe/           HTTP probe engine, thresholds, probe age
  state/           observed-state store, generations, watch channels
```

`proxy/`, `api/`, `metrics/` may depend on `pool/` **only through the `PoolManager` trait**
(enforced by a re-exported facade); `pool/` owns `runner/`, `sysd/`, `gpu/`, `probe/`,
`state/`. The multi-host cut line is the trait.

## Testing strategy

- **Unit:** state-machine transition table (every legal/illegal edge), scheduler decisions,
  config validation — all table-driven.
- **Integration (fast, no GPU):** mock-runner binary + real router process; covers lifecycle,
  parking, API, metrics. Runs anywhere.
- **On-box (real vLLM):** `#[ignore]`-tagged suite run manually on CT 107; covers S2's
  passing sleep cells, real streaming, chaos drills.
- **Load:** `oha`/`k6` scripts for stampede, sustained SSE fan-out, parked-cap enforcement.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sleep mode broken on sm_120 (TP=2 #21336, FP8-KV #39078) | S2 gate; per-service sleep opt-out; stop/start path is fully functional without sleep |
| Pingora can't read+replay body cleanly at 0.8.1 | S1 runs first; spec'd fallback (custom ingress app) decided before Phase 4 design |
| Pre-1.0 pingora API drift | exact `=0.8.1` pin; upgrade only deliberately, re-running Phase 4/6 suites |
| zbus/D-Bus quirks in unprivileged LXC | S3 gate; `systemctl` subprocess fallback for commands |
| Pinned-RAM exhaustion with several L1 sleepers | config validator enforces host-RAM budget across every allowed combination; L2 tier |
| CUDA zombie VRAM after crash | NVML drift detection + reconcile marks GPU unsafe until clean; unit `Restart=` policy |
| Parked-request pileup during long cold start | global/per-model caps + deadlines (spec'd), verified by stampede load test |

## Sequencing and effort (rough)

Phase 0 is ~2–3 focused sessions (S2 dominates — it's a test matrix on real hardware).
Phases 1–2 are mechanical (~2 sessions). Phase 3 is the largest correctness surface
(~3–4 sessions incl. mock-runner). Phase 4 ~2–3 sessions, Phase 5 ~1–2, Phase 6 ~2.
Phases 1+2 can start while S2 cells are still being ground through, since only Phase 3's
sleep sequences and service-definition defaults consume S2's output.
