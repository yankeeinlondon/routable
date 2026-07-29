# vLLM Router for Model Pooling

Use the `vllm` and `pingora` skills when reviewing or implementing this specification.

## Problem

We have limited GPU resources spread across multiple GPUs. We need to route each request to
a vLLM server serving the requested model and intelligently start or wake that model when
there is demand for it.

In the future, multiple inference runners (vLLM, llama.cpp, etc.) may contend for the same
GPUs, so resource ownership and scheduling must not be coupled to vLLM-specific process
details.

## Tech Stack

We will build a bespoke Rust binary using the Pingora crates. Pin the exact Pingora release
(currently 0.8.1) because it is pre-1.0 and callback signatures may change between minor
releases. The initial runner adapter targets the installed vLLM 0.26.0; upgrades require
rerunning lifecycle, API, and sleep-mode compatibility tests.

## Features

- Start, stop, sleep, and wake any configured vLLM service definition.
- Discover host GPUs through NVML and identify them by stable GPU UUID, not by mutable
  CUDA ordinal alone.
- Route model-specific OpenAI-compatible requests to an appropriate ready replica based on
  the `model` field in the request payload.
- Optimize load across the GPU pool:
    - enforce configurable idle timeouts after which an unused service is slept or stopped;
    - evict only services marked `evictable`, and never sleep or stop a service with an
      active or parked request;
    - coalesce concurrent cold-start requests for the same service into one idempotent
      start/wake operation;
    - cap parked requests globally and per model so cold-start demand cannot exhaust router
      memory or file descriptors;
    - expose an authenticated `/pool` control API to list service definitions and observed
      state, start/wake a service, drain/sleep/stop a service, and reconcile state.
- Proxy the supported vLLM OpenAI-compatible inference surface, including streaming SSE,
  without buffering upstream responses.
- Aggregate service state and health from periodic systemd, vLLM HTTP, and NVML probes:
    - router liveness must be separate from pool readiness;
    - an unhealthy optional model must appear in pool status but must not make the router
      process itself non-live;
    - include probe age so callers can distinguish a current observation from stale state.
- Expose Prometheus metrics for request counts and latency by model, parked-request count and
  wait time, lifecycle transitions, probe failures, active streams, and per-service state.

The router must not expose vLLM development or administrative endpoints such as `/sleep`,
`/wake_up`, or `/is_sleeping` on the public data-plane listener. Completion responses are not
HTTP-cached by the router.

In v1, "load a model" means start or wake a service definition whose base model is fixed.
Hot-swapping a base model inside a running vLLM process is out of scope. Multi-LoRA support,
if added later, must be modeled separately from base-model lifecycle and GPU placement.

## Future Requirement

Though not a v1 concern, the router should eventually provide similar functionality for
`llama.cpp` and possibly other runners. The pool manager therefore owns generic resource and
lifecycle concepts, while a runner adapter owns commands, readiness probes, API capabilities,
and sleep/wake behavior specific to a runner.

> For the short to mid term, all managed runners are on the same host. A later version may
> place runners on multiple hosts; v1 does not need distributed scheduling, consensus, or
> cross-host failure recovery.

**Dynamic service registration (future, decided 2026-07-29):** a later version will
support *adding* a new service definition at runtime (e.g. `POST /pool/v1/services`,
enabling a web UI to make a new model available alongside the existing inventory).
Registration is purely additive — new map entries, a new parking semaphore, a new
placement participant — so it does not reopen the stale-consumer invalidation class
that motivated immutable configuration. The candidate definition must pass the same
full validation as boot-time configuration, merged against the live config, and unit
generation plus `daemon-reload` run before the service becomes schedulable.
**Mutating or removing an existing definition stays restart-to-apply** (validated
restart / `-u` graceful upgrade); in-place mutation is explicitly out of scope at any
version. This dovetails with the multi-host evolution: service definitions become
control-plane-owned data with a single writer, not a file the process re-reads.

**Occupancy-aware placement (future, decided 2026-07-29).** Motivating example: M1
occupies 20 of 32 GB on GPU1 under constant load; M2 needs to span both GPUs and fits
the *aggregate* free VRAM (12 GB on GPU1 + 32 GB on GPU2), but a naive symmetric launch
(50/50) would OOM GPU1. A later version must place such a model gracefully:

- **Budget ledger:** at placement time compute per-GPU free budget = capacity − awake
  reservations − sleep residuals, from the same declaration-based accounting v1 uses.
- **Deploy-time parameterization:** the service definition becomes a template; the
  router computes the per-GPU memory assignment and GPU selection at start time and
  renders them into the generated unit. A computed binding is pinned in the ledger for
  as long as the service is running *or asleep* (a woken vLLM re-grabs its original
  allocation).
- **Runner-declared split capability:** asymmetric fitting is runner-specific — vLLM TP
  is symmetric-only (uniform `gpu_memory_utilization`; verified no per-GPU cap), vLLM
  PP can split unevenly by layers (`VLLM_PP_LAYER_PARTITION`), llama.cpp
  (`--tensor-split`) and accelerate-style servers (`max_memory` per device) support
  arbitrary splits natively. The `Runner` adapter therefore gains a planning hook
  ("given these per-GPU budgets, produce launch parameters or decline"); the pool stays
  generic.
- **Refusal:** a model that fits the inventory but not the *current* free budget gets
  the existing evict-or-503 treatment (`Retry-After`); a model that can never fit the
  inventory is rejected at registration/validation time. No placement may overcommit.
- **Isolation caveat:** memory fit is not performance isolation; consumer Blackwell has
  no MIG. Enabling awake co-residency requires the benchmarked interference profiles
  already demanded by the v1 resolutions, plus a per-service latency-priority signal to
  bias packing decisions.

---

## Decisions (2026-07-28)

1. **Architecture: single binary, split-brain internals ("B with C's seam").** One Pingora
   binary hosts both the data plane (ProxyHttp routing/streaming) and the control plane
   (pool manager). The pool manager is a self-contained module behind a trait
   (`PoolManager`: ensure_ready / drain / stop / sleep / status / subscribe) — the proxy side
   talks to it only through that interface. Lifecycle commands are idempotent. When
   multi-host arrives, the module becomes a per-host agent daemon and the trait
   implementation becomes a client; the data-plane routing contract remains unchanged.
2. **Process supervision is delegated to systemd.** Service definitions render to
   templated/generated units (`vllm@<name>.service`); the router starts/stops via systemd
   D-Bus (`zbus`) and never owns child processes. Avoid shelling out to `systemctl` in the
   steady-state control path. Router restarts/upgrades leave models serving. The pool manager
   must *reconcile* against observed state (health probes, NVML, and unit state), not just
   remember what it did. The configured service definitions are desired state; systemd,
   HTTP probes, and NVML are observed state.
3. **Cold start: park requests.** A request for a not-running model triggers a start (or
   wake) and is held in the async `upstream_peer` phase on a `tokio::sync::watch`/`Notify`
   until the model reports ready, then proxied normally. A configurable parking deadline
   returns an OpenAI-shaped error with HTTP 503 and `Retry-After` on expiry. The start/wake
   operation is single-flight per service. Disconnecting clients release their parked slot,
   but do not cancel a shared start needed by other waiters.
4. **Eviction: idle-timeout AND on-demand, both in v1.** Idle timer per service definition
   shuts down (or sleeps) unused models; when a request needs a GPU and none is free, the
   scheduler evicts the least-recently-used *eligible* model to make room. Eligibility
   requires `evictable=true`, zero active requests, zero parked requests, and no lifecycle
   transition already in progress. `/pool` also allows explicit drain and eviction. If no
   safe placement can be made before the request deadline, fail the request rather than
   overcommitting VRAM.
5. **Sleep mode is a first-class lifecycle state** (see findings below), gated on a local
   validation spike — there are known sm_120/Blackwell sleep bugs. Per-service-definition
   opt-out or failed validation falls back to plain stop/start.
6. **GPU accounting is slot/declaration-based, not measured.** Each service definition
   declares its GPU UUID assignment, parallelism, and VRAM reservation (including
   `--gpu-memory-utilization`, KV-cache sizing, and sleep residuals for any planned
   co-residency). NVML is used for ground truth and crash/drift detection, not as the sole
   placement algorithm. A woken vLLM re-grabs its original allocation, so every allowed
   combination of simultaneously awake residents must fit. Prefer TP=1 for models that fit
   on one RTX 5090; TP across these PCIe-only GPUs is a capacity tool, not a default
   throughput optimization.
7. **Replica selection is health- and load-aware.** Multiple ready services may advertise
   the same served model name. Maintain dynamic backend membership and health in a Pingora
   background service, then select only ready, enabled backends. Prefer least-inflight
   selection, with an optional stable affinity key for workloads that benefit from prefix
   cache locality.
8. **Streaming and retry behavior is explicit.** Pingora streams response chunks by default.
   Use the upstream `read_timeout` as an inter-chunk idle timeout, not as a total generation
   deadline. A connect failure may be retried before request bytes are sent; a model
   inference POST must not be retried after it may have reached an upstream, and no retry is
   possible after response headers or SSE data reach the client.
9. **Control and data planes have separate security policy.** Bind vLLM backends to loopback
   or otherwise firewall them from clients. The `/pool` API is authenticated and authorized,
   is preferably on a separate listener, and records an audit event for every mutating
   operation. Public API authentication is enforced at the router rather than relying on
   per-backend vLLM keys.

## Model lifecycle

```
stopped ──start (systemd)──▶ starting ──ready probe──▶ running
running ──idle t1──▶ asleep-L1 (weights → pinned RAM, ~sub-second wake)
asleep-L1 ──idle t2──▶ asleep-L2 (weights dropped, few-second wake via disk reload)
asleep-* ──request──▶ waking ──▶ running
running ──idle t3 / eviction / explicit──▶ draining ──zero active requests──▶ stopping ──▶ stopped
starting / waking ──failure or deadline──▶ failed ──bounded backoff──▶ stopped
```

Every transition must be serialized per service and carry a generation/operation ID so a
late probe or systemd event cannot complete an obsolete transition. `draining` removes the
service from new-request selection before waiting for its active-request count to reach zero.
The drain has a configurable deadline: on expiry, an automatic eviction is cancelled and an
explicit administrative stop follows its configured force policy. Router startup begins in
`reconciling`, reconstructs state from systemd, readiness/sleep probes, and NVML, and does not
assume that all services are stopped.

A service becomes `running` only after systemd reports it active, its vLLM readiness probe
passes the configured consecutive-success threshold, and a lightweight model check confirms
the expected served model name. Probe failures use consecutive-failure thresholds to prevent
flapping. Repeated start/wake failures enter exponential backoff with jitter; parked requests
receive the same terminal failure instead of each initiating another start.

Sleep-mode facts that shape the design (verified against vLLM 0.26 docs/repo, 2026-07-28):

- **Idle CPU is a non-issue on our version.** The engine busy-wait loops were removed in
  PR #28053 (merged 2026-03, in ≥0.17); `VLLM_SLEEP_WHEN_IDLE` no longer exists. Idle or
  slept servers sit near 0% CPU with the default MP executor. **Do not use the Ray executor**
  (still busy-spins by design).
- Enablement: `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode`; endpoints `POST /sleep?level=N`,
  `POST /wake_up` (L2 wake is a 3-step sequence incl. `reload_weights`), `GET /is_sleeping`.
  These endpoints are unauthenticated — must not be exposed past the router.
- Wake latency: L1 sub-second, L2 a few seconds. Cold start on a 5090 is tens of seconds to
  **minutes** in bad cases (documented ~277 s for a 30B NVFP4 model on sm_120; ~210 s of that
  is uncached warmup). This is the core argument for sleep.
- Costs: L1 pins ~weight-size CPU RAM (page-locked, unswappable — budget against 187 GB);
  ~1–3 GB residual VRAM per slept server (CUDA context/graphs — measure with nvidia-smi,
  torch counters lie); KV/prefix cache is discarded at both levels.
- Behavior: a slept server returns **503** to inference requests (they are not queued) — the
  router must wake-then-forward, which matches the parked-request design. Drain in-flight
  requests before sleeping (sleeping mid-request → CUDA crash). Do one warmup request after
  first start; validate whether each model/quantization also requires warmup after wake.
- **Validation spike required before relying on it:** known open Blackwell issues — TP=2
  init crash with `--enable-sleep-mode` (#21336) and FP8-KV wake crash (#39078). Test on this
  box with our exact vLLM version, model, quantization, TP mode, KV-cache dtype, and sleep
  level. The test must include generation correctness before and after repeated sleep/wake
  cycles, draining an active stream, residual VRAM/host RAM measurement, and failure
  recovery. Keep stop/start as the fallback path.

## Request routing and proxy behavior

- Route only endpoints whose semantics are understood. Model-bearing inference endpoints use
  the parsed `model` value; router-owned endpoints and non-model vLLM endpoints use explicit
  routing rules. Unknown models return an OpenAI-shaped 404 error without starting anything.
- Normalize configured aliases to one canonical served model name. Reject ambiguous aliases
  at configuration load time.
- Enforce a configurable maximum request-body size before buffering. This limit must account
  for multimodal image/audio payloads and must return 413 when exceeded. Reject malformed
  JSON or a missing/non-string `model` with a 400 error.
- Preserve the original request method, path/query, relevant headers, and exact body bytes
  when forwarding. Strip hop-by-hop headers, set the upstream `Host` deliberately, and add
  request/trace IDs. Do not enable response compression for SSE.
- Track the selected backend and an active-request guard in per-request Pingora context.
  Release it on every success, error, timeout, or client disconnect; Pingora's `logging`
  phase is the final accounting and access-log hook. Lifecycle safety requires exact
  per-service counters (for example, atomics plus RAII guards); approximate Count-Min-Sketch
  inflight estimates may inform load balancing but must not decide that a service is safe to
  sleep or stop.
- A service is considered idle only when it has no active streams, no non-streaming requests,
  and no parked waiters. Update its last-used time when a request is admitted, not only when
  a response completes.
- Client cancellation propagates upstream. Once downstream response streaming begins,
  preserve chunked/SSE delivery and do not buffer or transform the full response.

The Pingora body-buffer/replay mechanism is an implementation spike, not an assumed solved
detail. `upstream_peer` runs before the normal streamed request-body forwarding path, so the
router must prove that it can read enough of the downstream body to select a model and replay
the exact body without unbounded buffering. If Pingora cannot do this cleanly at the pinned
version, use a small custom request-ingress service in the same binary or require a trusted
model-routing header; do not silently forward a consumed or truncated body.

## Control and status API

Version the control API from the beginning (for example, `/pool/v1`).

- `GET /pool/v1/services` returns desired configuration plus observed lifecycle, systemd,
  readiness, sleep, GPU, active-request, parked-request, and last-error state.
- `POST /pool/v1/services/{name}/start` and `/wake` are idempotent and return an operation ID.
- `POST /pool/v1/services/{name}/drain`, `/sleep`, and `/stop` accept an explicit deadline and
  force policy. Destructive operations do not use `GET`.
- `GET /pool/v1/operations/{id}` exposes asynchronous operation progress and terminal error.
- `POST /pool/v1/reconcile` requests reconciliation but does not bypass placement or safety
  checks.
- `GET /livez` reports only router-process liveness. `GET /readyz` reports whether the router
  can accept and route requests. Pool degradation details belong in the pool status response
  and metrics rather than making liveness fail.

Mutating calls require authentication and are audit-logged. Lifecycle commands are
idempotent by design — repeating a start/wake/drain/stop converges on the same target state
and returns the already-running operation's ID — so client-supplied idempotency keys are not
required. Error responses use a stable machine-readable code in addition to human-readable
text.

## Service definition and configuration

Each immutable-at-runtime service definition includes at least:

- unique service name, runner type and absolute executable path, canonical served model name,
  optional aliases, model path, tokenizer path/options, quantization, and additional validated
  runner arguments;
- stable GPU UUIDs, TP/PP/DP settings, VRAM reservation per GPU, expected residual sleep VRAM,
  L1 pinned-host-memory reservation, and whether the service is evictable;
- loopback address/port, systemd unit, environment file, readiness path and thresholds,
  startup/wake deadlines, retry/backoff policy, and warmup request;
- sleep support/level policy, idle thresholds, drain deadline, parking deadline and queue
  limit, priority, and whether automatic start is allowed;
- vLLM compatibility settings such as served model name, maximum model length, KV-cache
  dtype, and GPU memory utilization.

Validate the entire configuration before applying it: reject duplicate names/aliases or
ports, unknown GPU UUIDs, impossible TP topology, overlapping exclusive reservations,
unsafe runner arguments, relative model paths, and host-RAM/VRAM overcommit under every
allowed awake/asleep combination. Configuration is immutable for the lifetime of a router
process. To apply changes, validate the replacement configuration with `vllm-router -t`
while the current process remains serving, then use either a supervised router restart or a
`-u` graceful upgrade. If the replacement configuration is invalid, leave the current
process serving its last valid snapshot. Generated systemd inputs must use allowlisted
fields rather than interpolating arbitrary shell fragments.

Ports are stable, declared configuration rather than allocated opportunistically. vLLM binds
only to loopback, and the router is the sole supported client-facing entry point.

## Availability and shutdown

- Run health checks, discovery/reconciliation, and lifecycle scheduling as registered Pingora
  background services; otherwise they will not execute periodically.
- Configure Pingora graceful shutdown/upgrade long enough for expected LLM streams. On router
  shutdown, stop accepting new requests and drain proxy traffic, but do not stop healthy
  systemd-managed vLLM services.
- Persist only configuration and audit/operation history if required. Reconstruct live
  service and allocation state from observations after every router restart.
- Bound all queues and timeouts. Slow model startup, a hung systemd job, stale NVML data, or
  a failed readiness probe must not block unrelated ready models.

## Acceptance criteria

- Concurrent requests for one stopped model cause exactly one start, remain within the
  configured parked-request limit, and all either route after readiness or receive the same
  bounded failure.
- Automatic policy never sleeps, stops, or selects a model as an eviction victim while it has
  an active or parked request. An explicit force-stop may terminate work only after its drain
  deadline and must report and audit the number of affected requests.
- A streaming completion passes through incrementally. A long-running but *active* stream is
  never terminated by the inter-chunk idle timeout, regardless of total duration; a stream
  whose inter-token stall exceeds the idle timeout fails with a clear error.
- Connect failures may fail over safely; a POST is never duplicated after any possibility
  that the upstream received it.
- Router restart and graceful binary upgrade preserve running models, reconstruct correct
  pool state, and do not admit traffic to an unready backend.
- Invalid or oversized model-bearing requests cannot trigger a lifecycle action.
- Public clients cannot reach `/pool` mutations or vLLM development endpoints.
- Sleep support remains disabled for a service until the hardware-specific validation matrix
  passes, and a failed wake falls back according to the configured stop/start policy.
- Metrics and status distinguish stopped, starting, running, draining, asleep, waking,
  stopping, failed/backoff, and stale/unknown observed state.

## v1 resolutions (2026-07-28, finalizing review)

- **`/pool` auth mechanism: static bearer token.** Loaded from a root-owned file referenced
  by config (never inline in config or unit files), required on every `/pool` request, served
  on a dedicated management listener whose bind address is deployment configuration. mTLS is
  deferred to the multi-host version.
- **Scheduler fairness:** cold-start contention for a GPU is resolved FIFO by demand arrival.
  Eviction victims are chosen among eligible services by lowest priority first, then least
  recently used. An administrative start may pass `evict=true` to displace lower-priority
  running services; without it, an unplaceable start fails fast with the blocking reason.
- **Model-routing header: deferred.** v1 routes by parsing the request body (gated by the
  Phase-0 spike below); the fallback chain in "Request routing and proxy behavior" applies if
  the spike fails. A trusted header may be added later as an optimization, with body
  validation remaining authoritative upstream.
- **Audit retention:** audit events are structured log lines on stderr, captured by journald
  under the router's systemd unit; retention follows the host journald policy. Request bodies
  and credentials are never logged.
- **GPU co-residency: not in v1.** At most one *awake* service owns a GPU (a TP service owns
  all its assigned GPUs while awake). Multiple *slept* services may retain residual VRAM on
  the same GPU when the validated residual budget fits. Benchmarked awake co-residency
  profiles are deferred.

## Gating spikes (Phase 0 of the implementation plan)

- Pingora request-body buffer/replay spike against pinned pingora 0.8.1.
- Sleep-mode validation matrix on this host (sm_120), per the lifecycle section.
- systemd control via D-Bus (`zbus`) from inside the unprivileged LXC.
