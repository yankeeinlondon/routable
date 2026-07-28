# vLLM Router for Model Pooling

use the 'vllm' skill


## Problem

We have limited GPU resources (which spread multiple GPUs) and want to be able to route a request for a particular model to 
the right vllm server which is serving it but also to intelligently start a model when there is demand for it. 

In the future we will have multiple LLM runners (vllm, llama.cpp, etc.) that we may want to utilize the GPU's too so 
need a way to address this contention too.

## Tech Stack

We will build a bespoke binary with Rust and leveraging the Pingora crates.

## Features

- ability to start and stop any vllm service definition
- able to identify the GPU availability of the host
- able to route requests to the appropriate server based on the `model` in the request payload
- ability to optimize load across the pool of GPU's:
    - allows for timeouts where a server will be shutdown automatically
    - exposes a `/pool` API endpoint for managing nodes in the resource pool
        - start vllm service
        - stop vllm service
        - load new model(s) on service
        - etc.
- expose the vLLM API surface so that it can be proxied back to the appropriate server
- provide aggregated data on endpoints like `/health` so the caller gets a full picture of the health across
  the
    - in this particular endpoint we should probably have some sort of polling so the router always
      knows exactly what servers are in what states

## Future Requirement

Though not a concern for now, we will in the future want to have a router be able to provide similar functionality
for `llama.cpp` (and possibly other runners too). We don't need to design for this per se but we should consider it
enough that we don't make our life more difficult than it needs to be later on when we do implement this.

> Note: for the short to mid term we can assume that all the runners we're managing are hosted on the same server
> but eventually we'll also want to branch out to allow proxying to multiple servers

---

## Decisions (2026-07-28)

1. **Architecture: single binary, split-brain internals ("B with C's seam").** One Pingora
   binary hosts both the data plane (ProxyHttp routing/streaming) and the control plane
   (pool manager). The pool manager is a self-contained module behind a trait
   (`PoolManager`: start / stop / sleep / wake / status / subscribe) — the proxy side talks
   to it only through that interface. When multi-host arrives, the module becomes a per-host
   agent daemon and the trait impl becomes an HTTP client; nothing on the proxy side changes.
2. **Process supervision is delegated to systemd.** Service definitions render to
   templated/generated units (`vllm@<name>.service`); the router starts/stops via D-Bus
   (`zbus`) or `systemctl` and never owns child processes. Router restarts/upgrades leave
   models serving. The pool manager must *reconcile* against observed state (health probes,
   NVML, unit state), not just remember what it did.
3. **Cold start: park requests.** A request for a not-running model triggers a start (or
   wake) and is held in the async `upstream_peer` phase on a `tokio::sync::watch`/`Notify`
   until the model reports ready, then proxied normally. A configurable parking deadline
   returns 503 + `Retry-After` on expiry.
4. **Eviction: idle-timeout AND on-demand, both in v1.** Idle timer per service definition
   shuts down (or sleeps) unused models; when a request needs a GPU and none is free, the
   scheduler evicts the least-recently-used evictable model to make room. `/pool` also allows
   explicit stop/evict.
5. **Sleep mode is a first-class lifecycle state** (see findings below), gated on a local
   validation spike — there are known sm_120/Blackwell sleep bugs. Per-service-definition
   opt-out falls back to plain stop/start.
6. **GPU accounting is slot/declaration-based, not measured.** Each service definition
   declares its GPU assignment, TP degree, and VRAM requirement (incl.
   `--gpu-memory-utilization` sizing for any planned co-residency). NVML is used for ground
   truth / crash detection, not placement math. Note: a woken vLLM re-grabs its original
   allocation, so co-resident servers must be sized so all *awake* residents fit.

## Model lifecycle

```
stopped ──start (systemd)──▶ starting ──ready probe──▶ running
running ──idle t1──▶ asleep-L1 (weights → pinned RAM, ~sub-second wake)
asleep-L1 ──idle t2──▶ asleep-L2 (weights dropped, few-second wake via disk reload)
asleep-* ──request──▶ waking ──▶ running
any ──idle t3 / eviction / explicit──▶ draining ──▶ stopped
```

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
  first start or every wake pays compile cost.
- **Validation spike required before relying on it:** known open Blackwell issues — TP=2
  init crash with `--enable-sleep-mode` (#21336) and FP8-KV wake crash (#39078). Test on this
  box with our exact quant formats; keep stop/start as the fallback path.

## Open items

- `/pool` auth scheme (endpoint can kill workloads).
- Port allocation scheme for spawned servers.
- Service definition format (name, model, runner type, quant, TP, GPU assignment, port,
  VRAM requirement, sleep opt-in, idle timeouts, parking deadline).
- Body-peek routing: reading the JSON `model` field before `upstream_peer` requires
  buffering the request body for replay — fine at LLM request sizes, but confirm the
  Pingora mechanics during implementation.

