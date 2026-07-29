# Acceptance status

Automated verification recorded 2026-07-29 (re-measured after the review-4 fix
batch; the previous 2026-07-28 record predates it):

- `cargo fmt --all -- --check`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --all-targets --all-features`: 168 pass, 3 ignored, 0 fail
- two-file `-t` configuration check: pass, against the live
  `/etc/vllm-router/router.yaml` as well as the in-repo example

The three ignored tests each require host resources a CI-style run does not
have: a live system bus and systemd, and the generated units on `/run`.

The fast suite covers strict startup configuration, model aliases and invalid
request rejection, generation-safe observed state, exact active/parked guards,
50-way single-flight start, shared failure fanout, FIFO contention, safe
priority/LRU eviction, drain deadlines, idle sleep tiers, least-inflight
admission, probe threshold/age behavior, vLLM readiness/model verification,
bounded HTTP calls, OpenAI warmup, and the vLLM 0.26 three-step L2 wake.

The review-4 batch added coverage for the one-awake-owner invariant under
concurrent reconciliation, force-stop blast-radius capture and its drained
latch, fail-closed admission against stale/asleep/incomplete facts, the
sleep-fact provenance that keeps an unproven `/is_sleeping` from aging as
fresh, joined-flight audit records, the deployment shutdown-signal contract,
and the probe-freshness configuration floor.

The custom Pingora ingress additionally enforces bounded body reads before
lifecycle admission, a public endpoint allowlist, hop-by-hop stripping,
loopback Host rewriting, request/trace IDs, global/per-model parking caps,
disconnect-aware parking, connect-only failover, inter-chunk response timeout,
and incremental upstream-to-downstream body writes.

## On-box Tier 4/5 run — 2026-07-29

Run on CT 107 with two RTX 5090s, vLLM 0.26.0, Pingora 0.8.1, and the production
systemd/router topology. Tier 4 is **failed** overall:

- **Pass:** negative admission (400 malformed JSON, 413 at 33,554,433 bytes, and
  404 for data-plane `/sleep`) caused no lifecycle transition.
- **Pass with audit gap:** a clean 50-way stopped-model stampede returned 50
  bounded 503s and created exactly one systemd invocation with zero restarts.
  An 80-way run enforced the 64-request parked cap exactly. No
  `joined_existing` audit records were emitted, so the audit half of the
  criterion remains failed.
- **Pass:** disconnecting a parked client released its slot immediately while
  the detached start continued.
- **Partial:** a healthy SSE stream survived beyond the configured 5 s
  inter-chunk timeout and counters returned to zero after cancellation. A
  7-second SIGSTOP stall terminated the client and released the counter, but
  the router logged only `error decoding response body`, not a clear timeout.
- **Pass:** killing the vLLM main PID broke the active stream cleanly; systemd
  restarted the unit and the router did not admit work until readiness.
- **Pass:** restarting the router broke the active stream as expected without
  changing either model unit's PID/invocation, then reconciled the ready model.
- **Fail:** Pingora listener-FD handoff worked, but the old process terminated
  an active SSE stream when the configured 10-second graceful-shutdown period
  expired.
- **Fail:** idle same-GPU eviction worked, but requesting `gemma31b` while
  `qwen35b` had eight active requests launched both units on GPU 1. This
  violates both busy-victim protection and the one-awake-owner invariant.
- **Fail:** force-stop was issued with eight active requests and the operation
  succeeded, but its audit record reported `affected_requests: 0`; the unit
  then restarted, so the requested stop was not durable.

Two cross-cutting observations were repeatable: a backend can become healthy
while the router remains in stale `starting` state until manual reconcile, and
the graceful-upgrade retry window is shorter than a leisurely operator
sequence unless SIGQUIT follows socket creation immediately.

Tier 5 is also **failed** overall; no tested service is eligible for
`hardware_validated: true`:

| TP | quantization | L1 (5 cycles) | L2 (5 cycles) | promotion | gating result |
|---:|---|---|---|---|---|
| 1 | NVFP4 | stable output; sleep ~1.02 s, wake ~3.03 s; 1,414 MiB residual | stable output; sleep ~1.02 s, wake ~5.05 s; 1,414 MiB residual | pass | fail: ~38.8–39.0 GB pinned host RAM exceeded the 30 GB declaration; asleep hang characterized |
| 2 | NVFP4 | not reached | not reached | not reached | fail at initialization: `last dim has to be multiple of 16, but got 2152` |
| 1 | FP8 | stable output; sleep/wake ~0.27 s; 944 MiB residual | stable output; sleep ~0.27 s, wake ~3.10 s; 944 MiB residual | pass | fail: ~24.53 GB pinned host RAM exceeded the 20 GB declaration; asleep hang characterized |
| 2 | FP8 | stable output; sleep 0.27–1.04 s, wake 2.32–2.33 s; 1,098 MiB/GPU residual | stable output; sleep ~0.27 s, wake 2.84–3.10 s; 1,098 MiB/GPU residual | pass | fail: ~20.22 GB pinned host RAM exceeded the 20 GB declaration; asleep hang characterized |

All initialized cells used the required L2 three-step sequence
(`wake_up?tags=weights`, `collective_rpc`, `wake_up?tags=kv_cache`). L1→L2
promotion first woke the L1 sleeper and then slept at L2. A 7,000-token active
request caused sleep to fail safely after the 10-second drain deadline on both
FP8 topologies. Direct inference against every sleeping backend was accepted
but timed out after 2 seconds with no bytes. This measured vLLM 0.26 behavior
means router safety cannot depend on a backend 503: admission must fail closed
on fresh sleep/readiness facts, wake and reprobe before forwarding, and bound a
post-admission stall with the inter-chunk idle timeout.

Per-model measurements are in `docs/qwen3.6-27b-nvfp4.md` and
`docs/meta-llama-3.1-8b-instruct-fp8.md`. Stop/start therefore remains the
production fallback. The failed Qwen NVFP4 TP=2 cell must also be recorded in
its service compatibility declaration; configuration validation then rejects
that topology instead of allowing another doomed cold start. Conservative L1
floors are 25,000,000,000 bytes for the measured FP8 topologies (1.25× the
provisional declaration) and 40,000,000,000 bytes for NVFP4 TP=1 (1.33×);
smaller future L1 declarations are rejected.
