# Acceptance status

Automated verification recorded 2026-07-29 (re-measured after the review-4 fix
batch; the previous 2026-07-28 record predates it):

- `cargo fmt --all -- --check`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --all-targets --all-features`: 168 pass, 3 ignored, 0 fail
- two-file `-t` configuration check: pass against a host-valid configuration

`-t` became a host check in review 5 (F23), so what it accepts changed. It now
resolves every inventory UUID through live NVML, renders the generated units, and
opens both bearer-token files. `src/config/example.yaml` therefore **fails** `-t`
by design — its placeholder `GPU-00000000-...` exists on no host — and a
deployment config must be migrated for the new required inference token before it
passes. Both are intended: the previous record's claim that the shipped example
passed `-t` was only true because the check could not see the host.

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

## On-box Tier 4 re-run — 2026-07-29 (post-fix, commit 00ad691)

Re-run on CT 107 against the deployed fix batch, same topology as the failed run
below. **Tier 4 passes.** Every drill that previously failed now passes on the
hardware, not merely in tests:

| Drill | Result | Evidence |
|---|---|---|
| Negative admission | pass | 400 / 413 at 33,554,433 B / 404 data-plane `/sleep` / OpenAI-shaped 404 / 401 unauthenticated; zero transition series |
| Stampede + audit (**F17**) | pass | 50-way cold stampede → 49 `joined_existing` records on one `operation_id`, exactly 1 systemd start, `NRestarts=0`; 50 bounded 503s at the 20 s parking deadline |
| Stall classification (**F17**) | pass | SIGSTOP mid-stream → `upstream response idle timeout after 120s while reading the response body`, fired at exactly 120 s; counters released to zero |
| Busy-owner protection (**F11**) | pass | 200 s sustained load: owner `active` across all 197 samples, contender unit never activated, 8×503, **118/118** owner streams completed; both services `evictable: true`, priority 0 |
| Deferred eviction | pass | a *draining* owner is evicted only after its work completes — 8/8 streams finished, stop followed in the same second; one awake owner throughout |
| Force-stop count (**F12**) | pass | `affected_requests: 8` against 8 in-flight (previously `0`); streams cut at `finish_reason: "abort"` with clean SSE termination |
| Force-stop durability (**F12**) | pass | latched `drained`, unit stayed `inactive` across 10 retry requests |
| Restart boundary | pass | after router restart: `drained` → `stopped`, demand restarts it — matches the documented scope limit |
| Crash recovery | pass | `kill -9` mid-stream → curl exit 18, `NRestarts=1`, observed `stale`, admission failed closed during reload |
| Disconnect-while-parked | pass | parked 1 → 0 immediately; detached start continued |
| Router restart under traffic | pass | **15 ms** with an active stream; model unit identical `InvocationID` and `MainPID` |
| Graceful upgrade (**F13**) | pass | SSE stream survived the handoff — `curl exit=0`, 1,458,323 B, `[DONE]`; post-handoff request 200; old process lingered in its grace window |

Measured incidentally: `qwen35b` cold start 2 m 16 s against a `typically ~180 s`
estimate; `qwen27b` ready ~2.5 min *after* systemd reported the unit active,
which is why `systemctl is-active` is not a readiness signal.

### Residual observations (none are regressions from this batch)

1. **The crash path still logs `upstream HTTP error: error decoding response
   body`.** F17's classifier special-cases `is_timeout()` only, so a mid-stream
   connection reset keeps reqwest's generic wrapper — the same unhelpful text
   F17 was filed about, on a different path.
2. **A detached placement demand outlives its client.** A request that 503s at
   its 20 s parking deadline leaves an eviction claim queued against
   `startup_ms` (900 s). Observed stopping a healthy model minutes later with no
   client waiting, and two competing demands on one GPU ping-pong, each
   displacing the other. Consider bounding the detached demand once no client
   remains.
3. **A process inside the grace window cannot be interrupted.** Pingora's
   graceful path is a plain `thread::sleep`; SIGINT and SIGTERM are ignored
   until it ends, so only SIGKILL shortens an 11-minute lingering upgrade
   process. Compounding this, the old process has already given its listener FDs
   away — killing the new process before it is supervised leaves *nobody*
   holding the ports. This produced a ~6-minute data-plane outage during this
   re-run's cleanup; model units were unaffected. The manual upgrade procedure
   needs a documented, supervised cutover.
4. **`gptoss20b` under-declares its VRAM reservation** — 31,097,880,576 B
   observed against 31,000,000,000 B declared — so NVML advisory drift fires
   continuously and degrades GPU observation for its GPU-sharing peer.
5. **`vllm_router_admission_rejections_total` is quieter than intended.**
   `choose_running` pre-filters candidates on lifecycle, so lifecycle-level
   reasons rarely reach `acquire_active`. The `stale_probe` reason — the
   fleet-outage signal it was added for — remains reachable and is the one to
   alert on.

## On-box Tier 4/5 run — 2026-07-29 (pre-fix, superseded for Tier 4)

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

Tier 5 was **not** re-run against the fix batch, deliberately. Its failures are
hardware/runtime findings that no code change in this batch addresses: the NVFP4
TP=2 FlashInfer initialization crash (F15) and the measured L1 host-RAM overage
(F16) are properties of this checkpoint/vLLM/GPU cell, now *recorded* in the
service compatibility declaration rather than fixed. Sleep also remains disabled
on every service (`hardware_validated: false`), so a re-run would first require
deliberately enabling it. The one sleep-path change here — the `SleepFact`
provenance fix that stops an unproven `/is_sleeping` from ageing as fresh — is
unreachable while sleep is off. Re-run Tier 5 when someone next attempts a fresh
hardware-validation cell; that fix is what it should exercise.

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
