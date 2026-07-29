# Implementation Review 2 — vLLM Pooling Router

Reviewed: the uncommitted working tree on top of `b081307` (+6,236/−1,228 across 18 files,
`src/probe/` deleted, `src/sysd/generator.rs` added), against `spec.md`, `plan.md`, and
every finding in `review-1.md`. Method mirrors review 1: full manual re-read of the
concurrency core (pool, state, proxy ingress, managed runner) plus two parallel deep
re-verifications of the config/API/metrics and observation/systemd/runner scopes, each
prior finding traced to its fix in the new source. Baseline: `cargo build` clean,
`clippy --all-targets --all-features` zero warnings, **116 tests pass** (113 lib + 3 bin,
up from 36; 1 host-only test ignored).

## Verdict

**All 26 review-1 findings are resolved** — 25 fixed outright, 1 (M5, config reload)
resolved by a deliberate and defensible design change that needs a one-line spec
amendment. The fix quality is high: fixes land with regression tests (the lost-wakeup
race has a deterministic barrier test; the operation-ID unification has a concurrent-
mutation test; the generator has injection-escape tests), and several go beyond the
letter of the finding. Two smaller items I raised in discussion *after* review 1 remain
open, and the re-review surfaced five minor new observations. Nothing blocks committing
this work; one pre-commit hazard and one small functional gap should be addressed with
it.

## Review-1 findings: disposition

| ID | Finding | Verdict |
|---|---|---|
| C1 | daemon fork kills tokio/zbus | **Fixed** — `-d` and pingora `daemon: true` are both rejected with clear errors (`main.rs:177,199-210`, tests); `-u` fd-handover upgrade (no fork) remains supported |
| C2 | no unit/env generation | **Fixed** — new `sysd/generator.rs`: typed fields → fixed flags, allowlisted `runner_args` with full exec-quoting (no shell anywhere), atomic 0644/0600 writes, traversal-safe unit names, `VLLM_SERVER_DEV_MODE`/`--enable-sleep-mode` emitted only when sleep is enabled *and* hardware-validated, `mp` executor default; installed + `daemon-reload` before any lifecycle command |
| C3 | global cold-start serialization | **Fixed** — placement is now a demand queue; FIFO applies only among *overlapping GPU sets*, and the mutex is released before every runner/readiness/warmup/eviction wait. Non-overlapping starts proceed concurrently |
| C4 | sleep level forgotten across restart | **Fixed** — unknown sleepers take the full L2 three-step reload wake (safe for L1 too, `runner/vllm.rs:169-198`); the L1 reporting assumption is used for capacity accounting only and never written back |
| C5 | NVML hard gate + unbounded hang | **Fixed** — GPU health is advisory (logs + `probe_failures`, no readiness/admission gate); NVML snapshots hard-bounded at 2 s with single-flight joining, so a wedged driver degrades observations instead of freezing the control loop |
| H1 | one HTTP timeout for all ops | **Fixed** — `VllmTimeouts{probe,sleep,wake,warmup}` mapped from per-service deadlines; tested with probe-timeout-exceeding latencies |
| H2 | stopped units stuck `Reconciling` | **Fixed** — `LoadUnit` replaces `GetUnit`; reconcile error paths restore prior lifecycle + record `last_error` |
| H3 | idle measured from admission | **Fixed** — `ActiveGuard::drop` refreshes `last_used` while holding the timestamp lock through the decrement |
| H4 | drain trap-state / undone by demand | **Fixed** — terminal `Drained` state: releases placement, survives reconcile, and only an admin start (`reactivate_drained: true`) revives it — data-plane demand cannot |
| H5 | serialized background tick | **Fixed** — reconciles run 8-way concurrent with per-source bounds; the reaper fires single-flighted transitions detached |
| H6 | two operation-ID namespaces | **Fixed** — `begin_lifecycle` on the trait returns the pool's single-flight ID; the store dedupes on it; concurrent starts observably share one ID end-to-end |
| H7 | `active` gauge double writer | **Fixed** — periodic snapshot `set()` is the sole writer; streaming gets its own RAII-guarded `vllm_router_active_streams` gauge |
| H8 | reconcile neutralizes backoff | **Fixed** — reconcile preserves `Failed`/`Backoff` (refresh without generation bump) and `Drained` |
| M1 | lost wakeup in `wait_until_idle` | **Fixed** — `pin!` + `enable()` + recheck, with a deterministic regression test |
| M2 | `/readyz` can't fail | **Fixed** — ready iff some enabled service is running or automatically reachable; degraded optional models don't fail it |
| M3 | `/v1/models` missing | **Fixed** — GET-only synthesis from configured models + aliases |
| M4 | failed sleep/stop bricks service | **Fixed** — `schedule_backoff` on those error paths too |
| M5 | config reload dead code | **Resolved by design change** — hot reload removed; config is an immutable process-lifetime snapshot, restart-to-apply (see below) |
| M6 | drain/sleep force policy | **Fixed** — plumbed end-to-end with force-fallback tests |
| M7 | fabricated probe age / model-match | **Fixed** — independent `healthy`/`served_model_matches`/`sleeping_level` tracking; real, aging probe timestamps; MainPID read over D-Bus |
| M8 | drift detection dead | **Fixed** — expected state built from reservations + MainPID + sleep residuals; drift in status, metrics, and logs |
| M9 | dead/divergent machinery | **Fixed** — `probe/`, `UnitObserver`, `publish_observation`, and the unused `PoolManager::admit` path all removed or unified |
| M10 | missing config validations | **Fixed** — cross-service unit/env-file uniqueness; evictability decoupled from idle policy (default no longer poisons) |
| M11 | hardcoded API deadlines | **Fixed** — defaults map to the target service's configured deadlines, request-overridable |
| M12 | ingress robustness trio | **Fixed** — replica-local acquire errors fall through to the next candidate; `redirect::Policy::none()` (with test); systemd `JobRemoved` results awaited with typed failures (subscribe-before-queue, table-tested) |
| M13 | metrics gaps | **Fixed** — 600 s-range histogram buckets (asserted in test), active-streams gauge, force-stop audits `affected_requests` in a terminal audit line per mutation |
| Low (all 6) | `Cow` model field, TOCTOU, bearer case, error-body cap, warmup fields, L1→L2 cell | **Fixed** — incl. open+fstat single-fd token load with a file-swap regression test; unknown-tier promotion wakes through the safe reload before re-sleeping; the S2 matrix note stands |

## The one design change to ratify: restart-to-apply configuration

`ConfigStore::reload` was removed rather than wired: configuration is now immutable for
the process lifetime, documented in-code, which structurally eliminates the whole class
of stale-consumer bugs (pool/auth/alias/semaphore maps can no longer diverge). Applying
changes means `-t` validation + supervised restart or `-u` graceful upgrade — under
which the spec's "invalid new configuration leaves the last valid configuration active"
holds *procedurally* (the old instance keeps serving unless the new binary validates).
Recommendation: amend the spec's reload sentence to say exactly that, and add a runbook
line ("always `-t` before restart"). With that amendment this is a clean resolution.

## Carried over (raised post-review-1, not yet addressed)

**CO1 — Endpoint allowlist still mismatches vLLM's real paths (medium).**
`MODEL_ENDPOINTS` (`proxy/ingress.rs:32-45`) still lists `/v1/tokenize`,
`/v1/detokenize`, `/v1/classify` — vLLM serves these at the **root** only (`/tokenize`,
`/detokenize`, `/classify`, verified against current vLLM docs 2026-07-28), so the
allowlisted paths 404 upstream while the real ones are refused at the router. Also
missing: `/v2/rerank`, Cohere `/v2/embed`, Anthropic `/v1/messages/count_tokens`.
One-list fix plus the README endpoint table row.

**CO2 — Router-generated errors are OpenAI-shaped on every dialect (low).**
A router-side 404/503 on `/v1/messages` uses the OpenAI envelope; Anthropic SDKs expect
`{"type":"error","error":{...}}`. Documented as a caveat in the README; a per-endpoint
error shaper would remove it.

## New findings (all minor)

- **N1 — pre-commit hazard: `src/sysd/generator.rs` is untracked.** Everything else is
  a modification; a habitual `git add -u` commit would ship a build that doesn't
  compile. Use `git add -A` (or add it explicitly).
- **N2 — hardcoded `VLLM_EXECUTABLE = "/opt/vllm-venv/bin/vllm"`** (`generator.rs:15`).
  Correct for CT 107; should become configuration before any second host.
- **N3 — generated units live in `/run/systemd/system`**, so after a host reboot model
  services can't start until the router has started once and regenerated them.
  Consistent with router-as-orchestrator; deserves a runbook line.
- **N4 — `/is_sleeping` transport errors read as "awake"** (`managed.rs:385-389`): a
  sleeping service whose probe errors shows failing health probes (`Stale`) until the
  probe recovers. `commanded_sleep` is correctly preserved, so wake behavior stays
  safe; status fidelity only.
- **N5 — cosmetic:** a service mid-flight is warn-logged `TransitionInProgress` by every
  reconcile tick during a long start; and `SleepUnsupported` from `begin_lifecycle`
  maps to 409 where 400 is arguably righter. A >1024-mutation burst can age a live
  operation out of the bounded ring (self-healing, honestly reported — informational).

## Remaining gates (unchanged, by design)

The hardware-dependent acceptance work is still deliberately unclaimed and is now
meaningful to run, since the code-level blockers are gone: the S2 sleep matrix on
sm_120 (including the L1→L2 tier-promotion cell), live D-Bus start/stop/crash
round-trips, `kill -9`/SIGSTOP/NVML-staleness/router-restart/graceful-upgrade chaos
drills, and sustained SSE/stampede load with the new metrics. Sleep remains disabled
until `hardware_validated: true` per service.

## Recommendation

Approve. Before committing: include the untracked `generator.rs` (N1), fix the
endpoint allowlist (CO1 — small and functional), and add the spec's restart-to-apply
amendment. CO2 and N2–N5 can ride the next change. Then proceed to the on-box
acceptance phase on CT 107.
