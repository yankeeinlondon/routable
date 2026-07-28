# Implementation Review 1 — vLLM Pooling Router

Reviewed: commit `b081307` ("Implement vLLM pooling router"), 7,643 lines of Rust, against
`spec.md` (finalized 2026-07-28) and `plan.md`. Method: full manual read of `pool/`,
`proxy/`, `state/` (concurrency core) plus two parallel deep reviews of `config/`/`api/`/
`metrics/` and `state/`/`sysd/`/`gpu/`/`probe/`/`runner/`/`main.rs`; every finding below was
traced to source, not pattern-matched. Baseline: `cargo build` clean, `clippy -D warnings`
clean, 36 tests pass / 1 ignored, `-t` config check works.

## Verdict

The skeleton is genuinely good: the `PoolManager` seam is respected, accounting is exact
RAII as spec'd, the state store is generation-guarded, single-flight is real (tested
50-way), the L2 three-step wake is correct with a sequence test, auth is timing-safe with
root-owned-file checks, config validation is broad and strict, and the Phase-0 findings
honestly document both the pingora body-replay fallback (legitimate: 64 KiB retry-buffer
ceiling at 0.8.1) and the deliberately unclaimed hardware gates. Sleep correctly stays
disabled without `hardware_validated: true`.

It is **not deployable yet**. Five critical findings — three of which would bite on CT 107
on day one — plus a cluster of high-severity lifecycle bugs need fixing before the on-box
acceptance phase is meaningful.

## Critical

**C1. Daemon mode breaks all systemd control (fork after tokio/zbus init).**
`main.rs` is a tokio runtime; `ManagedVllmRunner::connect` creates the zbus `Connection`
(whose I/O tasks live on that runtime) before `server.run_forever()` (`main.rs:26-44`,
`pool/managed.rs:66`). Pingora daemonizes by forking inside `run_forever`; the child keeps
only the forking thread, so the runtime workers polling the D-Bus socket are gone — every
start/stop/observe call times out forever when run with `-d`. Undaemonized it works by
accident. Fix: construct the pool (and any pre-fork async state) inside a post-fork
initialization path (e.g. lazily on first background-service tick), or reject `-d`.

**C2. Systemd unit/environment generation does not exist.**
Spec Decision #2 ("service definitions render to templated/generated units") and the
allowlisted-fields requirement are unimplemented — zero code renders units or environment
files (no `ExecStart`/`Environment`/`VLLM_SERVER_DEV_MODE` writer anywhere). Consequences:
the validated `runner_args` are dead configuration; `VLLM_SERVER_DEV_MODE=1` /
`--enable-sleep-mode` gating is unenforceable; the Ray-executor ban only guards a config
field nothing consumes; and a hand-written unit missing `--enable-sleep-mode` degrades
silently (`is_sleeping` 404 is swallowed by `.ok()` at `pool/managed.rs:246`) until a
`/sleep` command fails at runtime. Fix: implement the generator per spec, or explicitly
re-scope the spec to "operator-provided units" and validate unit content at startup.

**C3. All cold starts are globally serialized.**
`PoolInner.placement_fifo` (`pool/mod.rs:336`) is one global mutex, and
`do_ensure_ready` holds it across placement, eviction, `runner.start/wake`,
`wait_until_ready` polling, *and* warmup (`pool/mod.rs:757-841`). A ~277 s cold start
(documented on this hardware) blocks `ensure_ready` for every other service — including a
sub-second wake on a completely free GPU. Violates the spec's per-GPU FIFO resolution and
"slow model startup … must not block unrelated ready models"; the existing FIFO test only
covers same-GPU contention (`pool/mod.rs:1864-1901`). Fix: publish the `Starting` claim
(begin_transition) and then release the FIFO before awaiting the runner — the code comment
at `pool/mod.rs:768-770` already describes exactly this design; the guard just isn't
dropped — or scope the mutex per GPU set.

**C4. Sleep level is forgotten across router restart → corrupt wake.**
vLLM's `/is_sleeping` does not report the level, and the only memory is the in-process
`commanded_sleep` (`pool/managed.rs:250-258`, `remembered.unwrap_or(SleepLevel::L1)`).
After a router restart, an L2 sleeper is reconstructed as `AsleepL1`; the subsequent wake
takes the L1 path (`runner/vllm.rs:188`) and skips `reload_weights` — the service resumes
with discarded weights. Violates "reconstruct live state from observations". Fix: treat
unknown sleep level conservatively (always run the L2 wake sequence, which is safe for L1
sleepers, or stop/start), or persist the commanded level.

**C5. NVML is a hard gate and can hang the whole control loop.**
One failing NVML call fails the snapshot (`gpu/mod.rs:137-143,205-246`) → every
`GpuStatus.healthy=false` (`pool/managed.rs:232`) → readiness never passes
(`pool/mod.rs:981`) and running services demote to `Stale` (`pool/mod.rs:1452-1463`) →
admission refused pool-wide. This inverts spec Decision #6 (NVML is "ground truth and
drift detection, not the sole placement algorithm") and — per `phase0-findings.md` — NVML
does not currently work in this container, so as-built the router would refuse all traffic
on CT 107. Additionally `snapshot_async` is `spawn_blocking` with no timeout
(`gpu/mod.rs:148-158`); a wedged driver freezes `reconcile_all`'s `join_all` and the
single background tick forever. Fix: make GPU health advisory (affect status/metrics, not
admission), degrade gracefully when NVML is absent, and bound the snapshot.

## High

**H1. One HTTP timeout for probes, sleep, wake, and warmup.** The `VllmRunner` client
timeout is sourced from `readiness.timeout_ms` (`pool/managed.rs:78-82`,
`runner/vllm.rs:51-60`) and bounds *every* call — including L1 sleep (offloading tens of
GB to pinned RAM), L2 wake (disk reload), and warmup (~210 s documented on sm_120). Real
warmups will spuriously time out and drive services into `Failed`/backoff. Per-operation
timeouts are needed; the pool already passes proper deadlines, but reqwest aborts first.

**H2. Stopped services get stuck in `Reconciling`.** `unit_status` uses Manager `GetUnit`
(`sysd/mod.rs:143-153`), which returns `NoSuchUnit` for units systemd has garbage-collected
(any inactive, unreferenced unit). `observe` then errors, and `Pool::reconcile`'s error
path leaves the service in `Reconciling` with no `last_error` recorded
(`pool/mod.rs:514-525`) — every tick, forever. Status can never show "stopped", violating
the acceptance criterion that states be distinguishable. Fix: use `LoadUnit`, and restore
the prior lifecycle (with `last_error`) on observe failure.

**H3. Idle timeout measures from request start, not completion.** `acquire_active` touches
`last_used` (`state/mod.rs:365-372`) but `ActiveGuard::drop` does not
(`state/mod.rs:460-468`). Spec: update at admission, "not *only* when a response
completes" — i.e. both. A stream longer than `sleep_l1_after` ends with `idle_for` already
past threshold, so the reaper sleeps/stops the service on the next tick immediately after
a long request completes. Fix: touch on guard drop as well.

**H4. Standalone drain is a trap state that any client request silently undoes.**
`do_drain` succeeds by leaving lifecycle at `Draining` permanently (`pool/mod.rs:994-1027`):
`Draining` owns awake placement, is never evictable (`is_transitioning`), and is invisible
to the idle reaper — the GPU is pinned until an admin acts. Worse, `Draining` is not in
`ensure_ready`'s blocked set, so the next data-plane request for that model resurrects the
service — an admin drain preparing a stop is undone by any client. Fix: introduce a
terminal `Drained` state that (a) blocks automatic start and (b) is reachable by the reaper
/eviction, or make drain-only transitions require `automatic_start=false` semantics.

**H5. The single background tick serializes probing behind lifecycle work.**
`PoolBackground` runs `reconcile_all` then `reap_idle_once` in one loop (`main.rs:92-98`),
and `reap_idle_once` awaits `sleep_status`/`stop_status` inline up to each service's drain
deadline (`pool/mod.rs:549-612`). One slow transition (or C5's NVML hang) freezes
observation for all services — violating the Phase-2 exit criterion that one service's
probes can't block another's. Fix: spawn reap transitions detached (they're already
single-flighted), and reconcile services concurrently with per-service bounds.

**H6. Two disjoint operation-ID namespaces; idempotency contract broken.**
`spawn_mutation` mints a fresh API-layer UUID per HTTP request (`api/control.rs:255`) and
discards the pool's single-flight `TransitionResponse.operation_id`
(`control.rs:184,197,225,243`). Spec: repeating a lifecycle command "returns the
already-running operation's ID" — instead every retry gets a new ID, `GET
/pool/v1/operations/{id}` only resolves API-layer IDs, and audit lines log IDs the pool
never saw. Fix: propagate the pool's operation ID into the store (or key the store by it).

**H7. `active` gauge has two conflicting writers.** The ingress path `inc()`/`dec()`s it
(`proxy/ingress.rs:172-175,183-208`) while `publish_state_metrics` periodically `set()`s
it from state snapshots (`main.rs:132-135`). Interleaved absolute writes and deltas
corrupt the value. Pick one writer (the snapshot `set` is simpler and already exact).

**H8. Exponential backoff is neutralized by the periodic reconciler.** After a failed
start, `schedule_backoff` moves `Failed → Backoff`, sleeps, then force-stops
(`pool/mod.rs:1165-1244`). But the flight slot is already cleared, so the next
`reconcile_all` tick freely overwrites `Backoff` with observed state (e.g. `Stopped`),
making the service immediately retriable and orphaning the backoff task's later updates
(stale generation). Spec requires repeated failures to enter backoff. Fix: skip periodic
reconcile for services in `Failed`/`Backoff` (or teach reconcile to preserve those states
until the backoff task completes).

## Medium

**M1. Lost-wakeup race in `Activity::wait_until_idle`** (`state/mod.rs:437-453`):
`Notify::notified()` registers only on first poll, and the guards use `notify_waiters()`
(no permit). A guard dropped between the idle re-check and the first poll is missed, so
the drain burns its entire deadline before the timeout-path recheck saves correctness.
Combined with H5 this stalls the control loop. Fix: `pin!` the future and call
`.enable()` before the re-check.

**M2. `/readyz` can never fail** (`api/control.rs:51-58`): the whole-pool `status()` has
no error path, so readyz ≡ livez, contra the spec's liveness/readiness split.

**M3. `/v1/models` is not implemented.** The data-plane allowlist 404s it
(`proxy/ingress.rs:110-120`). Standard OpenAI clients call it for discovery; the router
should synthesize it from configured served models + aliases (spec: router-owned endpoints
use explicit rules).

**M4. Failed sleep/stop bricks the service until the next reconcile tick.**
`do_sleep`/`do_stop` error paths call `record_failure` but never `schedule_backoff`
(`pool/mod.rs:1072-1075,1132-1135`); `ensure_ready` returns `TerminalFailure` for
`Failed`, so parked demand fails until periodic reconcile happens to heal it. Also, an
admin `sleep` on a `Stopped` service proceeds (only the target level is checked,
`pool/mod.rs:1042-1045`), fails at the runner, and marks the service `Failed`.

**M5. Config reload is dead code, and half the stack couldn't see it anyway.**
`ConfigStore::reload` has zero production callers (no SIGHUP, no endpoint); the pool,
`TokenAuth`, alias/semaphore maps in `DataPlane` are all built once at boot
(`main.rs:43-44`, `proxy/ingress.rs:72-90`). Either wire reload end-to-end or document
restart-only reconfiguration and delete the dead path. (Related: a reloaded config adding
a model would hit the stale `model_parked` map and return `model_not_found`.)

**M6. Drain/sleep ignore the spec'd force policy** — only `deadline_ms` is plumbed
(`api/control.rs:188-227`); a client's `{"force":true}` on drain is silently accepted and
ignored.

**M7. Status fidelity gaps:** `probe.age` is hardcoded `Some(Duration::ZERO)`
(`pool/managed.rs:141`) — spec explicitly requires real probe age; `served_model_matches`
is set to `probe.ready` (`pool/managed.rs:137`), hiding "healthy but wrong model";
`main_pid` is always `None` (`pool/managed.rs:212`), which also guarantees the drift
detector's allowed-PID set can never populate.

**M8. NVML drift detection is dead** (spec Decision #6): `detect_drift` only iterates
`expected.devices` and the sole caller passes `ExpectedGpuState::default()`
(`gpu/mod.rs:248-277`, `pool/managed.rs:215`); the computed `drift` field is discarded.

**M9. Dead/duplicated machinery with drift risk:** `PoolManager::admit`/`Admission`
(`pool/mod.rs:1372-1443`) is a second, divergent admission implementation with zero
callers — `DataPlane::admit` reimplements it differently; the entire `probe/` engine,
`sysd::UnitObserver`, and `StateStore::publish_observation` are unwired (and the observer
would be inert anyway: `Manager.Subscribe()` is never called, and the unit path is
resolved once with no retry — `sysd/mod.rs:250-336`). Delete or wire each.

**M10. Missing config validations:** duplicate `systemd.unit`/`environment_file` across
services is accepted (one service's stop would kill another's process) —
`validate_systemd` checks only per-service syntax (`config/mod.rs:723-740`). Also
`evictable: false` alone makes a config invalid because `IdleConfig::default()` sets
`stop_after_ms: Some(3_600_000)` and the validator rejects the combination
(`config/mod.rs:376-384,1022-1026`); idle-sleep is inconsistently not gated on
`evictable`.

**M11. API mutation deadlines use hardcoded defaults** (300 s/120 s,
`api/control.rs:170,191,204,232`) instead of the target service's configured
`deadlines.*` — a service with `startup_ms: 600000` gets cut off at 300 s.

**M12. Ingress robustness:** `choose_running` aborts admission on any non-
`TransitionInProgress` acquire error instead of trying the next replica
(`proxy/ingress.rs:339-345`); the reqwest data-plane client leaves the default redirect
policy (follows up to 10) — a proxy should pass 3xx through (`Policy::none`); systemd
start/stop job results are never watched, so a unit that fails instantly still burns the
full `startup_deadline` against parked requests (`sysd/mod.rs:171-189`).

**M13. Metrics gaps:** default Prometheus buckets (5 ms–10 s) make parking/request
histograms useless for a workload with documented 100 s+ operations
(`metrics/mod.rs:36-59`); no "active streams" metric distinct from active requests (spec
§Features); force-stop does not audit the number of affected requests (spec acceptance).

## Low

- `ModelBody` borrows `&str` (`proxy/routing.rs:66-70`): valid JSON with escapes in the
  model name (`"a-b"`) fails deserialization → 400. Use `Cow`/`String`.
- `TokenAuth::load` metadata/read TOCTOU (`api/auth.rs:31-47`) — negligible here; use
  open+fstat. Bearer scheme match is case-sensitive contra RFC 7235.
- `runner/vllm.rs:107-113` buffers non-2xx bodies unbounded (loopback-only peer).
- `ManagedVllmRunner::warmup` silently drops all configured warmup-body fields except
  `prompt`/`max_tokens`/`temperature` (`pool/managed.rs:348-364`).
- The L1→L2 idle-tier transition (`/sleep?level=2` on an already-L1-sleeping server) is
  unverified against real vLLM and is missing from the S2 matrix — add it.
- Two TLS stacks are linked (openssl via pingora, rustls via reqwest) — informational.

## Test coverage

The 36 fast tests are well-aimed at the pool engine (single-flight, eviction safety, FIFO,
drain deadlines, L2 wake sequence) but:

- **The ingress has no behavioral tests at all** — its only test is a static allowlist
  check. Parking, disconnect-while-parked, connect-failover, streaming passthrough, and
  error mapping (the bulk of `acceptance.md`'s ingress claims) are untested even against a
  local mock HTTP server. The plan's mock-runner integration suite exists only as the
  in-process `MockRunner` inside pool tests.
- Config validation has ~40 rejection rules with no test (full list available: inventory,
  listener-vs-listener ports, policy zeros, alias edge cases, parallelism overflow, sleep
  pairing rules, budget overflow branches, …). The plan required table-driven coverage of
  every rule.
- The placement/FIFO test only exercises same-GPU contention, which is why C3 was not
  caught.

## Process notes

- Phase-0 artifacts exist and are honest; the ProxyHttp→custom-ingress fallback is
  spec-sanctioned and convincingly justified (64 KiB retry-buffer limit at pinned 0.8.1).
- The hardware gates (S2 sleep matrix incl. the missing L1→L2 cell, live D-Bus round-trip,
  chaos drills, SSE load) remain correctly unclaimed and must run on CT 107 — but only
  after C1–C5, or the on-box results will be dominated by known defects.

## Recommended remediation order

1. C5 (NVML advisory + bounded) and C1 (post-fork init) — the router cannot run on the
   target container at all otherwise.
2. C3 (drop FIFO after claim), H3 (touch on drop), H4 (drained state), H8 (reconcile vs
   backoff) — lifecycle correctness cluster, all small diffs in `pool/`/`state/`.
3. C4 + H1 (sleep restart-safety + per-op timeouts) — prerequisites for an honest S2 run.
4. C2 (unit generation) or a deliberate spec amendment to operator-provided units.
5. H2, H5, H6, H7, M1–M13 in roughly listed order; add ingress behavioral tests and the
   missing config-rule tests alongside.
