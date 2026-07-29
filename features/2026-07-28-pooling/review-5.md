# Implementation Review 5 — post-review-4 findings (2026-07-29)

Reviewed `00ad691` against `spec.md`, `review-4.md`, and the current acceptance
record. The fast baseline is clean:

- `cargo fmt --all -- --check`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --all-targets --all-features`: 168 pass, 3 host-only ignored

The review-4 implementation materially closes F11–F17: uncertain/reconciling owners
now reserve placement, reconciliation and admission are serialized, force-stop uses
`StopUnit` and captures a blast radius, the upgrade grace covers the stream target,
known-bad sleep topologies and measured L1 floors are validated, and joined flights
plus upstream idle timeouts are auditable. Sleep remains hardware-gated as required.
F14's fail-closed admission fix is not complete, however, and the new joined-flight
audit work exposed a policy-coalescing bug. Additional findings follow in severity
order.

## F18 — The public inference listener has no authentication

**Severity: critical (security and unbounded compute spend).**

Spec Decision 9 requires public API authentication to be enforced at the router.
`DataPlane` has no authentication dependency or configuration
(`proxy/ingress.rs:116-152`), and `process` reaches model discovery, body parsing, and
lifecycle admission without checking `Authorization` (`proxy/ingress.rs:155-190`).
Only the management listener loads `TokenAuth` (`main.rs:42-56`). With the documented
`0.0.0.0:6969` bind, an unauthenticated network client can enumerate models, cold-start
services, evict idle residents, and consume GPU inference.

Passing the client's `Authorization` header upstream is not router authentication;
generated vLLM services do not have a corresponding router-owned API key, and the
router makes the admission/lifecycle decision before any upstream could reject it.

**Fix:** add an inference-plane authentication policy (at minimum a separately
configured root-owned bearer-token file or token set) and reject unauthorized requests
before reading a body or starting lifecycle work. Keep management and inference
credentials separate, compare tokens in constant time, and deliberately strip or
replace the router credential before forwarding unless backend authentication is
configured as a separate concern. Add negative tests proving an unauthenticated valid
model request causes no parking, eviction, or start.

## F19 — Single-flight joins erase behavior-changing `force` and `evict` policy

**Severity: high (a nominally safe request can join a destructive operation).**

`get_or_spawn_flight` joins solely on `TransitionKind` equality
(`pool/mod.rs:1009-1040`), but that key retains `force` only for `Stop`:

- `Drain` drops `DrainOptions.force`;
- `Sleep(level)` drops `SleepOptions.force`;
- `EnsureReady` drops `evict` and `reactivate_drained`
  (`pool/mod.rs:438-473`).

The management API accepts all of those options. Consequently:

- a non-force sleep can join an existing force sleep and share the operation that
  force-stops active work after the drain deadline;
- a force sleep can join a non-force sleep and silently lose the requested fallback;
- an administrative `start` with `evict: false` can join data-plane demand running
  with `evict: true`, despite the spec requiring the non-evicting administrative call
  to fail fast rather than displace a resident.

The audit makes this invisible: joined drain/sleep records omit `force`, and the new
test explicitly asserts that a sleep join has no force field
(`pool/mod.rs:2783-2876`). This is not merely missing metadata—the underlying flight
really uses whichever caller arrived first.

**Fix:** include every behavior-changing option in the single-flight identity, or join
only after comparing an immutable operation-policy record and reject incompatible
callers with `TransitionInProgress`. Audit that stored policy, not the joining
request. Add both-order tests for force/non-force sleep and drain, plus
evicting/non-evicting ensure-ready.

## F20 — A failed sleep probe can reuse a timestamp that proved a different sleep fact

**Severity: high (residual F14 admission bypass).**

The new `SleepFact` distinguishes an unobserved response, but
`ProbeDebounce::record` still updates `sleeping_level` on every call while declining
to update only `observed_at` (`pool/managed.rs:112-145`). The timestamp and the fact it
proved can therefore diverge.

A concrete start/wake sequence is:

1. The last authoritative probe proves the old process inactive or asleep and stamps
   `observed_at`.
2. `start` or `wake` resets `SleepObservationState` to its default, but does not clear
   `ProbeDebounce`'s timestamp (`pool/managed.rs:670-680,729-749`).
3. `/is_sleeping` fails. `SleepObservationState::update(None)` now yields no level, so
   the observation is recorded as `Unobserved(None)`; health and model probes are
   nevertheless allowed to set `ready=true` (`pool/managed.rs:507-558`).
4. `wait_until_ready` checks systemd/ready/healthy/model only, not whether the awake
   fact was authoritatively observed (`pool/mod.rs:1436-1463`).
5. Admission sees `sleeping_level=None` and the still-young timestamp from step 1 and
   can accept the backend before that timestamp ages out.

Thus an old “this previous process is absent/asleep” timestamp can be presented as a
fresh “this current process is awake” fact—the exact sleeping-backend black-hole F14
must prevent. Optional warmup may mask the bug for some service definitions, but
warmup is not required by validation and is not the specified sleep proof.

**Fix:** store the authoritative sleep value, timestamp, and process/operation
generation as one unit. `Unobserved` must retain both the prior value and its timestamp;
it must never install a new value under an old timestamp. Starting or waking a process
must invalidate the old proof until `/is_sleeping` authoritatively returns false.
Readiness completion should require that current-process proof. Add regression tests
for inactive→start and asleep→wake where `/is_sleeping` fails but `/health` and
`/v1/models` succeed.

## F21 — `/readyz` remains green when fail-closed admission rejects every backend

**Severity: medium-high (load balancers route into a known 503-only fleet).**

The review-4 fix makes `acquire_active` reject stale, asleep, systemd-inactive,
unhealthy, or wrong-model facts. `routing_readiness`, however, reduces each service to
its lifecycle and counts every `Running` service as routable
(`api/control.rs:442-472`). Probe aging does not change `Lifecycle::Running`; it is
checked only in the admission predicate.

If all running services age past `maximum_probe_age_ms` after an observation failure,
every inference request is rejected with `stale_probe` while `/readyz` continues to
return 200 and report those services as routable. The same mismatch exists for the
other internally inconsistent running facts F14 now rejects.

**Fix:** derive readiness from the same non-mutating admissibility predicate used by
active acquisition (plus the intentionally parkable stopped/starting/asleep states),
including each service's configured maximum probe age. Add a test with one stale
`Running` service and no alternative route; `/readyz` must return 503.

## F22 — Client cancellation is not observed while waiting for upstream headers

**Severity: medium (orphaned inference and inflated exact activity).**

The ingress explicitly watches the downstream connection only while parking
(`proxy/ingress.rs:350-375`). After active admission, `proxy_upstream` awaits
`request.send()` and then upstream body reads; it does not poll the downstream until it
has headers or body bytes to write (`proxy/ingress.rs:489-575`).

If a client disconnects after its request has been forwarded but before vLLM emits
response headers, the router cannot notice the FIN while blocked on the upstream.
Reqwest therefore continues the inference and the exact active guard remains held
until the backend responds or the inter-chunk/read timeout expires. This violates the
spec's explicit cancellation-propagation and release-on-disconnect requirements. The
existing downstream-disconnect test covers a stream only after response bytes are
available, where a failed write happens to reveal the cancellation.

**Fix:** race upstream header/body progress against a downstream-cancellation signal
throughout proxying and drop the reqwest response future immediately on cancellation.
Add a test whose upstream accepts the request but never sends headers, then disconnect
the client and assert prompt upstream cancellation and active-count release.

## F23 — `-t` validates UUIDs only against self-declared inventory, not host NVML

**Severity: medium (the documented pre-upgrade safety check can approve an unusable
configuration).**

Static validation checks placement UUIDs against `inventory.gpus`
(`config/mod.rs:790-824`), so the same typo in both sections is considered valid.
Actual NVML discovery and UUID-to-index resolution happen in
`managed_pool_from_config`, after `server.bootstrap()` (`main.rs:34-51`). Pingora exits
inside `bootstrap()` for `-t`, so that host validation is unreachable.

This is directly reproducible with the checked-in example's placeholder
`GPU-00000000-...` UUID:

```text
$ target/debug/vllm-router -t -c deploy/pingora.yaml \
    --router-config src/config/example.yaml
Server Test passed, exiting
$ echo $?
0
```

That conflicts with the spec and README claim that `-t` rejects unknown GPU UUIDs
before a restart/upgrade applies a replacement configuration.

**Fix:** perform a read-only host inventory check before Pingora's test-mode exit (and
dry-run any UUID-dependent generated inputs), while leaving D-Bus connections and
runtime-owned production objects after `bootstrap()`. Add a `-t` subprocess test using
a syntactically valid but nonexistent host UUID.

## Recommendation

Do not mark the implementation fully accepted yet. F18 is a deployment security
blocker; F19 and F20 need regression tests before enabling sleep on any hardware cell.
F21–F23 should be closed before relying on readiness automation, cancellation safety,
or `-t` as the pre-upgrade gate. After the code fixes, rerun the Tier-4 busy-eviction,
force-stop, and graceful-upgrade drills to replace the pre-fix failure record with
measured closure evidence.
