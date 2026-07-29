# Implementation Review 3 — findings from test preparation and on-box acceptance

Opened 2026-07-29 during test preparation (per `testing-runbook.md`, failures and
pre-test discoveries land here). Numbered continuing from review-2's scheme.

## F1 — Aliases are broken end-to-end: generator omits them from `--served-model-name`

**Severity: medium-high (functional; every alias-named request fails).**

- `sysd/generator.rs:160-161` renders `--served-model-name <served_model>` with the
  canonical name only; the service definition's `aliases` are never passed to vLLM.
- The router advertises aliases in its `/v1/models` synthesis
  (`proxy/ingress.rs::configured_models`) and accepts them at admission
  (`parse_model_route` resolves alias → canonical).
- The data plane forwards the request body **byte-exact** (spec §Request routing;
  asserted by `parses_alias_without_touching_body`), so the upstream receives
  `{"model": "<alias>"}` — a name the vLLM instance does not serve → upstream 404
  for every alias-named request, after a potentially expensive cold start was
  triggered for it.

**Failure scenario:** configure `served_model: rdtand/Qwen3.6-35B-…` with alias
`qwen3.6-35b-q4.75`; `POST /v1/chat/completions {"model":"qwen3.6-35b-q4.75"}` →
router parks/starts/forwards → vLLM 404 "model not found". Discovered by comparing a
manually-started vLLM (launched with two served names; `/v1/models` correctly lists
both) against the generated unit.

**Fix:** the generator emits `--served-model-name <canonical> <alias…>` (vLLM accepts
multiple values; each becomes a served id). Byte-exact forwarding is preserved and the
readiness served-model check (canonical ∈ list) is unaffected. Add a generator test
asserting aliases are rendered, and a Tier-3 runbook step exercising an alias-named
request end-to-end (would have caught this).

**Scope clarification (2026-07-29):** the multi-entry listing is only acceptable as
*backend-internal plumbing*. vLLM's own `/v1/models` is reachable only on loopback and
is read only by the router's readiness probe — clients never see it — so emitting
canonical + aliases as `--served-model-name` values remains the correct F1 fix and its
duplicated listing there is harmless. The client-visible presentation problem is F2.

Note for v2: llama.cpp's `--alias` takes a single name, so multi-alias services are a
per-runner capability — validation should reject what a runner cannot honor
(v2 spec §1 complexity register pattern).

## F2 — Router `/v1/models` duplicates every alias as a full model entry

**Severity: medium (client-visible presentation; decided behavior).**

`proxy/ingress.rs::configured_models` emits one complete `ModelDescription` per alias
*and* per canonical name — identical in every meaningful field (`root`, `owned_by`,
`created`), differing only in `id`, with `parent: None` giving no hint of the
relationship. A client's model picker shows the same model N times. This mirrors the
raw vLLM behavior the operator has explicitly rejected as unwanted duplication.

**Decided fix (operator decision 2026-07-29):** the router lists **one entry per
service** — `id` = canonical `served_model`, `root` = canonical — and exposes the
abbreviated/alternate names on that single entry via an `aliases: ["…"]` extension
field (OpenAI clients ignore unknown fields, so compatibility is preserved). Aliases
remain fully *accepted* as request ids; they are simply not independent list entries.
Update the `/v1/models` test to assert exactly one entry per service and the presence
of the aliases field; document the extension field in the README's API table.

**Governing rule (refined 2026-07-29):** a list entry is justified by a distinct
*client contract*, not a distinct string. Two offerings of the same weights with
different `max_model_len` (e.g. a 32k co-residency-friendly variant vs a 128k
full-context variant) are legitimately two entries — modeled as two service
definitions sharing `model_path` with different `served_model` ids. An alias changes
nothing a client can act on and therefore collapses into the entry. Accordingly, the
synthesized entries must also **carry the contract**: populate `max_model_len` from
`compatibility.maximum_model_length` (currently omitted entirely, though real clients
use it to size prompts).

## F3 — Replica sets are not validated for contract homogeneity

**Severity: medium (latent nondeterministic failure).**

`config::RouterConfig::services_for_model` (config/mod.rs:74-82) treats every service
with the same `served_model` as an interchangeable replica, and the data plane selects
among them least-inflight. Nothing in `validate_services` requires those services to
share a client-visible contract: two replicas with different
`compatibility.maximum_model_length` (or kv-cache dtype) validate cleanly, and a
long-context request then succeeds or fails depending on which replica wins selection
— a nondeterministic contract violation that no test would catch until load shifts.

**Fix:** config validation rejects same-`served_model` services whose client-visible
compatibility fields differ (`maximum_model_length`, `kv_cache_dtype`; extend as the
contract grows). Different contracts must use different `served_model` ids — which is
exactly the F2 variant-modeling rule. Table-driven test alongside the existing
duplicate/alias rules.

**F2 wording correction (found while enumerating tests):** the invariant is **one
entry per unique served model**, not per service — replicas share a `served_model`
and must collapse into a single entry (well-defined precisely because F3 makes
replica contracts homogeneous). "One entry per service" would reintroduce the
duplication for replica sets.

## Required test coverage for F1–F3

**F1 — generator (unit):**
- renders `--served-model-name <canonical> <alias…>` in declaration order;
- no aliases → single name (pin existing behavior);
- alias values pass `exec_argument` escaping (an alias containing a quote/space is
  either rejected by config validation or rendered safely — pin whichever holds);
- runner observe: `served_model_matches` true when the backend's model list contains
  canonical **plus** extra names (multi-name listing), false when only aliases appear
  without the canonical.
- On-box (runbook Tier 3): alias-named completion succeeds end-to-end.

**F2 — models synthesis (unit):**
- two services, one with two aliases → exactly two entries; alias strings appear only
  in the `aliases` field, never as `id`s;
- **two replicas of one served model → exactly one entry** (the corrected invariant);
  its `aliases` is the union of the replicas' aliases;
- `max_model_len` equals `compatibility.maximum_model_length`;
- serialized shape keeps OpenAI-required fields (`object: "list"`, `data[].id/object/
  created/owned_by`) so standard clients parse it; extension fields present;
- `POST /v1/models` still 405 (pin existing behavior).

**F3 — config validation (table-driven):**
- same `served_model`, differing `maximum_model_length` → rejected, message names
  both services and the field;
- same `served_model`, differing `kv_cache_dtype` → rejected;
- same `served_model`, identical contract → accepted (replica set);
- different `served_model` sharing one `model_path` (the 32k/128k variant case) →
  accepted;
- decision to pin while writing these: whether one alias may be declared on two
  replicas of the same canonical (currently `alias_owners` likely rejects any
  cross-service duplicate even when the canonical matches — either allow-when-same-
  canonical or document declare-on-one-replica; test whichever is chosen).

## F4 — `run_forever` panics under `#[tokio::main]`: the router has never actually served

**Severity: critical (blocks all serving; found at first real launch, 2026-07-29).**

`main` is `#[tokio::main] async fn` (required today because `managed_pool_from_config`
awaits the zbus connection), but `pingora_core::server::Server::run_forever` performs a
`block_on` internally (pingora-core-0.8.1 `server/mod.rs:756`) — which tokio forbids
from inside a runtime: `"Cannot start a runtime from within a runtime"`. The panic is
deterministic on every real start.

**Why 140 passing tests missed it:** nothing ever executes `run_forever`. `-t` mode
exits inside `bootstrap` before it; unit tests never start the server; review-2's C1
fix addressed the *daemon-fork* hazard by rejecting `-d` but left the async-main
incompatibility in place, unexercised. Tier 3 of the runbook is the first code path in
the project's life to reach this line.

**Fix requirement:** `main` must be synchronous and `run_forever()` must be called
outside any tokio context, while the zbus connection's I/O tasks stay on a runtime
that lives for the whole process (the C1 lesson: those tasks must never lose their
runtime). Recommended shape: build the pool on a dedicated multi-thread runtime kept
alive for process lifetime (owned in `main` or intentionally leaked — never dropped),
then call `run_forever()` from plain `fn main`. Cross-runtime awaits of the pool from
pingora's worker runtimes are fine (futures are `Send`; zbus internals stay home).
Alternative shape: defer all async construction into a pingora background service with
a lazy `PoolManager` handle — larger diff, same guarantee.

**Regression test requirement:** an `#[ignore]`d host test that launches the built
binary with a minimal valid config on ephemeral ports, polls `/livez` until 200, and
sends SIGTERM — i.e. "the binary actually serves." Wire it into the runbook's Tier 1
so no future refactor of `main` can regress unserved again.

---

## Verification (2026-07-29, on CT 107)

All four findings verified against commit `8e4e955`:

- **F1 ✓** — the router-generated unit renders `--served-model-name
  "qwen3.6-27b-nvfp4" "qwen27b"` (canonical + alias, exec-quoted); generator tests
  cover ordering, escaping, and the replica alias union.
- **F2 ✓ (live)** — `GET /v1/models` against the running router returns exactly one
  entry: canonical `id`, `aliases: ["qwen27b"]`, `max_model_len: 32768`, OpenAI shape
  intact.
- **F3 ✓** — `validate_replica_contract` in place with table-driven tests; the
  alias-ownership decision went allow-when-same-canonical.
- **F4 ✓ (live)** — `main` is synchronous with a dedicated process-lifetime
  `vllm-router-pool` runtime; the new `tests/binary_serves.rs` regression passes as
  root, and the router **served for the first time**: `/livez`, `/readyz`
  (`routable_services: 1`), and the data plane all answered on 6969–6971.

Full suite: 141 tests pass (136 lib + 4 bin + binary_serves), clippy clean.

**Field note:** at first live startup the drift detector immediately flagged ~30 GB of
unmanaged usage per GPU (an operator-launched vLLM TP=2 instance occupying both
cards) — advisory-only, exactly as designed (C5). The Tier-3 cold-start smoke is
blocked until a GPU is freed; a cold start attempted now would OOM against the
unmanaged occupant, since v1 placement is declaration-based and does not treat
unmanaged processes as placement blockers (a deliberate v1 scope line worth a
runbook/README caveat: the router assumes it is the only GPU tenant besides declared
services).

## F5 — Graceful shutdown hangs: background service runtime never exits

**Severity: high (operational; SIGTERM/systemd restart takes the full 900 s timeout).**

Observed live (2026-07-29): after SIGTERM, and again after SIGINT, the router logged
`"waiting for service runtime BG pool observation and idle policy to exit"`
indefinitely and required SIGKILL. The `PoolBackground` loop itself breaks on the
shutdown watch, but the *runtime* hosting it does not exit — most likely lingering
detached tasks spawned onto it (reaper-fired lifecycle flights via
`get_or_spawn_flight`'s `tokio::spawn`, `schedule_backoff` sleeps of up to 60 s, or a
cross-runtime await parked on the pool runtime). With `graceful_shutdown_timeout` and
the unit's `TimeoutStopSec` both at 900 s, every restart/stop of an otherwise-idle
router stalls for 15 minutes then gets killed.

**Fix requirement:** the background service's runtime must quiesce promptly on
shutdown — spawn detached lifecycle tasks on the process-lifetime pool runtime (not
the pingora service runtime), or track and abort them on shutdown signal.
**Acceptance:** SIGTERM against an idle router exits within the grace period; add
this to the `binary_serves` regression (it currently SIGTERMs — extend it to assert
bounded exit time, which would have caught this).

## F6 — Drift warnings repeat every observation tick (log spam)

**Severity: low (operational noise).**

`NVML drift detected` and `GPU observation degraded` re-log identical facts every
~2 s (the observation interval) while an unmanaged process occupies a GPU — hundreds
of identical WARN lines per hour for a steady condition. Warn on *transitions*
(drift appears/changes/clears) and demote the steady-state repeat to debug; the
drift metric already carries the continuous signal.
