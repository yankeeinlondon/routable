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

## F7 — Generated environment omits required host toolchain variables

**Severity: high on this host (crash-loop for any FlashInfer-JIT model).**

First real cold start (2026-07-29): the mixed NVFP4/BF16 checkpoint hit FlashInfer's
JIT path, which died with `RuntimeError: Could not find nvcc and default
cuda_home='/usr/local/cuda' doesn't exist` — the venv's `nvcc` lives at
`.venv/lib/python3.12/site-packages/nvidia/cu13/bin` and is only reachable via the
PATH that `/root/serve-vllm.sh` exports; the router-generated environment file sets
only `CUDA_VISIBLE_DEVICES` (+ args). The unit crash-looped under `Restart=on-failure`
while the parked request burned its deadline. Also missing: `VLLM_CACHE_ROOT` (so
cold starts ignore the warm torch.compile caches under `/models/vllm`) and `HF_HOME`.
MXFP4 models (`openai/gpt-oss-20b`) hit the same path.

**Interim operator fix (deployed):** systemd drop-in
`/etc/systemd/system/vllm@.service.d/10-toolchain.conf` setting `PATH`, `HF_HOME`,
`VLLM_CACHE_ROOT` for every `vllm@*` instance; survives unit regeneration.

**Proper fix:** an allowlisted `environment:` map — per service and/or a global
default block — validated with the existing control-character/newline rules and
rendered into the generated env file, so host toolchain requirements are declared
configuration rather than out-of-band drop-ins. Document the drop-in pattern in the
README as the escape hatch for env the schema doesn't model.

## F8 — UUID-based `CUDA_VISIBLE_DEVICES` is discarded by vLLM: models land on the wrong GPU

**Severity: critical (placement is silently not enforced).**

Observed live: `vllm@qwen35b` with generated `CUDA_VISIBLE_DEVICES="GPU-06645389-…"`
allocated 27 GB on **GPU-d751861a** (the other card), starving `qwen27b`
(`Free memory on device cuda:0 (3.86/31.36 GiB)…`) into a crash loop. The worker
process env contained **no** `CUDA_VISIBLE_DEVICES` at all: vLLM's multiprocess
executor manages device assignment itself and only understands *numeric* CVD values —
a UUID string is dropped, workers see all GPUs, and rank 0 takes physical GPU 0.

**Fix:** the generator must translate GPU UUIDs to *indices* at render time (NVML
lookup, stable within a boot) and emit numeric `CUDA_VISIBLE_DEVICES` plus
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so index order matches NVML/nvidia-smi order. The
ledger/config stay UUID-based (spec Decision 6); only the rendered env translates.
Regenerate env on every router start (already the case) so indices refresh per boot.
Add a drift check: after readiness, verify via NVML that the unit's MainPID compute
process appears on exactly the assigned UUIDs — this turns wrong-GPU placement from
silent into `Failed` (would have caught this immediately).
**Interim (deployed):** hand-edited numeric CVD in both env files +
`CUDA_DEVICE_ORDER=PCI_BUS_ID` in the shared drop-in — NOTE: overwritten at every
router restart until the generator fix lands.

## F9 — Tool-calling flags half-allowlisted: agentic clients get 400s

**Severity: medium-high (breaks OpenAI-SDK agentic clients).**

`enable-auto-tool-choice` is in the generator's BOOL_ARGS allowlist but
`tool-call-parser` (VALUE_ARGS) is not — so tool calling cannot be enabled via
configuration at all, and any client sending `tool_choice: "auto"` receives vLLM's
400 (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`).
Observed live from the operator's chat client.

**Fix:** allowlist `tool-call-parser` and `reasoning-parser` (value-validated against
vLLM's known parser names or safe-charset), document that parsers are model-family
specific (hermes for Qwen; gpt-oss and Gemma need their own), and consider surfacing
tool-capability in the `/v1/models` entry (contract rule, F2). **Interim (deployed):**
`--enable-auto-tool-choice --tool-call-parser hermes` appended to both generated
units by hand — overwritten at every router restart.

**F8 root-cause correction (2026-07-29, later):** the primary defect is that the
generator emits `EnvironmentFile="/etc/vllm/<name>.env"` — systemd rejects the quoted
path (`EnvironmentFile= path is not absolute, ignoring`, warned at every daemon-reload
since first generation) and **silently loads no environment file at all**. Every
process therefore started with no `CUDA_VISIBLE_DEVICES` whatsoever; the UUID-vs-
numeric question was never actually exercised. Fix: emit the path unquoted (the
generator's exec-quoting was over-applied to a directive that must be bare), and add a
generator test asserting `systemd-analyze verify` passes on a rendered unit — that
check catches this class wholesale. Keep numeric CVD + `CUDA_DEVICE_ORDER=PCI_BUS_ID`
as the rendered form regardless (vLLM's own device bookkeeping is index-based), and
keep the post-readiness NVML placement check from the original F8 write-up — it would
have converted both variants of this failure from silent to loud. Interim: quotes
stripped by hand in both generated units (overwritten at router restart, as with the
other interim fixes).

## F10 — Overlapping failures orphan backoff recovery: service wedges in `failed` forever

**Severity: high (no automatic recovery; observed live).**

When multiple flights fail in close succession (here: a parked-demand flight and an
admin start, during the F8 chaos), each `record_failure` bumps the generation, and the
*earlier* `schedule_backoff` task's terminal `Failed → Stopped` update loses the
`update_if_current` race — by design for stale writers. But the *latest* failure's
backoff task can also be preempted the same way, leaving no pending recovery at all.
End state observed: unit `inactive`/`success`, no flight, no backoff task — and
lifecycle permanently `failed`, which `ensure_ready` rejects (`terminal_failure`) and
reconcile deliberately preserves (the H8 fix). The service is unreachable until an
operator issues an explicit `stop` (which does transition `failed → stopped`) and then
`start` — the escape hatch used live.

**Fix:** make failure recovery convergent rather than task-bound: reconcile should
transition `failed → stopped` when it observes systemd `inactive` **and** no flight
**and** no pending backoff for the current generation (that combination means the
failure is fully drained — preserving `failed` there serves nobody); alternatively,
`record_failure` re-arms a single keyed backoff task on every failure. Regression
test: two concurrent failing flights → the service reaches `stopped` automatically
within backoff-max.

**F9 amendment (2026-07-29, later):** the parser *value* is per-chat-template, not
merely per-family — Qwen documentation says `hermes`, but these Qwen3.6 checkpoints'
templates emit the XML function format requiring `qwen3_xml`, and thinking models
additionally need `--reasoning-parser qwen3` or tool parsing fails silently mid-stream
(observed live: streams ended at `</think>` with no tool_calls; `<think>` text leaked
into `content`). The config fix must therefore allowlist **both** `tool-call-parser`
and `reasoning-parser` with values validated against vLLM's registries (`--help=all`
enumerates tool parsers; `vllm.reasoning` registry for reasoning parsers), and the
runbook's Tier-3 smoke needs a tool-call + reasoning-separation assertion, which would
have caught both the wrong-parser and missing-parser variants.

## Bundled with the F5–F10 batch: short-park policy (spec Decision 3 amendment)

Not a defect — a decided policy change (2026-07-29, spec amended). With the batch:

- Config: `parking.deadline_ms` → ~20000 and `retry_after_seconds` → honest per-model
  startup estimates, in both live `/etc/vllm-router/router.yaml` services and
  `src/config/example.yaml`.
- Enhancement: the parking-expiry 503 body includes startup progress — service name,
  elapsed seconds, and a typical-duration estimate (from config or a rolling
  observation of recent start durations). Interactive clients display 503 message
  text, so this turns cold starts into visible progress instead of silent hangs.
- Test: parked request against a slow-starting mock exceeds the short deadline →
  503 with `Retry-After` and progress fields; the start continues; a follow-up
  request after readiness succeeds warm.

## F5–F10 implementation and verification (2026-07-29)

Implemented and landed in the working tree:

- **F5:** detached lifecycle and backoff work is spawned on the process-lifetime
  pool runtime, and the background observation tick is cancellation-safe while it
  is awaiting external I/O. Live testing exposed one additional shutdown cause:
  Pingora 0.8 waits the entire `graceful_shutdown_timeout_seconds` value after the
  grace period even when its tasks have exited. The deployed setting is therefore
  zero-after-grace (the existing 10-second grace remains available to active
  requests). A fixed-binary live `systemctl restart` completed in **10.00 s**;
  the host-only `binary_serves` SIGTERM regression completed in **0.28 s** with a
  zero-second test grace and a three-second bound.
- **F6:** each service retains its previous degraded reason and drift set. New,
  changed, cleared, and recovered conditions log at WARN; identical observations
  on later ticks log only at DEBUG. Metrics continue to expose the steady state.
- **F7/F8:** global and per-service allowlisted environment maps merge into the
  generated env files. Units use a bare `EnvironmentFile=` path; env files contain
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, numeric NVML UUID-to-index
  `CUDA_VISIBLE_DEVICES`, `PATH`, `HF_HOME`, and `VLLM_CACHE_ROOT`. A rendered-unit
  test passes `systemd-analyze verify`. Post-readiness placement checks all
  processes in the unit's MainPID-rooted process tree because vLLM's NVML-visible
  worker is a child of the API server, then requires its observed GPU UUID set to
  equal the assignment exactly.
- **F9:** `tool-call-parser` and `reasoning-parser` are string-valued runner
  arguments validated against the installed vLLM registry names. Both live
  services and the example use `qwen3_xml` + `qwen3` with automatic tool choice.
  The Tier-3 assertion checks a parsed `get_weather` call with a non-empty city,
  separated reasoning (`reasoning` in vLLM 0.26, with compatibility for
  `reasoning_content`), and no `<think>` leakage.
- **F10:** failure recovery is keyed by generation and re-armed for each failure;
  stale cleanup cannot clear newer intent. Reconciliation converges a fully
  inactive failed/backoff service with no current recovery to `stopped`. The
  two-overlapping-failures regression reaches `stopped` automatically.
- **Short parking:** the default/example deadline is 20 seconds. Live qwen27b and
  qwen35b use 20 seconds with honest `Retry-After` estimates of 150 and 120
  seconds, respectively. Expiry returns a 503 with service, elapsed seconds, and
  typical duration while the detached start continues. The regression verifies
  that a later request succeeds warm without launching a second start.

Acceptance outcomes:

- `just lint` — clean.
- `just test` — **143 passed, 2 host-only ignored** in the library target and
  **4 passed** in the binary target; no failures.
- `cargo test --test binary_serves -- --ignored --exact --nocapture` — passed.
- The ignored live generator comparison confirmed both rendered units are
  byte-equivalent to the hand-corrected `/run/systemd/system/` references, and
  live `systemd-analyze verify` completed without diagnostics.
- The generated env files map qwen27b to GPU index 0 and qwen35b to index 1.
  Both units are active; NVML reports each worker only on its assigned UUID; router
  readiness reports two routable services.
- Tier 3 passed model discovery, alias-routed completions for both services, and
  the strengthened tool-call/reasoning assertion on both services.
- The obsolete `10-toolchain.conf` hand-patch was removed from systemd and retained
  recoverably as `/tmp/10-toolchain.conf.review3-backup`. The router supervisor is
  enabled for boot.
