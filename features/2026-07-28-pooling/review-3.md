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
