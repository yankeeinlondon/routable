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

Note for v2: llama.cpp's `--alias` takes a single name, so multi-alias services are a
per-runner capability — validation should reject what a runner cannot honor
(v2 spec §1 complexity register pattern).
