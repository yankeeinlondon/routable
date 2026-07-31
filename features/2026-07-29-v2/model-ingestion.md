# Model Ingestion

Status: **in progress — being populated collaboratively.** Producer of the capability data
consumed by [`sleep-trait.md`](sleep-trait.md) and by [v2 §4 (dynamic
scheduling)](spec.md). Formalizes what [V2-S1](spec.md) treated as a one-off spike.

## Problem

The router schedules GPU memory using numbers a human typed into YAML. Those numbers are
not measurements, and in the deployed configuration they carry no model-specific
information at all.

**Observed, in the live `/etc/vllm-router/router.yaml`:** all four services — a 27B dense
NVFP4 model, a 35B mixture-of-experts, a 20B, and a 31B — declare the identical value

```yaml
vram_reservation_bytes_per_gpu: 31000000000
gpu_memory_utilization: 0.9
```

against a card of `vram_capacity_bytes: 34359738368`. That capacity times the utilization
target is 30,923,764,531 — so the declared reservation is the *utilization knob*, rounded
up. It says what vLLM was told to claim. It says nothing about what any of these models
needs.

This is tolerable in v1 because the one-awake-owner-per-GPU rule means only one model's
number ever has to be right, and it is right by construction. It fails the moment
placement has to answer "does B also fit alongside A," which is precisely what [v2
§4](spec.md) requires.

The same pattern appears in three other places:

- **`sleep.enabled` / `hardware_validated`** ask an operator to attest by hand to something
  only measurement establishes. A field that can be a lie should not be hand-authored.
- **`docs/{model-id}.md`** are hand-written prose recording measured facts —
  `docs/qwen3.6-27b-nvfp4.md` and `docs/meta-llama-3.1-8b-instruct-fp8.md` contain exactly
  the numbers a scheduler needs, in a form nothing can read.
- **`compatibility.minimum_l1_host_ram_bytes`** and
  `compatibility.unsupported_tensor_parallelism` are measured outputs that must be manually
  transcribed into config. Neither measured floor has actually been transcribed into the
  live configuration, so nothing currently enforces them.

### The recorded sleep failures were ingestion failures

Of the three reasons no service on this host is cleared for sleep:

| Recorded blocker | Actually |
|---|---|
| L1 host RAM measured 39.0 GB against a 30 GB declaration | A guess was wrong. Measurement would have produced 39.0 GB and there would be nothing to fail |
| Tensor-parallel degree 2 with NVFP4 dies in FlashInfer's FP4 quantizer | Capability discovery. One launch attempt finds it |
| A sleeping backend accepts a request and stalls rather than returning 503 | Not a capability limit. The router already handles this by failing closed on stale facts, waking and reprobing before forwarding, and bounding a post-admission stall with the inter-chunk idle timeout |

Two of three dissolve given a measurement pipeline. The third was never a blocker.

## What ingestion is

A gated pipeline that a model passes **before it can be configured for service**,
producing a machine-written capability record. Its outputs are facts about a specific
combination, never operator preferences.

The unit of measurement is the tuple:

> (checkpoint, runner + version, host, topology, operating point)

Change any element and the record does not transfer.

## What it must produce

> Draft. This is the shared data contract; `sleep-trait.md` reads it and must not redefine
> it.

**Identity** — ties to [`model-data/spec.md`](../2026-07-29-model-data/spec.md): the
operator-authored public id, the upstream repository for provenance, the pinned snapshot.

**Viability** — does this launch at all, at each topology attempted? Records the failure
verbatim when it does not, which is what populates
`compatibility.unsupported_tensor_parallelism`.

**Memory requirement**, decomposed rather than a single scalar:

- weights
- KV cache at the operating point
- fixed per-device overhead that does not split

[V2-S6](spikes/V2-S6.md) gap 7 established the need for this decomposition: `--tensor-split`
fractions alone cannot satisfy a byte ledger or an asymmetric-fit guarantee. A single
number is not sufficient for co-residency arithmetic.

**The suspend ladder** — for each rung the runner offers:

- retained GPU bytes
- retained host bytes
- cost to enter
- cost to restore
- output stability across a suspend/restore cycle

For vLLM that is three rungs; for llama.cpp, one. This is the table
[`sleep-trait.md`](sleep-trait.md) selects from.

**Effective endpoint capability set** — [V2-S6](spikes/V2-S6.md) gap 8: several
router-allowlisted paths return 404/501 on llama.cpp, and embeddings/reranking vary by
launch flag and by model. Replicas sharing a served model need compatible sets.

**Provenance** — checkpoint hash, runner version, driver/CUDA version, host identity,
measurement date. Without this the record cannot be invalidated correctly.

## Invalidation

A record is void when any element of its tuple changes: a different checkpoint snapshot, a
runner upgrade, a driver or CUDA change, a different topology, or a different operating
point. The validator should refuse a configuration whose record does not match the
configuration it is being applied to, rather than silently using stale numbers.

## Open questions

These are the decisions this document exists to settle.

- **Is ingestion blocking?** Must a model be measured before it can serve at all, or is
  there a degraded mode — serve, but ineligible for co-residency and suspension — that lets
  someone try a model without a measurement run first? Blocking is safer and matches the
  argument that capability must never be hand-asserted; non-blocking is far friendlier for
  evaluating a new checkpoint.
- **Who runs it, and when?** A `vllm-router ingest <model>` subcommand run by an operator,
  a first-use path inside the router, or an offline harness? The router owning it means the
  router needs exclusive GPU access mid-operation, which conflicts with serving.
- **How is the operating point chosen?** Memory requirement is a *function* of
  `max_model_len` and `max_num_seqs`, not a constant — the live config uses 32768 and 32.
  Does ingestion measure at one configured point, at several points to fit a curve, or
  record a formula? Measuring one point is cheap and brittle; a curve is expensive and
  reusable.
- **Where does the record live?** A generated file per model committed to the repo, a
  runtime state directory, or embedded in config? The `docs/{model-id}.md` files suggest
  the repo, but generated artifacts and hand-written documentation should probably not be
  the same file.
- **How long is a full run, and is that acceptable?** Five suspend/resume cycles per rung
  per topology, plus launch attempts, plus warm-up. On multi-minute cold starts this is
  plausibly tens of minutes per model.
- **What happens to the existing hand-authored fields?** `vram_reservation_bytes_per_gpu`,
  `minimum_l1_host_ram_bytes`, `unsupported_tensor_parallelism`, `sleep.*` — deleted and
  sourced from the record, or retained as operator overrides that the validator
  cross-checks against it?
- **Does measurement need an idle host?** Co-resident load perturbs both timing and
  page-cache state ([V2-S6](spikes/V2-S6.md) gap 10). If ingestion requires quiescence,
  that is an operational constraint worth stating explicitly.

## Relationship to existing work

- **[V2-S1](spec.md)** asks for per-profile ground truth as planner input data. This
  document argues that should be a standing pipeline rather than a one-time spike.
- **[v2 §4](spec.md)** has services declare an ordered list of pre-validated placement
  profiles. Under this proposal profiles remain, but become *measured outputs* rather than
  operator declarations — which is a change to §4 and needs to be reconciled.
- **[`model-data/spec.md`](../2026-07-29-model-data/spec.md)** defines what a model *is*
  (identity, naming, provenance). This defines what a model *can do* (capability, cost).
  Natural siblings; the identity fields here should not be redefined.
