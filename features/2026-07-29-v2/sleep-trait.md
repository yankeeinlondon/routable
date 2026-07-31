# Suspend and Resume as Router Semantics

Status: **in progress — being populated collaboratively.** Expands [v2 §1
(additional runners)](spec.md) and constrains [v2 §4 (dynamic
scheduling)](spec.md). Consumes the capability data defined in
[`model-ingestion.md`](model-ingestion.md); that document is the sole authority on how
those measurements are produced.

## Problem

"Sleep" is currently vLLM's word, and the router repeats it verbatim all the way out to
its own public API. `POST /pool/v1/services/{name}/sleep` accepts `{"level": "l1"|"l2"}`,
parsed straight off the wire in `api/control.rs:213-224`, and `SleepOptions.level`
(`pool/mod.rs:142`) carries that choice into the lifecycle engine.

The consequence is that **the caller decides the mechanism**. The pool — the one component
that knows current memory pressure, what else is contending for the GPU, and what each
option costs — never makes that decision. It executes an instruction.

This shows up three ways:

1. **The public API leaks a runner's vocabulary.** `l1`/`l2` are meaningful only for vLLM.
   A llama.cpp service cannot answer the question the API asks, because
   [V2-S6](spikes/V2-S6.md) measured `POST /sleep?level=1` returning `404` there.
2. **The idle configuration hard-codes vLLM's ladder.** `sleep_l1_after_ms`,
   `sleep_l2_after_ms`, and `stop_after_ms` name three specific mechanisms, so a runner
   with a different number of rungs has no valid way to express its policy.
3. **Capability and policy are conflated** in `sleep.enabled` / `hardware_validated`,
   letting an operator assert by hand something only measurement can establish. See
   [`model-ingestion.md`](model-ingestion.md).

What is **not** a problem, and should be preserved: the two trigger paths already converge.
`PoolManager::sleep` (`pool/mod.rs:2139`) and the idle reaper (`reap_idle_once`, calls at
`pool/mod.rs:981` and `:1000`) both reach `sleep_flight`, sharing the single-flight slot,
the drain-first guarantee, and generation guards. The plumbing is right; the signature is
wrong.

## The semantic model

Suspend and resume are **router verbs describing intent**, not passthroughs to a runner
endpoint. The router's `/sleep` is not vLLM's `/sleep`.

- **Suspend** expresses: *this model is not needed immediately, though it may be needed
  again soon.*
- **Resume** expresses: *this model is needed as soon as possible* — issued ahead of
  demand, as a pre-warm.

Both are meaningful for every runner, because both describe an operator's or a policy's
belief about near-future demand. That belief exists independently of whether the
underlying runtime happens to expose an endpoint called "sleep." How the intent is
satisfied is a **mapping concern**, resolved per runner, and invisible to the caller.

vLLM's L1 and L2 stop being API surface and become two rungs the router may select. They
remain semantically similar to each other — both say "not needed right now" — and differ
only in how strongly they commit to that belief and what they cost to reverse.

### Intent is the interface; measured cost is the implementation

The caller states *when it expects to need the model again*. The router chooses the
mechanism using:

- the rungs this service's runner actually offers, with measured cost to enter, cost to
  restore, and retained GPU/host bytes (from [`model-ingestion.md`](model-ingestion.md));
- how much memory needs reclaiming right now, and for whom;
- the expectation the caller expressed, or that idle time implies.

The router then **reports the mechanism it selected**. An operator debugging memory needs
to know a service is at a weights-discarded rung rather than a weights-retained one, even
though no caller had to ask for it.

### Idle policy is the same statement, made by the router

An explicit suspend call and an idle timeout are not two features. Idle duration is
*evidence* for exactly the intent a caller states explicitly: the longer a model goes
untouched, the stronger the belief it is not needed imminently. The reaper should
therefore generate the same intent internally and take the same path — which it already
does structurally, and would then also do semantically.

This reframes the idle configuration from *three named mechanisms* to *how confidence
decays with idleness*, letting the same policy express itself on a runner with one rung or
three.

## The runner ladder

Every runner declares an ordered ladder of suspended states. Two properties hold
universally:

- **Stop/start is the universal floor.** Every runner has it by construction, which is
  what makes suspend meaningful everywhere without requiring every runner to implement a
  sleep endpoint.
- **Rungs are ordered by the trade they make**: cheaper to hold, costlier to restore.

Known ladders:

| Runner | Rungs, ordered | Notes |
|---|---|---|
| vLLM | L1 (weights → host RAM) → L2 (weights discarded) → stopped | `--enable-sleep-mode` at launch; dev-mode endpoints |
| llama.cpp | stopped | `/sleep` and `/is_sleeping` return `404` ([V2-S6](spikes/V2-S6.md)); relies on mmap/page-cache for fast restart |

**Do not assume llama.cpp's floor is cheap.** V2-S6 measured 133–141 ms launch-to-health
but explicitly disclaims it as production economics: a 101 MiB, 135M-parameter, CPU-only
model with warm page cache. A production GGUF is orders of magnitude larger, and
page-cache pressure from co-resident models is part of the real cost. The router must not
encode an assumption that any particular runner is fast — it reads measured cost and
decides.

## Proposed shape

> Draft. The naming below is deliberately provisional.

```rust
pub enum SuspendIntent {
    /// Not needed immediately; may be needed again soon.
    NotNeededSoon { expected_idle: Option<Duration> },
    /// Reclaim this model's resources; restoring may be expensive.
    Reclaim,
}

pub struct SuspendOptions {
    pub intent: SuspendIntent,
    pub deadline: Instant,
    /// After the drain deadline, force rather than abandoning the transition.
    pub force: bool,
}
```

`SleepOptions.level` is **deleted, not deprecated** — a mechanism field on a semantic
call is the defect.

## Open questions

- **Naming.** `sleep`/`wake` are established in the API and understood by operators;
  `suspend`/`resume` are more accurate for a multi-runner ladder. Is the compatibility
  cost worth the precision? Does the HTTP verb need to match the internal trait name?
- **How intent is expressed on the wire.** A duration hint (`expected_idle`), a small
  named enum, or nothing at all with the router inferring purely from memory pressure?
- **Does an explicit caller ever get to force a rung?** An operator running a deliberate
  experiment has a real need the semantic API denies by design. Is that an escape hatch,
  or is that what the existing `stop` verb is for?
- **Resume-as-pre-warm.** Distinct enough from demand-driven waking to keep as its own
  verb, or a flag on the same one?
- **Idle policy expression.** What replaces the three `*_after_ms` keys? A decay curve, an
  ordered list of (idle duration → intent) pairs, or a single "reclaim after" with the
  router filling intermediate rungs opportunistically?
- **Reporting.** Does the selected rung appear in `GET /pool/v1/services` observed state,
  in metrics labels, or both? `vllm_router_service_state` currently one-hots
  `asleep_l1`/`asleep_l2` — runner-specific state names in a metric have the same leak
  problem as the API.
- **Backward compatibility.** v1 clients send `{"level":"l1"}`. Accept and translate,
  reject, or accept-with-deprecation?

## Non-goals

- Changing the single-flight, drain-first, or generation-guard semantics. Those are
  correct and stay.
- Awake co-residency. Suspended residuals sharing a GPU is v1 behavior and continues;
  benchmarked awake co-residency remains [v2 §4](spec.md).
