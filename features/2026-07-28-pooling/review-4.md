# Implementation Review 4 — Tier 4/5 defects (opened 2026-07-29)

Tier 4 and Tier 5 ran on CT 107 and **failed overall**; full drill-by-drill detail
lives in `acceptance.md` (§ On-box Tier 4/5 run) and the sleep matrix in
`phase0-findings.md`. Numbered findings for the fix cycle, severity-ordered:

- **F11 (critical) — busy eviction double-placed a GPU:** requesting `gemma31b`
  while `qwen35b` had eight active requests launched **both units on GPU 1**,
  violating busy-victim protection and the one-awake-owner invariant. Placement
  safety is the router's core promise; this is the top fix.
- **F12 (high) — force-stop not durable and audits `affected_requests: 0`** despite
  eight active requests (spec acceptance criterion explicitly requires the count).
- **F13 (high) — graceful upgrade kills long SSE streams:** listener handoff works
  but the old process terminates streams at the 10 s grace period; config must
  align `graceful_shutdown` with expected stream duration or drain differently
  (spec: streams survive the handoff).
- **F14 (high) — sleeping vLLM backends hang instead of returning 503:** contradicts
  the researched behavior our wake-then-forward design assumed; router must bound
  this (it does via inter-chunk timeout — but the drill shows requests reached a
  sleeping backend at all, which admission should prevent).
- **F15 (medium) — NVFP4 TP=2 init fails** (FlashInfer dimension assertion) — the
  predicted #21336-class risk confirmed; sleep cells for that combination blocked.
- **F16 (medium) — FP8/NVFP4 L1 sleep exceeds declared host-RAM budgets:** measured
  pinned RAM above weight-size declarations; budget validator inputs need the
  measured multipliers from `docs/meta-llama-3.1-8b-instruct-fp8.md` and
  `docs/qwen3.6-27b-nvfp4.md`.
- **F17 (low) — audit gaps and noise:** no `joined_existing` audit records on
  stampede joins; SIGSTOP stall logs `error decoding response body` instead of a
  clear timeout classification.

Passing behaviors (for scope): stampede single-flight, parking caps, disconnect
cleanup, crash recovery, restart reconciliation, repeatable sleep/wake output,
safe drain refusal, L1→L2 promotion, three-step L2 wake. `hardware_validated`
remains **false** for all services pending F14–F16.
