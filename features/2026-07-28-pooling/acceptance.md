# Acceptance status

Automated verification recorded 2026-07-28:

- `cargo fmt --all -- --check`: pass
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo test --all-targets --all-features`: 36 pass, 1 ignored, 0 fail
- two-file `-t` configuration check: pass

The fast suite covers strict startup configuration, model aliases and invalid
request rejection, generation-safe observed state, exact active/parked guards,
50-way single-flight start, shared failure fanout, FIFO contention, safe
priority/LRU eviction, drain deadlines, idle sleep tiers, least-inflight
admission, probe threshold/age behavior, vLLM readiness/model verification,
bounded HTTP calls, OpenAI warmup, and the vLLM 0.26 three-step L2 wake.

The custom Pingora ingress additionally enforces bounded body reads before
lifecycle admission, a public endpoint allowlist, hop-by-hop stripping,
loopback Host rewriting, request/trace IDs, global/per-model parking caps,
disconnect-aware parking, connect-only failover, inter-chunk response timeout,
and incremental upstream-to-downstream body writes.

## Pending on CT 107

These gates require hardware/services absent from this execution environment
and remain deliberately unclaimed:

- real vLLM exact-body and incremental SSE drill;
- the sm_120 TP/quantization/sleep validation matrix;
- live systemd D-Bus start/stop/crash observation;
- `kill -9`, SIGSTOP, NVML-staleness, router-restart, and graceful-upgrade chaos
  drills;
- measured metrics under sustained SSE/stampede load.

Sleep stays disabled unless a service explicitly records
`hardware_validated: true`; stop/start is the default production path until
the matrix in `phase0-findings.md` passes.
