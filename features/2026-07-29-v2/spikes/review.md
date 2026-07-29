# Section 7 orchestrated spike review

Date: 2026-07-29 (UTC)

Each V2-S1 through V2-S6 spike was delegated independently, then reviewed against
section 7's stated question and exit condition. Measurements had to be reproducible
host evidence; blocked cells remain blank rather than being estimated.

| Spike | Review disposition | Exit state | Planning consequence |
|---|---|---|---|
| [V2-S1](V2-S1.md) | Accepted after follow-up added public metric documentation and distinct vLLM load/compile extraction | Open | The router now records systemd→active, active→ready, and HTTP warmup histograms. Do not assume asymmetric PP=2 until the eight TP=1/PP=2 cells run. |
| [V2-S2](V2-S2.md) | Accepted | Open | The proposed qwen27b + gpt-oss pair is not packable before usable KV. MPS control works in the LXC, but vLLM attachment and curves are unproven. Keep `max_awake_owners_per_gpu: 1`. |
| [V2-S3](V2-S3.md) | Accepted | Open; conditional runner go | The HTTP/lifecycle shape is viable. Qwen-Image's transformer needs component-level Accelerate dispatch; pipeline-level balanced placement cannot split it. Full load, VRAM, stop-release, and generation latency still gate runner #2. |
| [V2-S4](V2-S4.md) | Accepted | Decision exited | Add-only registration is tractable but is a medium coordinated transaction, estimated at 880–1,460 gross changed lines before persistence/UI. Reject alias expansion on existing canonicals until request-body canonicalization exists. |
| [V2-S5](V2-S5.md) | Accepted after follow-up replaced teardown-contaminated samples with a request-driven resident window | Decision exited | Direct in-LXC power-limit writes are cut. On GPU 0, stopped idle averaged 12.309 W and gpt-oss resident idle averaged 51.168 W in the recorded 20-second windows. Lifecycle stopping remains the available stewardship mechanism. |
| [V2-S6](V2-S6.md) | Accepted after temporary fixture cleanup | Deferred; CUDA split open | Pinned llama.cpp b10176 satisfies health/models/slots and serves more routes than assumed. The 101 MiB CPU fixture restarted warm in 133–141 ms, but actual CUDA `--tensor-split` placement remains unverified. |

## Accepted code-spike changes

- S1 adds
  `vllm_router_start_phase_duration_seconds{service,phase,result}` and cold-start
  success/failure coverage without mixing wake paths into the histogram.
- S4 makes `ConfigStore` publish immutable validated snapshots and makes the
  data-plane parking registry additive/lazy. It deliberately does not expose a
  registration endpoint or pretend that pool/runner/systemd coordination is done.
- S3 leaves its minimal Qwen-Image HTTP server, mock contract tests, and systemd
  prototype under `V2-S3/`; it does not alter the production runner.

## Additional risk found during review

A management-plane start does not refresh a service's activity timestamp. If the
stored idle duration already exceeds `stop_after`, the idle reaper can stop the
newly started service on its next tick. S5 reproduced this twice; a normal data-plane
request avoids it because parked/active admission refreshes activity. Treat this as a
v1 lifecycle bug to fix independently of US9.

## Remaining pre-v2 gates

1. Run S1's eight controlled profile cells and decide whether vLLM placement is
   asymmetric-capable or symmetric-only.
2. Select a genuinely packable S2 pair, prove both workers attach under MPS (or record
   the runtime failure), and collect the full interference curves.
3. Run the complete S3 Qwen-Image pipeline on idle GPUs with a jointly budgeted
   component map and record load, ready/peak VRAM, generation latency, and stop
   release.
4. If llama.cpp remains a candidate after runner priority is settled, run S6's pinned
   CUDA build/GGUF matrix for `3,1`, `1,3`, and `1,1` tensor splits.
