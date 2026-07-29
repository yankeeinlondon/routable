# Qwen3.6-27B NVFP4 sleep-mode validation

Measured on 2026-07-29 on an RTX 5090 (sm_120), vLLM 0.26.0. Checkpoint:
`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`, pinned snapshot
`9b5389d4a1e207daab2d47732efea57d7e946dcf`.

## TP=1

Five L1 and five L2 cycles produced the same fixed-prompt output after every
wake. L1 sleep was approximately 1.02 s and wake approximately 3.03 s. L2
sleep was approximately 1.02 s and wake approximately 5.05 s. Sleeping
residual VRAM was 1,414 MiB. `MemoryCurrent` remained approximately
38.8–39.0 GB, above the provisional 30 GB L1 host-RAM declaration.

L2 wake used all three required calls: weights wake, `collective_rpc`
`reload_weights`, and KV-cache wake. L1→L2 promotion also succeeded. An
in-flight request prevented sleep until the 10-second drain deadline and the
router resumed `running`, preserving the request.

Direct inference while asleep did not return the required 503; it returned no
bytes before a 2-second client timeout in every cycle. Consequently this
topology is not hardware validated.

## TP=2

Initialization failed during vLLM's profile run in the FlashInfer NVFP4
quantizer:

```text
AssertionError: last dim has to be multiple of 16, but got 2152
```

No sleep cycles were possible. Keep sleep disabled and use stop/start.
