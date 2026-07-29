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

Direct inference while asleep was accepted but returned no bytes before a
2-second client timeout in every cycle. This measured vLLM 0.26 behavior means
the router must fail closed on fresh sleep/readiness facts, wake and reprobe
before forwarding, and bound a post-admission stall with its inter-chunk idle
timeout. This topology remains ineligible for `hardware_validated: true`
because its measured L1 host-RAM use exceeded the declared budget.

## TP=2

Initialization failed during vLLM's profile run in the FlashInfer NVFP4
quantizer:

```text
AssertionError: last dim has to be multiple of 16, but got 2152
```

No sleep cycles were possible. Keep sleep disabled and use stop/start.

This failure is specific to the tested checkpoint/runtime cell, so the router
does not impose a global NVFP4 restriction. Its service definition must record
the failed degree and select TP=1:

```yaml
placement:
  tensor_parallelism: 1
compatibility:
  minimum_l1_host_ram_bytes: 40000000000
  unsupported_tensor_parallelism: [2]
```

Configuration validation rejects a later change to TP=2 while this compatibility
declaration is present. Removing the declaration requires a fresh successful
hardware validation of this exact checkpoint, vLLM/FlashInfer version, and host.

The provisional L1 declaration was 30 GB and the maximum observed
`MemoryCurrent` was approximately 39.0 GB, a 1.30× measured ratio. The
40,000,000,000-byte floor above rounds that to a conservative 1.33× budget with
measurement margin. It is retained while `sleep.enabled` and
`sleep.hardware_validated` remain false. If L1 is eventually enabled,
`sleep.l1_host_ram_bytes` must meet this floor.
