# Meta-Llama-3.1-8B-Instruct FP8 sleep-mode validation

Measured on 2026-07-29 on two RTX 5090s (sm_120), vLLM 0.26.0. Checkpoint:
`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`, pinned snapshot
`12fd6884d2585dd4d020373e7f39f74507b31866`.

## TP=1

Five L1 and five L2 cycles produced identical fixed-prompt output within the
cell. L1 sleep and wake were approximately 0.27 s. L2 sleep was approximately
0.27 s and wake 3.095–3.098 s. Sleeping residual VRAM was 944 MiB and
`MemoryCurrent` approximately 24.53 GB, above the provisional 20 GB L1 host-RAM
declaration. Warm vLLM allocation was approximately 29.5 GB with
`gpu_memory_utilization: 0.85`.

## TP=2

Five L1 and five L2 cycles produced identical fixed-prompt output within the
cell. L1 sleep was 0.27–1.04 s (the first cycle was the outlier) and wake
2.32–2.33 s. L2 sleep was approximately 0.27 s and wake 2.84–3.10 s. Sleeping
residual VRAM was 1,098 MiB per GPU and `MemoryCurrent` approximately 20.22 GB,
just above the provisional 20 GB declaration. Warm allocation was
approximately 29.4 GB per GPU.

## Common result

Both topologies passed fixed-output recovery, safe 10-second refusal with a
7,000-token request active, L1→L2 promotion, and the required L2 weights
wake/`collective_rpc`/KV-cache wake sequence.

Every direct request to a sleeping backend was accepted but returned no bytes
before the 2-second client timeout. This characterizes the tested vLLM 0.26
behavior rather than providing a backend safety barrier: the router must fail
closed on fresh sleep/readiness facts, wake and reprobe before forwarding, and
bound a post-admission stall with its inter-chunk idle timeout. Neither topology
is eligible for `hardware_validated: true` because measured L1 host-RAM use
exceeded the declared budget; use stop/start.

## Budget declaration

The maximum measured ratio was 24.53 / 20 = 1.2265 at TP=1. Round this up to a
1.25× conservative multiplier and declare a 25,000,000,000-byte measured floor
for either topology:

```yaml
sleep:
  enabled: false
  hardware_validated: false
compatibility:
  minimum_l1_host_ram_bytes: 25000000000
```

The floor records the completed memory measurement without enabling sleep. If
the remaining hardware-validation failures are resolved and L1 is enabled,
`sleep.l1_host_ram_bytes` must be at least 25,000,000,000 or configuration
validation rejects the service.
