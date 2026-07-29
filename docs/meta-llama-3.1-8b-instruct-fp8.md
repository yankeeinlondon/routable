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

Both failed the asleep-inference criterion: every direct request to a sleeping
backend returned no bytes before the 2-second client timeout instead of a 503.
Neither topology is eligible for `hardware_validated: true`; use stop/start.
