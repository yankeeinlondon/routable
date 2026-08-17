# Host and per-model hardware optimization for LLM inference

This document records the host-level investigation performed on the AMD EPYC
9175F and dual RTX 5090 host on 2026-08-17. It also defines a repeatable process
for qualifying a new host and ingesting models for vLLM and llama.cpp.

The most important result is counterintuitive: on this host and the tested
llama.cpp revision, the native `-march=native` ggml CPU device is faster than
the optional ZenDNN device for BF16, Q8_0, and Q4_K_M. ZenDNN should remain an
opt-in comparison, not the default llama.cpp device. vLLM is different: its
ZenTorch integration works and should be used for BF16 serving.

## Recommended configuration on this host

| Runtime | Recommended baseline |
| --- | --- |
| Qwen3.8-27B-FP8 on CUDA | Both RTX 5090s, TP=2/PP=1, native 262,144-token context, BF16 KV cache by default, `max-num-seqs=32`, `max-num-batched-tokens=8192`, FlashInfer attention |
| vLLM online serving | CPU wheel with `vllm[zen]`, BF16, TP=1, 15 physical cores for inference, one physical core reserved for the front end |
| vLLM CPU set in this LXC | Process tree: `6,7,10-21,24,25`; OpenMP inference: `6,7,10-21,24` |
| vLLM allocation | Preload bundled tcmalloc and Intel OpenMP; size `VLLM_CPU_KVCACHE_SPACE` per model and workload |
| llama.cpp default | Native build, device `none`, 16 physical cores, strict affinity |
| llama.cpp quantization | Q4_K_M for best single-stream decode of the tested options; Q8_0 when prompt throughput or quality is more important |
| llama.cpp ZenDNN | Build it for A/B tests, use `ZENDNNL_MATMUL_ALGO=1`, but do not select it by default on the current revision |
| SMT | Exclude it initially; it reduced performance in the 4B llama.cpp tests |
| NUMA | Retain NPS1 and TP=1 until a controlled BIOS-level NPS1/NPS4 comparison is run |
| Frequency policy | No change needed: the visible governor says `powersave`, but `amd-pstate-epp` has EPP set to `performance` and boost is enabled |

These settings are a starting point, not a substitute for per-model ingestion.
Model architecture, quantization, context length, concurrency, latency SLO, and
runtime revision can all move the optimum.

## What was installed

- llama.cpp source: `/opt/llama.cpp`
- llama.cpp revision: `7c35571e5dd2e577eddaca7e0ffa7cf46a54fcf2`
  (build 10463)
- native llama.cpp build: `/opt/llama.cpp/build-native`
- ZenDNN llama.cpp build: `/opt/llama.cpp/build-zen`
- default native commands: `/usr/local/bin/llama-server`, `llama-bench`,
  `llama-cli`, and `llama-quantize`
- explicit comparison commands: `llama-server-zendnn`,
  `llama-bench-zendnn`, and `llama-cli-zendnn`
- isolated vLLM CPU environment: `/opt/vllm-cpu/.venv`
- vLLM `0.26.0+cpu`, PyTorch `2.11.0+cpu`, ZenTorch `2.11.0.0`, and
  Intel OpenMP `2024.2.1`
- Qwen3.8-27B-FP8 snapshot `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
  under `/models/hf/hub`, verified against the Hub manifest
- CUDA vLLM `0.26.0`, PyTorch `2.11.0+cu130`, FlashInfer `0.6.14`,
  Transformers `5.14.1`, and fastokens `0.3.1`

The original CPU experiments ran while the CUDA services were live, so those
results are suitable for choosing large configuration differences but should
not be treated as publication-quality peak numbers. The Qwen3.8 GPU matrix was
run with the other GPU services stopped and under the host experiment lock.

## Host inventory

### Hardware and execution boundary

| Property | Observed value |
| --- | --- |
| CPU | AMD EPYC 9175F, Zen 5 |
| Topology | 1 socket, 16 physical cores, 32 hardware threads |
| Cache | 512 MiB L3 reported as 16 distinct 32 MiB LLC instances |
| NUMA | one node (NPS1) |
| ISA | AVX-512, AVX512-BF16, AVX512-VNNI and AVX-VNNI present |
| Native compiler target | GCC resolves `-march=native` and `-mtune=native` to `znver5` |
| Visible memory | 187.5 GiB, no swap |
| Environment | unprivileged Proxmox LXC, Ubuntu 26.04 userspace |
| Effective container CPU set | `6-7,10-31` |
| GPU locality | both GPUs report NUMA node 0 and the same visible CPU affinity |
| GPUs | 2 x GeForce RTX 5090, 32 GiB each, Blackwell `sm_120` |
| GPU link | PCIe/NUMA-local; CUDA peer access is unavailable between the GPUs |
| Transparent huge pages | `madvise` |
| Performance counters | unavailable in the guest (`perf_event_paranoid=4`) |
| Locked-memory limit | 8 MiB |

The [AMD product page](https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9175f.html)
lists 16 cores, 512 MiB L3, 12 memory channels, and 614 GB/s theoretical
socket bandwidth. The live topology is more useful than the product sheet for
affinity decisions.

Each physical core reports a different L3 cache ID and only its SMT sibling
shares that ID. This is consistent with one active core in each of 16 CCD-sized
LLC domains. It provides a large, mostly uncontended LLC per core but makes
cross-core communication cross an LLC boundary. That is one reason thread
scaling must be measured on this unusual SKU.

### Container CPU IDs are not bare-metal CPU IDs

The LXC does not expose logical CPUs `0-5` or `8-9`. A one-thread-per-core set
must therefore select the visible sibling for those physical cores:

```text
physical cores represented once: 6,7,10-21,24,25
additional visible SMT siblings:  22,23,26-31
```

On an unrestricted bare-metal instance the equivalent conventional sets would
usually be `0-15` and `16-31`, but never hard-code that assumption. Inspect the
effective cgroup CPU set and sibling lists inside the actual service boundary:

```bash
cat /sys/fs/cgroup/cpuset.cpus.effective
lscpu -e=CPU,NODE,SOCKET,CORE,CACHE,ONLINE
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    test -r "$cpu/topology/thread_siblings_list" || continue
    printf '%s ' "${cpu##*cpu}"
    cat "$cpu/topology/thread_siblings_list"
done
numactl --hardware
```

Use exactly one *allowed* logical CPU for every unique `(socket, core)` pair.
Do not assume that the lower-numbered sibling is available in a container.

### Frequency policy

The sysfs governor is `powersave`, but this host uses `amd-pstate-epp` and every
visible CPU has `energy_performance_preference=performance`; boost is enabled.
With this driver, changing only the governor name can be misleading. Record all
three values and measure frequency under load before changing policy:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
cat /sys/devices/system/cpu/cpufreq/boost
```

## Qwen3.8-27B-FP8 GPU ingestion

### Pinned artifact and architecture

The deployed artifact is
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) at commit
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. The 81-file snapshot is 28.76 GiB,
including 66 weight shards, and is stored at:

```text
/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
```

Verify the local snapshot rather than trusting the existence of the directory:

```bash
/root/.venv/bin/hf cache verify Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --cache-dir /models/hf/hub
```

This is a dense 64-layer multimodal model with a repeating three Gated
DeltaNet linear-attention layers plus one full-attention layer. It therefore has
48 recurrent GDN layers and 16 full-attention layers. The full-attention path
uses 24 query heads and four KV heads; the GDN path uses 16 key and 48 value
heads with 128-wide state. All relevant head counts divide cleanly at TP=2.

These details change which knobs matter:

- The 28.76 GiB checkpoint does not leave credible production headroom on one
  32 GiB GPU after activations, CUDA graphs, encoder workspaces, and cache.
  TP=2 is required; CPU weight offload is unnecessary and would turn the PCIe
  link into a weight-streaming bottleneck.
- Full-attention KV at TP=2 costs 32 KiB/token/GPU in BF16 or 16 KiB/token/GPU
  in FP8. The 48 GDN layers also require about 73.4 MiB of FP32 recurrent and
  BF16 convolution state per active sequence per GPU. At `max-num-seqs=32`,
  that fixed recurrent state alone is about 2.29 GiB/GPU. For this model,
  `max-num-seqs` is a memory control, not merely a scheduler preference.
- Native context is 262,144 tokens. Both tested KV formats fit at least one full
  context, so there is no reason to reduce the advertised limit for the default
  service.
- The GPUs do not have CUDA peer access. TP collectives stage through the PCIe
  root complex and host memory; the EPYC's bandwidth helps, but does not remove
  that tax. Set `NCCL_P2P_DISABLE=1` and disable vLLM's custom all-reduce so the
  selected transport is explicit and repeatable.

### Qualified production profile

The default is the full multimodal model with BF16 KV cache. BF16 was selected
because it preserved native context, matched FP8 throughput in the controlled
matrix, and avoids the uncalibrated KV-scale warning emitted for this
checkpoint. FP8 KV remains a capacity profile for deployments that validate its
quality on their own evaluation set.

```bash
export HF_HOME=/models/hf
export VLLM_CACHE_ROOT=/models/vllm
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FASTOKENS=1
export FLASHINFER_CUDA_ARCH_LIST=12.0f

/root/.venv/bin/vllm serve \
  /models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --disable-custom-all-reduce \
  --attention-backend flashinfer
```

The parser names and multimodal TP mode follow the
[official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B). Do not add
`--trust-remote-code`; the architecture is native to the installed
Transformers/vLLM versions. vLLM auto-selects CUTLASS block-scaled FP8 GEMM on
`sm_120`, Triton/FLA for GDN prefill, FlashInfer for language-model attention,
and FlashAttention for the vision encoder. DeepGemm is intentionally
auto-disabled by vLLM for this architecture on Blackwell because its E8M0 scale
path can degrade accuracy.

Prefix caching sets vLLM's Mamba cache mode to experimental `align` for this
hybrid model. Keep it because repeated system/tool prefixes are a primary
router workload, but include prefix-cache correctness in upgrade tests. A
text-only service can add `--language-model-only`; qualify it independently
because it changes memory balance and removes image support.

### Measured placement and KV-cache matrix

Each cell used vLLM's online benchmark with 12 random requests per repetition,
1,024 input tokens, 128 forced output tokens, one warm-up, three repetitions,
and a different seed per repetition. Prefix caching stayed enabled, but random
prompts and fresh seeds prevented cross-repetition prefix reuse.

| Placement / KV | Concurrency | Output tok/s | Mean TTFT | Mean TPOT | Cache tokens | Full-context concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=2, PP=1, BF16 KV | 1 | 75.07 | 141.80 ms | 12.31 ms | 365,925 | 1.40x |
| TP=2, PP=1, BF16 KV | 8 | 288.50 | 790.05 ms | 15.67 ms | 365,925 | 1.40x |
| TP=2, PP=1, FP8 KV | 1 | 75.52 | 128.72 ms | 12.33 ms | 717,129 | 2.74x |
| TP=2, PP=1, FP8 KV | 8 | 284.02 | 866.93 ms | 15.47 ms | 717,129 | 2.74x |
| TP=1, PP=2, FP8 KV | 1 | 43.74 | 141.53 ms | 21.93 ms | 554,419 | 2.11x |
| TP=1, PP=2, FP8 KV | 8 | 185.70 | 1,078.27 ms | 25.12 ms | 554,419 | 2.11x |

TP=2 is the production placement. Despite the lack of peer access, PP=2 was
42% slower for a single stream and 35% slower at concurrency 8. PP=2 also
loaded the multimodal outside weights unevenly (14.71 versus 17.08 GiB), which
reduced cache capacity. The EPYC memory system makes host-staged TP viable, but
does not make pipeline bubbles cheaper than the TP collectives for this model.

At 92% GPU memory utilization, TP=2 loaded 14.71 GiB of model weights per GPU,
reserved 11.41 GiB of cache on the limiting rank, and used approximately
29.3 GiB/GPU after startup. BF16 KV delivered slightly higher aggregate output
throughput in this short-context matrix; the difference is small enough that
quality/capacity policy should decide between the formats.

FP8 KV doubles measured cache capacity, but vLLM 0.26 reports that the
checkpoint supplies no calibrated Q/K/V/probability cache scales and falls back
to 1.0. Treat `--kv-cache-dtype fp8` as an opt-in capacity profile until a
task-relevant accuracy evaluation accepts that fallback. Do not infer output
quality from the throughput benchmark.

### MTP and upgrade constraints

Do not enable the included one-layer MTP head on vLLM 0.26.0. The installed
GDN implementation predates
[vLLM PR #51812](https://github.com/vllm-project/vllm/pull/51812), which fixes
mixed-batch speculative decoding state ordering for Qwen3.8. Re-evaluate MTP
only after upgrading to a release that contains that fix, then run mixed
prefill/decode correctness tests as well as throughput tests. The model serving
profile must be requalified after any vLLM, Transformers, FlashInfer, CUDA,
driver, model-revision, KV-format, context, or parallel-placement change.

## llama.cpp

### Reproducible builds

Build a native baseline and a separate ZenDNN comparison. Pin a tested commit;
do not benchmark an unrecorded moving branch.

```bash
git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
git -C /opt/llama.cpp checkout 7c35571e5dd2e577eddaca7e0ffa7cf46a54fcf2

cmake -S /opt/llama.cpp -B /opt/llama.cpp/build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_ZENDNN=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build /opt/llama.cpp/build-native \
  --target llama-server llama-bench llama-cli llama-quantize -j16

cmake -S /opt/llama.cpp -B /opt/llama.cpp/build-zen \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_ZENDNN=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build /opt/llama.cpp/build-zen \
  --target llama-server llama-bench llama-cli -j16
```

`GGML_NATIVE=ON` emitted `-march=native`, and GCC 15.2 selected `znver5` with
the expected AVX-512 BF16 and VNNI features. Manually piling individual
`GGML_AVX512*` switches on top is unnecessary for an on-host native build.

Current llama.cpp downloads its pinned ZenDNN dependency when
`GGML_ZENDNN=ON`; this build selected ZenDNN commit
`1f399a75cc0993778374a51bea49b64a57879595`. Preserve that information with
the llama.cpp commit in benchmark records.

### Device selection matters

Compiling ZenDNN does not mean every run uses it. It appears as an optional
ggml device:

```bash
/opt/llama.cpp/build-zen/bin/llama-server --list-devices
```

Use these explicit controls in comparisons:

```text
-dev none      native CPU backend only
-dev ZenDNN    ZenDNN device
```

This corrects a subtle but important mistake in the initial research:
`GGML_ZENDNN=ON` followed by `--device none` explicitly excludes ZenDNN. The
[llama.cpp ZenDNN documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/ZenDNN.md)
currently describes FP32, BF16, and Q8_0 support; other quantizations fall back
to ordinary CPU operations. Even fallback can add device/copy overhead, so an
explicit ZenDNN run must still be compared against a native-only build.

### Benchmark method

The main matrix used Qwen3-4B from `unsloth/Qwen3-4B-GGUF`, pinned at Hub
revision `22c9fc8a8c7700b76a1789366280a6a5a1ad1120`. BF16, Q8_0, and Q4_K_M
were tested with 512 prompt tokens, 64 generated tokens, and three repetitions.
Long-prefill checks used 2,048 and 4,096 prompt tokens. The same prompts,
runtime revision, CPU set, and repetition count were used for every comparison.

Example native sweep:

```bash
MODEL=/models/gguf/qwen3-4b/Qwen3-4B-Q4_K_M.gguf
PHYSICAL_CPUS=6,7,10-21,24,25

taskset -c "$PHYSICAL_CPUS" \
  /opt/llama.cpp/build-native/bin/llama-bench \
    -m "$MODEL" -dev none -p 512 -n 64 -t 8,12,16 -r 3
```

For the 20-thread case, additional visible SMT siblings were deliberately
added. For strict server experiments, use the separate generation and batch
affinities exposed by the current server CLI. Check `llama-server --help` at the
pinned revision because option names have changed over time.

### Measured 4B results

Native ggml, tokens/second:

| Representation | Threads | Prompt 512 | Generate 64 |
| --- | ---: | ---: | ---: |
| BF16 | 8 | 479.37 | 28.35 |
| BF16 | 12 | 617.48 | 28.12 |
| BF16 | 16 physical | **791.14** | **28.56** |
| BF16 | 20 with SMT | 546.08 | 27.19 |
| Q8_0 | 8 | 537.72 | 43.63 |
| Q8_0 | 12 | 686.96 | 46.72 |
| Q8_0 | 16 physical | **840.62** | **47.21** |
| Q8_0 | 20 with SMT | 636.23 | 42.36 |
| Q4_K_M | 8 | 230.51 | 61.67 |
| Q4_K_M | 12 | 302.49 | 67.68 |
| Q4_K_M | 16 physical | **380.19** | **69.09** |
| Q4_K_M | 20 with SMT | 371.68 | 67.95 |

At 16 physical cores, native versus ZenDNN device:

| Representation | Native prompt | ZenDNN prompt | Native generate | ZenDNN generate |
| --- | ---: | ---: | ---: | ---: |
| BF16 | **791.14** | 576.21 | **28.56** | 25.30 |
| Q8_0 | **840.62** | 621.52 | **47.21** | 40.01 |
| Q4_K_M | **380.19** | 318.08 | **69.09** | 54.25 |

The native lead remained at long prompt lengths:

| Representation | Prompt length | Native tok/s | ZenDNN tok/s |
| --- | ---: | ---: | ---: |
| BF16 | 2,048 | **646.66** | 449.35 |
| BF16 | 4,096 | **521.87** | 343.26 |
| Q8_0 | 2,048 | **703.20** | 465.42 |
| Q8_0 | 4,096 | **550.04** | 355.15 |

ZenDNN algorithm values 0 through 3 were also tested for BF16 at 2,048 prompt
tokens and 16 threads. Algorithm 1 was the best ZenDNN choice, but only by a
small margin: 447.30 tok/s versus 445.35, 437.34, and 424.32. It did not close
the native gap. Explicit `OMP_PLACES`, `OMP_PROC_BIND`, and
`GOMP_CPU_AFFINITY` settings produced no consistent improvement over llama.cpp's
own thread control.

Conclusions from this revision and dense model:

- Sixteen physical cores are the best default. Adding SMT reduced or did not
  improve both prompt and generation rates.
- Q4_K_M has the best single-stream generation rate because reduced weight
  traffic matters more than ZenDNN/AVX512-BF16 utilization during decode.
- Q8_0 has the best prompt rate of the tested formats and substantially better
  generation than BF16. It is a useful quality/performance compromise.
- BF16 generation saturated early; extra threads mainly helped prompt
  processing.
- ZenDNN algorithm 1 is the right ZenDNN comparison, but native ggml is the
  production default here.
- Re-test ZenDNN on runtime upgrades and on MoE models. Its `MUL_MAT_ID` path
  could change the result for active-expert workloads; that was not tested here.

### Recommended llama.cpp server baseline

```bash
MODEL=/models/gguf/qwen3-4b/Qwen3-4B-Q4_K_M.gguf

taskset -c 6,7,10-21,24,25 \
  llama-server \
    --model "$MODEL" \
    --device none \
    --threads 16 \
    --threads-batch 16 \
    --cpu-mask 0x33ffcc0 \
    --cpu-mask-batch 0x33ffcc0 \
    --cpu-strict 1 \
    --cpu-strict-batch 1 \
    --host 0.0.0.0 \
    --port 8080
```

The mask has bits 6, 7, 10-21, 24, and 25 set. Verify the exact affinity syntax
with `llama-server --help`; the current CLI accepts hexadecimal CPU masks or a
single range, but older releases used different names. Use a mask for a
non-contiguous cgroup set like this one.

Do not enable `--mlock` in this LXC as currently configured. Its locked-memory
limit is only 8 MiB, so it cannot lock a real model. Raise `LimitMEMLOCK` and the
container capability/limit at the Proxmox host first, or rely on normal page
cache behavior. Also do not add `--numa distribute` while the guest has only one
NUMA node.

## vLLM CPU serving

### Isolate the CPU environment

Do not replace the CUDA vLLM installation with a CPU wheel. Create a dedicated
environment and install the matching Zen extra, following the
[vLLM CPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/):

```bash
uv venv /opt/vllm-cpu/.venv --python 3.12 --seed

VLLM_VERSION=0.26.0
uv pip install --python /opt/vllm-cpu/.venv/bin/python \
  "vllm[zen]==${VLLM_VERSION}" \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cpu" \
  --index-strategy first-index \
  --torch-backend cpu
```

Pinning the runtime version makes the ZenTorch and PyTorch pairing repeatable.
Upgrade them as a unit and rerun the acceptance matrix.

### LXC executable-stack wheel issue

The 0.26.0 CPU wheel marks all three vLLM CPU extension variants with an RWE
`PT_GNU_STACK` segment. The unprivileged LXC refuses to load them:

```text
cannot enable executable stack as shared object requires: Invalid argument
```

This was resolved in the isolated CPU venv by clearing only that ELF flag:

```bash
apt-get install -y patchelf

for so in \
  /opt/vllm-cpu/.venv/lib/python3.12/site-packages/vllm/_C.abi3.so \
  /opt/vllm-cpu/.venv/lib/python3.12/site-packages/vllm/_C_AVX2.abi3.so \
  /opt/vllm-cpu/.venv/lib/python3.12/site-packages/vllm/_C_AVX512.abi3.so
do
  readelf -lW "$so" | grep GNU_STACK
  patchelf --clear-execstack "$so"
done
```

Afterward the segment is `RW`, `import vllm._C` succeeds, and
`torch.ops._C.init_cpu_memory_env` is present. This patch is specific to the
installed wheel and must be rechecked after every reinstall. Prefer an upstream
wheel without the executable-stack flag when one becomes available; do not
weaken the entire container security profile to accommodate one library.

### Allocator and OpenMP preload

The wheel includes the expected libraries. Preload both as vLLM documents:

```bash
export LD_PRELOAD="/opt/vllm-cpu/.venv/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:/opt/vllm-cpu/.venv/lib/libiomp5.so"
```

Without this, vLLM warns that Intel OpenMP is not preloaded. Include the preload
in the eventual systemd unit rather than relying on an interactive shell.

### Recommended online launch

```bash
export LD_PRELOAD="/opt/vllm-cpu/.venv/lib/python3.12/site-packages/vllm/libs/libtcmalloc_minimal.so.4:/opt/vllm-cpu/.venv/lib/libiomp5.so"
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_OMP_THREADS_BIND='6,7,10-21,24'
export VLLM_CPU_KVCACHE_SPACE=2
export VLLM_ZENTORCH_WEIGHT_PREPACK=1

taskset -c 6,7,10-21,24,25 \
  /opt/vllm-cpu/.venv/bin/vllm serve Qwen/Qwen3-0.6B \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 4
```

The outer `taskset` constrains the complete process tree. The OpenMP set gives
the model worker 15 physical cores, leaving CPU 25 for the API/front-end work.
When explicit binding is used, `VLLM_CPU_NUM_OF_RESERVED_CPU` does not carve out
another core; that variable applies to `VLLM_CPU_OMP_THREADS_BIND=auto`.

For an offline comparison, giving the worker all 16 physical cores produced
35.48 output tok/s at concurrency 4 versus 35.67 tok/s with 15 inference cores.
The difference is noise, so reserving the front-end core is the better online
configuration.

Use BF16 on ZenTorch. Do not request FP16 merely because a model was stored in
FP16; the Zen optimized path is intended for BF16/FP32. Weight prepacking is
enabled by default in current vLLM, but setting it explicitly makes the intended
configuration visible.

### KV cache is a per-model setting

`VLLM_CPU_KVCACHE_SPACE` is in GiB. Do not copy a fixed value such as 40 GiB to
every model. It competes with model weights, compilation caches, page cache, and
other services inside the 187.5 GiB container limit.

For the 0.6B test, 2 GiB yielded 18,688 KV tokens, or 4.56 concurrent full
4,096-token sequences. Choose the value during ingestion:

1. Reserve operating-system and co-resident-service headroom.
2. Load the model and read vLLM's logged KV-token capacity.
3. Ensure capacity covers the intended context and concurrency with margin.
4. Increase only if the workload needs more live tokens; a larger idle cache
   does not make matrix multiplication faster.

### Verification

Do not accept a configuration merely because `lscpu` lists AVX-512. Verify the
runtime and placement:

```bash
VLLM_LOGGING_LEVEL=DEBUG /opt/vllm-cpu/.venv/bin/vllm serve MODEL \
  --dtype bfloat16 2>&1 | tee /tmp/vllm-cpu.log

grep 'AMD Zen CPU detected with zentorch installed' /tmp/vllm-cpu.log
grep 'OpenMP thread binding info' /tmp/vllm-cpu.log
grep 'CPU unquantized GEMM dispatch' /tmp/vllm-cpu.log
```

This host logged `ZenCpuPlatform`, the exact 15-core worker list, and strict
NUMA `MEMBIND` to node 0. `numactl --membind=0 true` also succeeds in the LXC,
so the usual container seccomp restriction on NUMA syscalls is not present.

ZenTorch emits `Failed to set thread affinity` during its own topology probe in
this non-contiguous LXC CPU set. The vLLM worker still logs the intended binding
and serves correctly. Treat the warning as a container/topology issue, but
re-evaluate it if process masks or measured utilization disagree.

### Measured vLLM online behavior

Qwen3-0.6B BF16 was served with TP=1, a 2 GiB KV cache, 15 inference cores, one
reserved core, 256 input tokens, 64 forced output tokens, and 12 requests per
point. Prefix caching was disabled and each point used a different seed to
prevent cache contamination.

| Concurrency | Output tok/s | Mean TTFT | Mean TPOT |
| ---: | ---: | ---: | ---: |
| 1 | 20.76 | 573 ms | 39.84 ms |
| 2 | 27.69 | 892 ms | 59.15 ms |
| 4 | **35.67** | 1,774 ms | 85.66 ms |
| 8 | 36.57 | 3,401 ms | 129.84 ms |

Concurrency 8 adds only 2.5% output throughput over 4 while nearly doubling
mean TTFT. For this small model and an interactive SLO, `--max-num-seqs 4` is a
better starting point. This is not a global vLLM limit: repeat the concurrency
sweep for every model and expected prompt/output distribution.

Prefix caching was initially left enabled while identical seeded prompts were
replayed. That inflated later concurrency results. Disable it during cold-path
comparisons, restart between variants, or ensure every variant receives unique
prefixes. Measure prefix caching separately with a representative hit rate.

## NUMA and NPS

The guest currently exposes one NUMA node. NPS1 interleaves the socket's memory
channels into one address space, which is a good control for a streaming LLM
workload. The correct current choices are:

```text
vLLM tensor parallel size = 1
llama.cpp NUMA mode        = none/default
```

Do not create multiple CPU tensor-parallel ranks within this single NUMA node;
that adds synchronization without adding a locality domain.

NPS4 remains a worthwhile BIOS-level experiment, not an assumed upgrade. It
would create four smaller locality domains, each with fewer cores and memory
channels. Run it only after preserving the NPS1 control results. For NPS4:

- bind one vLLM TP rank per NUMA node and compare TP=4 with TP=1;
- compare llama.cpp's NUMA distribution with explicit `numactl` placement;
- confirm first-touch allocation and memory policy from inside the service
  container;
- compare aggregate throughput, TTFT, and tail latency, not just bandwidth;
- restore NPS1 if the streaming decode workload loses aggregate channel access.

## Process for qualifying any new CPU host

### 1. Freeze the environment

Record the BIOS/NPS setting, microcode, kernel, container/VM boundary, cgroup
CPU and memory limits, runtime commits, compiler, model repository revisions,
quantizations, command line, environment variables, and background services.
Do not compare results collected from different software revisions as if only a
hardware knob changed.

### 2. Inventory topology from inside the workload boundary

Collect:

```bash
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE,CACHE,ONLINE
numactl --hardware
cat /sys/fs/cgroup/cpuset.cpus.effective
cat /sys/fs/cgroup/memory.max
cat /proc/meminfo
```

Map hardware threads to cores, cores to LLCs, NUMA nodes to memory, and GPUs to
PCIe/NUMA nodes. Build CPU sets from allowed logical CPUs rather than host-wide
numbering. Test NUMA syscalls inside the container.

### 3. Verify ISA and native code generation

Check CPU flags, then ask the compiler what native means:

```bash
lscpu | grep -i flags
gcc -march=native -Q --help=target | grep -E 'march=|mtune=|avx512|vnni|bf16'
```

Verify backend selection in runtime logs. If performance counters are
available, use `perf`, AMD uProf, or ProcessWatch to confirm instruction mix and
identify bandwidth, cache, branch, or vector bottlenecks. The current LXC does
not grant those counters; run such profiling on the Proxmox host or grant a
narrow, temporary profiling capability.

### 4. Establish frequency and memory controls

Record driver, governor, EPP, boost, THP, swap, memory speed/channel population,
and thermal/power limits. Warm the workload and check for frequency or power
throttling. Change one policy at a time. Avoid confusing a governor label with
the effective amd-pstate EPP policy.

### 5. Build at least two llama.cpp baselines

Always compare the native ggml CPU backend with any vendor backend. Test a
quality/performance ladder such as BF16, Q8_0, Q6_K, Q5_K_M, and Q4_K_M when
available. Include one dense and one MoE model if the fleet will serve both.

Sweep physical-core counts first, then add SMT explicitly. Measure prompt
processing and generation separately at short and long prompts. Select server
`threads` and `threads-batch` independently only after those curves are known.

### 6. Test vLLM as an online system

Confirm the intended platform and kernel dispatch, then sweep:

- inference physical cores, with one or two cores reserved for online work;
- request concurrency and arrival rate;
- `max-num-batched-tokens` and `max-num-seqs`;
- short/long context distributions;
- prefix-cache off, then a realistic cache-hit workload;
- KV-cache capacity;
- TP equal to actual NUMA domains, plus TP=1 as the control.

Record request throughput, output throughput, TTFT, TPOT/inter-token latency,
end-to-end latency, and p95/p99—not only tokens/second.

### 7. Control common benchmark errors

- Do not benchmark a container against CPU IDs it cannot use.
- Do not mix physical-core and SMT counts without labeling them.
- Do not replay cached prefixes accidentally.
- Do not compare different prompt/output lengths or sampling behavior.
- Do not infer vendor-backend use from compilation alone; select and verify the
  device.
- Do not let unrelated services float across the same cores during final runs.
- Do not use a model small enough to fit in cache as the only proxy for a large
  streaming workload.
- Run multiple repetitions, randomize variant order, warm code/model pages, and
  report variation or confidence intervals.
- Preserve raw machine-readable benchmark output with the exact configuration.

### 8. Turn results into two configuration layers

The future v2 design should keep these concerns separate:

**One-time host profile**

- allowed physical-core and SMT sibling sets;
- NUMA/PCIe mapping and NPS mode;
- allocator/OpenMP preload paths;
- frequency/EPP policy and limits;
- huge-page, swap, memlock, and profiling capabilities;
- native compiler target and runtime install locations;
- container syscall/security requirements.

**Per-model ingestion profile**

- runtime/backend/device and pinned version;
- model revision, representation, and quantization;
- generation and batch thread counts;
- context, KV cache, batch-token, and concurrency limits;
- NUMA/TP placement;
- measured TTFT, TPOT, prompt rate, generation rate, and memory use;
- the accepted benchmark/SLO and a revalidation trigger.

Revalidate a profile when the BIOS/NPS setting, kernel/microcode, compiler,
runtime, vendor backend, model files, quantization, or service CPU/memory limits
change.
