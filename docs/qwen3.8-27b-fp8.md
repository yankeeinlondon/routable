# Qwen3.8-27B-FP8 on this host

This document describes the `qwen38` service as it is actually configured on
this machine: what the checkpoint is, what the hardware is, and *why* each
serving option was chosen for this specific hardware rather than copied from a
generic recipe. Every acronym is expanded at first use, and a consolidated
glossary appears at the end.

The companion document
[`hardware-optimization.md`](./hardware-optimization.md) records the host-level
investigation and the qualification process. This document is the per-model
result of that process.

- **Deployed:** 2026-08-17, re-measured 2026-08-19
- **Router service name:** `qwen38`
- **Public model id:** `qwen3.8-27b-fp8` (alias `qwen38`)
- **Backend port:** `127.0.0.1:8104`
- **Systemd unit:** `vllm@qwen38.service`

---

## 1. Reading this document

Three vocabularies collide in an LLM serving profile: model architecture,
GPU runtime internals, and host/OS resource management. A configuration flag
usually only makes sense when you can see all three at once. The sections below
are ordered so that each constraint is established before the flags that
respond to it.

**LLM** — Large Language Model. **vLLM** is the inference server used here; the
name comes from "virtual memory LLM", after the paged KV-cache technique that
made it notable.

---

## 2. The hardware this is tuned for

Nothing in section 5 is a universal best practice. Each choice is a response to
one of the following facts.

| Property | Value | Why it matters here |
| --- | --- | --- |
| GPUs | 2 × NVIDIA GeForce RTX 5090 | 32,607 MiB each (~31.36 GiB usable per vLLM). The model does not comfortably fit on one. |
| GPU architecture | Blackwell, compute capability `sm_120` | Determines which quantized kernels exist and which are safe. |
| Driver | 595.71.05 | Part of the qualified cell; changing it triggers requalification. |
| GPU interconnect | PCIe, **no CUDA peer access** | The two GPUs cannot DMA directly to each other. Cross-GPU traffic is staged through the host. This is the single most important constraint on parallelism choice. |
| CPU | AMD EPYC 9175F (Zen 5) | 16 physical cores / 32 hardware threads, 1 socket. |
| CPU cache | 512 MiB L3, exposed as 16 separate 32 MiB LLC domains | One active core per LLC domain — unusually large per-core cache, but cross-core traffic crosses an LLC boundary. |
| Memory | 187.5 GiB visible, no swap, NPS1 (one NUMA node) | Ample host bandwidth, which is what makes host-staged GPU-to-GPU traffic tolerable. |
| Container | Unprivileged Proxmox LXC, Ubuntu 26.04 userspace | Effective CPU set is `6-7,10-31`; logical CPUs `0-5` and `8-9` are not visible. |

**PCIe** — Peripheral Component Interconnect Express, the serial bus connecting
GPUs to the CPU. **DMA** — Direct Memory Access, a device reading or writing
memory without CPU involvement. **NUMA** — Non-Uniform Memory Access, a memory
model where some memory is closer to some cores; **NPS1** ("Nodes Per Socket =
1") means AMD's firmware presents the whole socket as one memory domain, which
interleaves all twelve memory channels into a single address space. **LLC** —
Last Level Cache, the largest cache shared before main memory. **LXC** — Linux
Containers, the OS-level virtualization used by Proxmox VE.

The absence of CUDA peer access deserves emphasis. On a machine with NVLink,
tensor parallelism is nearly free and pipeline parallelism is rarely worth it.
Here, every cross-GPU collective operation crosses the PCIe root complex and
touches host memory. It would be reasonable to *predict* that pipeline
parallelism wins on this host. It was measured, and it does not — see section 6.

---

## 3. The checkpoint

The deployed artifact is [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
pinned at Hugging Face Hub commit `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`:

```text
/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
```

81 files, approximately 28.76 GiB, of which 64 are per-layer weight shards
(`layers-0.safetensors` … `layers-63.safetensors`), plus `outside.safetensors`
(5.6 GiB of embeddings, vision tower, and head) and `mtp.safetensors` (455 MiB,
unused — see section 7).

Verify the snapshot rather than trusting that the directory exists:

```bash
/root/.venv/bin/hf cache verify Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --cache-dir /models/hf/hub
```

### 3.1 Quantization

**FP8** — 8-bit floating point. This checkpoint uses the **E4M3** variant: one
sign bit, 4 exponent bits, 3 mantissa bits. Compared to **BF16** (bfloat16 —
1 sign, 8 exponent, 7 mantissa bits, 16 bits total), FP8 halves weight storage
and doubles effective memory bandwidth for weight reads, at some numerical cost.

The `quantization_config` in `config.json` declares:

- `quant_method: fp8`, `fmt: e4m3`
- `activation_scheme: dynamic` — activation scale factors are computed at
  runtime per tensor rather than baked in from a calibration pass. This removes
  a calibration dependency but means the weights, not the activations, carry the
  quantization decision.
- `modules_to_not_convert` — the first two vision-transformer blocks are kept in
  full precision. Early vision layers have wide dynamic range and quantizing
  them measurably hurts image understanding, so the checkpoint author excluded
  them.

The service passes `--quantization fp8` explicitly rather than relying on
auto-detection, so a checkpoint swap that changes format fails loudly instead of
silently reinterpreting weights.

### 3.2 Architecture: a hybrid attention model

This is the part that changes which knobs matter, so it is worth being precise.

The model is `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`), a dense
64-layer multimodal model. "Dense" means every parameter participates in every
forward pass — contrast **MoE** (Mixture of Experts), where a router activates
only a subset of expert layers per token. Dense models have predictable memory
and latency; MoE models trade memory for compute efficiency. This one is dense,
so the full 28.76 GiB is live on every token.

`layer_types` alternates with `full_attention_interval: 4`:

```text
linear, linear, linear, full,   linear, linear, linear, full,   … (×16)
```

That yields **48 linear-attention layers and 16 full-attention layers**.

**Full-attention layers** use standard scaled dot-product attention with a
**KV cache** — Key/Value cache, the stored per-token key and value projections
that let generation avoid recomputing history. KV cache grows linearly with
sequence length, and it is what limits context capacity.

- 24 query heads, 4 key/value heads → **GQA** (Grouped Query Attention): several
  query heads share one KV head, which cuts cache size 6× versus full
  multi-head attention.
- `head_dim: 256`, `attn_output_gate: true`
- **RoPE** (Rotary Position Embedding) with `rope_theta: 10,000,000` and
  `partial_rotary_factor: 0.25` — only a quarter of each head's dimensions carry
  rotary position information. The large theta is what makes 262,144-token
  context viable.
- **mRoPE** (multimodal RoPE) with `mrope_interleaved: true` and
  `mrope_section: [11, 11, 10]` — position is decomposed into temporal, height,
  and width components so image and video patches get spatially meaningful
  positions.

**Linear-attention layers** use **GDN** — Gated DeltaNet, a recurrent
(state-space-style) mechanism. Instead of a cache that grows with sequence
length, each layer carries a **fixed-size recurrent state** per active sequence.
Configuration: 16 key heads and 48 value heads at 128 dimensions each,
`linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: float32`.

**SSM** — State Space Model, the family GDN belongs to; vLLM reuses its "Mamba"
code paths for these layers, which is why Mamba appears in the logs.

The consequence for configuration is direct and non-obvious:

> On a pure-attention model, memory scales with *tokens*. On this model, memory
> scales with tokens **and** with *concurrent sequences*, because 48 layers pay a
> fixed per-sequence state cost regardless of how short the sequence is.

Derived from the config, per GPU at tensor parallelism 2:

| Component | Cost |
| --- | --- |
| Full-attention KV, BF16 | 32 KiB per token per GPU |
| Full-attention KV, FP8 | 16 KiB per token per GPU |
| GDN recurrent + convolution state | ~73.4 MiB per **active sequence** per GPU |

At `--max-num-seqs 32`, that fixed recurrent state alone reserves about
**2.29 GiB per GPU** before a single token is cached. This is why
`max-num-seqs` is treated as a memory control here, not merely a scheduler
preference (section 5.6).

### 3.3 Vision tower

A 27-layer **ViT** (Vision Transformer): hidden size 1152, 16-pixel patches,
`spatial_merge_size: 2`, `temporal_patch_size: 2`, projecting to the 5120-wide
text hidden size. `deepstack_visual_indexes` is empty, so the vision features
enter only at the input layer.

Text side: hidden size 5120, feed-forward intermediate size 17408, **SiLU**
(Sigmoid Linear Unit) activation in a gated feed-forward network, **RMSNorm**
(Root Mean Square Normalization, `eps: 1e-6`), vocabulary 248,320 tokens,
untied input/output embeddings, `max_position_embeddings: 262,144`.

---

## 4. Runtime versions

| Component | Version |
| --- | --- |
| vLLM | 0.26.0 |
| PyTorch | 2.11.0+cu130 |
| FlashInfer | 0.6.14 |
| Transformers | 5.14.1 |
| fastokens | 0.3.1 |
| xgrammar | 0.2.3 |
| Python venv | `/root/.venv` |

**FlashInfer** is a GPU attention-kernel library. **fastokens** is a fast
tokenizer implementation. **xgrammar** backs structured/constrained output.

This set is a *qualified cell*: the numbers in section 6 are only valid for this
combination. Any change requires rerunning the acceptance matrix.

---

## 5. The deployed configuration, option by option

The router generates the unit file; do not edit it by hand. The effective
command line is:

```text
/root/.venv/bin/vllm serve \
  /models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --host 127.0.0.1 --port 8104 \
  --served-model-name qwen3.8-27b-fp8 qwen38 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --tokenizer <snapshot path> --tokenizer-mode auto \
  --quantization fp8 \
  --max-model-len 262144 \
  --kv-cache-dtype auto \
  --distributed-executor-backend mp \
  --attention-backend flashinfer \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --mm-encoder-tp-mode data \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

with environment from `/etc/vllm/qwen38.env`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES="0,1"
FLASHINFER_CUDA_ARCH_LIST="12.0f"
HF_HOME="/models/hf"
NCCL_P2P_DISABLE="1"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
VLLM_CACHE_ROOT="/models/vllm"
VLLM_USE_FASTOKENS="1"
VLLM_WORKER_MULTIPROC_METHOD="spawn"
```

### 5.1 `--tensor-parallel-size 2`, `--pipeline-parallel-size 1`

**TP** — Tensor Parallelism: each layer's weight matrices are split across GPUs
and every GPU participates in every layer, synchronizing with an **all-reduce**
collective per block. **PP** — Pipeline Parallelism: whole layers are assigned
to different GPUs and activations are handed down the pipe. **DP** — Data
Parallelism: independent replicas serve different requests.

*Why TP=2 is required at all:* the 28.76 GiB checkpoint cannot share a 31.36 GiB
GPU with activations, CUDA graphs, encoder workspaces, and a usable KV cache.
Splitting the weights leaves 14.71 GiB per GPU and roughly 11–12 GiB for cache.
Offloading weights to CPU was rejected: it would turn the PCIe link into a
weight-streaming bottleneck on every token.

*Why not PP=2, given the missing peer access?* This was the open question and it
was measured. PP=2 was **42% slower single-stream and 35% slower at concurrency
8**. It also split the non-layer ("outside") weights unevenly — 14.71 versus
17.08 GiB — which shrank the cache on the heavier rank. On this host, PP's
pipeline bubbles cost more than TP's host-staged collectives. The EPYC's memory
bandwidth is what makes that true; do not assume it generalizes to a host with
fewer memory channels.

All relevant head counts divide cleanly by 2 (24 query heads → 12, 4 KV heads →
2, 16/48 GDN heads → 8/24), so TP=2 needs no padding.

### 5.2 `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`

**NCCL** — NVIDIA Collective Communications Library, which implements
all-reduce and friends. **P2P** — peer-to-peer, direct GPU-to-GPU transfer.

Peer access is unavailable on this host, so NCCL would probe for it, fail, and
fall back. Setting `NCCL_P2P_DISABLE=1` makes the fallback the *declared*
transport instead of a discovered one — the selected path is then explicit and
repeatable across restarts and driver updates.

`--disable-custom-all-reduce` turns off vLLM's hand-written all-reduce kernel,
which is designed for peer-capable topologies and is not a win here. The startup
log confirms the result: `Using ['PYNCCL'] all-reduce backends`.

### 5.3 `--attention-backend flashinfer` and `FLASHINFER_CUDA_ARCH_LIST=12.0f`

FlashInfer provides the language-model attention kernels. `12.0f` is the
`sm_120` (Blackwell) target; pinning it avoids compiling or loading kernels for
architectures this host does not have.

Confirmed at startup:

```text
FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16,
  decode_backend=flashinfer-native, kv_cache_dtype=torch.bfloat16, arch=sm120
```

vLLM independently selects FlashAttention for the vision encoder and Triton/FLA
kernels for GDN prefill. Those are not overridden.

### 5.4 FP8 GEMM kernel selection (automatic, but verify it)

**GEMM** — General Matrix Multiply, the core operation of a transformer layer.
**CUTLASS** is NVIDIA's template library for GEMM kernels. **DeepGEMM** is an
alternative FP8 GEMM implementation.

Startup log:

```text
Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod
Auto-disabled DeepGemm for model_type=qwen3_5_text on Blackwell. DeepGemm E8M0
  scale format causes accuracy degradation for this architecture. Falling back
  to CUTLASS.
```

**E8M0** is an 8-bit exponent-only scale format. vLLM disables DeepGEMM for this
architecture on Blackwell because that scale format degrades accuracy here. This
is automatic — but it is a *correctness* fallback, not a performance one, so
check for this line after any vLLM upgrade. Do not set `VLLM_USE_DEEP_GEMM=1`
for this model.

### 5.5 `--max-model-len 262144` and `--kv-cache-dtype auto`

262,144 tokens is the checkpoint's native `max_position_embeddings`. Both tested
cache formats fit at least one full context on this host, so there is no reason
to advertise less.

`auto` resolves to BF16 for this checkpoint. BF16 was chosen over FP8 KV cache
for three reasons, in priority order:

1. **The checkpoint supplies no calibrated KV scales.** vLLM 0.26 warns that
   query/key/value/probability cache scales are missing and falls back to 1.0.
   Un-calibrated FP8 KV is a quality risk that a throughput benchmark cannot
   detect.
2. **FP8 KV bought no measured speed.** In the controlled matrix it was within
   noise of BF16 (see section 6).
3. It preserves full native context comfortably.

FP8 KV remains available as a *capacity* profile — it roughly doubles cache
tokens (365,925 → 717,129) — but only for a deployment that validates quality on
its own evaluation set first.

### 5.6 `--max-num-seqs 32`

On a conventional model this caps scheduler concurrency. On this hybrid model it
also reserves memory: 32 × ~73.4 MiB ≈ **2.29 GiB per GPU** of GDN recurrent
state, allocated up front.

32 was chosen as the point where interactive latency, aggregate throughput, and
that fixed reservation balance on 32 GiB cards. Raising it trades KV-cache tokens
(and therefore usable context) for concurrency, one-for-one. Do not raise it
without re-reading the logged cache capacity.

### 5.7 `--max-num-batched-tokens 8192`

This is the **chunked prefill** budget: the maximum tokens processed in one
scheduler step, mixing new-prompt (prefill) work with ongoing generation
(decode) work. Confirmed at startup:
`Chunked prefill is enabled with max_num_batched_tokens=8192`.

8192 is a latency control. A larger budget finishes long prompts faster but lets
a single big prefill stall every in-flight decode, inflating inter-token latency
for everyone else. For a router serving interactive chat alongside occasional
long documents, 8192 keeps prefill chunks small enough that decode continues to
make steady progress. It also matches vLLM's compile range endpoint (`8192`), so
the compiled graphs cover the configured maximum.

### 5.8 `--enable-prefix-caching`

Reuses the cached KV of a shared prompt prefix across requests. The primary
router workload is repeated system prompts and tool definitions, so hit rates in
production logs run **50–65%**, which is a large latency saving on first token.

The caveat is specific to this architecture. Enabling it makes vLLM select an
experimental cache mode for the recurrent layers:

```text
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by
  default when prefix caching is enabled
Warning: Prefix caching in Mamba cache 'align' mode is currently enabled. Its
  support for Mamba layers is experimental.
```

The trade was accepted because of the measured hit rate, but **prefix-cache
correctness must be part of every upgrade test** for this model — not just
throughput.

`align` mode is also what forces the attention block size to 784 (section 6.2).

### 5.9 `--gpu-memory-utilization 0.92`

The fraction of each GPU vLLM may claim. 0.92 of 31.36 GiB is 28.85 GiB,
leaving roughly 2.5 GiB headroom per card.

0.92 rather than the more common 0.90 is deliberate: the model is large relative
to the cards, and each additional percent converts directly into KV-cache
tokens. It is not pushed higher because these are consumer GeForce cards with no
ECC and a display-capable driver stack, and because the vision encoder's
workspace varies with image size — a fragmentation spike near the limit would
produce an out-of-memory failure mid-request rather than at startup.

### 5.10 `--mm-encoder-tp-mode data`

**MM** — multimodal. The vision encoder is small relative to the language model.
Sharding it across both GPUs (tensor mode) would add collective operations to a
cheap component; `data` mode instead replicates the encoder and gives each GPU
different images. This follows the
[official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) for this model.

### 5.11 `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`

These make vLLM's OpenAI-compatible API split the model's raw output into the
right response fields: reasoning/thinking content into `reasoning_content`, and
tool invocations into structured `tool_calls`. The parser names are
model-family-specific and come from the official recipe. A wrong parser does not
crash — it silently returns thinking text as user-visible content, which is why
they are pinned explicitly.

### 5.12 `--distributed-executor-backend mp` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`

**MP** — multiprocessing. With two GPUs on one host there is no reason to
involve Ray (a distributed-computing framework) and its extra supervision
layer; plain multiprocessing has lower startup cost and fewer moving parts under
systemd.

`spawn` starts each worker as a fresh interpreter rather than forking. Forking a
process that has already initialized CUDA is unsafe, and the LXC's process and
namespace boundaries make fork-related failures harder to diagnose. `spawn`
costs a few seconds of startup and removes that class of bug entirely.

### 5.13 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

PyTorch's caching allocator normally carves fixed segments, which fragments
under variable-sized allocations. Vision encoder workspaces vary with image
resolution and prefill chunk shapes vary with traffic, so fragmentation is a
real risk at 92% utilization. Expandable segments let the allocator grow a
region instead of stranding memory between segments.

### 5.14 `--trust-remote-code` is deliberately absent

`Qwen3_5ForConditionalGeneration` is natively implemented in the installed
Transformers 5.14.1 / vLLM 0.26.0. Enabling remote code execution would run
arbitrary Python from the checkpoint repository for no benefit. The router
config records this explicitly as `tokenizer.trust_remote_code: false`.

### 5.15 `CUDA_DEVICE_ORDER=PCI_BUS_ID`

Without this, CUDA orders devices by a heuristic "fastest first" ranking that
can change between driver versions. `PCI_BUS_ID` makes device 0 and device 1
stable, so the GPU UUIDs recorded in the router's placement block stay correct.
The router pins by UUID (`GPU-d751861a-…`, `GPU-06645389-…`) rather than by
index for the same reason.

---

## 6. Measured behavior

### 6.1 Memory and capacity at startup

From the 2026-08-19 cold start, per GPU:

| Item | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Model weights | 14.71 GiB | 14.71 GiB |
| Peak activation | 2.08 GiB | 1.17 GiB |
| Non-torch memory | 0.06 GiB | 0.06 GiB |
| CUDA graph memory | 0.65 GiB | 0.65 GiB |
| KV cache | **11.41 GiB** | 12.32 GiB |
| Free at startup | 30.55 / 31.36 GiB | 30.55 / 31.36 GiB |

```text
GPU KV cache size: 365,925 tokens
Maximum concurrency for 262,144 tokens per request: 1.40x
```

Rank 0 measured 0.91 GiB more peak activation than rank 1, which makes it the
limiting rank: its 11.41 GiB sets the cache capacity for the whole service. The
cause of the imbalance was not isolated, so treat it as an observed asymmetry to
re-check after upgrades rather than a settled fact. 365,925 tokens is 1.40
full-length contexts — meaning one 262K-token conversation fits with room to
spare, but two do not. Typical traffic is far shorter, so this is a headroom
figure, not a concurrency limit.

**CUDA graph** — a pre-recorded, replayable sequence of GPU operations that
removes per-kernel launch overhead during decode. vLLM captured sizes
`[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` in `FULL_AND_PIECEWISE` mode in 2
seconds for 0.65 GiB.

### 6.2 The 784-token attention block size

```text
Setting attention block size to 784 tokens to ensure that attention page size
  is >= mamba page size.
Padding mamba page size by 0.13% to ensure that mamba page size and attention
  page size are exactly equal.
```

This unusual number is fully determined by the architecture. vLLM's hybrid
allocator uses one page pool for both layer types, so an attention page must be
at least as large as a GDN state page.

- GDN state per layer per sequence per GPU: 24 value heads × 128 × 128 × 4
  bytes (FP32 recurrent) + 5,120 × 3 × 2 bytes (BF16 convolution) =
  **1,603,584 bytes**
- Attention KV per token per layer per GPU: 2 (K and V) × 2 KV heads × 256
  head_dim × 2 bytes (BF16) = **2,048 bytes**
- 1,603,584 ÷ 2,048 = **783 tokens**, rounded up to 784

That rounding is exactly the reported padding: 784 × 2,048 ÷ 1,603,584 = 1.00128,
i.e. 0.13%. Reproducing the log line from `config.json` this way is the cheapest
available confirmation that the model is loading with the intended TP degree and
KV dtype — if it ever reports a very different block size, one of those changed.

### 6.3 Startup timing

| Phase | Duration |
| --- | --- |
| Engine init (profile, KV cache, warmup) | 30.97 s (11.37 s of it torch.compile) |
| **Process start → `Application startup complete`** | **~97 s** |

Repeated cold starts land in the **85–105 second** band. Both router deadlines
follow from this measurement: `deadlines.startup_ms: 900000` (15 min, a generous
supervision ceiling) and `parking.deadline_ms: 180000` (3 min, the window in
which the router holds a request while the model loads). The former default of
20 seconds guaranteed that an otherwise healthy cold start surfaced to clients
as `model_starting`.

`VLLM_CACHE_ROOT=/models/vllm` keeps the torch.compile artifact cache on
persistent storage so the 11.37-second compile is only paid on a cold cache.

### 6.4 Placement and KV-format matrix

Method: vLLM's online benchmark, 12 random requests per repetition, 1,024 input
tokens, 128 forced output tokens, one warm-up, three repetitions, distinct seed
per repetition. Prefix caching stayed on but random prompts and fresh seeds
prevented cross-repetition reuse. Other GPU services were stopped and the run
held the host experiment lock.

| Placement / KV | Concurrency | Output tok/s | Mean TTFT | Mean TPOT | Cache tokens | Full-context concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP=2, PP=1, BF16 KV** | 1 | **75.07** | 141.80 ms | 12.31 ms | 365,925 | 1.40x |
| **TP=2, PP=1, BF16 KV** | 8 | **288.50** | 790.05 ms | 15.67 ms | 365,925 | 1.40x |
| TP=2, PP=1, FP8 KV | 1 | 75.52 | 128.72 ms | 12.33 ms | 717,129 | 2.74x |
| TP=2, PP=1, FP8 KV | 8 | 284.02 | 866.93 ms | 15.47 ms | 717,129 | 2.74x |
| TP=1, PP=2, FP8 KV | 1 | 43.74 | 141.53 ms | 21.93 ms | 554,419 | 2.11x |
| TP=1, PP=2, FP8 KV | 8 | 185.70 | 1,078.27 ms | 25.12 ms | 554,419 | 2.11x |

**TTFT** — Time To First Token, the latency a user perceives before output
begins. **TPOT** — Time Per Output Token, the steady-state inter-token latency
that determines how fast text appears to stream.

Reading the table:

- TP=2 versus PP=2 is decisive, and it is the result that justified the
  configuration despite the missing peer access.
- BF16 versus FP8 KV is within noise on throughput. Since throughput does not
  choose, quality and capacity policy chose — and quality won (section 5.5).
- Do not infer output *quality* from any column here. This benchmark measures
  speed only.

---

## 7. Deliberately not enabled

### 7.1 MTP / speculative decoding

**MTP** — Multi-Token Prediction, a technique where a small auxiliary head
proposes several tokens at once and the main model verifies them, accelerating
decode. The checkpoint ships a one-layer MTP head
(`mtp_num_hidden_layers: 1`, `mtp.safetensors`, 455 MiB).

It is **off**. The GDN implementation in vLLM 0.26.0 predates
[vLLM PR #51812](https://github.com/vllm-project/vllm/pull/51812), which fixes
speculative-decoding state ordering for mixed prefill/decode batches on this
architecture. Enabling MTP before that fix risks silently corrupted recurrent
state. Re-evaluate only after upgrading past that PR, and then test mixed
prefill/decode **correctness**, not only throughput.

### 7.2 Sleep mode

```yaml
sleep:
  enabled: false
  hardware_validated: false
```

The router supports suspending a model's GPU memory rather than stopping the
process. It is not validated for this model on this host, so the router uses
stop/start and the one-hour idle timer instead.

### 7.3 `--language-model-only`

The checkpoint sets `language_model_only: false` and the service serves the full
multimodal model. A text-only variant would free the vision tower's share of the
weights, but it changes the memory balance and removes image support, so it
would need independent qualification.

### 7.4 CPU offload

Not used. See section 5.1 — PCIe weight streaming would dominate every token.

---

## 8. Router integration

The service definition lives in `/etc/vllm-router/router.yaml` under
`name: qwen38`. The fields whose values are specific to this model:

```yaml
placement:
  gpu_uuids: [GPU-d751861a-…, GPU-06645389-…]   # both GPUs, pinned by UUID
  tensor_parallelism: 2
  vram_reservation_bytes_per_gpu: 31000000000
idle:
  stop_after_ms: 3600000        # 1 hour
parking:
  deadline_ms: 180000           # covers the ~97 s cold start
  queue_limit: 64
  retry_after_seconds: 300
priority: 0
evictable: false
automatic_start: true
compatibility:
  maximum_model_length: 262144
  kv_cache_dtype: auto
  gpu_memory_utilization: 0.92
```

`compatibility.*` is not decorative: the router's systemd generator turns
`maximum_model_length` into `--max-model-len` and `gpu_memory_utilization` into
`--gpu-memory-utilization`. Change them there, not in the unit file.

### 8.1 Why `evictable: false`

This is the one setting whose rationale is about *router behavior* rather than
hardware, and it is easy to get wrong.

Qwen3.8 occupies **both** GPUs. If it were evictable, any background request for
a single-GPU model could evict it — and chat clients issue such requests
routinely, for conversation-title generation and similar housekeeping, even
while the visible conversation is using Qwen3.8. The next foreground prompt then
evicts that model and reloads Qwen3.8, at ~97 seconds a cycle. The result is
thrashing driven entirely by invisible background traffic.

`evictable: false` prevents that. The one-hour idle stop still releases both
GPUs eventually, so the resource is not held indefinitely.

### 8.2 Client timeouts

A client may still impose its own HTTP timeout shorter than the router's 3-minute
parking deadline. If the first request after an idle period must wait for a cold
load rather than be retried, configure the client's timeout **above three
minutes**.

### 8.3 Operations

```bash
# status and logs
systemctl status vllm@qwen38.service
journalctl -u vllm@qwen38.service -f

# verify the qualified kernel and backend selections after any upgrade
journalctl -u vllm@qwen38.service | grep -E \
  'CutlassFp8BlockScaled|FLASHINFER backend|Auto-disabled DeepGemm|all-reduce backends'

# verify capacity
journalctl -u vllm@qwen38.service | grep -E 'GPU KV cache size|Available KV cache memory'
```

---

## 9. Revalidation triggers

This profile is qualified for one exact cell. Rerun the acceptance matrix in
section 6 after **any** of:

- vLLM, PyTorch, FlashInfer, Transformers, or NVIDIA driver/CUDA version change
- model revision change (the pinned snapshot hash)
- KV-cache dtype change
- parallel placement change (TP, PP, or DP degree)
- context length or `max-num-seqs` change
- GPU hardware change, or the appearance of CUDA peer access between the cards
- host BIOS NPS mode change, or a change to the container's CPU/memory limits

For the second and later items, also re-run prefix-cache correctness checks —
the experimental `align` mode (section 5.8) is the most likely thing to break
quietly.

---

## 10. Glossary

| Term | Expansion and meaning |
| --- | --- |
| **all-reduce** | Collective operation summing a tensor across all GPUs and distributing the result to each. TP's per-layer synchronization primitive. |
| **BF16** | bfloat16 — 16-bit float with FP32's exponent range and reduced mantissa. The default compute type here. |
| **chunked prefill** | Splitting long prompt processing into bounded steps so decode requests are not starved. |
| **CUDA graph** | A pre-recorded, replayable GPU operation sequence that removes per-kernel launch overhead. |
| **CUTLASS** | NVIDIA's CUDA templates for linear algebra; source of the selected FP8 GEMM kernel. |
| **DeepGEMM** | Alternative FP8 GEMM library; auto-disabled here for accuracy reasons. |
| **DMA** | Direct Memory Access — device memory access without CPU involvement. |
| **DP** | Data Parallelism — independent model replicas serving different requests. |
| **E4M3** | FP8 layout: 1 sign, 4 exponent, 3 mantissa bits. This checkpoint's format. |
| **E8M0** | 8-bit exponent-only scale format used by DeepGEMM; inaccurate for this architecture. |
| **FLA** | Flash Linear Attention — kernel library used for GDN prefill. |
| **FlashInfer** | GPU attention-kernel library; the selected language-model attention backend. |
| **FP8** | 8-bit floating point. Halves weight memory versus BF16. |
| **GDN** | Gated DeltaNet — the recurrent linear-attention mechanism in 48 of the 64 layers. |
| **GEMM** | General Matrix Multiply — the dominant transformer operation. |
| **GQA** | Grouped Query Attention — several query heads share one KV head, shrinking the cache. |
| **KV cache** | Key/Value cache — stored per-token projections that let generation skip recomputing history. |
| **LLC** | Last Level Cache — largest cache before main memory (here, 16 × 32 MiB domains). |
| **LLM** | Large Language Model. |
| **LXC** | Linux Containers — the OS-level virtualization used by Proxmox. |
| **MM** | Multimodal — text plus image/video input. |
| **MoE** | Mixture of Experts — sparse architecture where a router activates a subset of experts. This model is *not* MoE. |
| **MP** | Multiprocessing — the distributed executor backend, as opposed to Ray. |
| **mRoPE** | Multimodal RoPE — position encoding decomposed into temporal/height/width sections. |
| **MTP** | Multi-Token Prediction — speculative decoding head. Present in the checkpoint, disabled here. |
| **NCCL** | NVIDIA Collective Communications Library — implements all-reduce and friends. |
| **NPS** | Nodes Per Socket — AMD firmware setting controlling NUMA domain granularity. NPS1 here. |
| **NUMA** | Non-Uniform Memory Access — memory model where locality affects bandwidth/latency. |
| **P2P** | Peer-to-peer — direct GPU-to-GPU transfer. **Unavailable on this host.** |
| **PCIe** | Peripheral Component Interconnect Express — the bus connecting the GPUs. |
| **PP** | Pipeline Parallelism — layers split across GPUs. Measured slower here than TP. |
| **prefix caching** | Reusing cached KV for shared prompt prefixes across requests. |
| **RMSNorm** | Root Mean Square Normalization — the normalization layer used throughout. |
| **RoPE** | Rotary Position Embedding — relative position encoding via rotation. |
| **SiLU** | Sigmoid Linear Unit — the feed-forward activation function. |
| **SLO** | Service Level Objective — the latency/throughput target a profile must meet. |
| **sm_120** | CUDA compute capability 12.0, the RTX 5090's Blackwell target. |
| **SMT** | Simultaneous Multi-Threading — two hardware threads per core. Excluded from CPU pinning here. |
| **SSM** | State Space Model — the recurrent family GDN belongs to; vLLM reuses its Mamba code paths. |
| **TP** | Tensor Parallelism — weight matrices split across GPUs. TP=2 here. |
| **TPOT** | Time Per Output Token — steady-state inter-token latency. |
| **TTFT** | Time To First Token — latency before output starts. |
| **ViT** | Vision Transformer — the 27-layer image/video encoder. |
| **vLLM** | The inference server. Name from "virtual memory LLM", after its paged KV cache. |
| **VRAM** | Video RAM — GPU memory. 32,607 MiB per card here. |
