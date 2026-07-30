# Model Discovery Data and Naming

Status: **requirements captured; implementation pending.**

This specification defines the identity and metadata contract exposed by the
router's `GET /v1/models` endpoint. It supersedes earlier alias-related decisions
in the v1 pooling specification and implementation reviews. Those documents remain
historical records of the behavior that shipped before this decision.

## Problem

The router currently conflates three different kinds of model identity:

- the short public name clients put in the request `model` field;
- the upstream repository from which the deployed model artifacts originated; and
- alternate public names accepted for the same locally managed model.

In particular, `id` and `root` currently contain the same shortened value, so
`root` loses useful Hugging Face provenance. Aliases add a second public naming
system without a demonstrated operational need. The response also contains
protocol constants such as `object` and `owned_by`; these must not become
per-model configuration merely because they appear in serialized output.

## Decisions

### 1. `id` is the operator-authored public model identifier

`id` is the canonical value clients use in the `model` field of inference
requests. It is intentionally authored by the server owner rather than derived
algorithmically from the upstream repository name.

Requirements:

- Every locally managed model contract has exactly one `id`.
- An `id` is unique in the router's active public model namespace.
- The server owner chooses the name. It should retain the characteristics useful
  to local clients, such as model family, size, important variant, and
  quantization, without having to reproduce every upstream repository detail.
- Once published to clients, an `id` is stable. Renaming it is a breaking API
  change.
- Different load configurations of the same model, including context-window and
  optimization choices, do not require new IDs. Those differences belong to
  default and active load metadata rather than the model's public identity.
- Configuration remains the source of truth. The current `served_model` value is
  emitted as `id` and is the only accepted model name passed to a runner.
- Validation must reject empty or duplicate IDs. The implementation should also
  enforce a conservative client-safe character set rather than allow whitespace
  or control characters; the precise character validator is an implementation
  detail so long as every currently selected ID below remains valid.

### 2. `root` preserves Hugging Face repository provenance

For a Hugging Face-backed model, `root` identifies the repository from which the
deployed artifact originated. It is distinct from both the public `id` and the
local filesystem `model_path`.

The canonical encoding is:

```text
{organization}/{repository} -> {organization}--{repository}
```

The transform replaces the repository ID's single namespace separator `/` with
`--`. All other characters, including capitalization, are preserved exactly.
Hugging Face repository IDs forbid `--`, so the encoding is unambiguous and
reversible under the current Hub naming rules.

Requirements:

- Hugging Face provenance is explicitly curated in service configuration; it is
  not inferred at response time from a cache directory or local model path.
- The implementation must add a dedicated configured value, provisionally named
  `model_root`, and emit it as `root`.
- `model_root` must differ conceptually from `model_path`. `model_path` is the
  absolute local artifact location used to start a runner and may include a
  snapshot revision, cache layout, mount point, or symlink. `model_root` is the
  portable upstream repository identity.
- Replicas sharing one `id` must declare the same `model_root`.
- For llama.cpp/GGUF models, `root` identifies the repository containing the
  actual deployed GGUF artifact, not an inferred base-model repository.
- A future genuinely local model with no Hugging Face origin must use an
  explicitly specified non-Hugging-Face provenance policy or a nullable root. The
  router must not fabricate an `org--repo` value.
- Repository identity alone is not exact artifact reproducibility. Snapshot or
  commit revision metadata remains a separate concern and is not folded into
  `root` by this specification.

The initial curated values are:

| `id` | `root` |
|---|---|
| `qwen3.6-27b-nvfp4` | `rdtand--Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| `qwen3.6-35b-q4.75` | `rdtand--Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm` |
| `gpt-oss-20b` | `openai--gpt-oss-20b` |
| `gemma-4-31b-nvfp4` | `LilaRest--gemma-4-31B-it-NVFP4-turbo` |

### 3. Aliases are removed

The router exposes and accepts one public name per model contract: `id`.

Requirements:

- Remove `aliases` from service configuration.
- Remove `aliases` from `GET /v1/models` entries rather than emitting an empty
  extension field.
- Remove alias lookup and canonicalization from request routing.
- Generated runner arguments contain only the canonical `id`. For vLLM this
  means one `--served-model-name` value. A llama.cpp adapter may pass that same
  canonical ID through its single runner-specific naming option.
- A request using a former alias is an unknown-model request and must fail before
  admission, parking, eviction, or model startup.
- `service.name` remains an independent operational identifier. Short service
  names such as `qwen27b` may continue to identify systemd units and management
  operations without becoming public inference model names.
- No permanent legacy alias mechanism is introduced by this feature. Any future
  compatibility mechanism requires a separate, explicit decision with a removal
  policy.

### 4. Response constants are emitted, not curated per model

Metadata curation distinguishes stored operator data, derived values, and
protocol constants. A key's presence in a JSON response does not imply that the
same key belongs in persistent service configuration.

| Response field | Curation class | Source |
|---|---|---|
| top-level `object` | response constant | always `"list"` |
| `data[].id` | operator-authored | configured `served_model` |
| `data[].root` | operator-authored provenance | configured `model_root` |
| `data[].object` | response constant | always `"model"` |
| `data[].owned_by` | response constant | always `"vllm-router"` |
| `data[].model_size` | derived artifact fact | exact deployed weight-file bytes |
| `data[].max_context_window` | resolved model capability | checkpoint configuration, validated by the runner |
| `data[].default_context_window` | operator-authored default | default load configuration |
| `data[].input_modalities` | curated effective capability | checkpoint plus runner/API validation |
| `data[].output_modalities` | curated effective capability | checkpoint plus runner/API validation |
| `data[].generation_defaults` | resolved default policy | effective runner defaults before request overrides |
| `data[].reasoning` | curated API capability | effective reasoning controls accepted by the deployment |
| `data[].tokenizer` | curated artifact identity | tokenizer repository/configuration |
| `data[].features` | curated structured capability/configuration | checkpoint, runner, and default load policy |
| `data[].created` | existing compatibility placeholder | emitted as `0`; not stored per model |
| `data[].parent` | existing compatibility placeholder | emitted as `null`; not stored per model |

`object` and `owned_by` therefore remain serialization concerns. They must not be
added to each service definition or persisted in a model registry while every
entry has the same value.

The treatment of `created` and `parent` is preserved only to keep this feature
focused. Whether they should remain in the public response is a separate metadata
decision. Likewise, exact source revision deserves its own field if the API later
needs reproducible artifact identity.

### 5. Catalog metadata is distinct from active load metadata

`GET /v1/models` is a catalog of models the router can potentially make
available. It is not an inventory of currently running processes.

Requirements:

- The catalog lists a model when at least one enabled service can provide its
  `id`, regardless of whether that service is running, sleeping, or stopped.
- A model backed only by disabled services is not cataloged as available.
- Catalog fields describe model capabilities and the default load policy.
- Concrete load facts belong to the management/status API. Multiple replicas may
  have different lifecycle states, so a single model card must not pretend that
  `active` is a model-level boolean.
- Dynamic Scheduling consumes placement-profile configuration and observed
  runtime state directly. It must not reverse-engineer scheduler inputs from the
  client-facing model card.

### 6. Model size

The public field is named `model_size`. Its unit is bytes even though the unit is
not encoded in the property name.

`model_size` is the exact sum of the deployed model weight artifacts, following
local links to their underlying files. For Safetensors this is the sum of the
weight shards; for GGUF this is the loaded GGUF artifact or shard set. Tokenizer,
README, and other small repository files are excluded.

Requirements:

- Store and serialize `model_size` as an integer without GB/GiB rounding.
- Record how the value was measured and recompute it when the artifact revision
  changes.
- Do not use `model_size` as the GPU placement reservation. Encoded weight bytes,
  resident VRAM, KV cache, graph/workspace allocations, and runner overhead are
  different quantities.
- Dynamic Scheduling continues to use measured per-placement-profile VRAM and
  host-memory reservations. `model_size` is useful for disk accounting, transfer
  planning, cold-start estimation, and plausibility checks.

The initial deployed weight sizes are:

| `id` | `model_size` |
|---|---:|
| `qwen3.6-27b-nvfp4` | `20171471784` |
| `qwen3.6-35b-q4.75` | `22882657048` |
| `gpt-oss-20b` | `13761316904` |
| `gemma-4-31b-nvfp4` | `19295277512` |

### 7. Context-window vocabulary

The previous `max_model_len` name is removed from the router-owned model metadata.
Context length always includes prompt and output tokens together.

- `max_context_window` is the greatest context window the model artifact allows
  under its validated model configuration. It is a model capability and does not
  describe the current deployment.
- `default_context_window` is the context window selected when a model is loaded
  without an alternate load-time value.
- `context_window` is the actual limit configured for a model that is loading or
  loaded. It is runtime/service-instance metadata and belongs in management
  status, not the possibly-available model catalog.

Requirements:

- `0 < context_window <= max_context_window` for every load.
- `0 < default_context_window <= max_context_window`.
- A placement profile may select a different `context_window` without creating a
  new public model `id`.
- The v1 `compatibility.maximum_model_length` value becomes the source of
  `default_context_window` and remains the value passed to vLLM's
  `--max-model-len` until the load/profile schema is implemented.
- The runner must report or verify the effective `context_window` after load so
  status reflects reality rather than the requested value alone.
- Replica routing must not send a request to a smaller-context replica unless
  admission has established that the request fits. Until context-aware admission
  exists, replicas of one `id` must use the same `context_window`.

### 8. Modalities

Model cards include:

```json
{
  "input_modalities": ["text", "image", "video"],
  "output_modalities": ["text"]
}
```

The initial controlled vocabulary is `text`, `image`, `audio`, `video`, and
`embedding`. Values describe what the checkpoint, runner, launch configuration,
and router endpoint surface can actually accept or produce together. A model-card
tag or architectural capability that the deployed API cannot exercise is not an
effective modality.

Tasks and protocol capabilities such as chat, reranking, tool calling, or
structured output are not modalities and may be modeled separately.

### 9. Generation defaults

`generation_defaults` contains the effective sampling values used when a request
omits them:

```json
{
  "generation_defaults": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20
  }
}
```

The values are resolved after applying runner built-ins, checkpoint
`generation_config.json`, configured runner overrides, and router/service policy.
Request properties have final precedence but do not change the advertised
defaults. The router must not copy raw checkpoint values without resolving this
precedence.

Only accepted generation parameters with known effective defaults are emitted.
An unset value is omitted rather than serialized as an ambiguous `null`.

### 10. Reasoning controls

Reasoning metadata describes the controls accepted through the deployed API:

```json
{
  "reasoning": {
    "supported": true,
    "effort_levels": ["low", "medium", "high"],
    "default_effort": "medium"
  }
}
```

`default_effort`, when present, must be a member of `effort_levels`. A model may
support reasoning without accepting a client-selectable effort level; in that
case `supported` is true and `effort_levels` is empty. Configuring a reasoning
output parser does not by itself prove that effort-level controls are accepted.

### 11. Tokenizer identity

`tokenizer` is an object because repository identity, revision, implementation
class, and vocabulary size are distinct facts:

```json
{
  "tokenizer": {
    "root": "rdtand--Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm",
    "revision": "9b5389d4a1e207daab2d47732efea57d7e946dcf",
    "class": "Qwen2Tokenizer",
    "vocab_size": 248320
  }
}
```

Requirements:

- `tokenizer.root` uses the same reversible `org--repo` encoding as model
  provenance. It may equal the model `root` when tokenizer artifacts are bundled
  with the checkpoint.
- `revision` pins the tokenizer artifacts when known.
- `class` is the effective tokenizer class selected by the runner.
- `vocab_size` is the validated tokenizer vocabulary size.
- Local tokenizer paths, cache layout, and `trust_remote_code` are operational
  configuration and are not exposed as portable tokenizer identity.
- The configured tokenizer must be validated as compatible with the model; Hub
  metadata is a source hint, not sufficient proof.

### 12. Structured features

`features` is an object keyed by stable capability families, not an array of
unparameterized marketing names. Every feature has a uniform envelope:

```json
{
  "features": {
    "speculative_decoding": {
      "supported": true,
      "enabled": true,
      "parameters": {
        "method": "draft_model",
        "draft_model": "draft-model-id",
        "num_speculative_tokens": 5
      }
    },
    "kv_cache_quantization": {
      "supported": true,
      "enabled": true,
      "parameters": {
        "implementation": "turboquant",
        "bits": 4
      }
    },
    "speculative_prefill": {
      "supported": true,
      "enabled": false,
      "parameters": {
        "method": "dflash"
      }
    }
  }
}
```

The feature key names the stable capability. A named implementation or technique
is represented in `parameters` where appropriate: MTP, EAGLE, n-gram, and a
separate draft model are speculative-decoding methods; TurboQuant is a KV-cache
quantization implementation; DFlash is a speculative-prefill method.

Requirements:

- `supported` means the exact checkpoint/runner/runtime combination can use the
  feature.
- In catalog metadata, `enabled` and `parameters` describe the default load.
- In active service status, they describe the effective loaded configuration.
- Feature parameters are validated by a schema registered for that feature key.
  `parameters` is not an unvalidated arbitrary JSON bag.
- References to another managed model, such as `draft_model`, use its canonical
  `id`.
- Features that affect placement contribute their resource requirements to the
  placement profile; the scheduler does not infer cost from feature names.

### 13. Hugging Face import boundary

Hugging Face provides several formal but separate metadata surfaces; there is no
single complete schema for this router's operational model card.

- Model-card YAML has a validated schema for high-level discovery fields such as
  license, library, tags, datasets, base-model lineage, pipeline task, and
  evaluation results. It also permits additional fields and custom tags, so
  presence and precision vary by repository.
- The Hub model API aggregates repository identity/revision, card data, detected
  tags/task/library, selected configuration data, file listings, and Safetensors
  parameter counts. Most fields are optional and some are inferred.
- Transformers `config.json`, `generation_config.json`,
  `tokenizer_config.json`, and processor files have library-level schemas, but
  model architectures may add fields or place common facts at different nesting
  levels.
- Safetensors headers formally describe tensors, shapes, dtypes, and byte
  offsets. They support reliable parameter accounting; encoded checkpoint byte
  size is still measured from the deployed files.

Import policy:

| Router metadata | Hugging Face usefulness | Curation rule |
|---|---|---|
| `root`, artifact revision | authoritative repo/API identity | import and pin |
| base-model lineage, license, library | usually structured card data | import when present |
| `model_size` | file metadata is useful | calculate from deployed weight files |
| `max_context_window` | often present in model config | resolve architecture-aware and validate with runner |
| tokenizer identity/class/vocabulary | usually present in repository files | import, then validate effective selection |
| generation defaults | standardized checkpoint file when present | merge with runner/router precedence |
| modalities | task tags and config are hints | curate and validate end to end |
| reasoning levels | rarely formalized | operator-curated and API-tested |
| structured `features` and parameters | tags may hint at support | operator-curated and runtime-validated |
| placement reservations and active state | not Hub facts | measure locally |

The import process records both the imported value and the artifact revision from
which it came. Missing or contradictory Hub metadata is surfaced for curation; it
is never silently invented.

## Target response

For the first configured model, `GET /v1/models` should contain an entry shaped
like:

```json
{
  "id": "qwen3.6-27b-nvfp4",
  "object": "model",
  "created": 0,
  "owned_by": "vllm-router",
  "root": "rdtand--Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm",
  "parent": null,
  "model_size": 20171471784,
  "max_context_window": 262144,
  "default_context_window": 32768,
  "input_modalities": ["text", "image", "video"],
  "output_modalities": ["text"],
  "generation_defaults": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20
  },
  "reasoning": {
    "supported": true,
    "effort_levels": [],
    "default_effort": null
  },
  "tokenizer": {
    "root": "rdtand--Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm",
    "revision": "9b5389d4a1e207daab2d47732efea57d7e946dcf",
    "class": "Qwen2Tokenizer",
    "vocab_size": 248320
  },
  "features": {}
}
```

There is no `aliases`, `max_model_len`, or `context_window` key. The last is an
active-load fact rather than possibly-available catalog metadata.

## Validation and acceptance

Implementation is complete when:

1. Every enabled service has a valid public `id` and explicitly curated
   provenance.
2. Duplicate IDs are rejected, while replicas may intentionally share one ID
   when their model-level metadata, including `root`, is identical. Until
   context-aware admission exists, their active `context_window` is also
   identical.
3. Every configured Hugging Face `model_root` follows the reversible
   `org--repo` encoding and preserves the repository ID's case.
4. `/v1/models` returns exactly one entry per unique ID having at least one
   enabled service, with the configured `root`, no `aliases` key, and response
   constants synthesized at serialization time.
5. Only the canonical ID succeeds in request routing and at the backend; every
   former alias returns the normal unknown-model error without causing lifecycle
   work.
6. Generated vLLM units contain exactly one `--served-model-name` value.
7. Configuration containing the removed `aliases` key is rejected rather than
   silently ignored.
8. Documentation, examples, tests, and operational smoke commands use canonical
   IDs for inference. Historical acceptance records may retain old names when
   clearly describing past observations.
9. Model metadata import records its Hugging Face source revision and requires
   explicit curation for missing or contradictory fields.
10. Context-window validation enforces the three-field vocabulary and runtime
    status reports the effective `context_window`.
11. Feature descriptors pass their registered parameter schemas and
    placement-affecting features contribute measured profile reservations.

## Non-goals

- Algorithmically generating shortened IDs from Hugging Face repository names.
- Treating the local filesystem path as portable provenance.
- Encoding a snapshot SHA into `root`.
- Supporting multiple simultaneous public names for one model contract.
- Using `model_size` as a substitute for measured placement reservations.
- Treating free-form Hugging Face tags as validated structured features.
- Exposing one flat, unparameterized list of feature names.
- Renaming operational service identifiers, systemd units, or environment files
  merely because their short names are no longer inference aliases.

## References

- [Hugging Face model cards and metadata](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face model-card metadata specification](https://github.com/huggingface/hub-docs/blob/main/modelcard.md)
- [Hugging Face `ModelCardData` reference](https://huggingface.co/docs/huggingface_hub/main/package_reference/cards#huggingface_hub.ModelCardData)
- [Safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing)
