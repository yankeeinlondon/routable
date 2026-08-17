# API Surfaces

The **LLM Router** exposes three separate API's (across three separate ports). This separation was done to address the following:

- provide full separation between "adminstrative" needs from normal "usage" needs
- allow for each API to be network bound separately in a way that is more restrictive to adminstrative needs than for general usage
- also allows for separate NIC's to be used for the different APIs so even if front facing data channel is saturated, admins can gain access to the administrative functions on a different network route.

The rest of this document will introduce the three API surfaces, what they are for, how to use them, and how they are impacted by CLI switches and ENV variables used when starting the service.

## Authentication and listener boundaries

- The inference API requires the inference bearer token on every request, including
  `GET /v1/models`. An OpenAI-compatible client should use the data-plane base URL (for
  the standard deployment, `http://<host>:6969/v1`) and configure the contents of
  `/etc/vllm-router/secrets/inference-token` as its API key.
- The management API uses a different bearer token from
  `/etc/vllm-router/secrets/management-token`. It is loopback-bound by default and that
  credential must not be given to inference clients.
- The metrics listener is independently loopback-bound. The management liveness and
  readiness endpoints are the only unauthenticated HTTP endpoints.

Both token files are persistent, root-owned, mode `0600`, non-empty, and distinct. Their
parent directory is root-owned mode `0700`. They must not be stored under `/run` or
`/var/run`, because those tmpfs paths are cleared at reboot. Before binding any listener,
the router validates each file's ownership, mode, contents, and distinct configured path.

## Inference API

The **Inference API** will be the API that most clients will interact with and it exposes:

- an OpenAI compatible API surface
- the Anthropic _messages_ API
- TODO: add cohere, others

In addition to these _standard_ API surfaces that client's will already be familiar with, it offers a small admin and information API that hangs off the `/router` path.

### Standards Based Surface

| Endpoint            | Protocol family                                                                              | 
|---------------------|----------------------------------------------------------------------------------------------|
| `GET /v1/models`    | Router model discovery (both OpenAI and Anthropic standards support this)                    |
| `GET /v1/models/{model-id}` | Get details on a specific model |
| `POST /v1/messages` | Anthropic's messages API for chat              |
| `POST /v1/messages/count_tokens` |  Anthropic endpoint to count the number of tokens in a request before sending   |
| `{verb} /v1/score`  | TODO: Cohere/Jina style API surface |
| `{verb} /v1/rerank` | TODO: Cohere/Jina style API surface |
| `{verb} /v2/rerank` | TODO: Cohere/Jina style API surface |
| `{verb} /v2/embed`  | TODO: Cohere/Jina style API surface |
| `{verb} /score`     | TODO: Cohere/Jina style API surface |
| `{verb} /rerank`    | TODO: Cohere/Jina style API surface |



| OpenAI-compatible | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses` |
| Anthropic Messages | `/v1/messages`, `/v1/messages/count_tokens` |
| Cohere/Jina-style pooling | `/v1/score`, `/v1/rerank`, `/v2/rerank`, `/v2/embed` (plus legacy `/score`, `/rerank`) |
| vLLM native | `/tokenize`, `/detokenize`, `/classify` |



### Router Surface

These are the endpoints that a typical client wouldn't know of "out of the box":

| Endpoint            | Protocol family                                                                              | 
|---------------------|----------------------------------------------------------------------------------------------|
| `POST /



| Endpoint                         | Purpose                                                                           |
|----------------------------------|-----------------------------------------------------------------------------------|
| `GET /v1/operations/{id}`        | progress / terminal result of an accepted mutation |
| `POST /v1/reconcile`             | re-observe systemd + probes + NVML and converge state |
| `GET /live`, `GET /ready`        | router process liveness / routing readiness (no auth — the only unauthenticated endpoints the router serves, and they are on the loopback management bind) |





| `POST /v1/services/{name}/start` | ensure running; body: `{"deadline_ms": …, "evict": true}` to allow displacing lower-priority idle models |
| `POST /v1/services/{name}/wake`  | same as start, for a sleeping service |
| `POST /v1/services/{name}/drain` | remove from selection and wait for in-flight work to finish |
| `POST /v1/services/{name}/sleep` | drain then sleep; body: `{"level": "l1"\|"l2"}` |
| `POST /v1/services/{name}/stop`  | drain then stop the unit; `{"force": true}` stops after the drain deadline even with work in flight and holds the service drained until an explicit start/wake |
| `GET /v1/services`               | desired config + observed state for every service (lifecycle, systemd, probes, GPU, active/parked counts, last error) |
