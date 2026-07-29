# Phase 0 findings

Recorded against the pinned source and this checkout on 2026-07-28.

## S1 — Pingora request-body replay

The pinned Pingora 0.8.1 `ProxyHttp` path cannot provide the general body
read/replay primitive required by this router:

- `upstream_peer` runs before normal request-body streaming.
- `ServerSession::read_request_body` can consume the body before peer selection.
- the only public replay facility is the retry buffer, whose hard-coded
  `BODY_BUF_LIMIT` is 64 KiB for both HTTP/1 and HTTP/2;
- there is no public API to inject an arbitrary consumed body back into the
  downstream session.

A 64-KiB ceiling would reject normal multimodal OpenAI payloads and therefore
does not satisfy the spec's configurable bounded-body requirement. The data
plane consequently uses the spec's fallback: a custom `HttpServerApp` ingress
inside the same Pingora binary. It buffers only the bounded downstream request,
preserves its bytes, and streams the upstream response incrementally.

The pure routing tests prove that malformed, missing-model, unknown-model, and
oversized requests fail before any lifecycle admission. End-to-end streaming
and delayed-readiness drills are included in the ignored on-box acceptance
suite because this container currently has no reachable vLLM GPU service.

## S2 — sm_120 sleep matrix

Not executable in the current environment: `nvidia-smi` cannot communicate
with the NVIDIA driver. Sleep remains opt-in per service and configuration
validation requires it to be explicitly marked validated. The default/fallback
path is systemd stop/start.

Required matrix remains:

| TP | quantization | level | result |
|---:|---|---:|---|
| 1 | FP8 | 1 | pending on-box |
| 1 | FP8 | 2 | pending on-box |
| 1 | FP8 | L1 → L2 | pending on-box |
| 1 | NVFP4 | 1 | pending on-box |
| 1 | NVFP4 | 2 | pending on-box |
| 1 | NVFP4 | L1 → L2 | pending on-box |
| 2 | FP8 | 1 | pending on-box |
| 2 | FP8 | 2 | pending on-box |
| 2 | FP8 | L1 → L2 | pending on-box |
| 2 | NVFP4 | 1 | pending on-box |
| 2 | NVFP4 | 2 | pending on-box |
| 2 | NVFP4 | L1 → L2 | pending on-box |

Each run must record five sleep/wake cycles, fixed-prompt output sanity, wake
latency, `nvidia-smi` residual VRAM, pinned host RAM, 503 while asleep,
drain-then-sleep, attempted sleep during a request, and the L2 reload sequence.
The L1 → L2 cells specifically validate promotion by waking the L1 sleeper and
then issuing `POST /sleep?level=2`, with no inference admitted between those
steps. Installed vLLM 0.26 source returns early from a sleep command when the
executor is already sleeping, so a direct second sleep call cannot perform the
promotion. The router's request sequence and lifecycle update are covered by
mock tests; VRAM release, output sanity, and transition latency remain gated on
the real sm_120 host.

## S3 — systemd D-Bus

The code uses zbus for `StartUnit`, `StopUnit`, status reads, and property-change
observation. The current container does not expose a usable systemd manager, so
the live round trip is an ignored on-box test. There is no `systemctl` call in
the steady-state implementation.
