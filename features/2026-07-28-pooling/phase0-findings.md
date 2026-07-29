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

Executed on CT 107 on 2026-07-29 with two RTX 5090s and vLLM 0.26.0. Sleep
remains opt-in per service and none of the tested cells qualifies for
`hardware_validated: true`. The default/fallback path remains systemd
stop/start.

| TP | quantization | level | result |
|---:|---|---:|---|
| 1 | FP8 | 1 | 5/5 output-stable cycles; fail because inference while asleep hung and host-RAM budget was exceeded |
| 1 | FP8 | 2 | 5/5 output-stable cycles and correct three-step wake; same gating failures |
| 1 | FP8 | L1 → L2 | promotion pass; cell remains unvalidated |
| 1 | NVFP4 | 1 | 5/5 output-stable cycles; fail because inference while asleep hung and host-RAM budget was exceeded |
| 1 | NVFP4 | 2 | 5/5 output-stable cycles and correct three-step wake; same gating failures |
| 1 | NVFP4 | L1 → L2 | promotion pass; cell remains unvalidated |
| 2 | FP8 | 1 | 5/5 output-stable cycles; fail because inference while asleep hung and host-RAM budget was exceeded |
| 2 | FP8 | 2 | 5/5 output-stable cycles and correct three-step wake; same gating failures |
| 2 | FP8 | L1 → L2 | promotion pass; cell remains unvalidated |
| 2 | NVFP4 | 1 | initialization failed before the matrix |
| 2 | NVFP4 | 2 | initialization failed before the matrix |
| 2 | NVFP4 | L1 → L2 | initialization failed before the matrix |

The initialized cells released VRAM repeatably, preserved fixed-prompt output,
refused sleep safely when the drain deadline expired, and exercised the
router's L1→L2 promotion and three-step L2 wake. The common behavioral failure
was the required 503 while asleep: vLLM accepted the connection but returned no
bytes before the 2-second test timeout. The TP=2 NVFP4 launch failed in
FlashInfer FP4 quantization with `last dim has to be multiple of 16, but got
2152`. The tested service must therefore declare TP=2 in
`compatibility.unsupported_tensor_parallelism` and run at TP=1; the boot-time
validator rejects selecting a declared failed degree. Detailed latency and memory
measurements are recorded in the per-model documents and `acceptance.md`.

The provisional weight-sized L1 declarations were also insufficient. The
conservative recorded floors are 25,000,000,000 bytes for FP8 (1.25× over the
20 GB input, covering the 24.53 GB maximum) and 40,000,000,000 bytes for NVFP4
(1.33× over 30 GB, covering the approximately 39.0 GB maximum). These floors may
be retained while sleep remains disabled, and config validation rejects any
future enabled-L1 reservation below them.

## S3 — systemd D-Bus

The code uses zbus for `StartUnit`, `StopUnit`, status reads, and property-change
observation. The current container does not expose a usable systemd manager, so
the live round trip is an ignored on-box test. There is no `systemctl` call in
the steady-state implementation.
