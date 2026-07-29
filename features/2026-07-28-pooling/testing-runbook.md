# v1 Testing Runbook

Ordered tiers, cheapest first; each tier assumes the previous passed. Status column
records what has already been verified on CT 107 (2026-07-29). The acceptance criteria
being exercised are `spec.md` §Acceptance criteria; results belong in `acceptance.md`.

Environment facts (verified 2026-07-29): both RTX 5090s visible via NVML
(`GPU-d751861a…`, `GPU-06645389…`); systemd + system D-Bus reachable from the
container; vLLM at `/root/.venv/bin/vllm` (NOT the example config's `/opt/vllm-venv`);
weights under `/models/hf/hub` (Qwen3.6-27B NVFP4, Qwen3.6-35B-A3B); warm
torch.compile caches under `/models/vllm/torch_compile_cache`.

## Tier 0 — fast suite (anywhere) — ✅ green

```console
just lint && just test        # 126 lib + 4 bin tests, 1 host-only ignored
```

## Tier 1 — host integration probes (read-only) — ✅ green

```console
cargo test --lib -- --ignored         # live systemd D-Bus round trip (passed 2026-07-29)
nvidia-smi -L                          # real GPU UUIDs for the config
```

## Tier 2 — real configuration validation

Write a real `router.yaml` from `src/config/example.yaml` with: the two real GPU
UUIDs and 32 GB capacities; `runner_executable: /root/.venv/bin/vllm`; one service for
the 27B NVFP4 model (TP=1, one GPU, `sleep.enabled: false`); backend on
`127.0.0.1:8100`; generous `deadlines.startup_ms` (≥ 600000 — NVFP4 warmup on sm_120
is documented at minutes-scale cold). Create the token:

```console
install -m 0600 /dev/null /run/secrets/vllm-router-token
openssl rand -hex 32 > /run/secrets/vllm-router-token
vllm-router -t -c deploy/pingora.yaml --router-config /etc/vllm-router/router.yaml
```

Note while configuring: check whether the generated environment can carry
`VLLM_CACHE_ROOT` so services reuse `/models/vllm/torch_compile_cache` — if the
generator's env is fully fixed, cold starts pay full compile each time (candidate
small gap; record what you find).

## Tier 3 — single-model smoke (first real end-to-end)

Run the router in the foreground first (`RUST_LOG=info`), systemd unit later.

1. `curl :8001/livez` and `/readyz` — ready via automatic-start reachability.
2. `curl -H "Authorization: Bearer $TOKEN" :8001/pool/v1/services | jq` — service
   `stopped`, real GPU observations, probe ages.
3. **Cold start + parking:** `time curl :8000/v1/chat/completions -d '{"model":…}'` —
   request parks, unit starts (`journalctl -u vllm@<name> -f`), answer arrives after
   readiness + warmup. Second request is warm and fast.
4. **Streaming:** `curl -N … "stream": true` — tokens arrive incrementally; check
   `vllm_router_active_streams` on `:8002/metrics` during the stream.
5. `/v1/models` lists the model + aliases; **an alias-named completion request must
   succeed end-to-end** (exercises `--served-model-name` alias rendering — review-3
   F1); unknown model → OpenAI-shaped 404;
   `/v1/messages` bad request → Anthropic-shaped error.
6. **Idle tier:** set `idle.stop_after_ms` low (e.g. 120000), watch the reaper drain
   and stop via `/pool/v1/services` + `vllm_router_transitions_total`.
7. Admin surface: `POST /pool/v1/services/<name>/stop` with deadline; operation ID
   polls to `succeeded`; audit lines in the journal; wrong/missing token → 401.

## Tier 4 — acceptance drills (map 1:1 to spec §Acceptance criteria)

- **Stampede:** 50 concurrent requests at a stopped model (`oha`/`xh`) → exactly one
  start (journal + `joined_existing` audits), all succeed or bounded-fail; parked
  caps enforced when exceeding `parking.queue_limit`.
- **Disconnect-while-parked:** Ctrl-C a parked curl → slot freed (metrics), start
  continues, other waiters unaffected.
- **Streaming timeouts:** a long active stream survives past `inter_chunk_timeout_ms`
  total duration; a stall (SIGSTOP the vLLM process mid-stream) kills the stream with
  a clear error and releases counters.
- **Crash:** `kill -9` the vLLM main PID mid-stream → clean downstream error, service
  observed failed/inactive, next request re-parks and restarts it.
- **Router restart under traffic:** `kill` the router during an active stream + idle
  model running → model units untouched; restarted router reconciles to correct
  state; stream (necessarily) broken — no admission to unready backends afterward.
- **Graceful upgrade:** new binary `-t`, then `-u`, SIGQUIT old — active SSE stream
  survives the handoff; models untouched.
- **Eviction:** two services pinned to the same GPU; request model B while A idles →
  A drains and stops, B starts, B's request parks throughout. Verify a busy A is
  never chosen (fire requests at A continuously and request B → B gets
  placement-unavailable/503 at deadline).
- **Force-stop:** `stop` with `{"force": true}` against in-flight requests → drain
  deadline passes, unit stops, audit line carries `affected_requests > 0`.
- **Negative:** oversized body (413), malformed JSON (400), `/sleep` on the data
  plane (404) — confirm none of them triggered any lifecycle action
  (`vllm_router_transitions_total` unchanged).

## Tier 5 — S2 sleep-mode matrix (gates `hardware_validated`)

Per `phase0-findings.md`: for each {TP=1, TP=2} × {FP8, NVFP4} × {L1, L2} cell — five
sleep/wake cycles with fixed-prompt output sanity, wake latency, residual VRAM
(nvidia-smi), pinned host RAM, 503-while-asleep, drain-then-sleep, sleep-mid-request
refusal, L2 three-step wake, **and the L1→L2 tier promotion**. Only passing cells get
`sleep: {enabled: true, hardware_validated: true}`. Record per-model results in
`docs/{model-id}.md` per convention — these measurements are also v2 planner inputs.

## Tier 6 — load & soak

`oha` sustained SSE fan-out at the warm model (watch histograms, active_streams,
memory of the router process); parked-cap stress (global and per-model 503s); leave
the idle-tier cycle running overnight and check for drift/leaks in the morning.

## Recording

Update `acceptance.md` per tier with pass/fail + date; failures become numbered
findings (review-3 if any). Keep `journalctl` extracts for the drills.
