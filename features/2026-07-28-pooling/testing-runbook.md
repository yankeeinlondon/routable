# v1 Testing Runbook

Ordered tiers, cheapest first; each tier assumes the previous passed. Status column
records what has already been verified on CT 107 (2026-07-29). The acceptance criteria
being exercised are `spec.md` §Acceptance criteria; results belong in `acceptance.md`.

Most commands below have `just` shortcuts (see `just --list`): `gpus`, `host-test`,
`token`, `validate`, `run`, `ready`, `pool`/`pool-raw`, `pool-op <svc> <action>
['{json}']`, `operation <id>`, `models`, `chat <model>`, `stream <model>`,
`stampede <model> [n]`, `metrics`, `unit-logs <svc>`. Ports default to 6969/6970/6971
and override per call (`DATA_PORT=9000 just run`) — `just run` patches a temp config
copy so the router and every client recipe agree; production uses the config file's
ports as-is. Full URLs/paths override via `DATA`/`MGMT`/`METRICS`/`ROUTER_CONFIG`/
`PINGORA_CONFIG`/`TOKEN_FILE`.

Environment facts (verified 2026-07-29): both RTX 5090s visible via NVML
(`GPU-d751861a…`, `GPU-06645389…`); systemd + system D-Bus reachable from the
container; vLLM at `/root/.venv/bin/vllm` (NOT the example config's `/opt/vllm-venv`);
weights under `/models/hf/hub` (Qwen3.6-27B NVFP4, Qwen3.6-35B-A3B); warm
torch.compile caches under `/models/vllm/torch_compile_cache`.

## Tier 0 — fast suite (anywhere) — ✅ green

```console
just lint && just test        # 136 passed lib + 4 bin; 2 host-only ignored (1 lib + 1 integration)
```

## Tier 1 — host integration probes — ✅ green

```console
cargo test --lib -- --ignored                                      # live systemd D-Bus round trip (passed 2026-07-29)
cargo test --test binary_serves -- --ignored --nocapture           # built binary reaches /livez, then exits on SIGTERM (passed 2026-07-29)
nvidia-smi -L                                                       # real GPU UUIDs for the config
```

The binary-serving probe creates one uniquely named generated systemd unit and unique
Pingora PID/upgrade-socket paths; its cleanup removes the unit and reloads systemd,
including when the assertion path fails.

## Tier 2 — real configuration validation

Write a real `router.yaml` from `src/config/example.yaml` with: the two real GPU
UUIDs and 32 GB capacities; `runner_executable: /root/.venv/bin/vllm`; one service for
the 27B NVFP4 model (TP=1, one GPU, `sleep.enabled: false`); backend on
`127.0.0.1:8100`; generous `deadlines.startup_ms` (≥ 600000 — NVFP4 warmup on sm_120
is documented at minutes-scale cold). Create the token:

```console
install -d -m 0700 /etc/vllm-router/secrets
openssl rand -out /etc/vllm-router/secrets/management-token -hex 32
chmod 0600 /etc/vllm-router/secrets/management-token
vllm-router -t -c deploy/pingora.yaml --router-config /etc/vllm-router/router.yaml
```

Set global `environment:` defaults for the CUDA toolchain `PATH`, `HF_HOME`, and
`VLLM_CACHE_ROOT`; inspect the generated service env to confirm the values and numeric
`CUDA_VISIBLE_DEVICES` are present.

## Tier 3 — single-model smoke (first real end-to-end)

Run the router in the foreground first (`RUST_LOG=info`), systemd unit later.

1. `curl :6970/livez` and `/readyz` — ready via automatic-start reachability.
2. `curl -H "Authorization: Bearer $TOKEN" :6970/pool/v1/services | jq` — service
   `stopped`, real GPU observations, probe ages.
3. **Cold start + short parking:** `time curl :6969/v1/chat/completions -d
   '{"model":…}'` — the request parks briefly while the unit starts
   (`journalctl -u vllm@<name> -f`). Fast wakes return the answer directly; a true
   cold start returns a progress-reporting 503 with `Retry-After` when the short
   parking deadline expires. The detached start must continue, and a retry after
   readiness is warm and fast.
4. **Streaming:** `curl -N … "stream": true` — tokens arrive incrementally; check
   `vllm_router_active_streams` on `:6971/metrics` during the stream.
5. `/v1/models` lists **exactly one entry per unique served model** (canonical `id`,
   aliases on the entry's `aliases` extension field — review-3 F2, no duplicate
   entries); **an alias-named completion (`just chat <configured-alias>`) must return
   a generated answer end-to-end** (exercises `--served-model-name` alias rendering
   — review-3 F1); unknown model → OpenAI-shaped 404;
   `/v1/messages` bad request → Anthropic-shaped error.
   Then run `just tool-smoke <canonical-or-alias>`: it must produce at least one
   structured `tool_calls` entry, non-empty separated `reasoning` (or the older
   `reasoning_content` spelling), and no leaked `<think>` tag in `content`
   (review-3 F9).
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
  a clear `upstream response idle timeout` error and releases counters.
- **Crash:** `kill -9` the vLLM main PID mid-stream → clean downstream error, service
  observed failed/inactive, next request re-parks and restarts it.
- **Router restart under traffic:** `systemctl restart` (which sends the unit's
  `KillSignal=SIGINT` fast shutdown) during an active stream + idle model running →
  the router exits promptly rather than waiting out the grace window; model units
  untouched; restarted router reconciles to correct state; stream (necessarily)
  broken — no admission to unready backends afterward. Time the stop: a restart that
  takes the full `grace_period_seconds` means the unit is delivering SIGTERM, which is
  the graceful path and always sleeps the whole window even with no traffic.
- **Graceful upgrade:** new binary `-t`, then `-u`, SIGQUIT old — active SSE stream
  survives the handoff; models untouched. The old process only exits after the full
  `grace_period_seconds`, so expect an 11-minute lingering process, and drive this by
  hand rather than through `systemctl restart`.
- **Eviction:** two services pinned to the same GPU; request model B while A idles →
  A drains and stops, B starts, B's request parks throughout. Verify a busy A is
  never chosen (fire requests at A continuously and request B → B gets
  placement-unavailable/503 at deadline).
- **Force-stop:** `stop` with `{"force": true}` against a known number of in-flight
  requests → drain deadline passes, the unit remains inactive despite retry traffic,
  and the terminal audit line carries that exact `affected_requests` count. Within the
  running router only an explicit start/wake clears the drained stop latch; lifecycle
  state is not persisted, so also confirm the documented boundary — restart the router
  and verify the force-stopped service reconciles to `stopped` and that demand traffic
  restarts it again. Keeping a model down across a router restart is a configuration
  change (`enabled: false` or `automatic_start: false`), not a runtime latch.
- **Negative:** oversized body (413), malformed JSON (400), `/sleep` on the data
  plane (404) — confirm none of them triggered any lifecycle action
  (`vllm_router_transitions_total` unchanged).

## Tier 5 — S2 sleep-mode matrix (gates `hardware_validated`)

Per `phase0-findings.md`: for each {TP=1, TP=2} × {FP8, NVFP4} × {L1, L2} cell — five
sleep/wake cycles with fixed-prompt output sanity, wake latency, residual VRAM
(nvidia-smi), pinned host RAM, direct-asleep behavior characterization, drain-then-sleep,
sleep-mid-request refusal, L2 three-step wake, **and the L1→L2 tier promotion**. The
router-level drill must prove that fresh fail-closed admission rejects stale or asleep
facts, wakes and reprobes before forwarding, and bounds any post-admission stall with
the inter-chunk idle timeout; do not expect or depend on a backend 503. Read the
refusals off `vllm_router_admission_rejections_total{reason=…}` rather than inferring
them from 503 bodies, and check `reason="stale_probe"` stays at zero during ordinary
traffic — a nonzero count there means observation is falling behind
`readiness.maximum_probe_age_ms` and would 503 every service, not just the asleep one. Only passing
cells get `sleep: {enabled: true, hardware_validated: true}`. Record per-model results
in `docs/{model-id}.md` per convention — these measurements are also v2 planner inputs.
Any topology that fails before the matrix must also be recorded in the service's
`compatibility.unsupported_tensor_parallelism`; configuration validation must reject
that degree until the exact checkpoint/runtime/host cell is revalidated successfully.
Round the maximum measured L1 `MemoryCurrent` ratio up with margin and store the
resulting byte floor in `compatibility.minimum_l1_host_ram_bytes` even if sleep remains
disabled. A future L1 declaration must meet that floor before the aggregate host-RAM
budget can validate.

## Tier 6 — load & soak

`oha` sustained SSE fan-out at the warm model (watch histograms, active_streams,
memory of the router process); parked-cap stress (global and per-model 503s); leave
the idle-tier cycle running overnight and check for drift/leaks in the morning.

## Recording

Update `acceptance.md` per tier with pass/fail + date; failures become numbered
findings (review-3 if any). Keep `journalctl` extracts for the drills.
