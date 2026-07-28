# vLLM pooling router

A single Pingora 0.8.1 binary that routes OpenAI-compatible model requests to a
bounded pool of systemd-managed vLLM 0.26 services. The pool manager owns
lifecycle and GPU placement; the ingress and control API depend on it through
the `PoolManager` trait.

The implementation follows
[`features/2026-07-28-pooling/plan.md`](features/2026-07-28-pooling/plan.md).
Hardware gates and their current status are recorded in
[`phase0-findings.md`](features/2026-07-28-pooling/phase0-findings.md).

## Development

```console
cargo run -- -c deploy/pingora.yaml --router-config src/config/example.yaml
cargo run -- -t -c deploy/pingora.yaml --router-config src/config/example.yaml
scripts/ci.sh
```

The public data listener, authenticated management listener, and Prometheus
listener are configured separately. Backend ports must be loopback-only.

## Deployment outline

1. Install the release binary as `/usr/local/bin/vllm-router`.
2. Install `deploy/vllm@.service` and `deploy/vllm-router.service`.
3. Put generated, mode-0600 vLLM environment files under `/etc/vllm`.
4. Put router configuration and a root-owned mode-0600 bearer token under
   `/etc/vllm-router`.
5. Run `vllm-router -t ...`, then `systemctl daemon-reload` and start the router.

The router never stops healthy model units during its own shutdown or upgrade.
Pingora's upgrade socket hands listening file descriptors to a `-u` process;
the old process drains active SSE streams within the configured grace period.
Perform that handoff from the deployment supervisor only after the new binary
passes `-t`; do not treat a bare `SIGQUIT` as a complete reload procedure.
