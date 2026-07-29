set shell := ["bash", "-euo", "pipefail", "-c"]


bold := '\033[1m'
dim := '\033[2m'
italic := '\033[3m'
reset := '\033[0m'
red := '\033[31m'
green := '\033[32m'
yellow := '\033[33m'
blue := '\033[34m'
magenta := '\033[35m'
cyan := '\033[36m'

default:
    #!/usr/bin/env bash
    set -euo pipefail

    echo
    echo -e "{{ bold }}Local AI Router{{ reset }}"
    echo "========================================================"
    echo
    just --list | grep -v 'default'
    echo


# Run the complete test suite.
test:
    cargo test --locked --all-targets --all-features

# Build an optimized router binary.
build:
    cargo build --locked --release --all-features

# Install the router with Cargo. Extra arguments are forwarded, e.g.
# `just install --root /usr/local`.
install *args:
    cargo install --path . --locked --force --all-features {{args}}

# Check formatting and fail on every Clippy warning.
lint:
    cargo fmt --all -- --check
    cargo clippy --locked --all-targets --all-features -- -D warnings


# Run `just _colors` if you want to exercise this (it does nothing).
_colors:
    @: {{ bold }} {{ italic }} {{ reset }} {{ red }} {{ green }} {{ yellow }} {{ blue }} {{ magenta }} {{ cyan }} {{ dim }} >/dev/null || true


# ── Testing & operations ────────────────────────────────────────────
# Override via environment, e.g. `MGMT=http://127.0.0.1:9001 just pool`.

router_config := env_var_or_default('ROUTER_CONFIG', '/etc/vllm-router/router.yaml')
pingora_config := env_var_or_default('PINGORA_CONFIG', '/etc/vllm-router/pingora.yaml')
token_file := env_var_or_default('TOKEN_FILE', '/run/secrets/vllm-router-token')

# Ports `just run` binds and every client recipe targets (override per call,
# e.g. `DATA_PORT=9000 just run`). The example-config conventions:
data_port := env_var_or_default('DATA_PORT', '8000')
mgmt_port := env_var_or_default('MGMT_PORT', '8001')
metrics_port := env_var_or_default('METRICS_PORT', '8002')

data := env_var_or_default('DATA', 'http://127.0.0.1:' + data_port)
mgmt := env_var_or_default('MGMT', 'http://127.0.0.1:' + mgmt_port)
metrics_url := env_var_or_default('METRICS', 'http://127.0.0.1:' + metrics_port)

# List host GPUs with the stable UUIDs the config needs.
gpus:
    nvidia-smi -L

# Run the host-only integration tests (live systemd D-Bus round trip).
host-test:
    cargo test --lib -- --ignored

# Create the management bearer token file (0600) unless it already exists.
token:
    @test -s {{token_file}} && echo "token already present: {{token_file}}" || \
        (mkdir -p "$(dirname {{token_file}})" && umask 077 && \
         openssl rand -hex 32 > {{token_file}} && echo "token written: {{token_file}}")

# Validate the full configuration without starting anything (-t mode).
validate:
    cargo run --locked --release --all-features -- -t -c {{pingora_config}} --router-config {{router_config}}

# Run the router in the foreground (info logs) on DATA_PORT/MGMT_PORT/METRICS_PORT
# (defaults 8000/8001/8002), patching a temp copy of the config to those ports.
# Production/systemd uses the config file's ports as-is — this override is dev-only.
run:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp="$(mktemp --suffix=.router.yaml)"
    trap 'rm -f "$tmp"' EXIT
    sed -E \
        -e 's#(data_plane:[[:space:]]*"?[^":]+:)[0-9]+#\1{{data_port}}#' \
        -e 's#(management:[[:space:]]*"?[^":]+:)[0-9]+#\1{{mgmt_port}}#' \
        -e 's#(metrics:[[:space:]]*"?[^":]+:)[0-9]+#\1{{metrics_port}}#' \
        {{router_config}} > "$tmp"
    echo "data-plane :{{data_port}}  management :{{mgmt_port}}  metrics :{{metrics_port}}"
    RUST_LOG=info cargo run --locked --release --all-features -- \
        -c {{pingora_config}} --router-config "$tmp"

# Router liveness and routing readiness.
ready:
    @curl -sS {{mgmt}}/livez | jq -c .
    @curl -sS {{mgmt}}/readyz | jq -c .

# Authenticated pool status: every service's desired + observed state.
pool:
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" \
        {{mgmt}}/pool/v1/services | jq '.services[] | {service, lifecycle, active_requests, parked_requests, last_error}'

# Full pool status, unabridged JSON.
pool-raw:
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" {{mgmt}}/pool/v1/services | jq .

# Run a lifecycle action (start|wake|drain|sleep|stop) with an optional JSON body.
pool-op service action body='{}':
    @curl -sS -X POST -H "Authorization: Bearer $(cat {{token_file}})" \
        -H 'content-type: application/json' -d '{{body}}' \
        {{mgmt}}/pool/v1/services/{{service}}/{{action}} | jq .

# Poll an asynchronous operation by id.
operation id:
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" \
        {{mgmt}}/pool/v1/operations/{{id}} | jq .

# The router's synthesized model list (what clients see).
models:
    @curl -sS {{data}}/v1/models | jq .

# One timed non-streaming completion; a cold model exercises parking.
chat model prompt='Say hello in five words.':
    time curl -sS -X POST {{data}}/v1/chat/completions \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"{{prompt}}"}],"max_tokens":64}' | jq -r '.choices[0].message.content // .'

# A streaming completion; tokens should render incrementally.
stream model prompt='Count slowly from one to twenty.':
    curl -sN -X POST {{data}}/v1/chat/completions \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"{{prompt}}"}],"max_tokens":128,"stream":true}'

# Fire n concurrent requests at a model and tally status codes (stampede drill).
stampede model n='20':
    seq {{n}} | xargs -P {{n}} -I_ curl -s -o /dev/null -w "%{http_code}\n" \
        -X POST {{data}}/v1/chat/completions \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
        | sort | uniq -c

# Current router metrics (Prometheus text).
metrics:
    @curl -sS {{metrics_url}}/metrics | grep '^vllm_router'

# Follow a managed model unit's journal.
unit-logs service:
    journalctl -u vllm@{{service}}.service -f

