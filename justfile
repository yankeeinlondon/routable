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




# ── Testing & operations ────────────────────────────────────────────
# Override via environment, e.g. `MGMT=http://127.0.0.1:9001 just pool`.

router_config := env_var_or_default('ROUTER_CONFIG', '/etc/vllm-router/router.yaml')
pingora_config := env_var_or_default('PINGORA_CONFIG', '/etc/vllm-router/pingora.yaml')
token_file := env_var_or_default('TOKEN_FILE', '/etc/vllm-router/secrets/management-token')
inference_token_file := env_var_or_default('INFERENCE_TOKEN_FILE', '/etc/vllm-router/secrets/inference-token')

# Ports `just run` binds and every client recipe targets (override per call,
# e.g. `DATA_PORT=9000 just run`). The example-config conventions:
data_port := env_var_or_default('DATA_PORT', '6969')
mgmt_port := env_var_or_default('MGMT_PORT', '6970')
metrics_port := env_var_or_default('METRICS_PORT', '6971')

data := env_var_or_default('DATA', 'http://127.0.0.1:' + data_port)
mgmt := env_var_or_default('MGMT', 'http://127.0.0.1:' + mgmt_port)
metrics_url := env_var_or_default('METRICS', 'http://127.0.0.1:' + metrics_port)

# List host GPUs with the stable UUIDs the config needs.
gpus:
    nvidia-smi -L

# Run the host-only integration tests (live systemd D-Bus round trip).
host-test:
    cargo test --lib -- --ignored

# Create both bearer token files (root-owned, 0600) unless they already exist.
# The router requires two distinct, persistent secrets: the management token authorizes
# lifecycle mutations, the inference token authorizes data-plane requests. Run as
# root — the router refuses a token file it does not own. Never place client
# credentials under /run: it is a tmpfs and is erased at reboot.
token:
    #!/usr/bin/env bash
    set -euo pipefail
    for file in {{token_file}} {{inference_token_file}}; do
        if [[ -s "$file" ]]; then
            echo "token already present: $file"
            continue
        fi
        mkdir -p "$(dirname "$file")"
        (umask 077 && openssl rand -hex 32 > "$file")
        echo "token written: $file"
    done

# Fail fast with bootstrap instructions when the config files are missing.
_need-config:
    @test -f {{router_config}} || { \
        echo "router config not found: {{router_config}}"; \
        echo "bootstrap:  mkdir -p /etc/vllm-router && cp src/config/example.yaml {{router_config}} && cp deploy/pingora.yaml {{pingora_config}}"; \
        echo "then edit the GPU UUIDs (just gpus), model paths, and runner_executable —"; \
        echo "or point elsewhere: ROUTER_CONFIG=path/to/router.yaml just run"; \
        exit 1; }
    @test -f {{pingora_config}} || { \
        echo "pingora config not found: {{pingora_config}} (cp deploy/pingora.yaml {{pingora_config}})"; \
        exit 1; }

# Validate the full configuration without starting anything (-t mode).
validate: _need-config
    cargo run --locked --release --all-features -- -t -c {{pingora_config}} --router-config {{router_config}}

# Run the router in the foreground (info logs) on DATA_PORT/MGMT_PORT/METRICS_PORT
# (defaults 6969/6970/6971), patching a temp copy of the config to those ports.
# Production/systemd uses the config file's ports as-is — this override is dev-only.
run: _need-config
    #!/usr/bin/env bash
    set -euo pipefail
    if curl -s -o /dev/null --max-time 2 {{mgmt}}/livez; then
        echo "a router is already serving at {{mgmt}} (PID $(pgrep -f 'vllm-router -c' | head -1))."
        echo "stop it first, or run on other ports: DATA_PORT=7969 MGMT_PORT=7970 METRICS_PORT=7971 just run"
        exit 1
    fi
    tmp="$(mktemp --suffix=.router.yaml)"
    trap 'rm -f "$tmp"' EXIT
    sed -E \
        -e 's#(data_plane:[[:space:]]*"?[^":]+:)[0-9]+#\1{{data_port}}#' \
        -e 's#(management:[[:space:]]*"?[^":]+:)[0-9]+#\1{{mgmt_port}}#' \
        -e 's#(metrics:[[:space:]]*"?[^":]+:)[0-9]+#\1{{metrics_port}}#' \
        {{router_config}} > "$tmp"
    echo
    echo -e "{{bold}}{{green}}vLLM router starting{{reset}}"
    echo -e "  data-plane  {{bold}}http://0.0.0.0:{{data_port}}{{reset}}"
    echo -e "  management  {{bold}}http://127.0.0.1:{{mgmt_port}}{{reset}}"
    echo -e "  metrics     {{bold}}http://127.0.0.1:{{metrics_port}}{{reset}}"
    echo
    RUST_LOG=info cargo run --locked --release --all-features -- \
        -c {{pingora_config}} --router-config "$tmp"

# Fail fast with a friendly message when the router isn't up.
_need-router:
    @curl -s -o /dev/null --max-time 2 {{mgmt}}/livez || { \
        echo "router is not reachable at {{mgmt}}."; \
        echo "start it with:  just run   (or check MGMT_PORT if you overrode ports)"; \
        exit 1; }

# Router liveness and routing readiness.
ready: _need-router
    @curl -sS {{mgmt}}/livez | jq -c .
    @curl -sS {{mgmt}}/readyz | jq -c .

# Authenticated pool status: every service's desired + observed state.
pool: _need-router
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" \
        {{mgmt}}/pool/v1/services | jq -c '.services[] | {name: .desired.name, lifecycle: .observed.lifecycle, active: .observed.active_requests, parked: .observed.parked_requests, error: .observed.last_error}'

# Full pool status, unabridged JSON.
pool-raw: _need-router
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" {{mgmt}}/pool/v1/services | jq .

# Run a lifecycle action (start|wake|drain|sleep|stop) with an optional JSON body.
pool-op service action body='{}': _need-router
    @curl -sS -X POST -H "Authorization: Bearer $(cat {{token_file}})" \
        -H 'content-type: application/json' -d '{{body}}' \
        {{mgmt}}/pool/v1/services/{{service}}/{{action}} | jq .

# Poll an asynchronous operation by id.
operation id: _need-router
    @curl -sS -H "Authorization: Bearer $(cat {{token_file}})" \
        {{mgmt}}/pool/v1/operations/{{id}} | jq .

# The router's synthesized model list (what clients see).
models: _need-router
    @curl -sS -H "Authorization: Bearer $(cat {{inference_token_file}})" \
        {{data}}/v1/models | jq .

# One timed non-streaming completion; a cold model exercises parking.
chat model prompt='Say hello in five words.': _need-router
    time curl -sS -X POST {{data}}/v1/chat/completions \
        -H "Authorization: Bearer $(cat {{inference_token_file}})" \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"{{prompt}}"}],"max_tokens":64}' | jq -r '.choices[0].message.content // .'

# A streaming completion; tokens should render incrementally.
stream model prompt='Count slowly from one to twenty.': _need-router
    curl -sN -X POST {{data}}/v1/chat/completions \
        -H "Authorization: Bearer $(cat {{inference_token_file}})" \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"{{prompt}}"}],"max_tokens":128,"stream":true}'

# Assert tool parsing and reasoning separation for a thinking chat template.
tool-smoke model: _need-router
    #!/usr/bin/env bash
    set -euo pipefail
    response="$(curl -sS -X POST {{data}}/v1/chat/completions \
        -H "Authorization: Bearer $(cat {{inference_token_file}})" \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"Use get_weather to check Paris, France. Do not answer without calling the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":512}')"
    jq . <<<"$response"
    jq -e '
        .choices[0].message as $message
        | ($message.tool_calls | type == "array" and length > 0)
          and ($message.tool_calls[0].function.name == "get_weather")
          and (($message.tool_calls[0].function.arguments | fromjson | .city)
            | type == "string" and length > 0)
          and (($message.reasoning // $message.reasoning_content) | type == "string" and length > 0)
          and (($message.content // "") | contains("<think>") | not)
    ' <<<"$response" >/dev/null
    echo "tool call parsed and reasoning separated"

# Fire n concurrent requests at a model and tally status codes (stampede drill).
stampede model n='20': _need-router
    seq {{n}} | xargs -P {{n}} -I__REQUEST__ curl -s -o /dev/null -w "%{http_code}\n" \
        -X POST {{data}}/v1/chat/completions \
        -H "Authorization: Bearer $(cat {{inference_token_file}})" \
        -H 'content-type: application/json' \
        -d '{"model":"{{model}}","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
        | sort | uniq -c

# Current router metrics (Prometheus text).
metrics: _need-router
    @curl -sS {{metrics_url}}/metrics | grep '^vllm_router'

# Follow a managed model unit's journal.
unit-logs service:
    journalctl -u vllm@{{service}}.service -f

# start claude code CLI
cc *args="":
  claude --dangerously-skip-permissions

# start codex CLI
codex *args="":
  codex --dangerously-bypass-approvals-and-sandbox


# Run `just _colors` if you want to exercise this (it does nothing).
_colors:
    @: {{ bold }} {{ italic }} {{ reset }} {{ red }} {{ green }} {{ yellow }} {{ blue }} {{ magenta }} {{ cyan }} {{ dim }} >/dev/null || true
