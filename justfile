set shell := ["bash", "-euo", "pipefail", "-c"]

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
