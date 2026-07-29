//! `-t` is the documented pre-upgrade gate, so it has to fail for every reason
//! the router would fail to start. These drive the real binary because the gap
//! being covered is *where* a check runs relative to Pingora's test-mode exit,
//! which only a subprocess can observe.

use std::{
    fs,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use tempfile::TempDir;
use vllm_router::{
    config::{GpuDefinition, RouterConfig},
    gpu::GpuInventory,
};

const ABSENT_GPU_UUID: &str = "GPU-00000000-0000-0000-0000-000000000001";

#[test]
fn test_mode_rejects_a_gpu_uuid_this_host_does_not_have() {
    let directory = tempfile::tempdir().expect("temporary directory should be available");
    // The checked-in example's placeholder UUID is the exact configuration that
    // used to pass `-t` and then fail at startup.
    let output = run_test_mode(&directory, |_| {});

    assert!(
        !output.status.success(),
        "-t accepted a GPU UUID no host has: {output:?}"
    );
    if GpuInventory::initialize().is_available() {
        assert!(
            stderr(&output).contains(ABSENT_GPU_UUID),
            "-t did not name the unknown GPU UUID: {}",
            stderr(&output)
        );
    }
}

#[test]
fn test_mode_rejects_an_unreadable_bearer_token_file() {
    let directory = tempfile::tempdir().expect("temporary directory should be available");
    let absent = directory.path().join("absent-inference-token");
    let output = run_test_mode(&directory, |config| {
        config.listeners.inference_bearer_token_file = absent;
    });

    assert!(
        !output.status.success(),
        "-t accepted a missing bearer-token file: {output:?}"
    );
    assert!(
        stderr(&output).contains("cannot read bearer-token file"),
        "-t did not report the unreadable token file: {}",
        stderr(&output)
    );
}

#[test]
fn test_mode_accepts_a_configuration_this_host_can_run() {
    let inventory = GpuInventory::initialize();
    // SAFETY: `geteuid` takes no arguments and cannot fail.
    if !inventory.is_available() || unsafe { libc::geteuid() } != 0 {
        eprintln!("skipped: a live NVML inventory and root-owned token files are required");
        return;
    }
    let gpu = inventory
        .inventory()
        .expect("available NVML should enumerate devices")
        .into_values()
        .next()
        .expect("this host test requires at least one GPU");

    let directory = tempfile::tempdir().expect("temporary directory should be available");
    let output = run_test_mode(&directory, |config| {
        config.inventory.gpus = vec![GpuDefinition {
            uuid: gpu.uuid.clone(),
            vram_capacity_bytes: gpu.total_memory_bytes,
        }];
        let service = config
            .services
            .first_mut()
            .expect("example defines a service");
        service.placement.gpu_uuids = vec![gpu.uuid.clone()];
        service.placement.vram_reservation_bytes_per_gpu = gpu.total_memory_bytes;
    });

    assert!(
        output.status.success(),
        "-t rejected a configuration this host can run: {}",
        stderr(&output)
    );
}

/// Run `-t` over the example configuration with `adjust` applied, plus token
/// files that exist so an unrelated check cannot be the reason for the verdict.
fn run_test_mode(directory: &TempDir, adjust: impl FnOnce(&mut RouterConfig)) -> Output {
    let mut config: RouterConfig = serde_yaml::from_str(include_str!("../src/config/example.yaml"))
        .expect("example configuration should parse");
    config.listeners.data_plane = SocketAddr::from(([127, 0, 0, 1], 16969));
    config.listeners.management = SocketAddr::from(([127, 0, 0, 1], 16970));
    config.listeners.metrics = SocketAddr::from(([127, 0, 0, 1], 16971));
    config.listeners.management_bearer_token_file =
        write_token(directory.path(), "management-token");
    config.listeners.inference_bearer_token_file = write_token(directory.path(), "inference-token");
    adjust(&mut config);

    let router_path = directory.path().join("router.yaml");
    fs::write(&router_path, serde_yaml::to_string(&config).unwrap()).unwrap();
    let pingora_path = directory.path().join("pingora.yaml");
    fs::write(
        &pingora_path,
        "---\nversion: 1\nthreads: 1\ndaemon: false\n",
    )
    .unwrap();

    Command::new(env!("CARGO_BIN_EXE_vllm-router"))
        .args([
            "-t",
            "-c",
            pingora_path.to_str().expect("path must be UTF-8"),
            "--router-config",
            router_path.to_str().expect("path must be UTF-8"),
        ])
        .env("RUST_LOG", "error")
        .output()
        .expect("router binary should run")
}

fn write_token(directory: &Path, name: &str) -> PathBuf {
    let path = directory.join(name);
    fs::write(&path, format!("{name}-secret\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    path
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
