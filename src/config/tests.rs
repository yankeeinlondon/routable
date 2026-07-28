use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;
type RejectionCase = (&'static str, fn(&mut RouterConfig), &'static str);

fn valid_config() -> RouterConfig {
    RouterConfig {
        listeners: Listeners {
            data_plane: SocketAddr::from(([0, 0, 0, 0], 8000)),
            management: SocketAddr::from(([127, 0, 0, 1], 8001)),
            metrics: SocketAddr::from(([127, 0, 0, 1], 8002)),
            management_bearer_token_file: PathBuf::from("/run/secrets/vllm-router-token"),
        },
        inventory: Inventory {
            host_ram_capacity_bytes: 128 * GIB,
            gpus: vec![
                GpuDefinition {
                    uuid: "GPU-a".into(),
                    vram_capacity_bytes: 32 * GIB,
                },
                GpuDefinition {
                    uuid: "GPU-b".into(),
                    vram_capacity_bytes: 32 * GIB,
                },
            ],
        },
        policies: RouterPolicies::default(),
        services: vec![valid_service("alpha", "model-a", 8100, "GPU-a")],
    }
}

fn valid_service(name: &str, served_model: &str, port: u16, gpu_uuid: &str) -> ServiceDefinition {
    ServiceDefinition {
        name: name.into(),
        enabled: true,
        runner: RunnerType::Vllm,
        served_model: served_model.into(),
        aliases: Vec::new(),
        model_path: PathBuf::from(format!("/models/{name}")),
        tokenizer: TokenizerConfig::default(),
        quantization: Some("fp8".into()),
        runner_args: BTreeMap::new(),
        placement: Placement {
            gpu_uuids: vec![gpu_uuid.into()],
            tensor_parallelism: 1,
            pipeline_parallelism: 1,
            data_parallelism: 1,
            vram_reservation_bytes_per_gpu: 24 * GIB,
        },
        backend: Backend {
            address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port,
        },
        systemd: SystemdConfig {
            unit: format!("vllm@{name}.service"),
            environment_file: PathBuf::from(format!("/etc/vllm/{name}.env")),
        },
        readiness: ReadinessConfig::default(),
        deadlines: DeadlineConfig::default(),
        retry: RetryConfig::default(),
        warmup: Some(WarmupRequest {
            path: "/v1/completions".into(),
            body: serde_json::json!({"prompt": "warmup", "max_tokens": 1}),
            after_wake: true,
        }),
        sleep: SleepConfig::default(),
        idle: IdleConfig::default(),
        parking: ParkingConfig::default(),
        priority: 0,
        evictable: true,
        automatic_start: true,
        compatibility: VllmCompatibility::default(),
    }
}

fn messages(config: &RouterConfig) -> String {
    config
        .validate()
        .expect_err("configuration should be rejected")
        .to_string()
}

#[test]
fn accepts_valid_configuration_and_resolves_models() {
    let mut config = valid_config();
    config.services[0].aliases = vec!["friendly".into()];
    config
        .services
        .push(valid_service("replica", "model-a", 8101, "GPU-b"));

    config.validate().unwrap();
    assert_eq!(config.canonical_model("friendly"), Some("model-a"));
    assert_eq!(config.services_for_model("friendly").len(), 2);
    assert_eq!(config.service("replica").unwrap().backend.port, 8101);
}

#[test]
fn checked_in_example_parses_and_validates() {
    let config: RouterConfig =
        serde_yaml::from_str(include_str!("example.yaml")).expect("example YAML should parse");
    config.validate().expect("example YAML should validate");
}

#[test]
fn rejects_identity_and_network_collisions_table() {
    let cases: &[RejectionCase] = &[
        (
            "service name",
            |config| {
                let service = valid_service("alpha", "model-b", 8101, "GPU-b");
                config.services.push(service);
            },
            "duplicate service name",
        ),
        (
            "backend port",
            |config| {
                let service = valid_service("beta", "model-b", 8100, "GPU-b");
                config.services.push(service);
            },
            "already used by service",
        ),
        (
            "listener port",
            |config| config.services[0].backend.port = 8000,
            "conflicts with listeners.data_plane",
        ),
        (
            "non-loopback backend",
            |config| config.services[0].backend.address = "10.0.0.4".parse().unwrap(),
            "is not loopback",
        ),
        (
            "ambiguous alias",
            |config| {
                config.services[0].aliases = vec!["shared".into()];
                let mut service = valid_service("beta", "model-b", 8101, "GPU-b");
                service.aliases = vec!["shared".into()];
                config.services.push(service);
            },
            "is ambiguous",
        ),
        (
            "alias versus canonical",
            |config| {
                config.services[0].aliases = vec!["model-b".into()];
                config
                    .services
                    .push(valid_service("beta", "model-b", 8101, "GPU-b"));
            },
            "conflicts with a canonical",
        ),
    ];

    for (name, mutate, expected) in cases {
        let mut config = valid_config();
        mutate(&mut config);
        assert!(messages(&config).contains(expected), "case {name}");
    }
}

#[test]
fn rejects_paths_topology_and_unsafe_fields_table() {
    let cases: &[RejectionCase] = &[
        (
            "relative token file",
            |config| {
                config.listeners.management_bearer_token_file = "token".into();
            },
            "management_bearer_token_file must be an absolute",
        ),
        (
            "relative model",
            |config| config.services[0].model_path = "model".into(),
            "model_path must be an absolute",
        ),
        (
            "relative tokenizer",
            |config| config.services[0].tokenizer.path = Some("tokenizer".into()),
            "tokenizer.path must be an absolute",
        ),
        (
            "relative environment",
            |config| config.services[0].systemd.environment_file = "service.env".into(),
            "environment_file must be an absolute",
        ),
        (
            "unknown GPU",
            |config| config.services[0].placement.gpu_uuids[0] = "GPU-missing".into(),
            "unknown GPU UUID",
        ),
        (
            "parallel topology",
            |config| config.services[0].placement.tensor_parallelism = 2,
            "TP × PP × DP requires 2",
        ),
        (
            "unsafe unit",
            |config| config.services[0].systemd.unit = "vllm;reboot.service".into(),
            "safely named .service",
        ),
        (
            "managed runner arg",
            |config| {
                config.services[0]
                    .runner_args
                    .insert("tensor-parallel-size".into(), RunnerArgValue::Integer(2));
            },
            "is not allowlisted",
        ),
        (
            "newline runner arg",
            |config| {
                config.services[0].runner_args.insert(
                    "revision".into(),
                    RunnerArgValue::String("main\nEnvironment=oops".into()),
                );
            },
            "forbidden control",
        ),
    ];

    for (name, mutate, expected) in cases {
        let mut config = valid_config();
        mutate(&mut config);
        assert!(messages(&config).contains(expected), "case {name}");
    }
}

#[test]
fn rejects_threshold_deadline_and_policy_errors_table() {
    let cases: &[RejectionCase] = &[
        (
            "probe threshold",
            |config| config.services[0].readiness.consecutive_successes = 0,
            "consecutive thresholds",
        ),
        (
            "probe timeout",
            |config| config.services[0].readiness.timeout_ms = 2_000,
            "timeout_ms must be less",
        ),
        (
            "deadline",
            |config| config.services[0].deadlines.wake_ms = 0,
            "deadlines must all",
        ),
        (
            "backoff ordering",
            |config| {
                config.services[0].retry.initial_backoff_ms = 10;
                config.services[0].retry.maximum_backoff_ms = 5;
            },
            "initial_backoff_ms <= maximum_backoff_ms",
        ),
        (
            "parking longer than startup",
            |config| {
                config.services[0].parking.deadline_ms =
                    config.services[0].deadlines.startup_ms + 1;
            },
            "cannot exceed deadlines.startup_ms",
        ),
        (
            "parking cap",
            |config| {
                config.services[0].parking.queue_limit =
                    config.policies.per_model_parked_request_limit + 1;
            },
            "exceeds policies.per_model",
        ),
        (
            "awake co-residency",
            |config| config.policies.max_awake_owners_per_gpu = 2,
            "must be 1",
        ),
    ];

    for (name, mutate, expected) in cases {
        let mut config = valid_config();
        mutate(&mut config);
        assert!(messages(&config).contains(expected), "case {name}");
    }
}

fn enable_l1(service: &mut ServiceDefinition, residual: u64, host: u64) {
    service.sleep = SleepConfig {
        enabled: true,
        hardware_validated: true,
        levels: vec![SleepLevel::L1],
        residual_vram_bytes_per_gpu: residual,
        l1_host_ram_bytes: host,
    };
    service.idle.sleep_l1_after_ms = Some(1_000);
    service.idle.stop_after_ms = Some(2_000);
}

#[test]
fn rejects_sleep_and_resource_budget_errors_table() {
    let cases: &[RejectionCase] = &[
        (
            "unvalidated sleep",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].sleep.hardware_validated = false;
            },
            "hardware_validated is false",
        ),
        (
            "sleep residual exceeds awake",
            |config| enable_l1(&mut config.services[0], 25 * GIB, 8 * GIB),
            "exceeds its awake reservation",
        ),
        (
            "host RAM",
            |config| {
                enable_l1(&mut config.services[0], GIB, 80 * GIB);
                let mut second = valid_service("beta", "model-b", 8101, "GPU-b");
                enable_l1(&mut second, GIB, 80 * GIB);
                config.services.push(second);
            },
            "L1 pinned host RAM overcommit",
        ),
        (
            "awake plus residual",
            |config| {
                enable_l1(&mut config.services[0], 10 * GIB, 8 * GIB);
                let mut second = valid_service("beta", "model-b", 8101, "GPU-a");
                enable_l1(&mut second, 10 * GIB, 8 * GIB);
                config.services.push(second);
            },
            "other residents are asleep",
        ),
        (
            "idle ordering",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].idle.stop_after_ms = Some(500);
            },
            "must be greater than sleep_l1_after_ms",
        ),
    ];

    for (name, mutate, expected) in cases {
        let mut config = valid_config();
        mutate(&mut config);
        assert!(messages(&config).contains(expected), "case {name}");
    }
}

#[test]
fn yaml_load_is_strict_and_reload_is_atomic() {
    let directory = tempfile::tempdir().unwrap();
    let good_path = directory.path().join("good.yaml");
    let bad_path = directory.path().join("bad.yaml");
    std::fs::write(&good_path, serde_yaml::to_string(&valid_config()).unwrap()).unwrap();
    std::fs::write(
        &bad_path,
        serde_yaml::to_string(&{
            let mut config = valid_config();
            config.services[0].backend.address = "192.0.2.1".parse().unwrap();
            config
        })
        .unwrap(),
    )
    .unwrap();

    let store = ConfigStore::load(&good_path).unwrap();
    let original = store.snapshot();
    assert!(store.reload(&bad_path).is_err());
    assert!(Arc::ptr_eq(&original, &store.snapshot()));

    std::fs::write(
        directory.path().join("unknown.yaml"),
        "listeners: {}\nunknown_field: true\n",
    )
    .unwrap();
    let error = RouterConfig::load(directory.path().join("unknown.yaml"))
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown field") || error.contains("missing field"));
}
