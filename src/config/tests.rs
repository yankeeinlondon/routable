use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use super::*;

const GIB: u64 = 1024 * 1024 * 1024;
type RejectionCase = (&'static str, fn(&mut RouterConfig), &'static str);
type ConfigAcceptanceCase = (&'static str, fn(&mut RouterConfig));
type IdleAcceptanceCase = (&'static str, fn(&mut ServiceDefinition));

fn valid_config() -> RouterConfig {
    RouterConfig {
        listeners: Listeners {
            data_plane: SocketAddr::from(([0, 0, 0, 0], 8000)),
            management: SocketAddr::from(([127, 0, 0, 1], 8001)),
            metrics: SocketAddr::from(([127, 0, 0, 1], 8002)),
            management_bearer_token_file: PathBuf::from(
                "/etc/vllm-router/secrets/management-token",
            ),
            inference_bearer_token_file: PathBuf::from("/etc/vllm-router/secrets/inference-token"),
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
        environment: BTreeMap::new(),
        services: vec![valid_service("alpha", "model-a", 8100, "GPU-a")],
    }
}

fn valid_service(name: &str, served_model: &str, port: u16, gpu_uuid: &str) -> ServiceDefinition {
    ServiceDefinition {
        name: name.into(),
        enabled: true,
        runner: RunnerType::Vllm,
        runner_executable: PathBuf::from("/opt/vllm-venv/bin/vllm"),
        served_model: served_model.into(),
        aliases: Vec::new(),
        model_path: PathBuf::from(format!("/models/{name}")),
        tokenizer: TokenizerConfig::default(),
        quantization: Some("fp8".into()),
        runner_args: BTreeMap::new(),
        environment: BTreeMap::new(),
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
fn validates_replica_client_contract_table() {
    let rejection_cases: &[RejectionCase] = &[
        (
            "maximum model length differs",
            |config| {
                config.services[0].compatibility.maximum_model_length = Some(32_768);
                let mut replica = valid_service("replica", "model-a", 8101, "GPU-b");
                replica.compatibility.maximum_model_length = Some(131_072);
                config.services.push(replica);
            },
            "compatibility.maximum_model_length",
        ),
        (
            "KV cache dtype differs",
            |config| {
                config.services[0].compatibility.kv_cache_dtype = Some("fp8".into());
                let mut replica = valid_service("replica", "model-a", 8101, "GPU-b");
                replica.compatibility.kv_cache_dtype = Some("auto".into());
                config.services.push(replica);
            },
            "compatibility.kv_cache_dtype",
        ),
    ];

    for (name, mutate, differing_field) in rejection_cases {
        let mut config = valid_config();
        mutate(&mut config);
        let errors = messages(&config);
        assert!(errors.contains(differing_field), "case {name}: {errors}");
        assert!(errors.contains("\"alpha\""), "case {name}: {errors}");
        assert!(errors.contains("\"replica\""), "case {name}: {errors}");
    }

    let acceptance_cases: &[ConfigAcceptanceCase] = &[
        ("replicas with identical contracts", |config| {
            config.services[0].compatibility.maximum_model_length = Some(32_768);
            config.services[0].compatibility.kv_cache_dtype = Some("fp8".into());
            let mut replica = valid_service("replica", "model-a", 8101, "GPU-b");
            replica.compatibility = config.services[0].compatibility.clone();
            config.services.push(replica);
        }),
        ("served-model variants sharing a model path", |config| {
            config.services[0].compatibility.maximum_model_length = Some(32_768);
            let mut variant = valid_service("variant", "model-a-128k", 8101, "GPU-b");
            variant.model_path = config.services[0].model_path.clone();
            variant.compatibility.maximum_model_length = Some(131_072);
            variant.compatibility.kv_cache_dtype = Some("fp8".into());
            config.services.push(variant);
        }),
    ];

    for (name, mutate) in acceptance_cases {
        let mut config = valid_config();
        mutate(&mut config);
        config
            .validate()
            .unwrap_or_else(|error| panic!("case {name}: {error}"));
    }
}

#[test]
fn validates_alias_ownership_across_replicas() {
    let mut replicas = valid_config();
    replicas.services[0].aliases = vec!["friendly".into()];
    let mut replica = valid_service("replica", "model-a", 8101, "GPU-b");
    replica.aliases = vec!["friendly".into()];
    replicas.services.push(replica);
    replicas
        .validate()
        .expect("one alias may be owned by replicas of the same canonical model");

    let rejection_cases: &[RejectionCase] = &[
        (
            "alias shared by different canonical models",
            |config| {
                config.services[0].aliases = vec!["shared".into()];
                let mut service = valid_service("beta", "model-b", 8101, "GPU-b");
                service.aliases = vec!["shared".into()];
                config.services.push(service);
            },
            "ambiguous between canonical models",
        ),
        (
            "alias conflicts with another canonical model",
            |config| {
                config.services[0].aliases = vec!["model-b".into()];
                config
                    .services
                    .push(valid_service("beta", "model-b", 8101, "GPU-b"));
            },
            "conflicts with a canonical served model",
        ),
    ];
    assert_rejections(rejection_cases);
}

#[test]
fn checked_in_example_parses_and_validates() {
    let config: RouterConfig =
        serde_yaml::from_str(include_str!("example.yaml")).expect("example YAML should parse");
    config.validate().expect("example YAML should validate");
}

#[test]
fn validates_allowlisted_environment_at_global_and_service_scope() {
    let mut config = valid_config();
    config
        .environment
        .insert("PATH".into(), "/opt/cuda/bin:/usr/bin".into());
    config.services[0]
        .environment
        .insert("HF_HOME".into(), "/models/hf".into());
    config.services[0]
        .environment
        .insert("VLLM_CACHE_ROOT".into(), "/models/vllm".into());
    config.validate().unwrap();

    config
        .environment
        .insert("CUDA_VISIBLE_DEVICES".into(), "1".into());
    config.services[0]
        .environment
        .insert("PATH".into(), "/bin\nEnvironment=oops".into());
    let errors = messages(&config);
    assert!(errors.contains("environment.CUDA_VISIBLE_DEVICES is not allowlisted"));
    assert!(errors.contains("environment.PATH contains a forbidden control character"));
}

#[test]
fn accepts_model_specific_gpu_environment_and_runner_arguments() {
    let mut config = valid_config();
    config.services[0].environment.extend([
        ("FLASHINFER_CUDA_ARCH_LIST".into(), "12.0f".into()),
        ("NCCL_P2P_DISABLE".into(), "1".into()),
        (
            "PYTORCH_CUDA_ALLOC_CONF".into(),
            "expandable_segments:True".into(),
        ),
        ("VLLM_USE_FASTOKENS".into(), "1".into()),
        ("VLLM_WORKER_MULTIPROC_METHOD".into(), "spawn".into()),
    ]);
    config.services[0].runner_args.extend([
        (
            "attention-backend".into(),
            RunnerArgValue::String("flashinfer".into()),
        ),
        (
            "mm-encoder-tp-mode".into(),
            RunnerArgValue::String("data".into()),
        ),
        (
            "disable-custom-all-reduce".into(),
            RunnerArgValue::Bool(true),
        ),
        ("language-model-only".into(), RunnerArgValue::Bool(false)),
    ]);

    config.validate().unwrap();
}

#[test]
fn validates_tool_and_reasoning_parsers_against_vllm_registries() {
    let mut config = valid_config();
    config.services[0].runner_args.extend([
        (
            "tool-call-parser".into(),
            RunnerArgValue::String("qwen3_xml".into()),
        ),
        (
            "reasoning-parser".into(),
            RunnerArgValue::String("qwen3".into()),
        ),
    ]);
    config.validate().unwrap();

    config.services[0].runner_args.insert(
        "tool-call-parser".into(),
        RunnerArgValue::String("not-a-parser".into()),
    );
    config.services[0]
        .runner_args
        .insert("reasoning-parser".into(), RunnerArgValue::Bool(true));
    let errors = messages(&config);
    assert!(errors.contains("tool-call-parser value \"not-a-parser\" is not registered"));
    assert!(errors.contains("reasoning-parser must be a string"));
}

#[test]
fn rejects_listener_inventory_and_global_policy_errors_table() {
    let cases: &[RejectionCase] = &[
        (
            "listener collision",
            |config| config.listeners.management = config.listeners.data_plane,
            "listeners.management uses port 8000, already used by listeners.data_plane",
        ),
        (
            "zero host RAM",
            |config| config.inventory.host_ram_capacity_bytes = 0,
            "host_ram_capacity_bytes must be greater than zero",
        ),
        (
            "empty GPU inventory",
            |config| config.inventory.gpus.clear(),
            "gpus must contain at least one GPU",
        ),
        (
            "empty GPU UUID",
            |config| config.inventory.gpus[0].uuid = " ".into(),
            "uuid must not be empty",
        ),
        (
            "duplicate GPU UUID",
            |config| config.inventory.gpus[1].uuid = config.inventory.gpus[0].uuid.clone(),
            "duplicate GPU UUID",
        ),
        (
            "zero GPU VRAM",
            |config| config.inventory.gpus[0].vram_capacity_bytes = 0,
            "vram_capacity_bytes must be greater than zero",
        ),
        (
            "zero body limit",
            |config| config.policies.maximum_request_body_bytes = 0,
            "maximum_request_body_bytes must be greater than zero",
        ),
        (
            "zero global parking limit",
            |config| config.policies.global_parked_request_limit = 0,
            "global_parked_request_limit must be greater than zero",
        ),
        (
            "zero per-model parking limit",
            |config| config.policies.per_model_parked_request_limit = 0,
            "per_model_parked_request_limit must be greater than zero",
        ),
        (
            "per-model parking exceeds global",
            |config| {
                config.policies.global_parked_request_limit = 1;
                config.policies.per_model_parked_request_limit = 2;
                config.services[0].parking.queue_limit = 1;
            },
            "per_model_parked_request_limit cannot exceed global_parked_request_limit",
        ),
        (
            "zero connect timeout",
            |config| config.policies.connect_timeout_ms = 0,
            "connection and inter-chunk timeouts must be greater than zero",
        ),
        (
            "unsupported awake co-residency",
            |config| config.policies.max_awake_owners_per_gpu = 2,
            "max_awake_owners_per_gpu must be 1",
        ),
        (
            "empty services",
            |config| config.services.clear(),
            "services must contain at least one service definition",
        ),
    ];

    assert_rejections(cases);
}

#[test]
fn rejects_identity_and_network_collisions_table() {
    let cases: &[RejectionCase] = &[
        (
            "empty service name",
            |config| config.services[0].name = " ".into(),
            "name must not be empty",
        ),
        (
            "unsafe service name",
            |config| config.services[0].name = "bad/name".into(),
            "name may contain only ASCII letters",
        ),
        (
            "service name",
            |config| {
                let service = valid_service("alpha", "model-b", 8101, "GPU-b");
                config.services.push(service);
            },
            "duplicate service name",
        ),
        (
            "empty served model",
            |config| config.services[0].served_model = " ".into(),
            "served_model must not be empty",
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
            "duplicate systemd unit",
            |config| {
                let mut service = valid_service("beta", "model-b", 8101, "GPU-b");
                service.systemd.unit = config.services[0].systemd.unit.clone();
                config.services.push(service);
            },
            "systemd.unit \"vllm@alpha.service\" is already used by service \"alpha\"",
        ),
        (
            "duplicate environment file",
            |config| {
                let mut service = valid_service("beta", "model-b", 8101, "GPU-b");
                service.systemd.environment_file =
                    config.services[0].systemd.environment_file.clone();
                config.services.push(service);
            },
            "systemd.environment_file \"/etc/vllm/alpha.env\" is already used by service \"alpha\"",
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
            "empty alias",
            |config| config.services[0].aliases = vec![" ".into()],
            "aliases must not contain an empty alias",
        ),
        (
            "alias repeats canonical model",
            |config| config.services[0].aliases = vec!["model-a".into()],
            "aliases contains canonical model name",
        ),
        (
            "duplicate alias within service",
            |config| config.services[0].aliases = vec!["friendly".into(), "friendly".into()],
            "aliases contains duplicate",
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

    assert_rejections(cases);
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
            "relative inference token file",
            |config| {
                config.listeners.inference_bearer_token_file = "inference-token".into();
            },
            "inference_bearer_token_file must be an absolute",
        ),
        (
            "shared management and inference token file",
            |config| {
                config.listeners.inference_bearer_token_file =
                    config.listeners.management_bearer_token_file.clone();
            },
            "must differ from listeners.management_bearer_token_file",
        ),
        (
            "relative runner executable",
            |config| config.services[0].runner_executable = "bin/vllm".into(),
            "runner_executable must be an absolute",
        ),
        (
            "unsafe runner executable",
            |config| {
                config.services[0].runner_executable = PathBuf::from("/opt/vllm\nEnvironment=oops");
            },
            "runner_executable contains a forbidden control character",
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
            "unsafe bare environment",
            |config| {
                config.services[0].systemd.environment_file = "/etc/vllm/bad path.env".into();
            },
            "safe for a bare EnvironmentFile= directive",
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
            "zero parallelism",
            |config| config.services[0].placement.pipeline_parallelism = 0,
            "parallelism values must all be greater than zero",
        ),
        (
            "parallelism product overflow",
            |config| {
                config.services[0].placement.tensor_parallelism = usize::MAX;
                config.services[0].placement.pipeline_parallelism = 2;
            },
            "parallelism product overflows usize",
        ),
        (
            "duplicate assigned GPU",
            |config| {
                config.services[0].placement.tensor_parallelism = 2;
                config.services[0].placement.gpu_uuids = vec!["GPU-a".into(), "GPU-a".into()];
            },
            "assigns GPU UUID \"GPU-a\" more than once",
        ),
        (
            "zero VRAM reservation",
            |config| config.services[0].placement.vram_reservation_bytes_per_gpu = 0,
            "vram_reservation_bytes_per_gpu must be greater than zero",
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
        (
            "non-boolean runner flag",
            |config| {
                config.services[0].runner_args.insert(
                    "enable-prefix-caching".into(),
                    RunnerArgValue::String("yes".into()),
                );
            },
            "enable-prefix-caching must have a boolean value",
        ),
        (
            "non-finite runner number",
            |config| {
                config.services[0]
                    .runner_args
                    .insert("rope-theta".into(), RunnerArgValue::Float(f64::NAN));
            },
            "rope-theta must have a finite numeric value",
        ),
        (
            "Ray executor",
            |config| {
                config.services[0].runner_args.insert(
                    "distributed-executor-backend".into(),
                    RunnerArgValue::String("RAY".into()),
                );
            },
            "cannot be `ray`",
        ),
        (
            "relative download directory",
            |config| {
                config.services[0].runner_args.insert(
                    "download-dir".into(),
                    RunnerArgValue::String("downloads".into()),
                );
            },
            "runner_args.download-dir must be an absolute path",
        ),
    ];

    assert_rejections(cases);
}

#[test]
fn rejects_threshold_deadline_and_policy_errors_table() {
    let cases: &[RejectionCase] = &[
        (
            "relative health path",
            |config| config.services[0].readiness.health_path = "health".into(),
            "health_path must be an absolute HTTP path",
        ),
        (
            "authority-style models path",
            |config| config.services[0].readiness.models_path = "//models".into(),
            "models_path must be an absolute HTTP path",
        ),
        (
            "probe threshold",
            |config| config.services[0].readiness.consecutive_successes = 0,
            "consecutive thresholds",
        ),
        (
            "zero probe interval",
            |config| config.services[0].readiness.interval_ms = 0,
            "intervals, timeout, and maximum probe age must be greater than zero",
        ),
        (
            "probe timeout",
            |config| config.services[0].readiness.timeout_ms = 2_000,
            "timeout_ms must be less",
        ),
        (
            "probe maximum age below interval",
            |config| config.services[0].readiness.maximum_probe_age_ms = 1_000,
            "maximum_probe_age_ms must be at least interval_ms",
        ),
        (
            "probe maximum age with no jitter margin",
            |config| {
                config.services[0].readiness.maximum_probe_age_ms =
                    config.services[0].readiness.interval_ms;
            },
            "is below the 5000 ms admission-freshness floor",
        ),
        (
            "deadline",
            |config| config.services[0].deadlines.wake_ms = 0,
            "deadlines must all",
        ),
        (
            "zero retry attempts",
            |config| config.services[0].retry.maximum_attempts = 0,
            "maximum_attempts must be greater than zero",
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
            "retry multiplier below one",
            |config| config.services[0].retry.multiplier = 0.5,
            "multiplier must be finite and at least 1",
        ),
        (
            "retry jitter outside unit interval",
            |config| config.services[0].retry.jitter_fraction = 1.1,
            "jitter_fraction must be finite and between 0 and 1",
        ),
        (
            "zero parking field",
            |config| config.services[0].parking.retry_after_seconds = 0,
            "parking deadline, queue limit, and retry-after must be greater than zero",
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
            "relative warmup path",
            |config| config.services[0].warmup.as_mut().unwrap().path = "warmup".into(),
            "warmup.path must be an absolute HTTP path",
        ),
        (
            "non-object warmup body",
            |config| config.services[0].warmup.as_mut().unwrap().body = serde_json::json!("prompt"),
            "warmup.body must be a JSON object",
        ),
        (
            "zero maximum model length",
            |config| config.services[0].compatibility.maximum_model_length = Some(0),
            "maximum_model_length must be greater than zero",
        ),
        (
            "invalid GPU memory utilization",
            |config| config.services[0].compatibility.gpu_memory_utilization = f64::NAN,
            "gpu_memory_utilization must be finite and in (0, 1]",
        ),
        (
            "zero unsupported tensor parallelism",
            |config| {
                config.services[0]
                    .compatibility
                    .unsupported_tensor_parallelism = vec![0];
            },
            "unsupported_tensor_parallelism values must be greater than zero",
        ),
        (
            "duplicate unsupported tensor parallelism",
            |config| {
                config.services[0]
                    .compatibility
                    .unsupported_tensor_parallelism = vec![2, 2];
            },
            "unsupported_tensor_parallelism contains duplicate degree 2",
        ),
    ];

    assert_rejections(cases);
}

#[test]
fn admission_freshness_floor_follows_the_shared_observation_cadence() {
    let mut config = valid_config();
    let mut second = valid_service("beta", "model-b", 8101, "GPU-b");
    // A longer per-service interval does not slow this service's refresh: one loop
    // reconciles everything at `alpha`'s 2 s cadence, so `beta`'s budget is measured
    // against that shared period rather than its own 4 s interval. Its own
    // `maximum_probe_age_ms >= interval_ms` check passes, which is exactly the
    // knife-edge the floor exists to catch.
    second.readiness.interval_ms = 4_000;
    second.readiness.timeout_ms = 1_500;
    second.readiness.maximum_probe_age_ms = 4_000;
    config.services.push(second);

    assert_eq!(
        config.observation_interval_ms(),
        2_000,
        "the shortest configured readiness interval is the observation cadence"
    );
    let errors = messages(&config);
    assert!(
        errors.contains(
            "services[1] (\"beta\").readiness.maximum_probe_age_ms 4000 is below the 5500 ms admission-freshness floor"
        ),
        "{errors}"
    );
    assert!(
        !errors.contains("services[0] (\"alpha\").readiness.maximum_probe_age_ms"),
        "a service that already clears the floor must not be reported: {errors}"
    );

    config.services[1].readiness.maximum_probe_age_ms = 5_500;
    config
        .validate()
        .expect("a budgeted freshness window validates");
}

#[test]
fn rejects_a_hardware_incompatible_tensor_parallel_topology() {
    let mut config = valid_config();
    // The tested NVFP4 checkpoint reports `compressed-tensors` rather than a
    // stable NVFP4 CLI label, so the safety decision comes from the explicit
    // service compatibility declaration.
    config.services[0].quantization = Some("compressed-tensors".into());
    config.services[0].placement.gpu_uuids = vec!["GPU-a".into(), "GPU-b".into()];
    config.services[0].placement.tensor_parallelism = 2;
    config.services[0]
        .compatibility
        .unsupported_tensor_parallelism = vec![2];

    let errors = messages(&config);
    assert!(
        errors.contains("placement.tensor_parallelism 2 is explicitly unsupported"),
        "{errors}"
    );
}

#[test]
fn omitted_unsupported_tensor_parallelism_remains_backward_compatible() {
    let yaml =
        include_str!("example.yaml").replace("      unsupported_tensor_parallelism: []\n", "");
    let config: RouterConfig = serde_yaml::from_str(&yaml).unwrap();

    assert!(
        config.services[0]
            .compatibility
            .unsupported_tensor_parallelism
            .is_empty()
    );
    config.validate().unwrap();
}

#[test]
fn omitted_l1_host_ram_floor_remains_backward_compatible_while_sleep_is_disabled() {
    let yaml = include_str!("example.yaml").replace("      minimum_l1_host_ram_bytes: 0\n", "");
    let config: RouterConfig = serde_yaml::from_str(&yaml).unwrap();

    assert_eq!(
        config.services[0].compatibility.minimum_l1_host_ram_bytes,
        0
    );
    config.validate().unwrap();
}

#[test]
fn measured_l1_host_ram_floors_reject_underdeclarations_and_accept_budgets() {
    const GB: u64 = 1_000_000_000;
    let cases = [("FP8", 20 * GB, 25 * GB), ("NVFP4", 30 * GB, 40 * GB)];

    for (name, provisional, measured_budget) in cases {
        let mut underdeclared = valid_config();
        enable_l1(&mut underdeclared.services[0], GIB, provisional);
        underdeclared.services[0]
            .compatibility
            .minimum_l1_host_ram_bytes = measured_budget;
        let errors = messages(&underdeclared);
        assert!(
            errors.contains(&format!(
                "l1_host_ram_bytes {provisional} is below compatibility.minimum_l1_host_ram_bytes {measured_budget}"
            )),
            "{name}: {errors}"
        );

        let mut measured = valid_config();
        enable_l1(&mut measured.services[0], GIB, measured_budget);
        measured
            .validate()
            .unwrap_or_else(|error| panic!("{name} measured budget should validate: {error}"));
    }
}

fn enable_l1(service: &mut ServiceDefinition, residual: u64, host: u64) {
    service.compatibility.minimum_l1_host_ram_bytes = host;
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

fn enable_l2(service: &mut ServiceDefinition, residual: u64) {
    service.sleep = SleepConfig {
        enabled: true,
        hardware_validated: true,
        levels: vec![SleepLevel::L2],
        residual_vram_bytes_per_gpu: residual,
        l1_host_ram_bytes: 0,
    };
    service.idle.sleep_l2_after_ms = Some(1_000);
    service.idle.stop_after_ms = Some(2_000);
}

#[test]
fn evictability_is_independent_from_valid_idle_policy_table() {
    let cases: &[IdleAcceptanceCase] = &[
        ("evictable with default idle stop", |_| {}),
        ("non-evictable with default idle stop", |service| {
            service.evictable = false;
        }),
        ("non-evictable with idle policy disabled", |service| {
            service.evictable = false;
            service.idle.stop_after_ms = None;
        }),
        (
            "non-evictable with validated idle sleep and stop",
            |service| {
                service.evictable = false;
                enable_l1(service, GIB, 8 * GIB);
            },
        ),
    ];

    for (name, mutate) in cases {
        let mut config = valid_config();
        mutate(&mut config.services[0]);
        config
            .validate()
            .unwrap_or_else(|error| panic!("case {name} should be valid: {error}"));
    }
}

#[test]
fn rejects_sleep_and_resource_budget_errors_table() {
    let cases: &[RejectionCase] = &[
        (
            "duplicate sleep level",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].sleep.levels.push(SleepLevel::L1);
            },
            "sleep.levels contains a duplicate level",
        ),
        (
            "unvalidated sleep",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].sleep.hardware_validated = false;
            },
            "hardware_validated is false",
        ),
        (
            "enabled sleep without levels",
            |config| {
                config.services[0].sleep.enabled = true;
                config.services[0].sleep.hardware_validated = true;
                config.services[0].sleep.residual_vram_bytes_per_gpu = GIB;
            },
            "sleep.levels must not be empty",
        ),
        (
            "zero sleep residual",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].sleep.residual_vram_bytes_per_gpu = 0;
            },
            "residual_vram_bytes_per_gpu must be greater than zero",
        ),
        (
            "zero L1 host memory",
            |config| enable_l1(&mut config.services[0], GIB, 0),
            "l1_host_ram_bytes must be greater than zero for L1",
        ),
        (
            "missing measured L1 host memory floor",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].compatibility.minimum_l1_host_ram_bytes = 0;
            },
            "minimum_l1_host_ram_bytes must record a measured conservative floor",
        ),
        (
            "missing L1 idle threshold",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].idle.sleep_l1_after_ms = None;
            },
            "sleep_l1_after_ms is required when L1 is enabled",
        ),
        (
            "L1 threshold without L1",
            |config| {
                enable_l2(&mut config.services[0], GIB);
                config.services[0].idle.sleep_l1_after_ms = Some(500);
            },
            "sleep_l1_after_ms requires sleep level L1",
        ),
        (
            "missing L2 idle threshold",
            |config| {
                enable_l2(&mut config.services[0], GIB);
                config.services[0].idle.sleep_l2_after_ms = None;
            },
            "sleep_l2_after_ms is required when L2 is enabled",
        ),
        (
            "L2 threshold without L2",
            |config| {
                enable_l1(&mut config.services[0], GIB, 8 * GIB);
                config.services[0].idle.sleep_l2_after_ms = Some(1_500);
            },
            "sleep_l2_after_ms requires sleep level L2",
        ),
        (
            "disabled sleep with residual policy",
            |config| config.services[0].sleep.hardware_validated = true,
            "sleep is disabled, so sleep resources, levels, and idle sleep thresholds must be unset",
        ),
        (
            "zero idle threshold",
            |config| config.services[0].idle.stop_after_ms = Some(0),
            "idle.stop_after_ms must be greater than zero",
        ),
        (
            "sleep residual exceeds awake",
            |config| enable_l1(&mut config.services[0], 25 * GIB, 8 * GIB),
            "exceeds its awake reservation",
        ),
        (
            "awake reservation exceeds GPU",
            |config| config.services[0].placement.vram_reservation_bytes_per_gpu = 33 * GIB,
            "awake VRAM reservation",
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
            "host RAM sum overflow",
            |config| {
                enable_l1(&mut config.services[0], 1, u64::MAX);
                let mut second = valid_service("beta", "model-b", 8101, "GPU-b");
                enable_l1(&mut second, 1, 1);
                config.services.push(second);
            },
            "L1 pinned host RAM reservations overflow u64",
        ),
        (
            "all-sleep residual overcommit",
            |config| {
                enable_l2(&mut config.services[0], 20 * GIB);
                let mut second = valid_service("beta", "model-b", 8101, "GPU-a");
                enable_l2(&mut second, 20 * GIB);
                config.services.push(second);
            },
            "residual sleep VRAM overcommit",
        ),
        (
            "all-sleep residual sum overflow",
            |config| {
                enable_l2(&mut config.services[0], u64::MAX);
                config.services[0].placement.vram_reservation_bytes_per_gpu = u64::MAX;
                let mut second = valid_service("beta", "model-b", 8101, "GPU-a");
                enable_l2(&mut second, 1);
                config.services.push(second);
            },
            "residual sleep VRAM reservations overflow u64",
        ),
        (
            "awake plus residual sum overflow",
            |config| {
                config.services[0].placement.vram_reservation_bytes_per_gpu = u64::MAX;
                let mut second = valid_service("beta", "model-b", 8101, "GPU-a");
                enable_l2(&mut second, 1);
                config.services.push(second);
            },
            "VRAM reservations overflow u64 when service \"alpha\" is awake",
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

    assert_rejections(cases);
}

fn assert_rejections(cases: &[RejectionCase]) {
    for (name, mutate, expected) in cases {
        let mut config = valid_config();
        mutate(&mut config);
        assert!(messages(&config).contains(expected), "case {name}");
    }
}

#[test]
fn yaml_load_is_strict_and_store_exposes_immutable_snapshots() {
    let directory = tempfile::tempdir().unwrap();
    let good_path = directory.path().join("good.yaml");
    std::fs::write(&good_path, serde_yaml::to_string(&valid_config()).unwrap()).unwrap();

    let store = ConfigStore::load(&good_path).unwrap();
    assert_eq!(store.snapshot().services[0].name, "alpha");

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

#[test]
fn additive_registration_validates_the_merged_config_before_publication() {
    let store = ConfigStore::new(valid_config()).unwrap();
    let original = store.snapshot();
    let mut candidate = valid_service("beta", "model-b", 8101, "GPU-b");
    candidate.aliases = vec!["model-a".into()];

    let error = store.register_service(candidate.clone()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("conflicts with a canonical served model")
    );
    assert_eq!(store.snapshot().services.len(), 1);

    candidate.aliases = vec!["friendly-b".into()];
    let merged = store.register_service(candidate).unwrap();
    assert_eq!(
        original.services.len(),
        1,
        "existing snapshots stay immutable"
    );
    assert_eq!(merged.services.len(), 2);
    assert_eq!(
        store.snapshot().canonical_model("friendly-b"),
        Some("model-b")
    );
}

#[test]
fn omitted_inference_token_file_fails_to_parse_rather_than_serving_unauthenticated() {
    let yaml = include_str!("example.yaml").replace(
        "  inference_bearer_token_file: /etc/vllm-router/secrets/inference-token\n",
        "",
    );
    let error = serde_yaml::from_str::<RouterConfig>(&yaml)
        .expect_err("a config without an inference token must not load")
        .to_string();

    assert!(
        error.contains("inference_bearer_token_file"),
        "unexpected parse error: {error}"
    );
}

#[test]
fn omitted_runner_executable_uses_the_backward_compatible_default() {
    let yaml = include_str!("example.yaml")
        .replace("    runner_executable: /opt/vllm-venv/bin/vllm\n", "");
    let config: RouterConfig = serde_yaml::from_str(&yaml).unwrap();

    assert_eq!(
        config.services[0].runner_executable,
        Path::new("/opt/vllm-venv/bin/vllm")
    );
    config.validate().unwrap();
}
