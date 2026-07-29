//! Router configuration schema, loading, validation, and immutable runtime access.
//!
//! The schema intentionally owns every vLLM option which affects placement or
//! lifecycle. `runner_args` is only an escape hatch for explicitly allowlisted,
//! non-placement options.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A fully parsed and validated router configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouterConfig {
    pub listeners: Listeners,
    pub inventory: Inventory,
    #[serde(default)]
    pub policies: RouterPolicies,
    /// Host environment defaults inherited by every generated runner service.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    pub services: Vec<ServiceDefinition>,
}

impl RouterConfig {
    /// Parse and validate a router YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config =
            serde_yaml::from_slice::<Self>(&bytes).map_err(|source| ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        config.validate().map_err(ConfigError::Validation)?;
        Ok(config)
    }

    /// Validate a programmatically constructed configuration.
    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = Vec::new();
        validate_listeners(self, &mut errors);
        validate_inventory(&self.inventory, &mut errors);
        validate_policies(&self.policies, &mut errors);
        validate_services(self, &mut errors);
        validate_resource_budgets(self, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrors(errors))
        }
    }

    /// Resolve a canonical model name or alias to its canonical name.
    pub fn canonical_model<'a>(&'a self, requested: &str) -> Option<&'a str> {
        self.services.iter().find_map(|service| {
            if service.served_model == requested
                || service.aliases.iter().any(|alias| alias == requested)
            {
                Some(service.served_model.as_str())
            } else {
                None
            }
        })
    }

    /// Return all configured replicas for a canonical model name or alias.
    pub fn services_for_model(&self, requested: &str) -> Vec<&ServiceDefinition> {
        let Some(canonical) = self.canonical_model(requested) else {
            return Vec::new();
        };
        self.services
            .iter()
            .filter(|service| service.served_model == canonical)
            .collect()
    }

    /// Return a service by its immutable configured name.
    pub fn service(&self, name: &str) -> Option<&ServiceDefinition> {
        self.services.iter().find(|service| service.name == name)
    }
}

/// A validated configuration snapshot that is immutable for the process lifetime.
///
/// Applying configuration changes requires a supervised router restart.
#[derive(Debug)]
pub struct ConfigStore {
    current: Arc<RouterConfig>,
}

impl ConfigStore {
    pub fn new(config: RouterConfig) -> Result<Self, ValidationErrors> {
        config.validate()?;
        Ok(Self {
            current: Arc::new(config),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::new(RouterConfig::load(path)?).map_err(ConfigError::Validation)
    }

    pub fn snapshot(&self) -> Arc<RouterConfig> {
        Arc::clone(&self.current)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Listeners {
    pub data_plane: SocketAddr,
    pub management: SocketAddr,
    pub metrics: SocketAddr,
    pub management_bearer_token_file: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub host_ram_capacity_bytes: u64,
    pub gpus: Vec<GpuDefinition>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuDefinition {
    /// Stable NVML UUID, for example `GPU-...`.
    pub uuid: String,
    pub vram_capacity_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouterPolicies {
    pub maximum_request_body_bytes: u64,
    pub global_parked_request_limit: usize,
    pub per_model_parked_request_limit: usize,
    pub connect_timeout_ms: u64,
    pub inter_chunk_timeout_ms: u64,
    /// v1 deliberately supports exactly one awake owner per GPU.
    pub max_awake_owners_per_gpu: usize,
}

impl Default for RouterPolicies {
    fn default() -> Self {
        Self {
            maximum_request_body_bytes: 32 * 1024 * 1024,
            global_parked_request_limit: 1_024,
            per_model_parked_request_limit: 128,
            connect_timeout_ms: 5_000,
            inter_chunk_timeout_ms: 120_000,
            max_awake_owners_per_gpu: 1,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceDefinition {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub runner: RunnerType,
    /// Absolute path to the runner binary used in this service's generated unit.
    #[serde(default = "default_vllm_executable")]
    pub runner_executable: PathBuf,
    pub served_model: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub model_path: PathBuf,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub runner_args: BTreeMap<String, RunnerArgValue>,
    /// Service-specific runner environment. Values override global defaults.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    pub placement: Placement,
    pub backend: Backend,
    pub systemd: SystemdConfig,
    #[serde(default)]
    pub readiness: ReadinessConfig,
    #[serde(default)]
    pub deadlines: DeadlineConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub warmup: Option<WarmupRequest>,
    #[serde(default)]
    pub sleep: SleepConfig,
    #[serde(default)]
    pub idle: IdleConfig,
    #[serde(default)]
    pub parking: ParkingConfig,
    #[serde(default)]
    pub priority: i32,
    /// Whether the scheduler may displace this service for competing demand.
    /// Explicit idle sleep/stop policy remains independent of eviction.
    #[serde(default = "default_true")]
    pub evictable: bool,
    #[serde(default = "default_true")]
    pub automatic_start: bool,
    #[serde(default)]
    pub compatibility: VllmCompatibility,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerType {
    #[default]
    Vllm,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokenizerConfig {
    pub path: Option<PathBuf>,
    pub mode: Option<String>,
    pub revision: Option<String>,
    pub trust_remote_code: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub gpu_uuids: Vec<String>,
    #[serde(default = "default_parallelism")]
    pub tensor_parallelism: usize,
    #[serde(default = "default_parallelism")]
    pub pipeline_parallelism: usize,
    #[serde(default = "default_parallelism")]
    pub data_parallelism: usize,
    pub vram_reservation_bytes_per_gpu: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Backend {
    pub address: IpAddr,
    pub port: u16,
}

impl Backend {
    pub fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.port)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SystemdConfig {
    pub unit: String,
    pub environment_file: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadinessConfig {
    pub health_path: String,
    pub models_path: String,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    pub interval_ms: u64,
    pub timeout_ms: u64,
    pub maximum_probe_age_ms: u64,
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            health_path: "/health".into(),
            models_path: "/v1/models".into(),
            consecutive_successes: 2,
            consecutive_failures: 3,
            interval_ms: 2_000,
            timeout_ms: 1_000,
            maximum_probe_age_ms: 10_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeadlineConfig {
    pub startup_ms: u64,
    pub wake_ms: u64,
    pub drain_ms: u64,
    pub systemd_job_ms: u64,
}

impl Default for DeadlineConfig {
    fn default() -> Self {
        Self {
            startup_ms: 300_000,
            wake_ms: 30_000,
            drain_ms: 120_000,
            systemd_job_ms: 30_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetryConfig {
    pub maximum_attempts: u32,
    pub initial_backoff_ms: u64,
    pub maximum_backoff_ms: u64,
    pub multiplier: f64,
    pub jitter_fraction: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            maximum_attempts: 3,
            initial_backoff_ms: 1_000,
            maximum_backoff_ms: 30_000,
            multiplier: 2.0,
            jitter_fraction: 0.2,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WarmupRequest {
    pub path: String,
    #[serde(default = "default_warmup_body")]
    pub body: serde_json::Value,
    #[serde(default)]
    pub after_wake: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SleepLevel {
    L1,
    L2,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SleepConfig {
    pub enabled: bool,
    /// Must be set only after the host/model validation matrix has passed.
    pub hardware_validated: bool,
    pub levels: Vec<SleepLevel>,
    pub residual_vram_bytes_per_gpu: u64,
    pub l1_host_ram_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct IdleConfig {
    pub sleep_l1_after_ms: Option<u64>,
    pub sleep_l2_after_ms: Option<u64>,
    pub stop_after_ms: Option<u64>,
}

impl Default for IdleConfig {
    fn default() -> Self {
        Self {
            sleep_l1_after_ms: None,
            sleep_l2_after_ms: None,
            stop_after_ms: Some(3_600_000),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParkingConfig {
    pub deadline_ms: u64,
    pub queue_limit: usize,
    pub retry_after_seconds: u64,
}

impl Default for ParkingConfig {
    fn default() -> Self {
        Self {
            deadline_ms: 20_000,
            queue_limit: 64,
            retry_after_seconds: 120,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VllmCompatibility {
    pub maximum_model_length: Option<u64>,
    pub kv_cache_dtype: Option<String>,
    pub gpu_memory_utilization: f64,
}

impl Default for VllmCompatibility {
    fn default() -> Self {
        Self {
            maximum_model_length: None,
            kv_cache_dtype: None,
            gpu_memory_utilization: 0.9,
        }
    }
}

/// A YAML scalar accepted for an allowlisted runner option.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RunnerArgValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read router configuration {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse router configuration {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error(transparent)]
    Validation(#[from] ValidationErrors),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationErrors(pub Vec<String>);

impl ValidationErrors {
    pub fn messages(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "router configuration has {} validation error(s):",
            self.0.len()
        )?;
        for error in &self.0 {
            writeln!(formatter, "- {error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

fn validate_listeners(config: &RouterConfig, errors: &mut Vec<String>) {
    let listeners = [
        ("data_plane", config.listeners.data_plane),
        ("management", config.listeners.management),
        ("metrics", config.listeners.metrics),
    ];
    let mut ports = HashMap::new();
    for (name, address) in listeners {
        if let Some(previous) = ports.insert(address.port(), name) {
            errors.push(format!(
                "listeners.{name} uses port {}, already used by listeners.{previous}",
                address.port()
            ));
        }
    }
    require_absolute(
        &config.listeners.management_bearer_token_file,
        "listeners.management_bearer_token_file",
        errors,
    );
}

fn validate_inventory(inventory: &Inventory, errors: &mut Vec<String>) {
    if inventory.host_ram_capacity_bytes == 0 {
        errors.push("inventory.host_ram_capacity_bytes must be greater than zero".into());
    }
    if inventory.gpus.is_empty() {
        errors.push("inventory.gpus must contain at least one GPU".into());
    }
    let mut uuids = HashSet::new();
    for (index, gpu) in inventory.gpus.iter().enumerate() {
        if gpu.uuid.trim().is_empty() {
            errors.push(format!("inventory.gpus[{index}].uuid must not be empty"));
        } else if !uuids.insert(gpu.uuid.as_str()) {
            errors.push(format!("duplicate GPU UUID {:?}", gpu.uuid));
        }
        if gpu.vram_capacity_bytes == 0 {
            errors.push(format!(
                "inventory.gpus[{index}].vram_capacity_bytes must be greater than zero"
            ));
        }
    }
}

fn validate_policies(policies: &RouterPolicies, errors: &mut Vec<String>) {
    if policies.maximum_request_body_bytes == 0 {
        errors.push("policies.maximum_request_body_bytes must be greater than zero".into());
    }
    if policies.global_parked_request_limit == 0 {
        errors.push("policies.global_parked_request_limit must be greater than zero".into());
    }
    if policies.per_model_parked_request_limit == 0 {
        errors.push("policies.per_model_parked_request_limit must be greater than zero".into());
    }
    if policies.per_model_parked_request_limit > policies.global_parked_request_limit {
        errors.push(
            "policies.per_model_parked_request_limit cannot exceed global_parked_request_limit"
                .into(),
        );
    }
    if policies.connect_timeout_ms == 0 || policies.inter_chunk_timeout_ms == 0 {
        errors.push("connection and inter-chunk timeouts must be greater than zero".into());
    }
    if policies.max_awake_owners_per_gpu != 1 {
        errors.push(
            "policies.max_awake_owners_per_gpu must be 1; awake GPU co-residency is not supported in v1"
                .into(),
        );
    }
}

fn validate_services(config: &RouterConfig, errors: &mut Vec<String>) {
    validate_environment(&config.environment, "environment", errors);
    if config.services.is_empty() {
        errors.push("services must contain at least one service definition".into());
        return;
    }

    let known_gpus: HashSet<&str> = config
        .inventory
        .gpus
        .iter()
        .map(|gpu| gpu.uuid.as_str())
        .collect();
    let listener_ports: HashMap<u16, &str> = [
        (config.listeners.data_plane.port(), "listeners.data_plane"),
        (config.listeners.management.port(), "listeners.management"),
        (config.listeners.metrics.port(), "listeners.metrics"),
    ]
    .into_iter()
    .collect();
    let canonical_models: HashSet<&str> = config
        .services
        .iter()
        .map(|service| service.served_model.as_str())
        .collect();
    let mut names = HashSet::new();
    let mut ports: HashMap<u16, &str> = HashMap::new();
    let mut alias_owners: HashMap<&str, &str> = HashMap::new();
    let mut replica_contracts: HashMap<&str, (&str, &VllmCompatibility)> = HashMap::new();
    let mut systemd_units: HashMap<&str, &str> = HashMap::new();
    let mut environment_files: HashMap<&Path, &str> = HashMap::new();

    for (index, service) in config.services.iter().enumerate() {
        let location = format!("services[{index}] ({:?})", service.name);
        if service.name.trim().is_empty() {
            errors.push(format!("{location}.name must not be empty"));
        } else if !is_safe_name(&service.name) {
            errors.push(format!(
                "{location}.name may contain only ASCII letters, digits, '.', '_', and '-'"
            ));
        } else if !names.insert(service.name.as_str()) {
            errors.push(format!("duplicate service name {:?}", service.name));
        }
        if service.served_model.trim().is_empty() {
            errors.push(format!("{location}.served_model must not be empty"));
        }
        validate_runner_executable(service, &location, errors);
        require_absolute(
            &service.model_path,
            &format!("{location}.model_path"),
            errors,
        );
        if let Some(path) = &service.tokenizer.path {
            require_absolute(path, &format!("{location}.tokenizer.path"), errors);
        }
        validate_aliases(
            service,
            &location,
            &canonical_models,
            &mut alias_owners,
            errors,
        );
        validate_placement(service, &location, &known_gpus, errors);
        validate_backend(service, &location, &listener_ports, &mut ports, errors);
        validate_systemd(service, &location, errors);
        if let Some(previous) =
            systemd_units.insert(service.systemd.unit.as_str(), service.name.as_str())
        {
            errors.push(format!(
                "{location}.systemd.unit {:?} is already used by service {previous:?}",
                service.systemd.unit
            ));
        }
        if let Some(previous) = environment_files.insert(
            service.systemd.environment_file.as_path(),
            service.name.as_str(),
        ) {
            errors.push(format!(
                "{location}.systemd.environment_file {:?} is already used by service {previous:?}",
                service.systemd.environment_file
            ));
        }
        validate_runner_args(service, &location, errors);
        validate_environment(
            &service.environment,
            &format!("{location}.environment"),
            errors,
        );
        validate_readiness(service, &location, errors);
        validate_timing(service, &location, config, errors);
        validate_sleep(service, &location, errors);
        validate_compatibility(service, &location, errors);
        validate_replica_contract(service, &mut replica_contracts, errors);
    }
}

fn validate_runner_executable(
    service: &ServiceDefinition,
    location: &str,
    errors: &mut Vec<String>,
) {
    let field = format!("{location}.runner_executable");
    require_absolute(&service.runner_executable, &field, errors);
    match service.runner_executable.to_str() {
        Some(path) if path.chars().any(char::is_control) => {
            errors.push(format!("{field} contains a forbidden control character"));
        }
        Some(_) => {}
        None => errors.push(format!("{field} must be valid UTF-8")),
    }
}

fn validate_aliases<'a>(
    service: &'a ServiceDefinition,
    location: &str,
    canonical_models: &HashSet<&str>,
    alias_owners: &mut HashMap<&'a str, &'a str>,
    errors: &mut Vec<String>,
) {
    let mut local = HashSet::new();
    for alias in &service.aliases {
        if alias.trim().is_empty() {
            errors.push(format!(
                "{location}.aliases must not contain an empty alias"
            ));
        } else if alias == &service.served_model {
            errors.push(format!(
                "{location}.aliases contains canonical model name {alias:?}"
            ));
        } else if canonical_models.contains(alias.as_str()) {
            errors.push(format!(
                "{location}.alias {alias:?} conflicts with a canonical served model"
            ));
        } else if !local.insert(alias.as_str()) {
            errors.push(format!("{location}.aliases contains duplicate {alias:?}"));
        } else if let Some(owner) = alias_owners.get(alias.as_str()) {
            if *owner != service.served_model.as_str() {
                errors.push(format!(
                    "alias {alias:?} is ambiguous between canonical models {owner:?} and {:?}",
                    service.served_model
                ));
            }
        } else {
            alias_owners.insert(alias, &service.served_model);
        }
    }
}

fn validate_replica_contract<'a>(
    service: &'a ServiceDefinition,
    replica_contracts: &mut HashMap<&'a str, (&'a str, &'a VllmCompatibility)>,
    errors: &mut Vec<String>,
) {
    let Some((other_name, other)) = replica_contracts.get(service.served_model.as_str()) else {
        replica_contracts.insert(
            service.served_model.as_str(),
            (service.name.as_str(), &service.compatibility),
        );
        return;
    };

    if other.maximum_model_length != service.compatibility.maximum_model_length {
        errors.push(format!(
            "services {other_name:?} and {:?} share served_model {:?} but compatibility.maximum_model_length differs ({:?} versus {:?})",
            service.name,
            service.served_model,
            other.maximum_model_length,
            service.compatibility.maximum_model_length,
        ));
    }
    if other.kv_cache_dtype != service.compatibility.kv_cache_dtype {
        errors.push(format!(
            "services {other_name:?} and {:?} share served_model {:?} but compatibility.kv_cache_dtype differs ({:?} versus {:?})",
            service.name,
            service.served_model,
            other.kv_cache_dtype,
            service.compatibility.kv_cache_dtype,
        ));
    }
}

fn validate_placement(
    service: &ServiceDefinition,
    location: &str,
    known_gpus: &HashSet<&str>,
    errors: &mut Vec<String>,
) {
    let placement = &service.placement;
    if placement.tensor_parallelism == 0
        || placement.pipeline_parallelism == 0
        || placement.data_parallelism == 0
    {
        errors.push(format!(
            "{location}.placement parallelism values must all be greater than zero"
        ));
    }
    let expected = placement
        .tensor_parallelism
        .checked_mul(placement.pipeline_parallelism)
        .and_then(|value| value.checked_mul(placement.data_parallelism));
    match expected {
        None => errors.push(format!(
            "{location}.placement parallelism product overflows usize"
        )),
        Some(expected) if expected != placement.gpu_uuids.len() => errors.push(format!(
            "{location}.placement assigns {} GPUs but TP × PP × DP requires {expected}",
            placement.gpu_uuids.len()
        )),
        _ => {}
    }
    let mut assigned = HashSet::new();
    for uuid in &placement.gpu_uuids {
        if !known_gpus.contains(uuid.as_str()) {
            errors.push(format!(
                "{location}.placement references unknown GPU UUID {uuid:?}"
            ));
        } else if !assigned.insert(uuid) {
            errors.push(format!(
                "{location}.placement assigns GPU UUID {uuid:?} more than once"
            ));
        }
    }
    if placement.vram_reservation_bytes_per_gpu == 0 {
        errors.push(format!(
            "{location}.placement.vram_reservation_bytes_per_gpu must be greater than zero"
        ));
    }
}

fn validate_backend<'a>(
    service: &'a ServiceDefinition,
    location: &str,
    listener_ports: &HashMap<u16, &str>,
    ports: &mut HashMap<u16, &'a str>,
    errors: &mut Vec<String>,
) {
    if !service.backend.address.is_loopback() {
        errors.push(format!(
            "{location}.backend.address {} is not loopback",
            service.backend.address
        ));
    }
    if let Some(listener) = listener_ports.get(&service.backend.port) {
        errors.push(format!(
            "{location}.backend.port {} conflicts with {listener}",
            service.backend.port
        ));
    }
    if let Some(previous) = ports.insert(service.backend.port, &service.name) {
        errors.push(format!(
            "{location}.backend.port {} is already used by service {previous:?}",
            service.backend.port
        ));
    }
}

fn validate_systemd(service: &ServiceDefinition, location: &str, errors: &mut Vec<String>) {
    let unit = service.systemd.unit.as_str();
    if !unit.ends_with(".service")
        || unit.is_empty()
        || !unit
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"@_.:-".contains(&byte))
    {
        errors.push(format!(
            "{location}.systemd.unit must be a literal, safely named .service unit"
        ));
    }
    require_absolute(
        &service.systemd.environment_file,
        &format!("{location}.systemd.environment_file"),
        errors,
    );
    match service.systemd.environment_file.to_str() {
        Some(path)
            if path
                .bytes()
                .any(|byte| byte.is_ascii_whitespace() || b"\\\"'%".contains(&byte)) =>
        {
            errors.push(format!(
                "{location}.systemd.environment_file must be safe for a bare EnvironmentFile= directive"
            ));
        }
        Some(_) => {}
        None => errors.push(format!(
            "{location}.systemd.environment_file must be valid UTF-8"
        )),
    }
}

fn validate_runner_args(service: &ServiceDefinition, location: &str, errors: &mut Vec<String>) {
    const VALUE_ARGS: &[&str] = &[
        "block-size",
        "compilation-config",
        "cpu-offload-gb",
        "cuda-graph-sizes",
        "distributed-executor-backend",
        "download-dir",
        "generation-config",
        "guided-decoding-backend",
        "logits-processor-pattern",
        "max-num-batched-tokens",
        "max-num-seqs",
        "max-seq-len-to-capture",
        "mm-processor-cache-gb",
        "ngram-prompt-lookup-max",
        "ngram-prompt-lookup-min",
        "num-lookahead-slots",
        "prefix-caching-hash-algo",
        "revision",
        "rope-scaling",
        "rope-theta",
        "scheduler-cls",
        "scheduler-delay-factor",
        "seed",
        "speculative-config",
        "swap-space",
        "tool-call-parser",
        "tokenizer-revision",
        "reasoning-parser",
    ];
    const BOOL_ARGS: &[&str] = &[
        "disable-log-requests",
        "disable-log-stats",
        "disable-sliding-window",
        "enable-auto-tool-choice",
        "enable-chunked-prefill",
        "enable-prefix-caching",
        "enforce-eager",
        "skip-tokenizer-init",
        "use-tqdm-on-load",
    ];

    for (name, value) in &service.runner_args {
        let allowed = if VALUE_ARGS.contains(&name.as_str()) {
            true
        } else if BOOL_ARGS.contains(&name.as_str()) {
            if !matches!(value, RunnerArgValue::Bool(_)) {
                errors.push(format!(
                    "{location}.runner_args.{name} must have a boolean value"
                ));
            }
            true
        } else {
            false
        };
        if !allowed {
            errors.push(format!(
                "{location}.runner_args.{name} is not allowlisted; placement, network, model, tokenizer, sleep, and served-name options must use typed fields"
            ));
        }
        if let RunnerArgValue::String(value) = value {
            if value.contains('\0') || value.contains('\n') || value.contains('\r') {
                errors.push(format!(
                    "{location}.runner_args.{name} contains a forbidden control character"
                ));
            }
        }
        if let RunnerArgValue::Float(value) = value {
            if !value.is_finite() {
                errors.push(format!(
                    "{location}.runner_args.{name} must have a finite numeric value"
                ));
            }
        }
        if name == "distributed-executor-backend"
            && matches!(value, RunnerArgValue::String(value) if value.eq_ignore_ascii_case("ray"))
        {
            errors.push(format!(
                "{location}.runner_args.distributed-executor-backend cannot be `ray`; v1 requires the multiprocessing executor"
            ));
        }
        if name == "download-dir" {
            if let RunnerArgValue::String(value) = value {
                require_absolute(
                    Path::new(value),
                    &format!("{location}.runner_args.download-dir"),
                    errors,
                );
            }
        }
        if name == "tool-call-parser" {
            validate_parser_value(
                value,
                TOOL_CALL_PARSERS,
                &format!("{location}.runner_args.tool-call-parser"),
                errors,
            );
        }
        if name == "reasoning-parser" {
            validate_parser_value(
                value,
                REASONING_PARSERS,
                &format!("{location}.runner_args.reasoning-parser"),
                errors,
            );
        }
    }
}

// Keep these synchronized with ToolParserManager.list_registered(), which is
// surfaced by `vllm serve --help=all` for --tool-call-parser.
const TOOL_CALL_PARSERS: &[&str] = &[
    "apertus",
    "cohere_command3",
    "cohere_command4",
    "deepseek_v3",
    "deepseek_v31",
    "deepseek_v32",
    "deepseek_v4",
    "ernie45",
    "functiongemma",
    "gemma4",
    "gigachat3",
    "glm45",
    "glm47",
    "granite",
    "granite-20b-fc",
    "granite4",
    "hermes",
    "hunyuan_a13b",
    "hy_v3",
    "inkling",
    "internlm",
    "jamba",
    "kimi_k2",
    "lfm2",
    "llama3_json",
    "llama4_json",
    "llama4_pythonic",
    "longcat",
    "mimo",
    "minicpm5",
    "minimax_m2",
    "minimax_m3",
    "mistral",
    "olmo3",
    "openai",
    "phi4_mini_json",
    "poolside_v1",
    "pythonic",
    "qwen3_coder",
    "qwen3_xml",
    "seed_oss",
    "step3",
    "step3p5",
    "xlam",
];

// Keep these synchronized with ReasoningParserManager.list_registered() in
// `vllm.reasoning`.
const REASONING_PARSERS: &[&str] = &[
    "cohere_command3",
    "cohere_command4",
    "deepseek_r1",
    "deepseek_v3",
    "deepseek_v4",
    "ernie45",
    "gemma4",
    "glm45",
    "glm47",
    "granite",
    "holo2",
    "hunyuan_a13b",
    "hy_v3",
    "inkling",
    "kimi_k2",
    "mimo",
    "minimax_m2",
    "minimax_m2_append_think",
    "minimax_m3",
    "mistral",
    "nemotron_v3",
    "olmo3",
    "openai_gptoss",
    "poolside_v1",
    "qwen3",
    "seed_oss",
    "step3",
    "step3p5",
];

fn validate_parser_value(
    value: &RunnerArgValue,
    registry: &[&str],
    location: &str,
    errors: &mut Vec<String>,
) {
    match value {
        RunnerArgValue::String(value) if registry.contains(&value.as_str()) => {}
        RunnerArgValue::String(value) => errors.push(format!(
            "{location} value {value:?} is not registered by this vLLM version"
        )),
        _ => errors.push(format!(
            "{location} must be a string naming a registered vLLM parser"
        )),
    }
}

fn validate_environment(
    environment: &BTreeMap<String, String>,
    location: &str,
    errors: &mut Vec<String>,
) {
    const ALLOWED: &[&str] = &["HF_HOME", "PATH", "VLLM_CACHE_ROOT"];

    for (name, value) in environment {
        if !ALLOWED.contains(&name.as_str()) {
            errors.push(format!(
                "{location}.{name} is not allowlisted; allowed variables are {}",
                ALLOWED.join(", ")
            ));
        }
        if value.chars().any(char::is_control) {
            errors.push(format!(
                "{location}.{name} contains a forbidden control character"
            ));
        }
    }
}

fn validate_readiness(service: &ServiceDefinition, location: &str, errors: &mut Vec<String>) {
    let readiness = &service.readiness;
    for (field, path) in [
        ("health_path", &readiness.health_path),
        ("models_path", &readiness.models_path),
    ] {
        if !path.starts_with('/') || path.starts_with("//") {
            errors.push(format!(
                "{location}.readiness.{field} must be an absolute HTTP path"
            ));
        }
    }
    if readiness.consecutive_successes == 0 || readiness.consecutive_failures == 0 {
        errors.push(format!(
            "{location}.readiness consecutive thresholds must be greater than zero"
        ));
    }
    if readiness.interval_ms == 0
        || readiness.timeout_ms == 0
        || readiness.maximum_probe_age_ms == 0
    {
        errors.push(format!(
            "{location}.readiness intervals, timeout, and maximum probe age must be greater than zero"
        ));
    }
    if readiness.timeout_ms >= readiness.interval_ms {
        errors.push(format!(
            "{location}.readiness.timeout_ms must be less than interval_ms"
        ));
    }
    if readiness.maximum_probe_age_ms < readiness.interval_ms {
        errors.push(format!(
            "{location}.readiness.maximum_probe_age_ms must be at least interval_ms"
        ));
    }
}

fn validate_timing(
    service: &ServiceDefinition,
    location: &str,
    config: &RouterConfig,
    errors: &mut Vec<String>,
) {
    let deadlines = &service.deadlines;
    if deadlines.startup_ms == 0
        || deadlines.wake_ms == 0
        || deadlines.drain_ms == 0
        || deadlines.systemd_job_ms == 0
    {
        errors.push(format!(
            "{location}.deadlines must all be greater than zero"
        ));
    }
    let retry = &service.retry;
    if retry.maximum_attempts == 0 {
        errors.push(format!(
            "{location}.retry.maximum_attempts must be greater than zero"
        ));
    }
    if retry.initial_backoff_ms == 0 || retry.initial_backoff_ms > retry.maximum_backoff_ms {
        errors.push(format!(
            "{location}.retry backoff must satisfy 0 < initial_backoff_ms <= maximum_backoff_ms"
        ));
    }
    if !retry.multiplier.is_finite() || retry.multiplier < 1.0 {
        errors.push(format!(
            "{location}.retry.multiplier must be finite and at least 1"
        ));
    }
    if !retry.jitter_fraction.is_finite() || !(0.0..=1.0).contains(&retry.jitter_fraction) {
        errors.push(format!(
            "{location}.retry.jitter_fraction must be finite and between 0 and 1"
        ));
    }
    if service.parking.deadline_ms == 0
        || service.parking.queue_limit == 0
        || service.parking.retry_after_seconds == 0
    {
        errors.push(format!(
            "{location}.parking deadline, queue limit, and retry-after must be greater than zero"
        ));
    }
    if service.parking.queue_limit > config.policies.per_model_parked_request_limit {
        errors.push(format!(
            "{location}.parking.queue_limit {} exceeds policies.per_model_parked_request_limit {}",
            service.parking.queue_limit, config.policies.per_model_parked_request_limit
        ));
    }
    if service.parking.deadline_ms > deadlines.startup_ms {
        errors.push(format!(
            "{location}.parking.deadline_ms cannot exceed deadlines.startup_ms"
        ));
    }
    if let Some(warmup) = &service.warmup {
        if !warmup.path.starts_with('/') || warmup.path.starts_with("//") {
            errors.push(format!(
                "{location}.warmup.path must be an absolute HTTP path"
            ));
        }
        if !warmup.body.is_object() {
            errors.push(format!("{location}.warmup.body must be a JSON object"));
        }
    }
}

fn validate_sleep(service: &ServiceDefinition, location: &str, errors: &mut Vec<String>) {
    let sleep = &service.sleep;
    let idle = &service.idle;
    let unique_levels: HashSet<_> = sleep.levels.iter().copied().collect();
    if unique_levels.len() != sleep.levels.len() {
        errors.push(format!(
            "{location}.sleep.levels contains a duplicate level"
        ));
    }

    if sleep.enabled {
        if !sleep.hardware_validated {
            errors.push(format!(
                "{location}.sleep is enabled but hardware_validated is false"
            ));
        }
        if sleep.levels.is_empty() {
            errors.push(format!(
                "{location}.sleep.levels must not be empty when sleep is enabled"
            ));
        }
        if sleep.residual_vram_bytes_per_gpu == 0 {
            errors.push(format!(
                "{location}.sleep.residual_vram_bytes_per_gpu must be greater than zero"
            ));
        }
        if sleep.levels.contains(&SleepLevel::L1) {
            if sleep.l1_host_ram_bytes == 0 {
                errors.push(format!(
                    "{location}.sleep.l1_host_ram_bytes must be greater than zero for L1"
                ));
            }
            if idle.sleep_l1_after_ms.is_none() {
                errors.push(format!(
                    "{location}.idle.sleep_l1_after_ms is required when L1 is enabled"
                ));
            }
        } else if idle.sleep_l1_after_ms.is_some() {
            errors.push(format!(
                "{location}.idle.sleep_l1_after_ms requires sleep level L1"
            ));
        }
        if sleep.levels.contains(&SleepLevel::L2) {
            if idle.sleep_l2_after_ms.is_none() {
                errors.push(format!(
                    "{location}.idle.sleep_l2_after_ms is required when L2 is enabled"
                ));
            }
        } else if idle.sleep_l2_after_ms.is_some() {
            errors.push(format!(
                "{location}.idle.sleep_l2_after_ms requires sleep level L2"
            ));
        }
    } else if sleep.hardware_validated
        || !sleep.levels.is_empty()
        || sleep.residual_vram_bytes_per_gpu != 0
        || sleep.l1_host_ram_bytes != 0
        || idle.sleep_l1_after_ms.is_some()
        || idle.sleep_l2_after_ms.is_some()
    {
        errors.push(format!(
            "{location}.sleep is disabled, so sleep resources, levels, and idle sleep thresholds must be unset"
        ));
    }

    let ordered = [
        ("sleep_l1_after_ms", idle.sleep_l1_after_ms),
        ("sleep_l2_after_ms", idle.sleep_l2_after_ms),
        ("stop_after_ms", idle.stop_after_ms),
    ];
    let mut previous: Option<(&str, u64)> = None;
    for (name, value) in ordered {
        if let Some(value) = value {
            if value == 0 {
                errors.push(format!("{location}.idle.{name} must be greater than zero"));
            }
            if let Some((previous_name, previous_value)) = previous {
                if value <= previous_value {
                    errors.push(format!(
                        "{location}.idle.{name} must be greater than {previous_name}"
                    ));
                }
            }
            previous = Some((name, value));
        }
    }
}

fn validate_compatibility(service: &ServiceDefinition, location: &str, errors: &mut Vec<String>) {
    let compatibility = &service.compatibility;
    if !compatibility.gpu_memory_utilization.is_finite()
        || !(0.0..=1.0).contains(&compatibility.gpu_memory_utilization)
        || compatibility.gpu_memory_utilization == 0.0
    {
        errors.push(format!(
            "{location}.compatibility.gpu_memory_utilization must be finite and in (0, 1]"
        ));
    }
    if compatibility.maximum_model_length == Some(0) {
        errors.push(format!(
            "{location}.compatibility.maximum_model_length must be greater than zero"
        ));
    }
}

fn validate_resource_budgets(config: &RouterConfig, errors: &mut Vec<String>) {
    let mut host_l1_total = 0_u64;
    for service in &config.services {
        if service.sleep.enabled && service.sleep.levels.contains(&SleepLevel::L1) {
            match host_l1_total.checked_add(service.sleep.l1_host_ram_bytes) {
                Some(total) => host_l1_total = total,
                None => {
                    errors.push("L1 pinned host RAM reservations overflow u64".into());
                    host_l1_total = u64::MAX;
                    break;
                }
            }
        }
    }
    if host_l1_total > config.inventory.host_ram_capacity_bytes {
        errors.push(format!(
            "L1 pinned host RAM overcommit: all sleepers reserve {host_l1_total} bytes, inventory capacity is {} bytes",
            config.inventory.host_ram_capacity_bytes
        ));
    }

    for gpu in &config.inventory.gpus {
        let assigned: Vec<_> = config
            .services
            .iter()
            .filter(|service| service.placement.gpu_uuids.contains(&gpu.uuid))
            .collect();
        for service in &assigned {
            if service.placement.vram_reservation_bytes_per_gpu > gpu.vram_capacity_bytes {
                errors.push(format!(
                    "service {:?} awake VRAM reservation on GPU {:?} is {} bytes, exceeding its {} byte capacity",
                    service.name,
                    gpu.uuid,
                    service.placement.vram_reservation_bytes_per_gpu,
                    gpu.vram_capacity_bytes
                ));
            }
            if service.sleep.residual_vram_bytes_per_gpu
                > service.placement.vram_reservation_bytes_per_gpu
            {
                errors.push(format!(
                    "service {:?} sleep residual VRAM on GPU {:?} exceeds its awake reservation",
                    service.name, gpu.uuid
                ));
            }
        }

        let all_asleep = checked_sum(assigned.iter().map(|service| {
            if service.sleep.enabled {
                service.sleep.residual_vram_bytes_per_gpu
            } else {
                0
            }
        }));
        match all_asleep {
            None => errors.push(format!(
                "GPU {:?} residual sleep VRAM reservations overflow u64",
                gpu.uuid
            )),
            Some(total) if total > gpu.vram_capacity_bytes => errors.push(format!(
                "GPU {:?} residual sleep VRAM overcommit: sleepers require {total} bytes but capacity is {} bytes",
                gpu.uuid, gpu.vram_capacity_bytes
            )),
            _ => {}
        }

        for awake in &assigned {
            let sleeping_others = checked_sum(assigned.iter().filter_map(|service| {
                if service.name != awake.name && service.sleep.enabled {
                    Some(service.sleep.residual_vram_bytes_per_gpu)
                } else {
                    None
                }
            }));
            let required = sleeping_others.and_then(|sleeping| {
                sleeping.checked_add(awake.placement.vram_reservation_bytes_per_gpu)
            });
            match required {
                None => errors.push(format!(
                    "GPU {:?} VRAM reservations overflow u64 when service {:?} is awake",
                    gpu.uuid, awake.name
                )),
                Some(total) if total > gpu.vram_capacity_bytes => errors.push(format!(
                    "GPU {:?} VRAM overcommit when service {:?} is awake and other residents are asleep: requires {total} bytes but capacity is {} bytes",
                    gpu.uuid, awake.name, gpu.vram_capacity_bytes
                )),
                _ => {}
            }
        }
    }
}

fn checked_sum(mut values: impl Iterator<Item = u64>) -> Option<u64> {
    values.try_fold(0_u64, u64::checked_add)
}

fn require_absolute(path: &Path, field: &str, errors: &mut Vec<String>) {
    if !path.is_absolute() {
        errors.push(format!("{field} must be an absolute path, got {path:?}"));
    }
}

fn is_safe_name(name: &str) -> bool {
    name.bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

const fn default_true() -> bool {
    true
}

fn default_vllm_executable() -> PathBuf {
    PathBuf::from("/opt/vllm-venv/bin/vllm")
}

const fn default_parallelism() -> usize {
    1
}

fn default_warmup_body() -> serde_json::Value {
    serde_json::json!({
        "model": "__served_model__",
        "prompt": "warmup",
        "max_tokens": 1
    })
}

#[cfg(test)]
mod tests;
