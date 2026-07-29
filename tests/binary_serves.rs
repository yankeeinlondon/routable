use std::{
    fs,
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

use tempfile::TempDir;
use uuid::Uuid;
use vllm_router::{
    config::{GpuDefinition, RouterConfig},
    gpu::GpuInventory,
};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(15);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

struct HostFixture {
    child: Option<Child>,
    unit_path: PathBuf,
    _directory: TempDir,
}

impl HostFixture {
    fn spawn(
        directory: TempDir,
        unit_path: PathBuf,
        pingora_config: &Path,
        router_config: &Path,
    ) -> io::Result<Self> {
        let mut fixture = Self {
            child: None,
            unit_path,
            _directory: directory,
        };
        fixture.child = Some(
            Command::new(env!("CARGO_BIN_EXE_vllm-router"))
                .args([
                    "-c",
                    pingora_config
                        .to_str()
                        .expect("temporary Pingora path must be UTF-8"),
                    "--router-config",
                    router_config
                        .to_str()
                        .expect("temporary router path must be UTF-8"),
                ])
                .env("RUST_LOG", "warn")
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()?,
        );
        Ok(fixture)
    }

    fn child(&mut self) -> &mut Child {
        self.child.as_mut().expect("router child must exist")
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child().try_wait()
    }

    fn stop_with_sigterm(&mut self) -> io::Result<ExitStatus> {
        let child = self.child();
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        send_signal(child.id(), libc::SIGTERM)?;
        if let Some(status) = wait_for_exit(child, SHUTDOWN_TIMEOUT)? {
            return Ok(status);
        }

        child.kill()?;
        let _ = child.wait();
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "router did not exit after SIGTERM",
        ))
    }

    fn cleanup_generated_unit(&mut self) -> io::Result<()> {
        match fs::remove_file(&self.unit_path) {
            Ok(()) => {
                let status = Command::new("systemctl").args(["daemon-reload"]).status()?;
                if !status.success() {
                    return Err(io::Error::other(format!(
                        "systemctl daemon-reload exited with {status}"
                    )));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Ok(())
    }
}

impl Drop for HostFixture {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = send_signal(child.id(), libc::SIGTERM);
                if wait_for_exit(child, Duration::from_secs(2))
                    .ok()
                    .flatten()
                    .is_none()
                {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
        if let Err(error) = self.cleanup_generated_unit() {
            eprintln!(
                "failed to clean generated host-test unit {}: {error}",
                self.unit_path.display()
            );
        }
    }
}

#[test]
#[ignore = "requires root, a host system bus, and systemd"]
fn binary_actually_serves() {
    let reservations = reserve_ports(4).expect("ephemeral ports should be available");
    let ports: Vec<u16> = reservations
        .iter()
        .map(|listener| listener.local_addr().unwrap().port())
        .collect();
    let management = SocketAddr::from(([127, 0, 0, 1], ports[1]));

    let directory = tempfile::tempdir().expect("temporary directory should be available");
    let unique = Uuid::new_v4().simple().to_string();
    let unit_name = format!("vllm-router-host-test-{unique}.service");
    let unit_path = Path::new("/run/systemd/system").join(&unit_name);
    let (pingora_config, router_config) = write_configs(directory.path(), &unit_name, &ports);

    // Hold all four ports until every config file is ready, then release them
    // immediately before the child binds its three listeners.
    drop(reservations);
    let mut fixture = HostFixture::spawn(directory, unit_path, &pingora_config, &router_config)
        .expect("router binary should launch");

    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        if let Some(status) = fixture
            .try_wait()
            .expect("router process status should be readable")
        {
            panic!("router exited before serving /livez: {status}");
        }
        if livez_returns_ok(management).unwrap_or(false) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "router did not serve /livez within {STARTUP_TIMEOUT:?}"
        );
        thread::sleep(Duration::from_millis(50));
    }

    let shutdown_started = Instant::now();
    let status = fixture
        .stop_with_sigterm()
        .expect("router should stop after SIGTERM");
    assert!(
        shutdown_started.elapsed() < SHUTDOWN_TIMEOUT,
        "router exceeded the bounded SIGTERM shutdown window"
    );
    assert!(status.success(), "router exited unsuccessfully: {status}");
    fixture
        .cleanup_generated_unit()
        .expect("generated host-test unit should be removed and systemd reloaded");
}

fn reserve_ports(count: usize) -> io::Result<Vec<TcpListener>> {
    (0..count)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)))
        .collect()
}

fn write_configs(directory: &Path, unit_name: &str, ports: &[u16]) -> (PathBuf, PathBuf) {
    assert_eq!(ports.len(), 4);

    let token_path = directory.join("token");
    fs::write(&token_path, b"host-test-token\n").unwrap();
    fs::set_permissions(&token_path, fs::Permissions::from_mode(0o600)).unwrap();

    let mut config: RouterConfig =
        serde_yaml::from_str(include_str!("../src/config/example.yaml")).unwrap();
    let gpu = GpuInventory::initialize()
        .inventory()
        .expect("host test requires a live NVML inventory")
        .into_values()
        .next()
        .expect("host test requires at least one GPU");
    config.inventory.gpus = vec![GpuDefinition {
        uuid: gpu.uuid.clone(),
        vram_capacity_bytes: gpu.total_memory_bytes,
    }];
    config.listeners.data_plane = SocketAddr::from(([127, 0, 0, 1], ports[0]));
    config.listeners.management = SocketAddr::from(([127, 0, 0, 1], ports[1]));
    config.listeners.metrics = SocketAddr::from(([127, 0, 0, 1], ports[2]));
    config.listeners.management_bearer_token_file = token_path;

    let service = config.services.first_mut().unwrap();
    service.name = "host-test".into();
    service.runner_executable = "/bin/true".into();
    service.model_path = "/tmp".into();
    service.tokenizer.path = None;
    service.placement.gpu_uuids = vec![gpu.uuid];
    service.placement.vram_reservation_bytes_per_gpu = gpu.total_memory_bytes;
    service.backend.port = ports[3];
    service.systemd.unit = unit_name.into();
    service.systemd.environment_file = directory.join("host-test.env");
    service.automatic_start = false;
    config
        .validate()
        .expect("minimal host-test router config should be valid");

    let router_path = directory.join("router.yaml");
    fs::write(&router_path, serde_yaml::to_string(&config).unwrap()).unwrap();

    let pingora_path = directory.join("pingora.yaml");
    fs::write(
        &pingora_path,
        format!(
            "---\n\
             version: 1\n\
             threads: 1\n\
             work_stealing: true\n\
             daemon: false\n\
             pid_file: {}\n\
             upgrade_sock: {}\n\
             grace_period_seconds: 0\n\
             graceful_shutdown_timeout_seconds: 0\n",
            directory.join("pingora.pid").display(),
            directory.join("pingora-upgrade.sock").display(),
        ),
    )
    .unwrap();

    (pingora_path, router_path)
}

fn livez_returns_ok(address: SocketAddr) -> io::Result<bool> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_millis(200))?;
    stream.set_read_timeout(Some(Duration::from_millis(200)))?;
    stream.set_write_timeout(Some(Duration::from_millis(200)))?;
    stream.write_all(b"GET /livez HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;

    let mut response = [0_u8; 1024];
    let length = stream.read(&mut response)?;
    Ok(response[..length].starts_with(b"HTTP/1.1 200"))
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn send_signal(pid: u32, signal: libc::c_int) -> io::Result<()> {
    let pid = libc::pid_t::try_from(pid)
        .map_err(|_| io::Error::other(format!("child PID {pid} does not fit pid_t")))?;
    // SAFETY: `pid` is the live child process ID returned by `Child::id`, and
    // the signal number is a libc constant.
    if unsafe { libc::kill(pid, signal) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
