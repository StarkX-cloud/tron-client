"""
TRON Configuration & Auto-Discovery
Auto-detects server, respects environment variables, zero-config ready.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class TronConfig:
    """Centralized TRON configuration with auto-discovery."""

    def __init__(self):
        self.server_url: Optional[str] = None
        self._discovered = False
        self._server_registry: dict[str, str] = {
            "local": "http://127.0.0.1:9000",
            "localhost": "http://127.0.0.1:9000",
        }
        self._local_server_process = None
        self._local_worker_process = None

    @property
    def url(self) -> str:
        """Get TRON server URL with auto-discovery fallback."""
        # 1. Check explicit environment variables
        if env_url := self._env_server_url():
            self.server_url = env_url.rstrip("/")
            return self.server_url

        # 2. Use explicit configured URL if present
        if self.server_url:
            return self.server_url

        # 3. Check if already discovered
        if self._discovered and self.server_url:
            return self.server_url

        # 4. Auto-discover on localhost
        self.server_url = self._auto_discover()
        self._discovered = True
        return self.server_url

    @staticmethod
    def _auto_discover() -> str:
        """Auto-discover TRON server on localhost."""
        common_ports = [9000, 8000, 8080, 5000]

        for port in common_ports:
            if TronConfig._is_server_alive(f"http://127.0.0.1:{port}"):
                url = f"http://127.0.0.1:{port}"
                print(f"[TRON] Auto-discovered server at {url}")
                return url

        # Fallback
        default = "http://127.0.0.1:9000"
        print(f"[TRON] No server found. Using default: {default}")
        return default

    @staticmethod
    def _is_server_alive(url: str, timeout: float = 0.5) -> bool:
        """Check if TRON server is responding."""
        try:
            import requests
            resp = requests.get(f"{url}/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _is_stream_supported(url: str, timeout: float = 1.0) -> bool:
        """Confirm the server supports the /stream SSE endpoint."""
        try:
            import requests
            from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout

            resp = requests.get(
                f"{url}/stream/unknown-job-id",
                stream=True,
                timeout=timeout,
            )
            if resp.status_code != 200:
                resp.close()
                return False

            it = resp.iter_content(chunk_size=64, decode_unicode=True)
            first_chunk = next(it, None)
            if not first_chunk or not ("retry:" in first_chunk or "data:" in first_chunk):
                resp.close()
                return False

            try:
                next(it)
            except ReadTimeout:
                resp.close()
                return True
            except (ChunkedEncodingError, ConnectionError):
                resp.close()
                return False
            except StopIteration:
                resp.close()
                return False

            resp.close()
            return True
        except Exception:
            return False

    @staticmethod
    def _is_bindable(host: str, port: int) -> bool:
        """Check whether a TCP port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True
        except OSError:
            return False

    @staticmethod
    def _find_free_port(start_port: int, host: str = "127.0.0.1", max_port: int = 9100) -> int:
        """Find a free local port for starting a TRON server."""
        for port in range(start_port, max_port + 1):
            if TronConfig._is_bindable(host, port):
                return port
        raise RuntimeError(f"No free port found between {start_port} and {max_port}")

    @staticmethod
    def _is_local_url(url: str) -> bool:
        return url.startswith("http://127.0.0.1") or url.startswith("http://localhost")

    def _env_server_url(self) -> Optional[str]:
        for env_var in ("TRON_URL", "TRON_SERVER"):
            if env_url := os.getenv(env_var):
                return env_url
        return None

    def add_server(self, name: str, url: str) -> None:
        """Register a named TRON server for later selection."""
        self._server_registry[name] = url.rstrip("/")

    def get_server(self, name: str) -> Optional[str]:
        """Return a registered server URL by name."""
        return self._server_registry.get(name)

    def list_servers(self) -> dict[str, str]:
        """List registered TRON servers."""
        return dict(self._server_registry)

    def use_server(self, name: str) -> str:
        """Switch the active TRON server to a registered named server."""
        url = self.get_server(name)
        if not url:
            raise ValueError(f"No TRON server registered with name '{name}'")
        self.set_url(url)
        self._discovered = True
        return url

    def start_local_server(
        self,
        port: int = 9000,
        host: str = "127.0.0.1",
        wait: bool = True,
        timeout: float = 10.0,
        reload_flag: bool = False,
    ) -> str:
        """Start a local TRON queue server process and connect the SDK to it."""
        if not self._is_bindable(host, port) and not self._is_server_alive(f"http://{host}:{port}"):
            original = port
            port = self._find_free_port(port + 1, host)
            print(f"[TRON] Port {original} unavailable; starting local server on port {port}")

        url = f"http://{host}:{port}"
        if self._is_server_alive(url):
            self.set_url(url)
            self._discovered = True
            self.add_server("local", url)
            return url

        queue_server_py = Path(__file__).resolve().parents[1] / "queue_server.py"
        if not queue_server_py.exists():
            raise FileNotFoundError(
                f"Unable to start local server; {queue_server_py} not found"
            )

        env = os.environ.copy()
        env["TRON_HOST"] = host
        env["TRON_PORT"] = str(port)
        env["TRON_RELOAD"] = str(reload_flag).lower()

        process = subprocess.Popen(
            [sys.executable, str(queue_server_py)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._local_server_process = process

        if wait:
            self._wait_for_server(url, timeout=timeout)

        self.set_url(url)
        self._discovered = True
        self.add_server("local", url)
        return url

    def _ensure_local_runtime(self, url: str) -> None:
        """Ensure a local runtime is available for developer convenience."""
        if not self._is_local_url(url):
            return
        if self._local_worker_process is None:
            try:
                self.start_local_worker(queue_url=url)
            except Exception:
                pass

    def start_local_worker(
        self,
        queue_url: Optional[str] = None,
    ) -> str:
        """Start a local TRON worker process and connect it to the active server."""
        queue_url = queue_url or self.url
        worker_py = Path(__file__).resolve().parents[1] / "worker.py"
        if not worker_py.exists():
            raise FileNotFoundError(
                f"Unable to start local worker; {worker_py} not found"
            )

        if not self._is_server_alive(queue_url):
            raise RuntimeError(f"Cannot start worker; TRON server not reachable at {queue_url}")

        env = os.environ.copy()
        env["TRON_URL"] = queue_url

        process = subprocess.Popen(
            [sys.executable, str(worker_py)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._local_worker_process = process
        return queue_url

    def start_local_environment(
        self,
        port: int = 9000,
        host: str = "127.0.0.1",
        wait: bool = True,
        timeout: float = 10.0,
        reload_flag: bool = False,
    ) -> str:
        """Start a local TRON server and worker for local development."""
        url = self.start_local_server(
            port=port,
            host=host,
            wait=wait,
            timeout=timeout,
            reload_flag=reload_flag,
        )
        self._ensure_local_runtime(url)
        return url

    def stop_local_worker(self) -> bool:
        """Stop the local TRON worker process started by this SDK."""
        process = self._local_worker_process
        if not process:
            return False
        process.terminate()
        process.wait(timeout=5)
        self._local_worker_process = None
        return True

    def ensure_server(
        self,
        port: int = 9000,
        host: str = "127.0.0.1",
        wait: bool = True,
        timeout: float = 10.0,
        reload_flag: bool = False,
    ) -> str:
        """Ensure a TRON server is available, else start a local one."""
        # If a configured server is already available, use it.
        current_url = self._env_server_url() or self.server_url
        if current_url:
            current_url = current_url.rstrip("/")
            if self._is_server_alive(current_url):
                if self._is_stream_supported(current_url):
                    self.set_url(current_url)
                    self._discovered = True
                    if self._is_local_url(current_url):
                        self._ensure_local_runtime(current_url)
                    return current_url
                print(f"[TRON] Server at {current_url} is alive but does not support /stream; launching a local server.")

        # If a named local server is registered and alive, use it.
        local_url = self.get_server("local")
        if local_url and self._is_server_alive(local_url) and self._is_stream_supported(local_url):
            self.set_url(local_url)
            self._discovered = True
            self._ensure_local_runtime(local_url)
            return local_url

        # Otherwise, launch a local server.
        url = self.start_local_server(
            port=port,
            host=host,
            wait=wait,
            timeout=timeout,
            reload_flag=reload_flag,
        )
        self._ensure_local_runtime(url)
        return url

    def stop_local_server(self) -> bool:
        """Stop the local TRON server process started by this SDK."""
        process = self._local_server_process
        if not process:
            return False
        process.terminate()
        process.wait(timeout=5)
        self._local_server_process = None
        return True

    def _wait_for_server(self, url: str, timeout: float = 10.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
            if self._is_server_alive(url):
                return
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for TRON server at {url}")

    def set_url(self, url: str) -> None:
        """Explicitly set server URL."""
        self.server_url = url.rstrip("/")

    def __repr__(self) -> str:
        return f"<TronConfig url={self.url}>"


# Global config instance
_config = TronConfig()


def get_config() -> TronConfig:
    """Get global TRON config."""
    return _config


def set_config_url(url: str) -> None:
    """Set TRON server URL."""
    _config.set_url(url)


def add_server(name: str, url: str) -> None:
    """Register a named TRON server."""
    _config.add_server(name, url)


def get_server(name: str) -> Optional[str]:
    """Return a registered server URL by name."""
    return _config.get_server(name)


def list_servers() -> dict[str, str]:
    """List registered TRON servers."""
    return _config.list_servers()


def use_server(name: str) -> str:
    """Switch the active TRON server to a named server."""
    return _config.use_server(name)


def start_local_server(
    port: int = 9000,
    host: str = "127.0.0.1",
    wait: bool = True,
    timeout: float = 10.0,
    reload_flag: bool = False,
) -> str:
    """Start a local TRON server process and connect the SDK to it."""
    return _config.start_local_server(
        port=port,
        host=host,
        wait=wait,
        timeout=timeout,
        reload_flag=reload_flag,
    )


def ensure_server(
    port: int = 9000,
    host: str = "127.0.0.1",
    wait: bool = True,
    timeout: float = 10.0,
    reload_flag: bool = False,
) -> str:
    """Ensure the SDK has a connected TRON server, starting local only if needed."""
    return _config.ensure_server(
        port=port,
        host=host,
        wait=wait,
        timeout=timeout,
        reload_flag=reload_flag,
    )


def stop_local_server() -> bool:
    """Stop the local TRON server process started by the SDK."""
    return _config.stop_local_server()


def start_local_worker(queue_url: Optional[str] = None) -> str:
    """Start a local TRON worker process and connect it to the active server."""
    return _config.start_local_worker(queue_url=queue_url)


def stop_local_worker() -> bool:
    """Stop the local TRON worker process started by the SDK."""
    return _config.stop_local_worker()


def start_local_environment(
    port: int = 9000,
    host: str = "127.0.0.1",
    wait: bool = True,
    timeout: float = 10.0,
    reload_flag: bool = False,
) -> str:
    """Start a local TRON server and worker for local development."""
    return _config.start_local_environment(
        port=port,
        host=host,
        wait=wait,
        timeout=timeout,
        reload_flag=reload_flag,
    )
