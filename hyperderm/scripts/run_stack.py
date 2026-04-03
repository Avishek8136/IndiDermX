from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _spawn(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(command, env=env)


def main() -> None:
    env = os.environ.copy()
    env.setdefault("HYPERDERM_API_URL", "http://127.0.0.1:8000")

    processes: list[subprocess.Popen] = []

    def _shutdown(*_: object) -> None:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    python = sys.executable

    processes.append(
        _spawn(
            [
                python,
                "-m",
                "uvicorn",
                "hyperderm.api.app:create_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            env,
        )
    )
    processes.append(_spawn([python, "-m", "hyperderm.mcp.run_http"], env))
    processes.append(
        _spawn(
            [
                python,
                "-m",
                "uvicorn",
                "frontend_service.app:create_frontend_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ],
            env,
        )
    )

    print("Started backend: http://127.0.0.1:8000")
    print("Started MCP HTTP: http://127.0.0.1:9001")
    print("Started frontend: http://127.0.0.1:8080")
    print("Press Ctrl+C to stop all services.")

    while True:
        for proc in processes:
            if proc.poll() is not None:
                print(f"Process exited with code {proc.returncode}. Stopping stack.")
                _shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
