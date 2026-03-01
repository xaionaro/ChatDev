"""Utility tools to manage Go projects inside the workspace."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.\-/+=<>!@:]+$")
_DEFAULT_TIMEOUT = float(os.getenv("GO_CMD_TIMEOUT", "120"))


class GoWorkspaceContext:
    """Resolve the workspace root from the injected runtime context."""

    def __init__(self, ctx: Dict[str, Any] | None):
        if ctx is None:
            raise ValueError("_context is required for Go tools")
        self.workspace_root = self._require_workspace(ctx.get("python_workspace_root"))

    @staticmethod
    def _require_workspace(raw_path: Any) -> Path:
        if raw_path is None:
            raise ValueError("python_workspace_root missing from _context")
        path = Path(raw_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


def _coerce_timeout(timeout_seconds: Any) -> float | None:
    if timeout_seconds is None:
        return None
    if isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be a number")
    if isinstance(timeout_seconds, (int, float)):
        value = float(timeout_seconds)
    elif isinstance(timeout_seconds, str):
        raw = timeout_seconds.strip()
        if not raw:
            raise ValueError("timeout_seconds cannot be empty")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("timeout_seconds must be a number") from exc
    else:
        raise ValueError("timeout_seconds must be a number")
    if value <= 0:
        raise ValueError("timeout_seconds must be positive")
    return value


def _validate_packages(packages: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for pkg in packages:
        if not isinstance(pkg, str):
            raise ValueError("package entries must be strings")
        stripped = pkg.strip()
        if not stripped:
            raise ValueError("package names cannot be empty")
        if not _SAFE_PACKAGE_RE.match(stripped):
            raise ValueError(f"unsafe characters detected in package spec: {pkg}")
        if stripped.startswith("-"):
            raise ValueError(f"flags are not allowed in packages list: {pkg}")
        normalized.append(stripped)
    if not normalized:
        raise ValueError("at least one package is required")
    return normalized


def _validate_args(args: Sequence[str] | None) -> List[str]:
    if not args:
        return []
    normalized: List[str] = []
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError("args entries must be strings")
        stripped = arg.strip()
        if not stripped:
            raise ValueError("args entries cannot be empty")
        normalized.append(stripped)
    return normalized


def _validate_env(env: Mapping[str, str] | None) -> Dict[str, str]:
    if env is None:
        return {}
    result: Dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise ValueError("environment variable keys must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError("environment variable values must be strings")
        result[key] = value
    return result


def _run_go_command(
    cmd: List[str],
    workspace_root: Path,
    *,
    step: str | None = None,
    env: Dict[str, str] | None = None,
    timeout: float | None = None,
) -> Dict[str, Any]:
    timeout_value = _DEFAULT_TIMEOUT if timeout is None else timeout
    env_vars = {**os.environ}
    if env:
        env_vars.update(env)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout_value,
            check=False,
            env=env_vars,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "'go' command not found in PATH. Ensure Go is installed."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        return {
            "command": cmd,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": None,
            "step": step,
            "timed_out": True,
            "timeout": timeout_value,
            "error": f"go command ({step}) timed out after {timeout_value}s",
        }

    return {
        "command": cmd,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "returncode": completed.returncode,
        "step": step,
    }


def init_go_module(
    module_name: str,
    *,
    go_version: str | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Initialize a Go module in the workspace with go mod init."""

    ctx = GoWorkspaceContext(_context)
    name = module_name.strip()
    if not name:
        raise ValueError("module_name cannot be empty")

    cmd: List[str] = ["go", "mod", "init", name]
    result = _run_go_command(cmd, ctx.workspace_root, step="go mod init")

    if go_version and result.get("returncode") == 0:
        edit_cmd = ["go", "mod", "edit", f"-go={go_version.strip()}"]
        edit_result = _run_go_command(edit_cmd, ctx.workspace_root, step="go mod edit -go")
        result["go_version_step"] = edit_result

    return result


def go_get(
    packages: Sequence[str],
    *,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Add Go dependencies using go get."""

    ctx = GoWorkspaceContext(_context)
    safe_packages = _validate_packages(packages)
    cmd: List[str] = ["go", "get"]
    cmd.extend(safe_packages)
    return _run_go_command(cmd, ctx.workspace_root, step="go get")


def go_build(
    *,
    entry_point: str = ".",
    output: str | None = None,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a Go program inside the workspace."""

    ctx = GoWorkspaceContext(_context)
    timeout_seconds = _coerce_timeout(timeout_seconds)

    cmd: List[str] = ["go", "build"]
    if output:
        cmd.extend(["-o", output.strip()])
    cmd.extend(_validate_args(args))
    cmd.append(entry_point.strip() or ".")

    return _run_go_command(
        cmd,
        ctx.workspace_root,
        step="go build",
        env=_validate_env(env),
        timeout=timeout_seconds,
    )


def go_run(
    *,
    entry_point: str = ".",
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a Go program inside the workspace using go run."""

    ctx = GoWorkspaceContext(_context)
    timeout_seconds = _coerce_timeout(timeout_seconds)

    cmd: List[str] = ["go", "run", entry_point.strip() or "."]
    cmd.extend(_validate_args(args))

    result = _run_go_command(
        cmd,
        ctx.workspace_root,
        step="go run",
        env=_validate_env(env),
        timeout=timeout_seconds,
    )
    result["workspace_root"] = str(ctx.workspace_root)
    return result


def start_file_server(
    *,
    port: int = 7000,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Start a background HTTP file server in the workspace directory.

    Writes a minimal Go HTTP server, builds it, and starts it as a
    background process.  Returns the URL where files are served.
    """
    ctx = GoWorkspaceContext(_context)

    server_dir = ctx.workspace_root / "_fileserver"
    server_dir.mkdir(exist_ok=True)

    server_src = server_dir / "main.go"
    server_src.write_text(
        'package main\n'
        '\n'
        'import (\n'
        '\t"flag"\n'
        '\t"log"\n'
        '\t"net/http"\n'
        ')\n'
        '\n'
        'func main() {\n'
        '\tdir := flag.String("dir", ".", "directory to serve")\n'
        '\tport := flag.String("port", "7000", "port to listen on")\n'
        '\tflag.Parse()\n'
        '\tlog.Printf("Serving %s on http://0.0.0.0:%s", *dir, *port)\n'
        '\tlog.Fatal(http.ListenAndServe(":"+*port, http.FileServer(http.Dir(*dir))))\n'
        '}\n'
    )

    # Init module (ignore error if go.mod already exists)
    subprocess.run(
        ["go", "mod", "init", "fileserver"],
        cwd=str(server_dir),
        capture_output=True,
    )

    binary = server_dir / "server"
    build = subprocess.run(
        ["go", "build", "-o", str(binary), "."],
        cwd=str(server_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if build.returncode != 0:
        return {
            "error": f"failed to build file server: {build.stderr}",
            "returncode": build.returncode,
        }

    proc = subprocess.Popen(
        [str(binary), "-dir", str(ctx.workspace_root), "-port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return {
        "url": f"http://0.0.0.0:{port}",
        "pid": proc.pid,
        "workspace": str(ctx.workspace_root),
    }


def go_test(
    *,
    package: str = "./...",
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    _context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run Go tests inside the workspace using go test."""

    ctx = GoWorkspaceContext(_context)
    timeout_seconds = _coerce_timeout(timeout_seconds)

    cmd: List[str] = ["go", "test"]
    cmd.extend(_validate_args(args))
    cmd.append(package.strip() or "./...")

    result = _run_go_command(
        cmd,
        ctx.workspace_root,
        step="go test",
        env=_validate_env(env),
        timeout=timeout_seconds,
    )
    result["workspace_root"] = str(ctx.workspace_root)
    return result
