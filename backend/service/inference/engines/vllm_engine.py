"""vLLM engine: launches and health-checks an OpenAI-compatible vLLM server."""

import glob
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from typing import IO, Any

import httpx
import psutil
from openai import APIError, OpenAI

from ...config_models import InferenceConfig
from ...settings import (
    VLLM_CLIENT_HOST,
    VLLM_ENABLE_LOG_REQUESTS,
    VLLM_HEALTH_TIMEOUT,
    VLLM_LOGGING_LEVEL,
    VLLM_OPENAI_API_KEY,
    VLLM_PORT,
    VLLM_SERVED_MODEL_NAME,
    VLLM_SERVER_HOST,
    VLLM_SERVER_PROJECT_DIR,
    VLLM_STARTUP_SWEEP,
    VLLM_SWEEP_PORTS,
    configure_logging,
)
from .base_engine import BaseEngine
from .vllm_error_classifier import (
    VllmErrorReport,
    classify_stderr,
)

logger = configure_logging(__name__)

_REQUIRED_NVIDIA_RUNTIME_LIBS = (
    ("libcudnn.so.9", "nvidia-cudnn-cu13"),
    ("libcusparseLt.so.0", "nvidia-cusparselt-cu13"),
    ("libnccl.so.2", "nvidia-nccl-cu13"),
    ("libnvshmem_host.so.3", "nvidia-nvshmem-cu13"),
)


# end of imports
def sweep_stale_vllm_processes() -> dict[str, Any]:
    """
    Clean up leftover vLLM serve processes at service startup.

    Controlled by environment variables:
    - VLLM_STARTUP_SWEEP: 1/true/yes/on to enable (enabled by default)
    - VLLM_SWEEP_PORTS: comma-separated port list (default 5000)
    """

    enabled = VLLM_STARTUP_SWEEP
    if not enabled:
        return {"enabled": False, "ports": [], "killed": 0, "pids": []}

    ports = VLLM_SWEEP_PORTS

    if not ports:
        return {"enabled": True, "ports": [], "killed": 0, "pids": []}

    target_ports = set(ports)
    killed_pids: list[int] = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.cmdline() or []).lower()
            if "vllm" not in cmdline or "serve" not in cmdline:
                continue

            owns_target_port = False
            for conn in proc.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port in target_ports:
                    owns_target_port = True
                    break

            if not owns_target_port:
                continue

            logger.warning(
                "[StartupSweep] Found stale vLLM process pid=%s cmd=%s",
                proc.pid,
                cmdline,
            )

            try:
                pgid = os.getpgid(proc.pid)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
                os.killpg(pgid, signal.SIGKILL)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                    OSError,
                ):
                    continue

            killed_pids.append(proc.pid)
            time.sleep(0.2)
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            continue

    return {
        "enabled": True,
        "ports": sorted(target_ports),
        "killed": len(killed_pids),
        "pids": killed_pids,
    }


@dataclass
class VllmRuntimeContext:
    """
    All mutable state of the vLLM engine across load -> inference -> unload.

    Collecting the previously scattered instance attributes means:

    1. ``unload`` only needs ``self.runtime = VllmRuntimeContext()`` to reset
       everything, so no field is left stale (e.g. an old ``client``).
    2. A ``VllmRuntimeContext`` can be built standalone and fed to helpers in
       unit tests, without wiring up a full engine and multiprocessing queues.
    3. Every lifecycle-related field is declared in one place.

    Note: ``config`` (``InferenceConfig``) still lives on ``BaseEngine``.
    """

    base_url: str = ""
    api_key: str = ""
    health_timeout_s: float = 0.0
    served_model_name: str | None = None
    current_request_id: str | None = None

    client: OpenAI | None = None
    server_process: subprocess.Popen | None = None
    stdout_pump_thread: threading.Thread | None = None
    stderr_pump_thread: threading.Thread | None = None

    stderr_buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    # Recent stderr from the vLLM subprocess, fed to the error classifier.
    # vLLM v1 multi-process (APIServer + EngineCore) prints two tracebacks
    # totalling 200+ lines per failure; too small a maxlen pushes the root
    # cause (e.g. OOM) out of the buffer.
    stderr_recent_lines: deque[str] = field(default_factory=lambda: deque(maxlen=500))


class VllmEngine(BaseEngine):
    """
    vLLM OpenAI-compatible engine.

    Lifecycle policy:
    - `load_model`: launches `vllm serve` from the backend process and waits for
      the `/v1/models` health check to succeed.
    - `unload`: stops the vLLM server process this engine started and releases
      local resources.
    """

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ) -> None:
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.config: InferenceConfig | None = None
        # All mutable state tied to the vLLM server lifecycle lives in runtime.
        # Defaults are seeded here; load_model overwrites base_url / api_key /
        # health_timeout_s.
        self.runtime: VllmRuntimeContext = VllmRuntimeContext(
            api_key=VLLM_OPENAI_API_KEY,
            health_timeout_s=VLLM_HEALTH_TIMEOUT,
        )

    def _log_context(self, *, stream_label: str) -> dict[str, Any]:
        """
        Build the structured fields passed as logger ``extra``.

        Every field tolerates None / absence, so external systems such as ELK
        only index keys that actually exist.
        """
        ctx: dict[str, Any] = {"stream": stream_label}
        proc = self.runtime.server_process
        if proc is not None:
            ctx["vllm_pid"] = proc.pid
        try:
            ctx["vllm_port"] = self._resolve_vllm_port()
        except Exception:
            # _resolve_vllm_port raises on misconfiguration; logging must not blow up
            pass
        if self.runtime.served_model_name:
            ctx["model_name"] = self.runtime.served_model_name
        elif self.config is not None:
            ctx["model_name"] = self.config.model_name
        return ctx

    def _pump_server_logs(self, stream: "IO[str] | None", is_stderr: bool = False) -> None:
        """Forward vLLM subprocess output to this service's logger with structured fields."""
        if stream is None:
            return
        stream_label = "stderr" if is_stderr else "stdout"
        try:
            for line in iter(stream.readline, ""):
                txt = (line or "").rstrip()
                if not txt:
                    continue

                extra = self._log_context(stream_label=stream_label)
                if is_stderr:
                    with self.runtime.stderr_buffer_lock:
                        self.runtime.stderr_recent_lines.append(txt)
                    logger.warning("[vLLM][stderr] %s", txt, extra=extra)
                else:
                    logger.info("[vLLM][stdout] %s", txt, extra=extra)
        except Exception as e:
            logger.debug(
                "[vLLM] log pump stopped: %s",
                e,
                extra=self._log_context(stream_label=stream_label),
            )
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _build_base_url(self) -> str:
        return f"http://{VLLM_CLIENT_HOST}:{self._resolve_vllm_port()}/v1"

    def _resolve_vllm_port(self) -> int:
        port = VLLM_PORT
        if not (1 <= port <= 65535):
            raise RuntimeError(f"VLLM_PORT out of range: {port}")
        return port

    def _is_vllm_request_logging_enabled(self) -> bool:
        return VLLM_ENABLE_LOG_REQUESTS

    def _recreate_client(self) -> None:
        self.runtime.client = OpenAI(base_url=self.runtime.base_url, api_key=self.runtime.api_key)

    def _find_port_owner(self, port: int) -> psutil.Process | None:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr and conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None

    def _cleanup_port(self, port: int) -> None:
        owner = self._find_port_owner(port)
        if owner is None:
            return

        try:
            cmdline = " ".join(owner.cmdline() or []).lower()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            cmdline = ""
        if "vllm" not in cmdline or "serve" not in cmdline:
            raise RuntimeError(
                f"Port {port} is occupied by non-vLLM-serve process (PID={owner.pid}, name={owner.name()})."
            )

        logger.warning("[Worker] Cleaning stale vLLM process on port=%s pid=%s", port, owner.pid)
        try:
            pgid = os.getpgid(owner.pid)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
            os.killpg(pgid, signal.SIGKILL)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
        except (ProcessLookupError, PermissionError, OSError):
            try:
                owner.kill()
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
                OSError,
            ):
                pass
        time.sleep(1.0)

    def _detect_multimodal_support(self, config: InferenceConfig) -> bool:
        """
        Detect offline whether the model has a multimodal (vision/audio) architecture.

        Reads config.json from the model folder and checks:
        1. whether ``architectures`` contains a multimodal naming pattern
           (``ForConditionalGeneration`` etc.)
        2. whether a ``vision_config`` sub-object exists (strong signal)

        Returns False whenever detection fails or config.json is missing — a
        conservative default that avoids adding meaningless flags to text-only
        models.
        """
        model_source = config.model_path or config.model_name
        if not model_source:
            return False

        # Local paths only; an HF repo id cannot be read offline
        config_path = os.path.join(model_source, "config.json")
        if not os.path.isfile(config_path):
            logger.debug(
                "[Worker] Multimodal auto-detect skipped: config.json not found at %s",
                config_path,
            )
            return False

        try:
            with open(config_path, encoding="utf-8") as f:
                hf_cfg = json.load(f)
        except (OSError, ValueError) as e:
            logger.debug("[Worker] Multimodal auto-detect failed to read config: %s", e)
            return False

        # Signal 1: vision_config sub-object (strongest signal)
        if isinstance(hf_cfg.get("vision_config"), dict):
            logger.debug("[Worker] Multimodal detected via vision_config field")
            return True

        # Signal 2: architectures contains a known multimodal naming pattern
        mm_patterns = (
            "ForConditionalGeneration",
            "ChatModel",
            "VLForCausalLM",
            "VLMForCausalLM",
            "Phi3V",
            "Molmo",
            "Ovis",
            "MultiModalLM",
        )
        archs = hf_cfg.get("architectures") or []
        if isinstance(archs, list):
            for arch in archs:
                if isinstance(arch, str) and any(p in arch for p in mm_patterns):
                    logger.debug(
                        "[Worker] Multimodal detected via architecture pattern: %s",
                        arch,
                    )
                    return True

        return False

    def _core_args(self, config: InferenceConfig) -> list[str]:
        """
        Core required flags: model / address / dtype / memory budget / max len /
        served name / TP.

        These are the fixed CLI arguments passed on every vLLM server start.
        """
        model_source = config.model_path or config.model_name
        max_model_len = config.vllm_max_model_len or config.n_ctx
        served_model_name = VLLM_SERVED_MODEL_NAME or model_source
        return [
            "--model",
            model_source,
            "--host",
            VLLM_SERVER_HOST,
            "--port",
            str(self._resolve_vllm_port()),
            "--dtype",
            config.vllm_dtype,
            "--gpu-memory-utilization",
            str(config.vllm_gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--cpu-offload-gb",
            str(config.vllm_cpu_offload_gb),
            "--served-model-name",
            served_model_name,
            "--tensor-parallel-size",
            str(config.vllm_tensor_parallel_size),
        ]

    def _perf_args(self, config: InferenceConfig) -> list[str]:
        """Conditional performance/behaviour flags: quantization, KV cache, prefix cache, logging."""
        args: list[str] = []
        if config.vllm_quantization:
            args.extend(["--quantization", config.vllm_quantization])
        if config.vllm_kv_cache_dtype:
            args.extend(["--kv-cache-dtype", config.vllm_kv_cache_dtype])
        if config.vllm_kv_offloading_size is not None and config.vllm_kv_offloading_size > 0:
            args.extend(
                [
                    "--kv-offloading-size",
                    str(config.vllm_kv_offloading_size),
                    "--disable-hybrid-kv-cache-manager",
                ]
            )
        if config.vllm_max_num_seqs is not None:
            args.extend(["--max-num-seqs", str(config.vllm_max_num_seqs)])

        if config.vllm_max_num_batched_tokens is not None:
            args.extend(["--max-num-batched-tokens", str(config.vllm_max_num_batched_tokens)])
        if config.vllm_enforce_eager:
            args.append("--enforce-eager")
        if config.trust_remote_code:
            args.append("--trust-remote-code")
        if not self._is_vllm_request_logging_enabled():
            args.append("--no-enable-log-requests")
        return args

    def _mm_args(self, config: InferenceConfig) -> list[str]:
        """
        Multimodal flags: the image/audio/video budgets of ``--limit-mm-per-prompt``.

        If the user set no explicit image limit and the model is detected as
        multimodal, image=1 is added automatically; vLLM ignores the flag on
        text-only models, so the risk is contained.
        """
        # getattr keeps compatibility with older InferenceConfig versions that
        # may not define these fields
        mm_image_limit = getattr(config, "vllm_mm_image_limit", None)
        mm_audio_limit = getattr(config, "vllm_mm_audio_limit", None)
        mm_video_limit = getattr(config, "vllm_mm_video_limit", None)

        if mm_image_limit is None and self._detect_multimodal_support(config):
            logger.info(
                "[Worker] Detected multimodal model; "
                "auto-enabling --limit-mm-per-prompt image=1 "
                "(override via InferenceConfig.vllm_mm_image_limit)"
            )
            mm_image_limit = 1

        mm_limits: dict[str, int] = {}
        if mm_image_limit is not None:
            mm_limits["image"] = int(mm_image_limit)
        if mm_audio_limit is not None:
            mm_limits["audio"] = int(mm_audio_limit)
        if mm_video_limit is not None:
            mm_limits["video"] = int(mm_video_limit)

        if not mm_limits:
            return []
        return ["--limit-mm-per-prompt", json.dumps(mm_limits)]

    def _template_args(self, config: InferenceConfig) -> list[str]:
        """
        architectures override and custom chat template.

        ``vllm_hf_overrides`` force-overrides the architectures in ``config.json``,
        e.g. ``'{"architectures":["Gemma4ForConditionalGeneration"]}'``.
        ``vllm_chat_template`` is for when ``tokenizer_config.json`` has no
        chat_template.
        """
        args: list[str] = []

        hf_overrides = getattr(config, "vllm_hf_overrides", None)
        if hf_overrides:
            if isinstance(hf_overrides, (dict, list)):
                hf_overrides = json.dumps(hf_overrides)
            args.extend(["--hf-overrides", str(hf_overrides)])

        chat_template = getattr(config, "vllm_chat_template", None)
        if chat_template:
            args.extend(["--chat-template", str(chat_template)])

        return args

    def _build_server_cmd(self, config: InferenceConfig) -> list[str]:
        """
        Assemble the full ``vllm serve <args...>`` command.

        Each group (core/perf/mm/template) has its own method to ease unit testing.
        """
        cmd: list[str] = ["vllm", "serve"]
        cmd.extend(self._core_args(config))
        cmd.extend(self._perf_args(config))
        cmd.extend(self._mm_args(config))
        cmd.extend(self._template_args(config))
        return cmd

    def _resolve_vllm_server_dir(self) -> str:
        """Resolve the project directory of the isolated vllm_server environment."""
        project_dir = VLLM_SERVER_PROJECT_DIR
        if not os.path.isdir(project_dir):
            raise RuntimeError(f"VLLM_SERVER_PROJECT_DIR does not exist: {project_dir}")
        return project_dir

    def _resolve_launch_prefixes(self) -> list[list[str]]:
        """
        Build the list of usable vllm launcher prefixes (without ``serve <args>``).

        Return order is the preference order: the venv's
        ``python -m vllm.entrypoints.cli.main`` >
        system ``uv run --project ... python -m vllm.entrypoints.cli.main``.

        The ``.venv/bin/vllm`` console script is deliberately not executed, so
        that containers mounting the venv at a different absolute path still
        work: that file's shebang hardcodes the interpreter path from venv
        creation time, and a moved path easily yields ``ENOENT`` at subprocess
        spawn.

        If neither is available this raises ``RuntimeError`` so the caller sees
        the failure at the very start of ``load_model`` instead of panicking
        when the subprocess is actually spawned.
        """
        project_dir = self._resolve_vllm_server_dir()

        prefixes: list[list[str]] = []

        # Try both "python" and "python3" — uv venv may only create one depending
        # on the host system. Using the venv binary directly avoids uv project
        # discovery walking up the tree and picking up the backend's pyproject.toml.
        venv_bin = os.path.join(project_dir, ".venv", "bin")
        found_python_bin: str | None = None
        for py_name in ("python", "python3"):
            candidate = os.path.join(venv_bin, py_name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                found_python_bin = candidate
                prefixes.append([candidate, "-m", "vllm.entrypoints.cli.main"])
                break

        uv_path = shutil.which("uv")
        if uv_path:
            # Use --no-project so uv does not traverse up the directory tree and
            # accidentally pick up the backend's pyproject.toml (now at repo root).
            # Explicitly pass --python to pin to the isolated venv's interpreter.
            uv_cmd = [uv_path, "run", "--no-project"]
            if found_python_bin:
                uv_cmd += ["--python", found_python_bin]
            uv_cmd += ["python", "-m", "vllm.entrypoints.cli.main"]
            prefixes.append(uv_cmd)

        if not prefixes:
            raise RuntimeError(
                f"vLLM launcher unavailable: no executable at "
                f"{venv_bin} and `uv` not found on PATH. "
                "Please set up the vllm_server isolated environment first."
            )
        return prefixes

    def _discover_runtime_library_dirs(self, project_venv: str) -> list[str]:
        """Find shared-library dirs in the isolated venv that must join ``LD_LIBRARY_PATH``."""
        discovered: list[str] = []
        seen: set[str] = set()
        search_roots = [
            os.path.join(project_venv, "lib"),
            os.path.join(project_venv, "lib64"),
        ]
        wanted_prefixes = (
            "libcudnn",
            "libcublas",
            "libcudart",
            "libcusolver",
            "libcusparse",
            "libnccl",
            "libnvJitLink",
            "libnvrtc",
        )

        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                if not filenames:
                    continue

                normalized_dir = dirpath.replace(os.sep, "/")
                has_cuda_libs = any(
                    ".so" in filename and filename.startswith(wanted_prefixes)
                    for filename in filenames
                )
                is_torch_lib = normalized_dir.endswith("/site-packages/torch/lib")
                is_nvidia_lib = (
                    "/site-packages/nvidia/" in normalized_dir and normalized_dir.endswith("/lib")
                )

                if not (has_cuda_libs or is_torch_lib or is_nvidia_lib):
                    continue

                abs_dir = os.path.abspath(dirpath)
                if abs_dir in seen:
                    continue
                seen.add(abs_dir)
                discovered.append(abs_dir)

        return discovered

    def _find_shared_library(self, project_venv: str, pattern: str) -> str | None:
        """Locate a given shared library inside the isolated venv."""
        for lib_root_name in ("lib", "lib64"):
            lib_root = os.path.join(project_venv, lib_root_name)
            if not os.path.isdir(lib_root):
                continue
            for candidate in glob.iglob(os.path.join(lib_root, "**", pattern), recursive=True):
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
        return None

    def _repair_isolated_nvidia_runtime(self, project_dir: str) -> None:
        """Repair NVIDIA runtime wheels that were not fully unpacked in the isolated vLLM env."""
        project_venv = os.path.join(project_dir, ".venv")
        missing_pairs = [
            (lib_name, package_name)
            for lib_name, package_name in _REQUIRED_NVIDIA_RUNTIME_LIBS
            if not self._find_shared_library(project_venv, lib_name)
        ]
        if not missing_pairs:
            return

        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError(
                "vLLM isolated env is missing required NVIDIA runtime libraries and `uv` is unavailable for repair."
            )

        python_bin = os.path.join(project_venv, "bin", "python")
        if not (os.path.isfile(python_bin) and os.access(python_bin, os.X_OK)):
            raise RuntimeError(f"vLLM isolated env missing python interpreter: {python_bin}")

        repair_env = os.environ.copy()
        active_venv = repair_env.get("VIRTUAL_ENV")
        if active_venv and os.path.abspath(active_venv) != os.path.abspath(project_venv):
            repair_env.pop("VIRTUAL_ENV", None)

        missing_packages = list(dict.fromkeys(package for _, package in missing_pairs))

        cmd = [
            uv_path,
            "pip",
            "install",
            "--python",
            python_bin,
        ]
        for package_name in missing_packages:
            cmd.extend(["--reinstall-package", package_name])
        cmd.extend(missing_packages)
        logger.warning(
            "[Worker] Missing NVIDIA runtime libraries %s in isolated vLLM env; repairing with: %s",
            ", ".join(lib_name for lib_name, _ in missing_pairs),
            " ".join(cmd),
        )
        try:
            completed = subprocess.run(
                cmd,
                env=repair_env,
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            stdout_excerpt = (completed.stdout or "").strip()
            if stdout_excerpt:
                logger.info(
                    "[Worker] NVIDIA runtime repair stdout: %s",
                    stdout_excerpt[-1000:],
                )
        except subprocess.CalledProcessError as exc:
            stderr_excerpt = ((exc.stderr or "") or (exc.stdout or "")).strip()
            raise RuntimeError(
                "Failed to repair vLLM NVIDIA runtime via uv pip; "
                f"stderr_excerpt={stderr_excerpt[-1000:]}"
            ) from exc

        unresolved = [
            lib_name
            for lib_name, _ in _REQUIRED_NVIDIA_RUNTIME_LIBS
            if not self._find_shared_library(project_venv, lib_name)
        ]
        if unresolved:
            raise RuntimeError(
                "vLLM isolated env still missing NVIDIA runtime libraries after uv pip repair: "
                + ", ".join(unresolved)
            )

        logger.info(
            "[Worker] NVIDIA runtime repair complete for libs: %s",
            ", ".join(lib_name for lib_name, _ in missing_pairs),
        )

    def _build_server_env(self) -> dict[str, str]:
        """Build the environment variables for the vLLM subprocess."""
        env = os.environ.copy()
        env.setdefault("VLLM_LOGGING_LEVEL", VLLM_LOGGING_LEVEL)
        env["HF_HUB_OFFLINE"] = "1"

        project_venv = os.path.join(self._resolve_vllm_server_dir(), ".venv")
        active_venv = env.get("VIRTUAL_ENV")
        if active_venv and os.path.abspath(active_venv) != os.path.abspath(project_venv):
            logger.info(
                "[Worker] Clearing inherited VIRTUAL_ENV=%s for isolated vLLM env %s",
                active_venv,
                project_venv,
            )
            env.pop("VIRTUAL_ENV", None)

        extra_lib_dirs = self._discover_runtime_library_dirs(project_venv)
        if extra_lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                extra_lib_dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
            )

        env["PATH"] = os.pathsep.join([os.path.join(project_venv, "bin"), env.get("PATH", "")])
        return env

    def _validate_runtime_environment(self) -> None:
        """
        Full environment check before load_model proceeds, to surface actionable errors early.

        Checks:
        1. ``VLLM_SERVER_PROJECT_DIR`` is set and points at an existing directory.
          2. At least one launcher is available (venv
              ``python -m vllm.entrypoints.cli.main`` or system uv).
        3. ``VLLM_PORT`` is within 1-65535.

        Failures raise directly; the caller's ``load_model`` try/except reports
        the error to the main process via status_queue.
        """
        # 1+2: resolve project dir and launcher prefixes (both raise on failure)
        project_dir = self._resolve_vllm_server_dir()
        self._resolve_launch_prefixes()
        self._repair_isolated_nvidia_runtime(project_dir)
        # 3: port range
        self._resolve_vllm_port()

    def _start_server_process(self, config: InferenceConfig) -> None:
        port = self._resolve_vllm_port()
        self._cleanup_port(port)

        # Key step: this is where load_model actually starts the vLLM
        # OpenAI-compatible server.
        raw_cmd = self._build_server_cmd(config)
        # raw_cmd has the form ["vllm", "serve", ...]; drop the leading "vllm"
        # when combining with a launcher, because each prefix already goes up to
        # the vLLM module (venv `python -m vllm.entrypoints.cli.main` or
        # `uv run ... python -m vllm.entrypoints.cli.main`).
        serve_args = raw_cmd[1:] if raw_cmd and raw_cmd[0] == "vllm" else raw_cmd

        launch_candidates: list[list[str]] = [
            [*prefix, *serve_args] for prefix in self._resolve_launch_prefixes()
        ]

        last_error: Exception | None = None
        env = self._build_server_env()
        with self.runtime.stderr_buffer_lock:
            self.runtime.stderr_recent_lines.clear()

        for cmd in launch_candidates:
            try:
                logger.info(
                    "[Worker] Starting vLLM server (isolated env): %s",
                    " ".join(cmd),
                )
                self.runtime.server_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )

                self.runtime.stdout_pump_thread = threading.Thread(
                    target=self._pump_server_logs,
                    args=(self.runtime.server_process.stdout, False),
                    daemon=True,
                )
                self.runtime.stderr_pump_thread = threading.Thread(
                    target=self._pump_server_logs,
                    args=(self.runtime.server_process.stderr, True),
                    daemon=True,
                )
                self.runtime.stdout_pump_thread.start()
                self.runtime.stderr_pump_thread.start()
                return
            except FileNotFoundError as e:
                last_error = e
                continue

        raise RuntimeError(f"Failed to start vLLM server. {last_error}")

    def _stop_server_process(self) -> None:
        if self.runtime.server_process and self.runtime.server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.runtime.server_process.pid), signal.SIGINT)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
                self.runtime.server_process.wait(timeout=10)
            except (
                ProcessLookupError,
                PermissionError,
                OSError,
                subprocess.TimeoutExpired,
            ):
                try:
                    os.killpg(os.getpgid(self.runtime.server_process.pid), signal.SIGTERM)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
                    self.runtime.server_process.wait(timeout=5)
                except (
                    ProcessLookupError,
                    PermissionError,
                    OSError,
                    subprocess.TimeoutExpired,
                ):
                    try:
                        os.killpg(os.getpgid(self.runtime.server_process.pid), signal.SIGKILL)  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; vLLM runs on Linux
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

        self.runtime.server_process = None

    def _wait_for_port_release(
        self, port: int, timeout: float = 15.0, poll_interval: float = 0.5
    ) -> bool:
        """
        Poll until the given port has no owner, or until timeout.

        Hot-swap case: when ``unload`` is immediately followed by ``load_model``,
        the OS may still hold a socket in TIME_WAIT after the old vLLM process
        was SIGKILLed, or GPU memory may not be flushed yet, so the next
        ``_cleanup_port`` hits leftovers. This method provides an explicit
        synchronization point so unload only returns once the port is truly free.

        Returns:
            ``True`` if released within the timeout; ``False`` if still occupied.
        """
        deadline = time.time() + max(0.0, timeout)
        last_owner_pid: int | None = None
        while time.time() < deadline:
            owner = self._find_port_owner(port)
            if owner is None:
                return True
            try:
                last_owner_pid = owner.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                last_owner_pid = None
            time.sleep(poll_interval)

        logger.warning(
            "[Worker] vLLM port %s still occupied after %.1fs (owner_pid=%s); "
            "next load_model will attempt cleanup.",
            port,
            timeout,
            last_owner_pid,
            extra=self._log_context(stream_label="lifecycle"),
        )
        return False

    def _drain_stderr_pump(self, timeout: float = 2.0) -> None:
        """
        Wait for the stderr pump thread to drain the buffered pipe contents.

        The kernel marks the pipe EOF the moment the subprocess dies, but the
        reader thread has not yet read every leftover line; calling
        ``_classify_recent_stderr`` right away would see an incomplete buffer
        (with vLLM v1 typically only the APIServer wrapper's last two lines,
        dropping the real OOM root cause). Call this helper after detecting
        subprocess death to give the reader thread a moment to finish.
        """
        thread = self.runtime.stderr_pump_thread
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout)

    def _classify_recent_stderr(self) -> VllmErrorReport:
        """Read the recent stderr and hand it to the error classifier."""
        with self.runtime.stderr_buffer_lock:
            lines = list(self.runtime.stderr_recent_lines)
        return classify_stderr(lines)

    def _get_error_reason(self) -> str:
        """Return the plain-text error summary for RuntimeError (backward-compatible API)."""
        return self._classify_recent_stderr().to_text()

    def _normalize_messages(self, prompt: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - arbitrary client-supplied prompt payload
        """
        Convert a frontend prompt into OpenAI chat messages format.

        content may be:
        - str: plain text
        - list[dict]: multimodal multi-part, e.g.
            [{"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:..."}}]
        The list structure must be preserved verbatim — coercing with str()
        would drop the images.
        """
        if isinstance(prompt, list):
            normalized: list[dict[str, Any]] = []
            for msg in prompt:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")

                # If content is a list (multimodal multi-part), keep it as is
                if isinstance(content, list):
                    normalized.append({"role": role, "content": content})
                elif isinstance(content, dict):
                    # Some clients send a single dict
                    normalized.append({"role": role, "content": [content]})
                else:
                    normalized.append({"role": role, "content": str(content)})

            if normalized:
                return normalized

        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": str(prompt)}]

    def _check_server_alive_or_raise(self) -> None:
        """
        Raise a classified RuntimeError immediately if the vLLM subprocess died.

        Called before each health-check poll iteration, so a crashed server is
        reported at once instead of spinning until ``health_timeout_s`` expires.
        """
        proc = self.runtime.server_process
        if proc is None:
            return
        if proc.poll() is None:
            return

        # Key step: let the reader thread finish the leftover pipe contents
        # (including the EngineCore traceback) first, otherwise
        # _classify_recent_stderr only sees the APIServer wrapper lines.
        self._drain_stderr_pump(timeout=2.0)

        exit_code = proc.returncode
        report = self._classify_recent_stderr()
        raise RuntimeError(
            "vLLM process exited during startup "
            f"(code={exit_code}, category={report.category}); "
            f"stderr_excerpt={report.to_text()}"
        )

    def _resolve_served_model_name(self, config: InferenceConfig) -> str:
        preferred = (
            VLLM_SERVED_MODEL_NAME
            or (config.model_path if config.model_path else None)
            or config.model_name
        )

        start = time.time()
        last_error: str | None = None

        while time.time() - start < self.runtime.health_timeout_s:
            # Check for subprocess death at the start of every poll (including
            # the iteration after a sleep)
            self._check_server_alive_or_raise()

            try:
                # Key step: use the OpenAI-compatible `models.list()` to verify
                # the vLLM server is reachable.
                if self.runtime.client is None:
                    raise RuntimeError("OpenAI client not initialized")
                models_response = self.runtime.client.models.list(timeout=5.0)
                model_ids = [
                    item.id
                    for item in getattr(models_response, "data", [])
                    if getattr(item, "id", None)
                ]

                if preferred in model_ids:
                    return str(preferred)
                if model_ids:
                    return str(model_ids[0])
                return str(preferred)
            except (APIError, OSError, ValueError, TypeError) as e:
                last_error = str(e)
                time.sleep(1.0)

        # After the timeout, check the process state one last time and attach the
        # classified stderr summary
        self._check_server_alive_or_raise()
        report = self._classify_recent_stderr()
        url = f"{self.runtime.base_url}/models"
        details_parts: list[str] = []
        if last_error:
            details_parts.append(f"last_error={last_error}")
        details_parts.append(f"category={report.category}")
        details_parts.append(f"stderr_excerpt={report.to_text()}")
        details = "; " + "; ".join(details_parts)
        raise RuntimeError(
            f"vLLM OpenAI-compatible server startup timeout: "
            f"{self.runtime.health_timeout_s:.0f}s, endpoint={url}{details}"
        )

    def load_model(self, config: InferenceConfig) -> None:
        """Validate environment, launch the vLLM server, and await readiness."""
        self.config = config
        self.runtime.api_key = VLLM_OPENAI_API_KEY
        self.runtime.health_timeout_s = VLLM_HEALTH_TIMEOUT
        try:
            # Check the environment early: raise straight away when no launcher
            # is available or the port is misconfigured, instead of wasting time
            # on client creation, subprocess start and health polling.
            self.status_queue.put({"status": "loading", "stage": "vllm_env_validate"})
            self._validate_runtime_environment()

            self.status_queue.put({"status": "loading", "stage": "vllm_server_start"})
            self.runtime.base_url = self._build_base_url()
            self._recreate_client()

            self._start_server_process(config)

            self.status_queue.put({"status": "loading", "stage": "vllm_server_connect"})
            self.runtime.served_model_name = self._resolve_served_model_name(config)

            self.status_queue.put(
                {
                    "status": "ready",
                    "message": "vLLM OpenAI-compatible server started and connected",
                    "device": "vllm-local-server",
                    "device_map_summary": f"endpoint: {self.runtime.base_url}",
                    "total_modules": None,
                    "layer_lines": [],
                    "memory_usage": None,
                }
            )
            logger.info(
                "[Worker] vLLM server ready. endpoint=%s model=%s",
                self.runtime.base_url,
                self.runtime.served_model_name,
            )
        except Exception as e:
            startup_excerpt = self._get_error_reason()
            logger.exception("[Worker] vLLM startup failed: %s", e)
            logger.exception("[Worker] vLLM stderr excerpt:\n%s", startup_excerpt)
            self._stop_server_process()
            self.runtime.served_model_name = None
            raise RuntimeError(f"vLLM startup failed: {e}; stderr_excerpt={startup_excerpt}") from e

    def _normalize_prompt(self, prompt: Any, params: dict[str, Any] | None = None) -> str:  # noqa: ANN401 - arbitrary client-supplied prompt payload
        _ = params
        messages = self._normalize_messages(prompt)

        def _content_to_text(content: Any) -> str:  # noqa: ANN401 - OpenAI message content is str or multi-part list

            # Multimodal multi-part case: keep only text, represent image/audio
            # with a placeholder
            if isinstance(content, list):
                parts = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "text":
                        parts.append(str(p.get("text", "")))
                    elif ptype == "image_url":
                        parts.append("[image]")
                    elif ptype in ("audio_url", "input_audio"):
                        parts.append("[audio]")
                    elif ptype == "video_url":
                        parts.append("[video]")
                return " ".join(parts)
            return str(content)

        return "\n".join(f"{m['role']}: {_content_to_text(m['content'])}" for m in messages)

    # NOTE: generate() / generate_stream() intentionally removed — vLLM is
    # served directly over async HTTP from the main process (see
    # service/inference/async_server_client.py), never through the worker.
    # The BaseEngine defaults emit a clear error if this is ever misrouted.

    def unload(self) -> None:
        """Stop the managed vLLM server and reset all runtime state."""
        # Key step: unload stops the vLLM server this engine started.
        # A failed port resolution (misconfigured env var) must not block unload,
        # hence the try.
        try:
            port = self._resolve_vllm_port()
        except Exception:
            port = None

        self._stop_server_process()

        # On hot swap, load_model follows unload immediately; make sure the old
        # server really released the port so the next _cleanup_port does not hit
        # TIME_WAIT or a leftover process.
        port_released = True
        if port is not None:
            port_released = self._wait_for_port_release(port)

        # Single wholesale reset: every vLLM lifecycle field goes back to zero,
        # so none is missed (the previous manual 5-line reset risked drifting).
        self.runtime = VllmRuntimeContext(
            api_key=VLLM_OPENAI_API_KEY,
            health_timeout_s=VLLM_HEALTH_TIMEOUT,
        )
        self.config = None

        unload_msg = "vLLM engine unloaded and server stopped"
        if not port_released:
            unload_msg += f" (warning: port {port} not yet released)"

        self.status_queue.put(
            {
                "status": "unloaded",
                "message": unload_msg,
                "memory_usage": None,
            }
        )
        logger.info("[Worker] %s", unload_msg)

    def apply_chat_template(self, request: dict[str, Any]) -> None:
        """Flatten chat messages into a plain-text prompt for logging/debug."""
        request_id = request.get("request_id")
        try:
            messages = request.get("messages", [])
            prompt = self._normalize_prompt(messages, request.get("template_kwargs", {}))
            self.data_queue.put(
                {
                    "type": "result",
                    "request_id": request_id,
                    "result": prompt,
                }
            )
        except (TypeError, ValueError) as e:
            self.data_queue.put(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": str(e),
                }
            )

    def _vllm_admin_url(self, path: str) -> str:
        """Build a vLLM server admin endpoint URL (without the /v1 prefix)."""
        # base_url looks like http://host:port/v1; admin endpoints sit under root.
        base = self.runtime.base_url
        root = base[:-3] if base.endswith("/v1") else base
        if not path.startswith("/"):
            path = "/" + path
        return f"{root.rstrip('/')}{path}"

    def cleanup_generation_memory(self) -> None:
        """
        Ask the vLLM server to reset its prefix cache and free accumulated KV memory.

        The vLLM OpenAI-compatible server exposes the ``POST /reset_prefix_cache``
        admin endpoint, which clears prefix cache entries (in-flight requests are
        unaffected). If the server is not up yet or the request fails, log a
        warning instead of propagating, so cleanup cannot take the service down.
        """
        # Report directly when the vLLM server is not ready (avoids a pointless HTTP call)
        if not self.runtime.base_url or self.runtime.client is None:
            self.data_queue.put(
                {
                    "type": "cleanup",
                    "result": "vLLM cleanup skipped: server not loaded",
                }
            )
            return

        url = self._vllm_admin_url("/reset_prefix_cache")
        headers = (
            {"Authorization": f"Bearer {self.runtime.api_key}"} if self.runtime.api_key else {}
        )

        try:
            with httpx.Client(timeout=5.0) as http:
                response = http.post(url, headers=headers)
                response.raise_for_status()
            logger.info("[Worker] vLLM reset_prefix_cache OK (%s)", url)
            self.data_queue.put(
                {
                    "type": "cleanup",
                    "result": "vLLM prefix cache reset",
                }
            )
        except httpx.HTTPStatusError as e:
            # The endpoint may be disabled in some vLLM builds; not fatal
            status_code = e.response.status_code if e.response is not None else None
            logger.warning(
                "[Worker] vLLM reset_prefix_cache returned %s; cleanup degraded.",
                status_code,
            )
            self.data_queue.put(
                {
                    "type": "cleanup",
                    "result": (
                        f"vLLM cleanup degraded: HTTP {status_code} "
                        "(endpoint may be disabled in this vLLM build)"
                    ),
                }
            )
        except (httpx.HTTPError, OSError) as e:
            logger.warning("[Worker] vLLM reset_prefix_cache error: %s", e)
            self.data_queue.put(
                {
                    "type": "cleanup",
                    "result": f"vLLM cleanup failed: {e}",
                }
            )
