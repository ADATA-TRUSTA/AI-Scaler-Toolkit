"""
Strata engine: a managed Strata server, reached over its OpenAI-compatible API.

Strata (github.com/Niko1221/Strata) serves a MoE model with the experts in DRAM and the
hot ones cached on the GPU. Its own ``serve/server.py`` wraps the engine binary in an
OpenAI-compatible HTTP server. The worker starts that server and waits for it here;
generations then go from the main process straight to it (async_server_client), as for
llama-server and vLLM.

The model is described by the run config Strata's setup writes (``strata-<model>.json``:
the engine executable, the pack, the GGUF shards, the vision encoder). The engine copies
it into STRATA_RUNTIME_DIR with the load request's overrides applied and starts the
server with that copy, so the user's file is never changed.
"""

import json
import os
import re
import signal
import socket
import subprocess
import time
from collections import deque
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import IO, Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from ...config_models import InferenceConfig
from ...settings import (
    STRATA_ENGINE_BINARY,
    STRATA_HEALTH_TIMEOUT,
    STRATA_HOME,
    STRATA_PORT,
    STRATA_PYTHON,
    STRATA_RUNTIME_DIR,
    configure_logging,
)
from .base_engine import BaseEngine

logger = configure_logging(__name__)

STRATA_HOST = "127.0.0.1"
# Loading a 50 GiB expert arena through the page cache pinned it while the kernel kept
# reclaiming, and the session's memory pressure stayed above systemd-oomd's 50% long
# enough for it to kill the desktop session (measured 69.8%). Read unbuffered and without
# transparent huge pages (whose fault-time compaction alone drove 44.8%), the same load
# peaked at 0.2% and read faster (9.1-10.0 GiB/s against 6.4). Set only when the caller
# did not choose: either variable in the environment wins.
_LOAD_ENV_DEFAULTS = {"STRATA_UNBUFFERED_LOAD": "1", "STRATA_NO_ARENA_THP": "1"}
_LOG_TAIL_LINES = 40


def strata_setup_problem() -> str | None:
    """Why engine=strata cannot start on this host, or None when it can."""
    if not STRATA_HOME.strip():
        return (
            "Strata is not set up: STRATA_HOME is empty. Point it at a Strata checkout "
            "prepared by Strata's own setup (the directory holding serve/server.py), and "
            "STRATA_PYTHON at its interpreter."
        )
    server = Path(STRATA_HOME).expanduser() / "serve" / "server.py"
    if not server.is_file():
        return f"STRATA_HOME has no serve/server.py: {server}"
    python = strata_python()
    if not Path(python).is_file():
        return f"The Strata interpreter does not exist: {python} (STRATA_PYTHON)"
    # The run config names the engine Strata's setup built, which has no deep sleep; this
    # backend always starts with --sleep-mode, so it runs only the build named here.
    if not STRATA_ENGINE_BINARY.strip():
        return (
            "STRATA_ENGINE_BINARY is empty: point it at the Strata engine built with deep "
            "sleep (--sleep-mode)."
        )
    if not Path(STRATA_ENGINE_BINARY).expanduser().is_file():
        return f"STRATA_ENGINE_BINARY does not exist: {STRATA_ENGINE_BINARY}"
    return None


def strata_python() -> str:
    """STRATA_PYTHON, else the checkout's own .venv interpreter."""
    if STRATA_PYTHON.strip():
        return str(Path(STRATA_PYTHON).expanduser())
    venv = Path(STRATA_HOME).expanduser() / ".venv"
    return str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def _set_flag(args: list[str], flag: str, value: str) -> list[str]:
    """``args`` with ``flag value`` replacing every earlier occurrence (or appended)."""
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        if args[i].startswith(flag + "="):
            i += 1
            continue
        out.append(args[i])
        i += 1
    return [*out, flag, value]


def _same_file(a: str, b: str) -> bool:
    try:
        # resolve(): a Hugging Face snapshot path and the blob it links to are the same file
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    except OSError:
        return False


def find_run_config(gguf_path: str) -> Path | None:
    """
    The run config in STRATA_HOME whose model is ``gguf_path``, or None.

    Strata's setup writes one strata-<model>.json per installed model beside serve/, naming
    the model's first GGUF shard in --native. Matching on it lets a load name the model the
    way the model registry lists it, without knowing the run config exists.
    """
    home = Path(STRATA_HOME).expanduser()
    for candidate in sorted(home.glob("strata-*.json")):
        try:
            run = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        args = run.get("args") if isinstance(run, dict) else None
        if not isinstance(args, list) or "--native" not in args[:-1]:
            continue
        if _same_file(str(args[args.index("--native") + 1]), gguf_path):
            return candidate
    return None


def free_vram_mib(gpu: int) -> int:
    """
    Free VRAM on ``gpu`` in MiB, read through NVML or nvidia-smi.

    Not through torch: that would open a CUDA context in this worker and hold a few
    hundred MiB of the very VRAM being measured for as long as the worker lives.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(gpu))
            return int(info.free) // 2**20
        finally:
            pynvml.nvmlShutdown()
    except Exception as nvml_error:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            return int(out.strip().splitlines()[0])
        except Exception as e:
            raise RuntimeError(
                f"strata_vram_limit_gb needs the free VRAM of GPU {gpu}, and neither NVML "
                f"({nvml_error}) nor nvidia-smi ({e}) reported it; leave it unset to use the "
                "run config's --vram-reserve-mib"
            ) from e


def vram_reserve_for_limit(limit_gb: float, free_mib: int) -> int:
    """--vram-reserve-mib that holds the engine to ``limit_gb`` of what is free now."""
    # Strata fills what is free at start-up less the reserve, its vision encoder included:
    # an 8 GB limit measured 8.3 GB in use (engine 6.5 + vision 1.6), so the cap lands
    # within a few hundred MiB of what was asked.
    reserve = free_mib - int(limit_gb * 1024)
    if reserve < 0:
        logger.warning(
            "[Strata] strata_vram_limit_gb=%s but only %.1f GB of VRAM is free; using all of it",
            limit_gb,
            free_mib / 1024,
        )
    return max(reserve, 0)


def build_run_config(
    config: InferenceConfig,
    source: dict[str, Any],
    log_path: str,
    vram_reserve_mib: int | None = None,
) -> dict[str, Any]:
    """
    The run config the server is started with: the model's own, with this load's choices.

    Only the keys the load request owns change; everything else (the pack, the shards,
    the expert profile, the MTP head, KV format, sampling defaults) stays as Strata's setup
    wrote it. ``vram_reserve_mib`` is strata_vram_limit_gb already turned into a reserve.
    """
    run = dict(source)
    args = [str(a) for a in run.get("args") or []]
    if config.strata_max_context is not None:
        args = _set_flag(args, "--max-context", str(config.strata_max_context))
    if vram_reserve_mib is not None:
        args = _set_flag(args, "--vram-reserve-mib", str(vram_reserve_mib))
    args = [a for a in args if a != "--sleep-mode"]  # the run config's sleep_mode key decides
    run["args"] = args
    # Always on: it costs nothing until the first sleep, and freeing the GPU for another
    # program is what this engine is chosen for.
    run["sleep_mode"] = True
    run["exe"] = str(Path(STRATA_ENGINE_BINARY).expanduser())
    run["log"] = log_path
    run.pop("port", None)  # the server is told its port on the command line
    return run


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "model"


class StrataEngine(BaseEngine):
    """Starts Strata's server for the model and stops it again (worker side)."""

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ) -> None:
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.server_process: subprocess.Popen | None = None
        self.base_url = f"http://{STRATA_HOST}:{STRATA_PORT}"
        self.runtime_dir = Path(STRATA_RUNTIME_DIR)
        self.slot_dir = self.runtime_dir / "slots"
        self.server_log_path = self.runtime_dir / "strata-server.log"
        self._log_handle: IO[str] | None = None
        self._model_key = "model"

    # ---------------------- HTTP ----------------------
    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> tuple[int, dict[str, Any]]:
        """One call to the managed server -> (HTTP status, JSON body)."""
        data = json.dumps(payload or {}).encode("utf-8") if method.upper() == "POST" else None
        req = urllib_request.Request(  # noqa: S310 - the loopback server this engine started
            f"{self.base_url}{path}", data=data, method=method.upper()
        )
        req.add_header("Content-Type", "application/json")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - as above
                body = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            status = e.code
        try:
            return status, (json.loads(body) if body.strip() else {})
        except json.JSONDecodeError:
            return status, {"raw": body}

    def _health(self) -> dict[str, Any] | None:
        try:
            status, body = self._request("GET", "/health", timeout=3.0)
        except (urllib_error.URLError, OSError):
            return None
        return body if status == 200 else None

    # ---------------------- process ----------------------
    def _log_tail(self) -> str:
        try:
            with open(self.server_log_path, encoding="utf-8", errors="replace") as f:
                return "".join(deque(f, maxlen=_LOG_TAIL_LINES)).strip()
        except OSError:
            return ""

    def _start_server(self, run_config_path: Path) -> None:
        home = Path(STRATA_HOME).expanduser()
        # Ask whether anything is listening rather than trying to bind: a reload right after
        # an unload finds the last server's connections in TIME_WAIT, which fail a plain
        # bind() although the new server (bound with SO_REUSEADDR) starts fine on the port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            if probe.connect_ex((STRATA_HOST, STRATA_PORT)) == 0:
                raise RuntimeError(
                    f"STRATA_PORT {STRATA_PORT} is already in use on {STRATA_HOST}; stop what "
                    "holds it or set another STRATA_PORT"
                )
        cmd = [
            strata_python(),
            str(home / "serve" / "server.py"),
            "--engine",
            "strata",
            "--config",
            str(run_config_path),
            "--host",
            STRATA_HOST,
            "--port",
            str(STRATA_PORT),
            # an OpenAI client's max_tokens beyond the room left is shortened, as
            # llama-server does, instead of refused with a 400
            "--fit-max-tokens",
            "--slot-save-path",
            str(self.slot_dir),
        ]
        env = os.environ.copy()
        for key, value in _LOAD_ENV_DEFAULTS.items():
            env.setdefault(key, value)
        logger.info("[Strata] starting the server: %s", " ".join(cmd))
        self._log_handle = open(self.server_log_path, "a", encoding="utf-8")  # noqa: SIM115 - held for the process's life
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # its own process group: the server, the engine and the vision encoder end together
            kwargs["start_new_session"] = True
        self.server_process = subprocess.Popen(
            cmd,
            cwd=str(home),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            **kwargs,
        )

    def _wait_ready(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            proc = self.server_process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"the Strata server exited with code {proc.returncode} while loading. "
                    f"Its log ({self.server_log_path}) ends with:\n{self._log_tail()}"
                )
            health = self._health()
            if health and health.get("loaded"):
                return health
            time.sleep(1.0)
        raise RuntimeError(
            f"the Strata server did not report the model loaded within {timeout_sec:.0f}s "
            f"(STRATA_HEALTH_TIMEOUT). Its log ends with:\n{self._log_tail()}"
        )

    def _stop_server(self) -> None:
        proc, self.server_process = self.server_process, None
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError) as e:
                    logger.warning("[Strata] the server did not stop: %s", e)
            logger.info("[Strata] server stopped")
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    # ---------------------- slots ----------------------
    def _slot_file(self) -> str:
        return f"{self._model_key}.slot-0.bin"

    def _restore_slot(self) -> dict[str, Any]:
        if not (self.slot_dir / self._slot_file()).is_file():
            return {"restored": False, "reason": "no saved conversation"}
        status, body = self._request(
            "POST", "/slots/0?action=restore", {"filename": self._slot_file()}, timeout=600
        )
        if status != 200:
            logger.warning("[Strata] the saved conversation was not restored: %s", body)
            return {"restored": False, "reason": body.get("error", body)}
        logger.info("[Strata] restored the saved conversation: %s", body)
        return {"restored": True, "tokens": body.get("n_restored")}

    def _save_slot(self) -> None:
        try:
            status, state = self._request("GET", "/is_sleeping", timeout=5)
            if status == 200 and state.get("is_sleeping"):
                # A save would have to wake it, taking back VRAM another program holds
                logger.warning("[Strata] asleep at unload: the conversation is not saved")
                return
            status, body = self._request(
                "POST", "/slots/0?action=save", {"filename": self._slot_file()}, timeout=600
            )
        except (urllib_error.URLError, OSError) as e:
            logger.warning("[Strata] the conversation was not saved: %s", e)
            return
        if status == 200:
            logger.info("[Strata] saved the conversation: %s", body)
        else:
            logger.warning("[Strata] the conversation was not saved: %s", body)

    # ---------------------- lifecycle ----------------------
    def load_model(self, config: InferenceConfig) -> None:
        """Start Strata's server with the model's run config and wait until it is loaded."""
        self.config = config
        problem = strata_setup_problem()
        if problem:
            raise RuntimeError(problem)
        if config.strata_config_path:
            source_path = Path(config.strata_config_path).expanduser()
        elif config.model_path:
            found = find_run_config(config.model_path)
            if found is None:
                raise RuntimeError(
                    f"no Strata run config in {STRATA_HOME} is for {config.model_path}: install "
                    "the model with Strata's setup (scripts/linux/install_strata.sh), or name "
                    "its run config in strata_config_path"
                )
            source_path = found
        else:
            raise RuntimeError(
                "engine=strata needs model_path (the model's first GGUF shard, as the model "
                "registry lists it) or strata_config_path (the strata-<model>.json Strata's "
                "setup wrote for it)"
            )
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RuntimeError(f"cannot read the Strata run config {source_path}: {e}") from e
        if not isinstance(source, dict) or not source.get("exe") or not source.get("args"):
            raise RuntimeError(f"{source_path} is not a Strata run config (no exe / args)")

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        self._model_key = _safe_name(source_path.stem)
        self._stop_server()  # before the VRAM is measured: a stale server would hold it
        reserve = None
        if config.strata_vram_limit_gb is not None:
            gpu_key = source.get("gpu")
            gpu = gpu_key if isinstance(gpu_key, int) and not isinstance(gpu_key, bool) else 0
            reserve = vram_reserve_for_limit(config.strata_vram_limit_gb, free_vram_mib(gpu))
        run = build_run_config(
            config, source, str(self.runtime_dir / "strata-engine.log"), vram_reserve_mib=reserve
        )
        run_path = self.runtime_dir / f"{self._model_key}.run.json"
        run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")

        self._start_server(run_path)
        try:
            health = self._wait_ready(STRATA_HEALTH_TIMEOUT)
        except Exception:
            self._stop_server()
            raise
        slot = self._restore_slot()

        capabilities = ["chat"]
        if health.get("images"):
            capabilities += ["vision", "multimodal"]
        self.status_queue.put(
            {
                "status": "ready",
                "config": config.model_dump(),
                "device": "cuda",
                # read by the model card and /inference/status like llama-server's tags
                "llama_capabilities": capabilities,
                "served_context_length": health.get("max_context"),
                "slot_restore_summary": slot,
            }
        )
        logger.info(
            "[Strata] %s loaded (context %s, images %s)",
            health.get("model"),
            health.get("max_context"),
            bool(health.get("images")),
        )

    def unload(self) -> None:
        """Save the conversation, stop the server, and reset state."""
        if self.config is not None and self.server_process:
            self._save_slot()
        self._stop_server()
        self.config = None
        self.stop_generation_flag.clear()
        self.status_queue.put({"status": "unloaded", "message": "Strata engine unloaded"})
