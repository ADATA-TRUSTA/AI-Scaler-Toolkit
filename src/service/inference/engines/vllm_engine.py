import json
import os
import glob
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass
import psutil
import httpx
from openai import OpenAI, APIError

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
def sweep_stale_vllm_processes() -> Dict[str, Any]:
    """在服務啟動時清理殘留 vLLM serve 進程。

    透過環境變數控制：
    - VLLM_STARTUP_SWEEP: 1/true/yes/on 啟用（預設啟用）
    - VLLM_SWEEP_PORTS: 逗號分隔 port 清單（預設 5000）
    """

    enabled = VLLM_STARTUP_SWEEP
    if not enabled:
        return {"enabled": False, "ports": [], "killed": 0, "pids": []}

    ports = VLLM_SWEEP_PORTS

    if not ports:
        return {"enabled": True, "ports": [], "killed": 0, "pids": []}

    target_ports = set(ports)
    killed_pids: List[int] = []

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
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
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
    """彙整 vLLM engine 在「載入 → 推論 → 卸載」期間的所有可變狀態。

    把原本散落的 instance 屬性集中，

    1. ``unload`` 只需 ``self.runtime = VllmRuntimeContext()``
       就把所有狀態歸零，避免漏掉某個欄位繼續殘留（例如保留舊 ``client``）。
    2. 可獨立構造 ``VllmRuntimeContext`` 餵給某些 helper
       做單元測試，不必綁住完整 engine 與 multiprocessing queue。
    3. 所有與生命週期相關的欄位列在同一處。

    注意：``config`` （``InferenceConfig``）仍維持在 ``BaseEngine`` 上
    """

    base_url: str = ""
    api_key: str = ""
    health_timeout_s: float = 0.0
    served_model_name: Optional[str] = None
    current_request_id: Optional[str] = None

    client: Optional[OpenAI] = None
    server_process: Optional[subprocess.Popen] = None
    stdout_pump_thread: Optional[threading.Thread] = None
    stderr_pump_thread: Optional[threading.Thread] = None

    stderr_buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    # vLLM 子程序最近一段 stderr，用來給 ErrorClassifier 分類。
    # vLLM v1 multi-process（APIServer + EngineCore）一次失敗會打出
    # 兩段 traceback 共 200+ 行；maxlen 設過小會把根因（如 OOM）推出去。
    stderr_recent_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=500))


class VllmEngine(BaseEngine):
    """vLLM OpenAI-compatible engine.

    生命週期策略：
    - `load_model`: 由後端程序啟動 `vllm serve`，並等待 `/v1/models` 健康檢查成功。
    - `unload`: 關閉由本 engine 啟動的 vLLM server 進程並釋放本地資源。
    """

    def __init__(
        self,
        status_queue: Queue,
        data_queue: Queue,
        stop_event: EventClass,
        stop_generation_flag: EventClass,
    ):
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.config: Optional[InferenceConfig] = None
        # 所有與 vLLM server 生命週期相關的可變狀態統一收進 runtime。
        # 預設值在此填入：實際 load_model 會覆寫 base_url / api_key / health_timeout_s。
        self.runtime: VllmRuntimeContext = VllmRuntimeContext(
            api_key=VLLM_OPENAI_API_KEY,
            health_timeout_s=VLLM_HEALTH_TIMEOUT,
        )

    def _log_context(self, *, stream_label: str) -> Dict[str, Any]:
        """組出供 logger ``extra`` 使用的結構化欄位。

        所有欄位皆容忍 None / 缺漏，方便 ELK 等外部系統建索引時僅看到實際存在的鍵。
        """
        ctx: Dict[str, Any] = {"stream": stream_label}
        proc = self.runtime.server_process
        if proc is not None:
            ctx["vllm_pid"] = proc.pid
        try:
            ctx["vllm_port"] = self._resolve_vllm_port()
        except Exception:
            # _resolve_vllm_port 在設定錯誤時會 raise；log 用途下不應該因此爆炸
            pass
        if self.runtime.served_model_name:
            ctx["model_name"] = self.runtime.served_model_name
        elif self.config is not None:
            ctx["model_name"] = self.config.model_name
        return ctx

    def _pump_server_logs(self, stream, is_stderr: bool = False) -> None:
        """將 vLLM 子程序輸出轉發到當前 service logger，附帶結構化欄位。"""
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

    def _recreate_client(self):
        self.runtime.client = OpenAI(
            base_url=self.runtime.base_url, api_key=self.runtime.api_key
        )

    def _find_port_owner(self, port: int) -> Optional[psutil.Process]:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr and conn.laddr.port == port:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None

    def _cleanup_port(self, port: int):
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

        logger.warning(
            "[Worker] Cleaning stale vLLM process on port=%s pid=%s", port, owner.pid
        )
        try:
            pgid = os.getpgid(owner.pid)
            os.killpg(pgid, signal.SIGKILL)
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
        """離線判斷模型是否為多模態（視覺/音訊）架構。

        透過讀取模型資料夾的 config.json 檢查：
        1. architectures 欄位是否含多模態命名模式（ForConditionalGeneration 等）
        2. 是否存在 vision_config 子物件（強訊號）

        判斷失敗或找不到 config.json 時一律回傳 False（保守策略，
        避免在純文字模型上加上無意義的旗標）。
        """
        model_source = config.model_path or config.model_name
        if not model_source:
            return False

        # 僅支援本地路徑；HF repo id 離線無法讀取
        config_path = os.path.join(model_source, "config.json")
        if not os.path.isfile(config_path):
            logger.debug(
                "[Worker] Multimodal auto-detect skipped: config.json not found at %s",
                config_path,
            )
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                hf_cfg = json.load(f)
        except (OSError, ValueError) as e:
            logger.debug("[Worker] Multimodal auto-detect failed to read config: %s", e)
            return False

        # 訊號 1：vision_config 子物件（最強訊號）
        if isinstance(hf_cfg.get("vision_config"), dict):
            logger.debug("[Worker] Multimodal detected via vision_config field")
            return True

        # 訊號 2：architectures 含已知多模態命名模式
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

    def _core_args(self, config: InferenceConfig) -> List[str]:
        """核心必備旗標：模型/位址/dtype/記憶體配額/最大長度/served 名稱/TP。

        這些是每次啟動 vLLM server 都會送的固定 CLI 參數。
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

    def _perf_args(self, config: InferenceConfig) -> List[str]:
        """效能/行為相關的條件式旗標：量化、KV cache、prefix cache、log 行為。"""
        args: List[str] = []
        if config.vllm_quantization:
            args.extend(["--quantization", config.vllm_quantization])
        if config.vllm_kv_cache_dtype:
            args.extend(["--kv-cache-dtype", config.vllm_kv_cache_dtype])
        if (
            config.vllm_kv_offloading_size is not None
            and config.vllm_kv_offloading_size > 0
        ):
            args.extend(
                [
                    "--kv-offloading-size",
                    str(config.vllm_kv_offloading_size),
                    "--disable-hybrid-kv-cache-manager",
                ]
            )
        if config.vllm_max_num_seqs is not None:
            args.extend(["--max-num-seqs", str(config.vllm_max_num_seqs)])
        if config.vllm_enforce_eager:
            args.append("--enforce-eager")
        if config.trust_remote_code:
            args.append("--trust-remote-code")
        if not self._is_vllm_request_logging_enabled():
            args.append("--no-enable-log-requests")
        return args

    def _mm_args(self, config: InferenceConfig) -> List[str]:
        """多模態相關旗標：``--limit-mm-per-prompt`` 的 image/audio/video 配額。

        若使用者未顯式設定 image limit 且模型偵測為多模態，會自動補 image=1；
        純文字模型即使誤加此旗標 vLLM 也會忽略，故風險可控。
        """
        # 使用 getattr 以相容舊版 InferenceConfig（未定義對應欄位時不會報錯）
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

        mm_limits: Dict[str, int] = {}
        if mm_image_limit is not None:
            mm_limits["image"] = int(mm_image_limit)
        if mm_audio_limit is not None:
            mm_limits["audio"] = int(mm_audio_limit)
        if mm_video_limit is not None:
            mm_limits["video"] = int(mm_video_limit)

        if not mm_limits:
            return []
        return ["--limit-mm-per-prompt", json.dumps(mm_limits)]

    def _template_args(self, config: InferenceConfig) -> List[str]:
        """architectures override 與自訂 chat template。

        ``vllm_hf_overrides`` 用於強制覆寫 ``config.json`` 的 architectures，
        例：``'{"architectures":["Gemma4ForConditionalGeneration"]}'``。
        ``vllm_chat_template`` 用於 ``tokenizer_config.json`` 缺 chat_template 時。
        """
        args: List[str] = []

        hf_overrides = getattr(config, "vllm_hf_overrides", None)
        if hf_overrides:
            if isinstance(hf_overrides, (dict, list)):
                hf_overrides = json.dumps(hf_overrides)
            args.extend(["--hf-overrides", str(hf_overrides)])

        chat_template = getattr(config, "vllm_chat_template", None)
        if chat_template:
            args.extend(["--chat-template", str(chat_template)])

        return args

    def _build_server_cmd(self, config: InferenceConfig) -> List[str]:
        """組出完整 ``vllm serve <args...>`` 命令。

        各分組（core/perf/mm/template）由獨立 method 負責，便於單元測試。
        """
        cmd: List[str] = ["vllm", "serve"]
        cmd.extend(self._core_args(config))
        cmd.extend(self._perf_args(config))
        cmd.extend(self._mm_args(config))
        cmd.extend(self._template_args(config))
        return cmd

    def _resolve_vllm_server_dir(self) -> str:
        """解析 vllm_server 隔離環境專案目錄。"""
        project_dir = VLLM_SERVER_PROJECT_DIR
        if not os.path.isdir(project_dir):
            raise RuntimeError(f"VLLM_SERVER_PROJECT_DIR does not exist: {project_dir}")
        return project_dir

    def _resolve_launch_prefixes(self) -> List[List[str]]:
        """產生可用的 vllm launcher 前綴清單（不含 ``serve <args>`` 部分）。

        回傳順序代表優先序：venv 內
        ``python -m vllm.entrypoints.cli.main`` >
        系統 ``uv run --project ... python -m vllm.entrypoints.cli.main``。

        為了相容容器內掛載到不同絕對路徑的情境，不直接執行
        ``.venv/bin/vllm`` console script：該檔案的 shebang 會寫死建立虛擬環境
        當下的 interpreter 絕對路徑，搬移路徑後容易在 subprocess spawn 階段
        直接得到 ``ENOENT``。

        若兩者皆不可用直接拋出 ``RuntimeError``，呼叫端可在 ``load_model``
        最早期就接到失敗，而非到實際 spawn 子程序時才 panic。
        """
        project_dir = self._resolve_vllm_server_dir()

        prefixes: List[List[str]] = []

        python_bin = os.path.join(project_dir, ".venv", "bin", "python")
        if os.path.isfile(python_bin) and os.access(python_bin, os.X_OK):
            prefixes.append([python_bin, "-m", "vllm.entrypoints.cli.main"])

        uv_path = shutil.which("uv")
        if uv_path:
            prefixes.append(
                [
                    uv_path,
                    "run",
                    "--project",
                    project_dir,
                    "python",
                    "-m",
                    "vllm.entrypoints.cli.main",
                ]
            )

        if not prefixes:
            raise RuntimeError(
                f"vLLM launcher unavailable: no executable at "
                f"{python_bin} and `uv` not found on PATH. "
                "Please set up the vllm_server isolated environment first."
            )
        return prefixes

    def _discover_runtime_library_dirs(self, project_venv: str) -> List[str]:
        """搜尋隔離 venv 內需要加入 ``LD_LIBRARY_PATH`` 的共享函式庫目錄。"""
        discovered: List[str] = []
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
                    "/site-packages/nvidia/" in normalized_dir
                    and normalized_dir.endswith("/lib")
                )

                if not (has_cuda_libs or is_torch_lib or is_nvidia_lib):
                    continue

                abs_dir = os.path.abspath(dirpath)
                if abs_dir in seen:
                    continue
                seen.add(abs_dir)
                discovered.append(abs_dir)

        return discovered

    def _find_shared_library(self, project_venv: str, pattern: str) -> Optional[str]:
        """在隔離 venv 中尋找指定共享函式庫。"""
        for lib_root_name in ("lib", "lib64"):
            lib_root = os.path.join(project_venv, lib_root_name)
            if not os.path.isdir(lib_root):
                continue
            for candidate in glob.iglob(
                os.path.join(lib_root, "**", pattern), recursive=True
            ):
                if os.path.isfile(candidate):
                    return os.path.abspath(candidate)
        return None

    def _repair_isolated_nvidia_runtime(self, project_dir: str) -> None:
        """修復 vLLM 隔離環境中未完整展開的 NVIDIA runtime wheels。"""
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
            raise RuntimeError(
                f"vLLM isolated env missing python interpreter: {python_bin}"
            )

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

    def _build_server_env(self) -> Dict[str, str]:
        """建立 vLLM 子程序環境變數。"""
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
                extra_lib_dirs
                + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
            )

        env["PATH"] = os.pathsep.join(
            [os.path.join(project_venv, "bin"), env.get("PATH", "")]
        )
        return env

    def _validate_runtime_environment(self) -> None:
        """在 load_model 進入子流程前先做完整環境體檢，提早給出可動作的錯誤。

        檢查項目：
        1. ``VLLM_SERVER_PROJECT_DIR`` 設定且為實際存在的資料夾。
          2. 至少一種 launcher 可用（venv ``python -m vllm.entrypoints.cli.main``
              或系統 uv）。
        3. ``VLLM_PORT`` 在 1–65535 範圍。

        失敗會直接 raise；呼叫端的 ``load_model`` try/except 會把錯誤透過
        status_queue 通知主程序。
        """
        # 1+2：解析 project dir 與 launcher 前綴（這兩步本身就會 raise）
        project_dir = self._resolve_vllm_server_dir()
        self._resolve_launch_prefixes()
        self._repair_isolated_nvidia_runtime(project_dir)
        # 3：port 範圍
        self._resolve_vllm_port()

    def _start_server_process(self, config: InferenceConfig):
        port = self._resolve_vllm_port()
        self._cleanup_port(port)

        # 關鍵：這裡在 load_model 階段實際啟動 vLLM OpenAI-compatible server。
        raw_cmd = self._build_server_cmd(config)
        # raw_cmd 的形式為 ["vllm", "serve", ...]；組合 launcher 時要去掉 leading "vllm"
        # 因為各 prefix 本身已含到 vLLM module 為止
        # （venv `python -m vllm.entrypoints.cli.main` 或
        # `uv run ... python -m vllm.entrypoints.cli.main`）。
        serve_args = raw_cmd[1:] if raw_cmd and raw_cmd[0] == "vllm" else raw_cmd

        launch_candidates: List[List[str]] = [
            [*prefix, *serve_args] for prefix in self._resolve_launch_prefixes()
        ]

        last_error: Optional[Exception] = None
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

    def _stop_server_process(self):
        if self.runtime.server_process and self.runtime.server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.runtime.server_process.pid), signal.SIGINT)
                self.runtime.server_process.wait(timeout=10)
            except (
                ProcessLookupError,
                PermissionError,
                OSError,
                subprocess.TimeoutExpired,
            ):
                try:
                    os.killpg(
                        os.getpgid(self.runtime.server_process.pid), signal.SIGTERM
                    )
                    self.runtime.server_process.wait(timeout=5)
                except (
                    ProcessLookupError,
                    PermissionError,
                    OSError,
                    subprocess.TimeoutExpired,
                ):
                    try:
                        os.killpg(
                            os.getpgid(self.runtime.server_process.pid), signal.SIGKILL
                        )
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

        self.runtime.server_process = None

    def _wait_for_port_release(
        self, port: int, timeout: float = 15.0, poll_interval: float = 0.5
    ) -> bool:
        """主動 poll 直到指定 port 無人佔用，或 timeout 為止。

        熱替換情境：``unload`` 立即接著 ``load_model`` 時，若舊 vLLM 進程
        SIGKILL 後 OS 仍有殘留的 socket TIME_WAIT 或 GPU memory 尚未刷掉，
        下一次 ``_cleanup_port`` 會撞上殘骸。本方法提供一個明確的同步點，
        讓 unload 在 port 真正釋放後才返回。

        Returns:
            ``True`` 表示在 timeout 內成功釋放；``False`` 表示逾時仍被佔用。
        """
        deadline = time.time() + max(0.0, timeout)
        last_owner_pid: Optional[int] = None
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
        """等 stderr pump thread 把 pipe 內 buffered 內容 drain 完。

        子程序死亡瞬間 kernel 會把 pipe 標 EOF，但 reader thread 還沒 readline
        到所有殘留內容；若立即 ``_classify_recent_stderr`` 會抓到不完整的 buffer
        （vLLM v1 常見只剩 APIServer wrapper 的最後兩行，根因 OOM 被略掉）。
        在偵測子程序死亡後呼叫此 helper，給 reader thread 短時間把剩餘內容讀完。
        """
        thread = self.runtime.stderr_pump_thread
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout)

    def _classify_recent_stderr(self) -> VllmErrorReport:
        """讀取最近的 stderr 並交由 ErrorClassifier 分類。"""
        with self.runtime.stderr_buffer_lock:
            lines = list(self.runtime.stderr_recent_lines)
        return classify_stderr(lines)

    def _get_error_reason(self) -> str:
        """回傳供 RuntimeError 使用的純文字錯誤摘要（向後相容介面）。"""
        return self._classify_recent_stderr().to_text()

    def _normalize_messages(self, prompt: Any) -> List[Dict[str, Any]]:
        """將前端 prompt 轉為 OpenAI chat messages 格式。

        content 可能是：
        - str: 純文字
        - list[dict]: 多模態 multi-part，例如
            [{"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:..."}}]
        必須原樣保留 list 結構，不可 str() 強轉，否則圖片會丟失。
        """
        if isinstance(prompt, list):
            normalized: List[Dict[str, Any]] = []
            for msg in prompt:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")

                # 若 content 是 list（多模態 multi-part），保留原結構
                if isinstance(content, list):
                    normalized.append({"role": role, "content": content})
                elif isinstance(content, dict):
                    # 某些客戶端會送單一 dict
                    normalized.append({"role": role, "content": [content]})
                else:
                    normalized.append({"role": role, "content": str(content)})

            if normalized:
                return normalized

        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": str(prompt)}]

    def _reorder_multimodal_content(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """依 Gemma 4 官方 best practice 將 image/audio/video 排在 text 之前。

        若 content 不是 list（純文字情況）則原樣返回。
        未知 type 的 part 一律保留在最後以避免遺漏。
        """
        media_types = {
            "image_url",
            "image",
            "audio_url",
            "audio",
            "input_audio",
            "video_url",
            "video",
        }
        text_types = {"text"}

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            media_parts: List[Any] = []
            text_parts: List[Any] = []
            other_parts: List[Any] = []
            for p in content:
                if not isinstance(p, dict):
                    other_parts.append(p)
                    continue
                ptype = p.get("type")
                if ptype in media_types:
                    media_parts.append(p)
                elif ptype in text_types:
                    text_parts.append(p)
                else:
                    other_parts.append(p)

            # 只有同時存在 media 與 text 時才重排，避免改動無意義
            if media_parts and text_parts:
                msg["content"] = media_parts + text_parts + other_parts

        return messages

    def _check_server_alive_or_raise(self) -> None:
        """若 vLLM 子程序已死，立即拋出帶分類資訊的 RuntimeError。

        在 health check 輪詢的每次迭代之前呼叫；避免 server crash 後仍空轉
        直到整個 ``health_timeout_s`` 用完才回報。
        """
        proc = self.runtime.server_process
        if proc is None:
            return
        if proc.poll() is None:
            return

        # 關鍵：先讓 reader thread 讀完 pipe 殘留內容（含 EngineCore traceback），
        # 否則 _classify_recent_stderr 只看得到 APIServer wrapper 那幾行
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
        last_error: Optional[str] = None

        while time.time() - start < self.runtime.health_timeout_s:
            # 每輪輪詢都先檢查子程序是否已死（含 sleep 後的下一輪）
            self._check_server_alive_or_raise()

            try:
                # 關鍵：使用 OpenAI-compatible `models.list()` 驗證 vLLM server 可用性。
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

        # Timeout 之後再做最後一次 process 狀態判定，並附上分類後的 stderr 摘要
        self._check_server_alive_or_raise()
        report = self._classify_recent_stderr()
        url = f"{self.runtime.base_url}/models"
        details_parts: List[str] = []
        if last_error:
            details_parts.append(f"last_error={last_error}")
        details_parts.append(f"category={report.category}")
        details_parts.append(f"stderr_excerpt={report.to_text()}")
        details = "; " + "; ".join(details_parts)
        raise RuntimeError(
            f"vLLM OpenAI-compatible server 啟動 timeout: "
            f"{self.runtime.health_timeout_s:.0f}s, endpoint={url}{details}"
        )

    def _prepare_payload(
        self, prompt: Any, params: Optional[Dict[str, Any]] = None, stream: bool = False
    ) -> Dict[str, Any]:
        params = params or {}

        logger.debug(
            "[vLLM] _prepare_payload raw prompt type=%s preview=%.300s",
            type(prompt).__name__,
            str(prompt),
        )

        messages = self._reorder_multimodal_content(self._normalize_messages(prompt))

        logger.debug("[vLLM] _prepare_payload messages: %s", messages)
        # Debug log：確認 multi-part content 未被意外 str 化
        # 正常多模態：[('user', 'list', 2)]；純文字：[('user', 'str', None)]
        if logger.isEnabledFor(10):  # DEBUG level
            try:
                structure = [
                    (
                        m.get("role"),
                        type(m.get("content")).__name__,
                        (
                            len(m["content"])
                            if isinstance(m.get("content"), list)
                            else None
                        ),
                    )
                    for m in messages
                ]
                logger.debug("[vLLM] payload messages structure: %s", structure)
            except Exception:
                pass

        model_name = self.runtime.served_model_name
        if model_name is None and self.config is not None:
            model_name = self.config.model_name
        if model_name is None:
            model_name = "vllm-server"

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max(1, int(params.get("max_new_tokens", 512))),
            "temperature": max(0.0, min(float(params.get("temperature", 0.7)), 2.0)),
            "top_p": max(0.0, min(float(params.get("top_p", 0.9)), 1.0)),
            "stream": stream,
        }

        # OpenAI function-calling 格式；None / 空 list 不送，避免干擾純聊天請求
        tools = params.get("tools")
        if tools:
            payload["tools"] = tools

        tool_choice = params.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        extra_body: Dict[str, Any] = {}
        repetition_penalty = params.get("repetition_penalty")
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = float(repetition_penalty)

        top_k = params.get("top_k")
        if top_k is not None:
            extra_body["top_k"] = int(top_k)

        enable_thinking = params.get("enable_thinking")
        if enable_thinking is not None:
            chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
            chat_template_kwargs["enable_thinking"] = bool(enable_thinking)

        if extra_body:
            payload["extra_body"] = extra_body

        return payload

    def load_model(self, config: InferenceConfig):
        self.config = config
        self.runtime.api_key = VLLM_OPENAI_API_KEY
        self.runtime.health_timeout_s = VLLM_HEALTH_TIMEOUT
        try:
            # 提早做環境體檢：launcher 不可用、port 設錯時直接 raise，
            # 不浪費時間走後面的 client 建立、子程序啟動、health poll。
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
            logger.error("[Worker] vLLM startup failed: %s", e)
            logger.error("[Worker] vLLM stderr excerpt:\n%s", startup_excerpt)
            self._stop_server_process()
            self.runtime.served_model_name = None
            raise RuntimeError(
                f"vLLM startup failed: {e}; stderr_excerpt={startup_excerpt}"
            ) from e

    def _normalize_prompt(
        self, prompt: Any, params: Optional[Dict[str, Any]] = None
    ) -> str:
        _ = params
        messages = self._normalize_messages(prompt)

        def _content_to_text(content: Any) -> str:
            # 多模態 multi-part 情況：只抽取 text，圖片/音訊以 placeholder 表示
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

        return "\n".join(
            f"{m['role']}: {_content_to_text(m['content'])}" for m in messages
        )

    def _build_sampling_params(self, params: Dict[str, Any]):
        return self._prepare_payload(prompt="", params=params, stream=False)

    def _abort_request(self, request_id: Optional[str]):
        logger.info("[Worker] vLLM abort requested for request_id=%s", request_id)

    def generate(self, request: Dict[str, Any]):
        request_id = request.get("request_id")
        prompt = request.get("prompt")
        params = request.get("params", {})
        request_stop_flag = request.get("request_stop_flag")
        if not self.config or not self.runtime.served_model_name:
            self.data_queue.put(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": "vLLM engine not loaded",
                }
            )
            return

        # 在發出 HTTP 請求前先檢查 stop flag（並發場景下可能已被設定）
        _req_stopped = (
            isinstance(request_stop_flag, threading.Event)
            and request_stop_flag.is_set()
        )
        if self.stop_generation_flag.is_set() or _req_stopped:
            logger.info("[vLLM] generate aborted before start (request_id=%s)", request_id)
            self.data_queue.put(
                {
                    "type": "result",
                    "request_id": request_id,
                    "result": "",
                    "stopped": True,
                }
            )
            return

        start = time.perf_counter()
        try:
            payload = self._prepare_payload(prompt=prompt, params=params, stream=False)
            timeout_s = float(params["total_timeout"])

            # 關鍵：改為 OpenAI SDK 呼叫 vLLM OpenAI-compatible server。
            if self.runtime.client is None:
                raise RuntimeError("OpenAI client not initialized")

            completion = self.runtime.client.chat.completions.create(
                timeout=timeout_s,
                **payload,
            )

            result_text = ""
            if completion.choices:
                message = completion.choices[0].message
                result_text = str(message.content or "")

            usage = completion.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            gen_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            elapsed = max(1e-6, time.perf_counter() - start)
            response = {
                "type": "result",
                "request_id": request_id,
                "result": result_text,
            }
            if total_tokens is not None:
                response["total_tokens"] = total_tokens
            if gen_tokens is not None:
                response["gen_tokens"] = gen_tokens
                response["gen_tps"] = float(gen_tokens) / elapsed
            if prompt_tokens is not None:
                response["prompt_tokens"] = prompt_tokens

            self.data_queue.put(response)
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.error("[Worker] vLLM generate error: %s", e)
            self.data_queue.put(
                {"type": "error", "request_id": request_id, "error": str(e)}
            )

    def generate_stream(self, request: Dict[str, Any]):
        request_id = request.get("request_id")
        prompt = request.get("prompt")
        params = request.get("params", {})
        request_stop_flag = request.get("request_stop_flag")

        # print debug log with prompt and params preview (truncated to 300 chars to avoid log flooding)
        logger.debug(
            "[vLLM] generate_stream request_id=%s prompt type=%s preview=%.300s params preview=%.300s",
            request_id,
            type(prompt).__name__,
            str(prompt),
            str(params),
        )

        if not self.config or not self.runtime.served_model_name:
            self.data_queue.put(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": "vLLM engine not loaded",
                }
            )
            return

        start = time.perf_counter()
        prompt_tokens = None
        gen_tokens = 0
        total_tokens = None

        try:
            payload = self._prepare_payload(prompt=prompt, params=params, stream=True)
            timeout_s = float(params["total_timeout"])

            # 關鍵：串流由 OpenAI SDK iterator 處理，不再手動解析 SSE `data:` 行。
            if self.runtime.client is None:
                raise RuntimeError("OpenAI client not initialized")

            logger.info("[vLLM] Using client.chat.completions.create")
            stream = self.runtime.client.chat.completions.create(
                timeout=timeout_s,
                stream_options={"include_usage": True},
                **payload,
            )

            for event in stream:
                _req_stopped = (
                    isinstance(request_stop_flag, threading.Event)
                    and request_stop_flag.is_set()
                )
                if self.stop_generation_flag.is_set() or _req_stopped:
                    logger.info(
                        "[vLLM] Stream aborted (request_id=%s, global=%s, local=%s)",
                        request_id,
                        self.stop_generation_flag.is_set(),
                        _req_stopped,
                    )
                    break

                chunk_text = ""
                if event.choices:
                    delta = event.choices[0].delta
                    chunk_text = str((delta.content or ""))

                usage = getattr(event, "usage", None)
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        prompt_tokens = int(usage.prompt_tokens)
                    if getattr(usage, "completion_tokens", None) is not None:
                        gen_tokens = int(usage.completion_tokens)
                    if getattr(usage, "total_tokens", None) is not None:
                        total_tokens = int(usage.total_tokens)

                if chunk_text:
                    # 注意：vLLM 在 stream_options={"include_usage": True} 時
                    # 僅於最後一筆事件回傳完整 usage（prompt/completion/total tokens），
                    # 不會逐 chunk 提供 token 計數。為了避免使用 split-based 估算
                    # 造成 CJK / BPE tokenizer 嚴重失真，這裡刻意省略 chunk_tokens；
                    # 下游消費端對缺漏 chunk_tokens 已是 if not None 防呆。
                    # 真實 token 計數一律由結尾 done_payload 中的 usage 欄位提供。
                    self.data_queue.put(
                        {
                            "type": "stream_chunk",
                            "request_id": request_id,
                            "chunk": chunk_text,
                            "done": False,
                        }
                    )

            elapsed = max(1e-6, time.perf_counter() - start)
            gen_tps = float(gen_tokens) / elapsed if gen_tokens else 0.0
            prompt_tps = float(prompt_tokens) / elapsed if prompt_tokens else 0.0
            done_payload = {
                "type": "stream_chunk",
                "request_id": request_id,
                "chunk": "",
                "done": True,
                "gen_tokens": gen_tokens,
                "gen_tps": gen_tps,
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
                "prompt_tps": prompt_tps,
            }
            if total_tokens is not None:
                done_payload["total_tokens"] = total_tokens
            elif prompt_tokens is not None:
                done_payload["total_tokens"] = int(prompt_tokens) + int(gen_tokens)
            else:
                done_payload["total_tokens"] = gen_tokens
            _req_stopped_final = (
                isinstance(request_stop_flag, threading.Event)
                and request_stop_flag.is_set()
            )
            if self.stop_generation_flag.is_set() or _req_stopped_final:
                done_payload["stopped"] = True

            self.data_queue.put(done_payload)
        except (APIError, OSError, ValueError, TypeError, RuntimeError) as e:
            logger.error("[Worker] vLLM generate_stream error: %s", e)
            self.data_queue.put(
                {"type": "error", "request_id": request_id, "error": str(e)}
            )

    def unload(self):
        # 關鍵：在 unload 階段關閉由本 engine 啟動的 vLLM server。
        # 解析 port 失敗（環境變數設錯）時不該擋住 unload，故包 try。
        try:
            port = self._resolve_vllm_port()
        except Exception:
            port = None

        self._stop_server_process()

        # 熱替換時 unload 緊接 load_model；確保舊 server 真的釋放 port，
        # 避免下一次 _cleanup_port 撞上 TIME_WAIT 或殘留進程。
        port_released = True
        if port is not None:
            port_released = self._wait_for_port_release(port)

        # 一次性整體重置：所有 vLLM 生命週期狀態歸零，
        # 避免漏掉某個欄位（之前手動 5 行 reset 出過不對齊風險）。
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

    def apply_chat_template(self, request: Dict[str, Any]):
        request_id = request.get("request_id")
        try:
            messages = request.get("messages", [])
            prompt = self._normalize_prompt(
                messages, request.get("template_kwargs", {})
            )
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
        """組出 vLLM server 管理 endpoint URL（不含 /v1 前綴）。"""
        # base_url 形如 http://host:port/v1，admin endpoints 位於 root 之下。
        base = self.runtime.base_url
        root = base[:-3] if base.endswith("/v1") else base
        if not path.startswith("/"):
            path = "/" + path
        return f"{root.rstrip('/')}{path}"

    def cleanup_generation_memory(self):
        """請 vLLM server 重置 prefix cache 以釋放累積的 KV 記憶體。

        vLLM OpenAI-compatible server 提供 ``POST /reset_prefix_cache`` admin
        endpoint，呼叫後會清空 prefix cache 條目（不影響當前在跑的請求）。
        若 server 尚未啟動或網路錯誤，記錄 warning 但不向上拋出，避免清理動作
        把整個服務拖垮。
        """
        # 在 vLLM server 尚未就緒時直接回報（避免空打 HTTP）
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
            {"Authorization": f"Bearer {self.runtime.api_key}"}
            if self.runtime.api_key
            else {}
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
            # endpoint 在某些 vLLM 版本可能尚未啟用；不視為致命錯誤
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
