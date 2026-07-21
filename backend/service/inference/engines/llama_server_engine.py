import os
import re
import json
import hashlib
import threading
import subprocess
import time
import glob
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from collections import deque
from typing import Dict, Any, List, Optional
from multiprocessing import Queue
from multiprocessing.synchronize import Event as EventClass

from openai import APIError, OpenAI

from ...config_models import InferenceConfig
from ...settings import (
    configure_logging,
    HF_HOME,
    LLAMA_SERVER_URL,
    LLAMA_SERVER_API_KEY,
    LLAMA_SERVER_TIMEOUT,
    LLAMA_SERVER_BINARY,
)
from ..generate.llama_server_runner import (
    handle_llama_server_generate,
    handle_llama_server_stream,
    handle_server_log_line,
)
from .base_engine import BaseEngine

logger = configure_logging(__name__)


class LlamaServerEngine(BaseEngine):
    """Llama Server 推理引擎（OpenAI-compatible API）。"""

    def __init__(self, status_queue: Queue, data_queue: Queue, stop_event: EventClass, stop_generation_flag: EventClass):
        super().__init__(status_queue, data_queue, stop_event, stop_generation_flag)
        self.base_url: str = LLAMA_SERVER_URL
        self.api_key: Optional[str] = LLAMA_SERVER_API_KEY
        self.timeout_sec: int = LLAMA_SERVER_TIMEOUT
        self.client: Optional[OpenAI] = None
        self.served_model_name: Optional[str] = None
        self.server_process: Optional[subprocess.Popen] = None
        self.managed_process: bool = False
        self.last_start_error: Optional[str] = None
        self._stdout_pump_thread: Optional[threading.Thread] = None
        self._stderr_pump_thread: Optional[threading.Thread] = None
        self._stderr_buffer_lock = threading.Lock()
        self._stderr_recent_lines = deque(maxlen=120)
        self._timing_lock = threading.Lock()
        self._active_timing_slot: Optional[int] = None
        self._slot_timings: Dict[int, Dict[str, Any]] = {}
        self._trace_lock = threading.Lock()
        self._pending_request_ids = deque()
        self._request_trace: Dict[str, Dict[str, Any]] = {}
        self._task_to_request: Dict[int, str] = {}
        self._last_runtime_error_signature: Optional[str] = None
        self._served_model_capabilities: List[str] = []
        self._prefill_strategy: str = "cache_prompt"
        self._detected_mmproj_path: Optional[str] = None

    def _pump_server_logs(self, stream, is_stderr: bool = False) -> None:
        """將 llama-server 子程序輸出轉發到當前 service logger。"""
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                txt = (line or "").rstrip()
                if not txt:
                    continue

                # 直接轉發原始行，讓 main/service log 看得到 llama-server 的 timing 與 slot 訊息
                if is_stderr:
                    with self._stderr_buffer_lock:
                        self._stderr_recent_lines.append(txt)
                    handle_server_log_line(self, txt)
                    logger.warning(f"[LlamaServer][stderr] {txt}")
                else:
                    logger.info(f"[LlamaServer][stdout] {txt}")
        except Exception as e:
            logger.debug(f"[LlamaServer] log pump stopped: {e}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _build_startup_error_summary(self, rc: int) -> str:
        """從最近 stderr 內容提取對使用者最有幫助的啟動失敗原因。"""
        with self._stderr_buffer_lock:
            lines = list(self._stderr_recent_lines)

        if not lines:
            return f"process exited with code {rc}"

        # 優先挑明確錯誤關鍵字
        keywords = [
            "out of memory",
            "cudamalloc failed",
            "unable to allocate",
            "error loading model",
            "failed to load model",
            "failed to load",
            "main: exiting",
            "error",
            "failed",
        ]

        picked: List[str] = []
        for line in reversed(lines):
            low = line.lower()
            if any(k in low for k in keywords):
                picked.append(line)
            if len(picked) >= 6:
                break

        if not picked:
            picked = lines[-4:]

        picked.reverse()
        compact = " | ".join(picked)
        if len(compact) > 1800:
            compact = compact[-1800:]
        return f"process exited with code {rc}; stderr={compact}"

    def _recent_stderr_excerpt(self, max_lines: int = 12, max_chars: int = 2000) -> str:
        """回傳最近 stderr 摘要，供 runtime error 診斷。"""
        with self._stderr_buffer_lock:
            lines = list(self._stderr_recent_lines)

        if not lines:
            return ""

        excerpt = " | ".join(lines[-max(1, max_lines):])
        if len(excerpt) > max_chars:
            excerpt = excerpt[-max_chars:]
        return excerpt

    def _is_runtime_oom_text(self, text: Optional[str]) -> bool:
        low = (text or "").lower()
        keywords = [
            "out of memory",
            "cuda error: out of memory",
            "cudamalloc failed",
            "unable to allocate",
            "failed to allocate",
            "cuMemCreate",
            "ggml_abort",
            "cuda error",
        ]
        return any(k.lower() in low for k in keywords)

    def _build_openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def _discover_mmproj_file(self, model_file: str, extra_args: List[str]) -> str:
        """自動尋找對應的多模態 projector 檔案。"""
        if self.config is None:
            return ""

        mmproj_path = self._resolve_local_path(self.config.llama_server_mmproj, prefer_hf_home=True)
        has_mmproj_arg = any(arg == "--mmproj" for arg in extra_args)
        if mmproj_path or has_mmproj_arg:
            return mmproj_path

        candidate_paths: List[str] = []

        def _push_candidate(path_value: Optional[str]) -> None:
            path_text = str(path_value or "").strip()
            if not path_text:
                return
            norm_path = os.path.normpath(path_text)
            if norm_path not in candidate_paths:
                candidate_paths.append(norm_path)

        def _push_mmproj_files_from_dir(dir_path: Optional[str]) -> None:
            dir_text = str(dir_path or "").strip()
            if not dir_text or not os.path.isdir(dir_text):
                return

            try:
                candidates = [
                    os.path.join(dir_text, fname)
                    for fname in os.listdir(dir_text)
                    if "mmproj" in fname.lower() and fname.lower().endswith(".gguf")
                ]
            except OSError as exc:
                logger.debug(f"[LlamaServer] Failed to scan mmproj candidates in {dir_text}: {exc}")
                return

            def _rank_candidate(path_value: str) -> tuple[int, int, str]:
                name = os.path.basename(path_value).lower()
                return (
                    0 if "bf16" in name else 1,
                    0 if "f16" in name else 1,
                    name,
                )

            for candidate in sorted(candidates, key=_rank_candidate):
                _push_candidate(candidate)

        model_path = Path(model_file)
        search_dirs = [model_path.parent]
        if model_path.parent != model_path.parent.parent:
            search_dirs.append(model_path.parent.parent)

        for search_dir in search_dirs:
            _push_mmproj_files_from_dir(str(search_dir))

        model_name = (self.config.model_name or "").strip()
        if "/" in model_name:
            repo_cache_dir_name = f"models--{model_name.replace('/', '--')}"
            hub_roots: List[Path] = []

            for parent in model_path.parents:
                if parent.name == "hub":
                    hub_roots.append(parent)

            hf_home = os.getenv("HF_HOME", "").strip()
            if hf_home:
                hub_candidate = Path(hf_home) / "hub"
                if hub_candidate not in hub_roots:
                    hub_roots.append(hub_candidate)

            for hub_root in hub_roots:
                snapshot_glob = hub_root / repo_cache_dir_name / "snapshots" / "*"
                for snapshot_dir in sorted(glob.glob(str(snapshot_glob))):
                    _push_mmproj_files_from_dir(snapshot_dir)

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                logger.info(f"[LlamaServer] Auto-discovered mmproj file: {candidate}")
                return candidate

        return ""

    def _recreate_client(self) -> None:
        self.client = OpenAI(
            base_url=self._build_openai_base_url(),
            api_key=self.api_key or "EMPTY",
        )

    def _list_models(self, timeout_sec: float = 5.0) -> List[str]:
        if self.client is None:
            self._recreate_client()
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        response = self.client.models.list(timeout=max(0.2, float(timeout_sec)))
        return [
            str(item.id)
            for item in getattr(response, "data", [])
            if getattr(item, "id", None)
        ]

    def _probe_server_alive(self, timeout_sec: float = 1.0) -> bool:
        try:
            self._list_models(timeout_sec=timeout_sec)
            return True
        except (APIError, OSError, RuntimeError, ValueError, TypeError):
            pass

        return False

    def build_runtime_error_payload(self, exc: Exception) -> Dict[str, Any]:
        """將 llama-server runtime 例外轉成可跨進程傳遞的結構化錯誤。"""
        raw_error = str(exc).strip() or exc.__class__.__name__
        stderr_excerpt = self._recent_stderr_excerpt()
        combined_text = f"{raw_error}\n{stderr_excerpt}" if stderr_excerpt else raw_error
        combined_low = combined_text.lower()

        exit_code: Optional[int] = None
        if self.managed_process and self.server_process is not None:
            try:
                exit_code = self.server_process.poll()
            except Exception:
                exit_code = None

        disconnected = any(
            marker in combined_low
            for marker in [
                "server disconnected without sending a response",
                "remoteprotocolerror",
                "connection reset",
                "broken pipe",
                "connection aborted",
            ]
        )
        is_oom = self._is_runtime_oom_text(combined_text)
        server_alive = self._probe_server_alive(timeout_sec=0.6)
        fatal = bool(exit_code is not None or is_oom or (disconnected and not server_alive))

        if is_oom:
            error_type = "LlamaServerOOM"
        elif exit_code is not None:
            error_type = "LlamaServerProcessExited"
        elif disconnected and not server_alive:
            error_type = "LlamaServerDisconnected"
        else:
            error_type = exc.__class__.__name__ or "LlamaServerRuntimeError"

        parts: List[str] = []
        if exit_code is not None:
            parts.append(f"llama-server process exited with code {exit_code}")
        parts.append(raw_error)
        if stderr_excerpt and (fatal or is_oom or disconnected):
            parts.append(f"recent_stderr={stderr_excerpt}")

        error_message = "; ".join([p for p in parts if p]) or raw_error

        return {
            "error": error_message,
            "error_type": error_type,
            "error_traceback": stderr_excerpt or None,
            "is_oom": is_oom,
            "recoverable": False,
            "fatal": fatal,
            "process_alive": server_alive,
        }

    def report_runtime_error(self, error_payload: Dict[str, Any]) -> None:
        """將 fatal runtime error 同步到 status_queue，讓 /inference/error_details 可讀到。"""
        if not isinstance(error_payload, dict) or not error_payload.get("fatal"):
            return

        signature = "|".join(
            [
                str(error_payload.get("error_type") or ""),
                str(error_payload.get("error") or ""),
            ]
        )
        if signature and signature == self._last_runtime_error_signature:
            return
        self._last_runtime_error_signature = signature or None

        try:
            self.status_queue.put(
                {
                    "status": "error",
                    "error": error_payload.get("error"),
                    "error_type": error_payload.get("error_type"),
                    "error_traceback": error_payload.get("error_traceback"),
                    "is_oom": error_payload.get("is_oom", False),
                }
            )
        except Exception as e:
            logger.debug(f"[LlamaServer] failed to publish runtime error status: {e}")

    def _normalize_base_url(self, raw_url: Optional[str]) -> str:
        url = (raw_url or "").strip() or LLAMA_SERVER_URL
        normalized = url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3].rstrip("/")
        return normalized

    def _resolve_model_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        if self.config is None:
            return ""
        return (self.config.llama_server_model or self.config.model_name or "").strip()

    def _resolve_served_model_name(self, timeout_sec: float = 5.0) -> str:
        preferred = self._resolve_model_name()
        model_ids = self._list_models(timeout_sec=timeout_sec)

        if preferred and preferred in model_ids:
            return preferred
        if model_ids:
            return model_ids[0]
        return preferred

    def _resolve_local_path(self, raw_path: Optional[str], *, prefer_hf_home: bool = True) -> str:
        """解析本地檔案路徑，支援相對路徑與 HF_HOME 目錄。"""
        path_text = str(raw_path or "").strip()
        if not path_text:
            return ""

        normalized_text = os.path.normpath(os.path.expanduser(path_text))
        candidate_path = Path(normalized_text)
        candidates: List[Path] = [candidate_path]

        if prefer_hf_home and HF_HOME and not candidate_path.is_absolute():
            hf_home_path = Path(HF_HOME).expanduser()
            hf_parts = [part.lower() for part in candidate_path.parts]

            candidates.append(hf_home_path / candidate_path)
            if hf_parts[:1] == ["hub"]:
                trimmed_parts = candidate_path.parts[1:]
                if trimmed_parts:
                    candidates.append(hf_home_path / Path(*trimmed_parts))
            else:
                candidates.append(hf_home_path / "hub" / candidate_path)

        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)

            if candidate.exists():
                return str(candidate.resolve())

        return normalized_text

    def _resolve_model_file(self) -> str:
        if self.config is None:
            return ""
        candidate = (self.config.model_path or self.config.model_name or "").strip()
        if not candidate:
            return ""
        return self._resolve_local_path(candidate, prefer_hf_home=True)

    def _resolve_slot_save_path(self, model_file: str) -> str:
        """回傳 llama-server slot 持久化目錄，預設使用模型所在目錄。"""
        model_dir = os.path.dirname(model_file or "")
        if model_dir and os.path.isdir(model_dir):
            return model_dir
        return ""

    def _build_server_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _request_server_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        req = urllib_request.Request(self._build_server_url(path), method=method.upper())
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            with urllib_request.urlopen(req, data=data, timeout=max(1.0, float(self.timeout_sec))) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore").strip()
            detail = body or str(e)
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e

        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def _refresh_served_model_capabilities(self) -> List[str]:
        """從 llama-server `/v1/models` 讀取目前模型能力。"""
        capabilities: List[str] = []

        try:
            payload = self._request_server_json("GET", "/v1/models")
        except Exception as e:
            logger.warning(f"[LlamaServer] failed to fetch model capabilities: {e}")
            self._served_model_capabilities = []
            return self._served_model_capabilities

        candidates: List[Dict[str, Any]] = []
        for key in ("data", "models"):
            values = payload.get(key)
            if isinstance(values, list):
                candidates.extend(item for item in values if isinstance(item, dict))

        preferred_names = {
            str(self.served_model_name or "").strip(),
            str(self._resolve_model_name() or "").strip(),
        }
        preferred_names.discard("")

        chosen: Optional[Dict[str, Any]] = None
        for item in candidates:
            item_names = {
                str(item.get("id") or "").strip(),
                str(item.get("name") or "").strip(),
                str(item.get("model") or "").strip(),
            }
            if preferred_names.intersection(name for name in item_names if name):
                chosen = item
                break

        if chosen is None and candidates:
            chosen = candidates[0]

        if isinstance(chosen, dict):
            raw_caps = chosen.get("capabilities")
            if not isinstance(raw_caps, list):
                raw_caps = chosen.get("meta", {}).get("capabilities") if isinstance(chosen.get("meta"), dict) else None
            if isinstance(raw_caps, list):
                capabilities = [str(item).strip().lower() for item in raw_caps if str(item).strip()]

        self._served_model_capabilities = capabilities
        return capabilities

    def _is_multimodal_model(self) -> bool:
        if any(cap == "multimodal" for cap in self._served_model_capabilities):
            return True
        if self._detected_mmproj_path:
            return True
        if self.config is not None and (self.config.llama_server_mmproj or "").strip():
            return True

        with self._stderr_buffer_lock:
            lines = list(self._stderr_recent_lines)

        for raw in reversed(lines):
            line = (raw or "").lower()
            if "loaded multimodal model" in line or "projector:" in line or "vision hparams" in line:
                return True

        return False

    def _coerce_slot_id(self, value: Any) -> Optional[int]:
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    def _resolve_request_slot_id(self, request: Optional[Dict[str, Any]] = None) -> int:
        request = request or {}
        max_slots = max(1, int(self.config.llama_server_np or 1)) if self.config is not None else 1

        explicit_slot = self._coerce_slot_id(request.get("slot"))
        if explicit_slot is not None:
            return explicit_slot % max_slots

        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        for key in ("session_id", "conversation_id", "thread_id"):
            raw_value = request.get(key) or params.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                digest = hashlib.sha256(raw_value.strip().encode("utf-8")).hexdigest()
                return int(digest[:8], 16) % max_slots

        return 0

    def _build_slot_cache_filename(self, slot_id: int) -> str:
        model_file = self._resolve_model_file()
        model_name = os.path.splitext(os.path.basename(model_file or "model"))[0] or "model"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._") or "model"
        return f"{safe_name}.slot-{slot_id}.bin"

    def _save_slot_cache(self, slot_id: int) -> bool:
        filename = self._build_slot_cache_filename(slot_id)
        try:
            payload = self._request_server_json("POST", f"/slots/{slot_id}?action=save", {"filename": filename})
            logger.info(f"[LlamaServer] saved slot cache: slot={slot_id}, filename={filename}, result={payload}")
            return True
        except Exception as e:
            logger.warning(f"[LlamaServer] save slot cache failed: slot={slot_id}, filename={filename}, error={e}")
            return False

    def _restore_slot_cache(self, slot_id: int) -> bool:
        filename = self._build_slot_cache_filename(slot_id)
        try:
            payload = self._request_server_json("POST", f"/slots/{slot_id}?action=restore", {"filename": filename})
            logger.info(f"[LlamaServer] restored slot cache: slot={slot_id}, filename={filename}, result={payload}")
            return True
        except Exception as e:
            logger.warning(f"[LlamaServer] restore slot cache failed: slot={slot_id}, filename={filename}, error={e}")
            return False

    def _restore_slot_caches(self) -> Dict[str, Any]:
        total_slots = max(1, int(self.config.llama_server_np or 1)) if self.config is not None else 1
        restored_slots: List[int] = []
        for slot_id in range(total_slots):
            if self._restore_slot_cache(slot_id):
                restored_slots.append(slot_id)
        return {
            "restored": len(restored_slots),
            "total": total_slots,
            "slots": restored_slots,
        }

    def _save_slot_caches(self) -> Dict[str, Any]:
        total_slots = max(1, int(self.config.llama_server_np or 1)) if self.config is not None else 1
        saved_slots: List[int] = []
        for slot_id in range(total_slots):
            if self._save_slot_cache(slot_id):
                saved_slots.append(slot_id)
        return {
            "saved": len(saved_slots),
            "total": total_slots,
            "slots": saved_slots,
        }

    def prepare_request_for_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        prepared_request = dict(request or {})
        extra_body = dict(prepared_request.get("extra_body") or {})
        extra_body["cache_prompt"] = True

        if self._is_multimodal_model():
            self._prefill_strategy = "cache_prompt"
            prepared_request.pop("slot", None)
        else:
            self._prefill_strategy = "slot"
            slot_id = self._resolve_request_slot_id(prepared_request)
            extra_body.setdefault("id_slot", slot_id)
            prepared_request["slot"] = slot_id

        prepared_request["extra_body"] = extra_body
        return prepared_request

    def _wait_server_ready(self, timeout_sec: int) -> bool:
        deadline = time.time() + max(1, timeout_sec)
        while time.time() < deadline:
            # 若為托管子程序，先檢查是否提前退出
            if self.managed_process and self.server_process is not None:
                rc = self.server_process.poll()
                if rc is not None:
                    self.last_start_error = self._build_startup_error_summary(rc)
                    logger.error(
                        f"[LlamaServer] managed subprocess exited early with code {rc}. {self.last_start_error}"
                    )
                    return False

            try:
                self._list_models(timeout_sec=2.0)
                return True
            except (APIError, OSError, RuntimeError, ValueError, TypeError):
                pass

            time.sleep(0.5)

        return False

    def _collect_layer_allocation_from_logs(self) -> Dict[str, Any]:
        """從 llama-server `load_tensors` 日誌推導 layer allocation。

        目標是對齊舊本地 GGUF 引擎的輸出格式：
        - `device_map_summary`: `GPU: x layers, CPU: y layers`
        - `total_modules`: 總層數
        - `layer_lines`: `Layers a-b -> GPU/CPU`
        """
        with self._stderr_buffer_lock:
            lines = list(self._stderr_recent_lines)

        total_layers: Optional[int] = None
        gpu_layers: Optional[int] = None
        repeating_gpu_layers: Optional[int] = None
        output_layer_on_gpu = False
        memory_usage: Dict[str, Any] = {}

        for raw in lines:
            line = (raw or "").strip()
            low = line.lower()

            # 例: load_tensors: offloaded 5/37 layers to GPU
            m_offloaded = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+gpu", low)
            if m_offloaded:
                try:
                    gpu_layers = int(m_offloaded.group(1))
                    total_layers = int(m_offloaded.group(2))
                except Exception:
                    pass

            # 例: load_tensors: offloading 4 repeating layers to GPU
            m_repeating = re.search(r"offloading\s+(\d+)\s+repeating\s+layers\s+to\s+gpu", low)
            if m_repeating:
                try:
                    repeating_gpu_layers = int(m_repeating.group(1))
                except Exception:
                    pass

            # 例: load_tensors: offloading output layer to GPU
            if "offloading output layer to gpu" in low:
                output_layer_on_gpu = True

            # 例:
            # load_tensors:   CPU_Mapped model buffer size = 62328.33 MiB
            # load_tensors:        CUDA0 model buffer size =  7784.52 MiB
            m_buf = re.search(
                r"load_tensors:\s*([A-Za-z0-9_]+)\s+model\s+buffer\s+size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
                line,
            )
            if m_buf:
                dev_name = m_buf.group(1)
                mib_val = float(m_buf.group(2))
                memory_usage[dev_name] = {
                    "mib": mib_val,
                    "gb": round(mib_val / 1024.0, 2),
                }

        # 無法解析時，回退到原本資訊
        if total_layers is None or gpu_layers is None:
            return {
                "device": "llama-server",
                "device_map_summary": "remote: llama-server",
                "total_modules": None,
                "layer_lines": [],
                "memory_usage": memory_usage or None,
            }

        total_layers = max(0, total_layers)
        gpu_layers = max(0, min(gpu_layers, total_layers))

        # 與舊本地 GGUF 引擎一致的摘要格式
        if total_layers == 0:
            device_label = "llama-server"
            device_summary = "remote: llama-server"
            layer_lines: List[str] = []
        elif gpu_layers >= total_layers:
            device_label = "GPU (Full)"
            device_summary = f"GPU: {total_layers}/{total_layers} layers (100%)"
            layer_lines = [f"Layers 0-{total_layers - 1} -> GPU"]
        elif gpu_layers == 0:
            device_label = "CPU"
            device_summary = f"CPU: {total_layers}/{total_layers} layers (100%)"
            layer_lines = [f"Layers 0-{total_layers - 1} -> CPU"]
        else:
            cpu_layers = total_layers - gpu_layers
            device_label = "Mixed"
            device_summary = f"GPU: {gpu_layers} layers, CPU: {cpu_layers} layers"

            # 若日志顯示「repeating + output」的分佈，優先反映更真實的分配
            if (
                isinstance(repeating_gpu_layers, int)
                and repeating_gpu_layers >= 0
                and output_layer_on_gpu
                and total_layers >= 2
                and repeating_gpu_layers + 1 == gpu_layers
            ):
                rep = min(repeating_gpu_layers, total_layers - 1)
                layer_lines = []
                if rep > 0:
                    layer_lines.append(f"Layers 0-{rep - 1} -> GPU")
                if rep <= total_layers - 2:
                    layer_lines.append(f"Layers {rep}-{total_layers - 2} -> CPU")
                layer_lines.append(f"Layers {total_layers - 1}-{total_layers - 1} -> GPU")
            else:
                # 回退：使用相同的連續切分表示
                layer_lines = [
                    f"Layers 0-{gpu_layers - 1} -> GPU",
                    f"Layers {gpu_layers}-{total_layers - 1} -> CPU",
                ]

        return {
            "device": device_label,
            "device_map_summary": device_summary,
            "total_modules": total_layers,
            "layer_lines": layer_lines,
            "memory_usage": memory_usage or None,
        }

    def _start_managed_server(self) -> None:
        if self.config is None:
            raise RuntimeError("Engine config not initialized")

        model_file = self._resolve_model_file()
        if not model_file:
            raise RuntimeError("llama-server auto-start requires model_path (or model_name as local gguf path)")

        if not os.path.exists(model_file):
            raise RuntimeError(f"llama model file not found: {model_file}")

        binary = (self.config.llama_server_binary or LLAMA_SERVER_BINARY).strip()
        if not os.path.isfile(binary):
            raise RuntimeError(f"llama-server binary not found: {binary}")
        if not os.access(binary, os.X_OK):
            raise RuntimeError(f"llama-server binary is not executable: {binary}")

        binary_dir = os.path.dirname(binary) or "."

        # 某些環境下子程序找不到同目錄/相鄰 lib，補上 LD_LIBRARY_PATH 可提高穩定性
        env = os.environ.copy()
        ld_paths: List[str] = []
        if os.path.isdir(binary_dir):
            ld_paths.append(binary_dir)
            maybe_lib_dir = os.path.abspath(os.path.join(binary_dir, "..", "lib"))
            if os.path.isdir(maybe_lib_dir):
                ld_paths.append(maybe_lib_dir)

        old_ld = env.get("LD_LIBRARY_PATH", "")
        if ld_paths:
            prefix = ":".join(ld_paths)
            env["LD_LIBRARY_PATH"] = f"{prefix}:{old_ld}" if old_ld else prefix

        host = self.config.llama_server_host
        port = self.config.llama_server_port

        cmd: List[str] = [
            binary,
            "-m", model_file,
            "--host", str(host),
            "--port", str(port),
            "-np", str(self.config.llama_server_np),
            "-c", str(self.config.n_ctx),
            "-b", str(self.config.n_batch),
            "--alias", "trusta-ast-default",
        ]

        if self.config.n_gpu_layers is not None:
            cmd.extend(["-ngl", str(self.config.n_gpu_layers)])

        extra_args = [str(x) for x in (self.config.llama_server_extra_args or []) if str(x).strip()]
        # has_cache_type_k_arg = any(arg == "--cache-type-k" for arg in extra_args)
        # has_cache_type_v_arg = any(arg == "--cache-type-v" for arg in extra_args)
        has_slot_save_path_arg = any(arg == "--slot-save-path" for arg in extra_args)

        # if not has_cache_type_k_arg:
        #     extra_args.extend(["--cache-type-k", "q8_0"])

        # if not has_cache_type_v_arg:
        #     extra_args.extend(["--cache-type-v", "q8_0"])

        if not has_slot_save_path_arg:
            slot_save_path = self._resolve_slot_save_path(model_file)
            if slot_save_path:
                extra_args.extend(["--slot-save-path", slot_save_path])
                logger.info(f"[LlamaServer] Auto-enabled slot save path: {slot_save_path}")

        mmproj_path = self._discover_mmproj_file(model_file, extra_args)
        has_mmproj_arg = any(arg == "--mmproj" for arg in extra_args)

        if mmproj_path and not has_mmproj_arg:
            if not os.path.isfile(mmproj_path):
                raise RuntimeError(f"llama mmproj file not found: {mmproj_path}")
            self._detected_mmproj_path = mmproj_path
            cmd.extend(["--mmproj", mmproj_path])

        if extra_args:
            cmd.extend(extra_args)

        logger.info(f"[LlamaServer] Starting managed subprocess: {' '.join(cmd)}")
        self.last_start_error = None
        with self._stderr_buffer_lock:
            self._stderr_recent_lines.clear()
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=binary_dir if os.path.isdir(binary_dir) else None,
            env=env,
        )
        self.managed_process = True

        # 啟動日誌轉發執行緒，避免 PIPE 堵塞並將 llama-server log 顯示到 service log
        self._stdout_pump_thread = threading.Thread(
            target=self._pump_server_logs,
            args=(self.server_process.stdout, False),
            daemon=True,
        )
        self._stderr_pump_thread = threading.Thread(
            target=self._pump_server_logs,
            args=(self.server_process.stderr, True),
            daemon=True,
        )
        self._stdout_pump_thread.start()
        self._stderr_pump_thread.start()

    def _stop_managed_server(self) -> None:
        if not self.managed_process or self.server_process is None:
            return

        proc = self.server_process
        self.server_process = None
        self.managed_process = False

        if proc.poll() is not None:
            return

        try:
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("[LlamaServer] managed subprocess terminated")
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
                logger.info("[LlamaServer] managed subprocess killed")
            except Exception as e:
                logger.warning(f"[LlamaServer] failed to stop managed subprocess: {e}")

    def load_model(self, config: InferenceConfig):
        self.config = config
        if config.llama_server_url and config.llama_server_url.strip():
            self.base_url = self._normalize_base_url(config.llama_server_url)
        else:
            self.base_url = self._normalize_base_url(f"http://{config.llama_server_host}:{config.llama_server_port}")
        self.api_key = config.llama_server_api_key or LLAMA_SERVER_API_KEY
        self.timeout_sec = config.llama_server_timeout or LLAMA_SERVER_TIMEOUT
        self.served_model_name = None
        self._recreate_client()

        logger.info(f"[Worker] Initializing llama-server engine: {self.base_url}")

        # 每次 load 先確保沒有遺留子程序
        self._stop_managed_server()

        # 動態模式：由引擎啟動/管理子程序
        if config.llama_server_auto_start:
            self._start_managed_server()

        # 等待服務可用（無論是自啟或外部既有服務）
        if not self._wait_server_ready(timeout_sec=config.llama_server_health_timeout):
            self._stop_managed_server()
            detail = f"; startup_error={self.last_start_error}" if self.last_start_error else ""
            raise RuntimeError(
                f"Failed to connect llama-server at {self.base_url} within {config.llama_server_health_timeout}s{detail}"
            )

        self.served_model_name = self._resolve_served_model_name(timeout_sec=5.0)
        capabilities = self._refresh_served_model_capabilities()
        slot_restore_summary = {"restored": 0, "total": 0, "slots": []}

        if self._is_multimodal_model():
            self._prefill_strategy = "cache_prompt"
            logger.info("[LlamaServer] multimodal model detected, using cache_prompt route")
        else:
            self._prefill_strategy = "slot"
            slot_restore_summary = self._restore_slot_caches()
            logger.info(
                f"[LlamaServer] text model detected, using slot route; restore summary={slot_restore_summary}"
            )

        allocation = self._collect_layer_allocation_from_logs()

        self.status_queue.put({
            "status": "ready",
            "config": config.model_dump(),
            "device": allocation.get("device"),
            "device_map_summary": allocation.get("device_map_summary"),
            "total_modules": allocation.get("total_modules"),
            "layer_lines": allocation.get("layer_lines"),
            "memory_usage": allocation.get("memory_usage"),
            "llama_capabilities": capabilities,
            "prefill_strategy": self._prefill_strategy,
            "slot_restore_summary": slot_restore_summary,
        })

    def generate(self, request: Dict[str, Any]):
        handle_llama_server_generate(self.prepare_request_for_generation(request), self)

    def generate_stream(self, request: Dict[str, Any]):
        handle_llama_server_stream(self.prepare_request_for_generation(request), self)

    def unload(self):
        if not self._is_multimodal_model():
            slot_save_summary = self._save_slot_caches()
            logger.info(f"[LlamaServer] slot save summary before unload: {slot_save_summary}")

        # 若為自啟模式，這裡會停止子程序
        self._stop_managed_server()
        self.client = None
        self.served_model_name = None
        self._served_model_capabilities = []
        self._prefill_strategy = "cache_prompt"
        self._detected_mmproj_path = None
        self.config = None
        self.stop_generation_flag.clear()
        self.status_queue.put({
            "status": "unloaded",
            "message": "llama-server engine unloaded",
        })

    def apply_chat_template(self, request: Dict[str, Any]):
        # llama-server 直接接收 messages，無需本地 tokenizer template
        request_id = request.get("request_id")
        messages = request.get("messages", [])
        try:
            self.data_queue.put({
                "type": "result",
                "request_id": request_id,
                "result": messages,
            })
        except Exception as e:
            self.data_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": str(e),
            })

    def cleanup_generation_memory(self, request: Optional[Dict[str, Any]] = None):
        try:
            slot = None
            if isinstance(request, dict):
                slot_val = request.get("slot")
                if isinstance(slot_val, int):
                    slot = slot_val
                elif isinstance(slot_val, str) and slot_val.isdigit():
                    slot = int(slot_val)

            summary = {
                "cleared": 0,
                "total": 1 if slot is not None else 0,
                "slots": [slot] if slot is not None else [],
                "message": (
                    "cleanup skipped: llama-server handles KV cache automatically by slot/prefix matching"
                ),
            }
            logger.info("[LlamaServer] cleanup_generation_memory skipped (auto slot/prefix cache management)")

            self.data_queue.put({
                "type": "cleanup",
                "result": summary,
            })
        except Exception as e:
            self.data_queue.put({
                "type": "cleanup",
                "result": {
                    "cleared": 0,
                    "total": 0,
                    "slots": [],
                    "message": f"cleanup failed: {e}",
                },
            })
