"""
System resources monitoring utility
Consolidates CPU, GPU, Memory, and Disk monitoring logic.
Supports stateful speed calculation for Disk I/O.
"""

import ctypes
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, cast

from ..config_models import (
    CPUInfo,
    DiskDevice,
    DiskMount,
    DiskResource,
    GPUInfo,
    GPUResource,
    MemoryInfo,
)

logger = logging.getLogger(__name__)

PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", ctypes.c_uint),
        ("doubleValue", ctypes.c_double),
    ]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PDH_FMT_COUNTERVALUE),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_longlong),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", ctypes.c_uint32),
        ("HighPart", ctypes.c_int32),
    ]


class _D3DKMT_QUERYVIDEOMEMORYINFO(ctypes.Structure):
    _fields_ = [
        ("AdapterLuid", _LUID),
        ("NodeOrdinal", ctypes.c_uint32),
        ("PhysicalAdapterIndex", ctypes.c_uint32),
        ("MemorySegmentGroup", ctypes.c_uint32),
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
    ]


def _bytes_to_gb(x: int) -> float:
    return round(x / (1024**3), 2)


class SystemMonitor:
    """Singleton monitor for CPU, GPU, memory, and disk resources."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "SystemMonitor":
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.last_io_time = time.time()
        self.last_disk_io = {}  # per device

        # Initialize last_disk_io to avoid huge spikes on first call
        try:
            import psutil

            self.last_disk_io = psutil.disk_io_counters(perdisk=True)
        except Exception:
            pass

        self._skip_fs_types = {
            "proc",
            "sysfs",
            "devtmpfs",
            "tmpfs",
            "cgroup",
            "cgroup2",
            "overlay",
            "squashfs",
            "ramfs",
            "autofs",
            "debugfs",
            "tracefs",
            "configfs",
            "devpts",
            "securityfs",
            "pstore",
            "binfmt_misc",
            "fusectl",
            "fuse.lxcfs",
        }

        # Windows GPU (PDH wildcard) monitor state
        self._igpu_pdh: Any = None  # ctypes WinDLL (pdh.dll) handle, Windows-only
        self._igpu_query: ctypes.c_void_p | None = None
        self._igpu_engine_counter: ctypes.c_void_p | None = None
        self._igpu_shared_counter: ctypes.c_void_p | None = None
        self._igpu_dedicated_counter: ctypes.c_void_p | None = None

        self._windows_cpu_ram_spec_cache = None
        self._windows_disk_spec_cache = None

        self._initialized = True

    @staticmethod
    def _parse_nspid_status_content(status_content: str) -> set[int]:
        """Parse all visible PID namespace values from /proc/<pid>/status."""
        pid_values: set[int] = set()
        for line in status_content.splitlines():
            if not line.startswith("NSpid:"):
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                return pid_values
            for raw_value in parts[1].split():
                try:
                    value = int(raw_value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    pid_values.add(value)
            break
        return pid_values

    def _get_pid_namespace_candidates(self, pid: int) -> set[int]:
        """Return all known PID namespace identities for a Linux process."""
        candidates = {pid} if pid > 0 else set()
        if os.name != "posix" or pid <= 0:
            return candidates

        status_path = f"/proc/{pid}/status"
        try:
            with open(status_path, encoding="utf-8") as status_file:
                status_content = status_file.read()
        except OSError:
            return candidates

        candidates.update(self._parse_nspid_status_content(status_content))
        return candidates

    def _expand_process_identity_set(self, pids: set[int]) -> set[int]:
        """Expand container PIDs to include host-visible PID namespace aliases."""
        expanded: set[int] = set(pids)
        for pid in pids:
            expanded.update(self._get_pid_namespace_candidates(pid))
        return expanded

    def _get_process_tree_pid_candidates(self) -> set[int]:
        """Return current process-tree PIDs with host-visible namespace aliases."""
        pids = set()
        try:
            import psutil

            pids.add(os.getpid())
            for child in psutil.Process().children(recursive=True):
                pids.add(child.pid)
        except Exception:
            pids.add(os.getpid())
        return self._expand_process_identity_set(pids)

    @staticmethod
    def _get_visible_gpu_tokens() -> list[str]:
        """Return visible GPU tokens from CUDA/NVIDIA environment variables."""
        raw_value = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES") or ""
        raw_value = raw_value.strip()
        if not raw_value or raw_value.lower() in {"all", "void", "none"}:
            return []
        return [token.strip() for token in raw_value.split(",") if token.strip()]

    def _get_torch_physical_gpu_index_map(
        self, physical_gpu_uuids: dict[int, str] | None = None
    ) -> dict[int, int]:
        """Map torch logical device indices to physical GPU indices."""
        visible_tokens = self._get_visible_gpu_tokens()
        if not visible_tokens:
            return {}

        logical_to_physical: dict[int, int] = {}
        normalized_uuid_map = {
            str(uuid_value).strip().lower(): index
            for index, uuid_value in (physical_gpu_uuids or {}).items()
            if uuid_value is not None
        }

        for logical_index, token in enumerate(visible_tokens):
            if token.isdigit():
                logical_to_physical[logical_index] = int(token)
                continue

            normalized_token = token.lower()
            if normalized_token.startswith("gpu-"):
                physical_index = normalized_uuid_map.get(normalized_token)
                if physical_index is not None:
                    logical_to_physical[logical_index] = physical_index

        return logical_to_physical

    def _get_torch_cuda_process_memory_map(
        self, physical_gpu_uuids: dict[int, str] | None = None
    ) -> dict[int, int]:
        """
        Best-effort current-process CUDA memory by device, in bytes.

        Uses torch allocator stats as a fallback when NVML/nvidia-smi per-process
        accounting is unavailable inside containers.
        """
        try:
            import torch
        except Exception:
            return {}

        try:
            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return {}
        except Exception:
            return {}

        memory_by_index: dict[int, int] = {}
        logical_to_physical = self._get_torch_physical_gpu_index_map(physical_gpu_uuids)
        try:
            device_count = int(torch.cuda.device_count() or 0)
        except Exception:
            return {}

        for index in range(device_count):
            try:
                reserved = int(torch.cuda.memory_reserved(index) or 0)
            except Exception:
                reserved = 0
            try:
                allocated = int(torch.cuda.memory_allocated(index) or 0)
            except Exception:
                allocated = 0

            used_bytes = max(reserved, allocated)
            if used_bytes > 0:
                physical_index = logical_to_physical.get(index, index)
                memory_by_index[physical_index] = max(
                    memory_by_index.get(physical_index, 0), used_bytes
                )

        return memory_by_index

    @staticmethod
    def _get_nvml_gpu_uuid_map(pynvml_module: Any, count: int) -> dict[int, str]:  # noqa: ANN401 - pynvml module handle, duck-typed
        """Return physical GPU index to UUID map from NVML."""
        uuid_map = {}
        for index in range(count):
            try:
                handle = pynvml_module.nvmlDeviceGetHandleByIndex(index)
                uuid_bytes = pynvml_module.nvmlDeviceGetUUID(handle)
                uuid_map[index] = (
                    uuid_bytes.decode("utf-8") if isinstance(uuid_bytes, bytes) else str(uuid_bytes)
                )
            except Exception:
                continue
        return uuid_map

    @staticmethod
    def _get_nvidia_smi_gpu_uuid_map() -> dict[int, str]:
        """Return physical GPU index to UUID map from nvidia-smi."""
        uuid_map = {}
        try:
            u_cmd = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
            u_out = subprocess.check_output(u_cmd, stderr=subprocess.STDOUT, text=True)
        except Exception:
            return uuid_map

        for line in u_out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                uuid_map[int(parts[0])] = parts[1]
            except (TypeError, ValueError):
                continue
        return uuid_map

    def _init_windows_igpu_usage_monitor(self) -> bool:
        """Initialize Windows PDH GPU monitoring (wildcard counter enumeration, same source as Task Manager)."""
        if os.name != "nt":
            return False
        if self._igpu_pdh is not None and self._igpu_query is not None:
            return True

        try:
            pdh = ctypes.windll.pdh
            query = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return False

            def _add_counter(path: str) -> ctypes.c_void_p | None:
                counter = ctypes.c_void_p()
                ret = pdh.PdhAddEnglishCounterW(query, path, 0, ctypes.byref(counter))
                if ret == 0:
                    return counter
                return None

            # Use a wildcard counter directly; all instances are read later via PdhGetFormattedCounterArrayW
            engine_counter = _add_counter(r"\GPU Engine(*)\Utilization Percentage")
            shared_counter = _add_counter(r"\GPU Adapter Memory(*)\Shared Usage")
            dedicated_counter = _add_counter(r"\GPU Adapter Memory(*)\Dedicated Usage")

            if engine_counter is None and shared_counter is None and dedicated_counter is None:
                try:
                    pdh.PdhCloseQuery(query)
                except Exception:
                    pass
                return False

            # prime counters
            pdh.PdhCollectQueryData(query)
            time.sleep(0.2)
            pdh.PdhCollectQueryData(query)

            self._igpu_pdh = pdh
            self._igpu_query = query
            self._igpu_engine_counter = engine_counter
            self._igpu_shared_counter = shared_counter
            self._igpu_dedicated_counter = dedicated_counter
            return True
        except Exception:
            return False

    def _read_pdh_counter_array(self, counter: ctypes.c_void_p | None) -> list[tuple[str, float]]:
        """Read every instance value of a wildcard counter, returning [(instance_name, value), ...]."""
        if counter is None:
            return []
        try:
            buf_size = ctypes.c_ulong(0)
            item_count = ctypes.c_ulong(0)
            ret = self._igpu_pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_DOUBLE,
                ctypes.byref(buf_size),
                ctypes.byref(item_count),
                None,
            )

            if (ret & 0xFFFFFFFF) not in (0, PDH_MORE_DATA):
                return []

            if buf_size.value <= 0 or item_count.value <= 0:
                return []

            raw_buf = ctypes.create_string_buffer(buf_size.value)
            ret2 = self._igpu_pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_DOUBLE,
                ctypes.byref(buf_size),
                ctypes.byref(item_count),
                ctypes.cast(raw_buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W)),
            )
            if (ret2 & 0xFFFFFFFF) != 0:
                return []

            arr_ptr = ctypes.cast(raw_buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
            out = []
            for i in range(int(item_count.value)):
                item = arr_ptr[i]
                try:
                    name = str(item.szName or "")
                    val = float(item.FmtValue.doubleValue)
                    out.append((name, val))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _get_windows_gpu_pdh_snapshot(self) -> dict[str, dict[str, float]]:
        """Return util/shared/dedicated per LUID (same source as Task Manager)."""
        if os.name != "nt":
            return {}
        if not self._init_windows_igpu_usage_monitor():
            return {}

        try:
            self._igpu_pdh.PdhCollectQueryData(self._igpu_query)
        except Exception:
            return {}

        by_luid = {}

        def _get_luid_record(instance_name: str) -> dict[str, float]:
            m = re.search(r"(luid_0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)", instance_name)
            luid = m.group(1).lower() if m else "unknown"
            return by_luid.setdefault(
                luid, {"util_percent": 0.0, "shared_used_bytes": 0.0, "dedicated_used_bytes": 0.0}
            )

        # GPU Util: take the max engine utilization per LUID (the busiest-engine logic Task Manager uses)
        for name, v in self._read_pdh_counter_array(self._igpu_engine_counter):
            if v is not None:
                cur = _get_luid_record(name)
                cur["util_percent"] = max(cur["util_percent"], min(100.0, v))

        # Shared/Dedicated Usage: Bytes
        for name, v in self._read_pdh_counter_array(self._igpu_shared_counter):
            if v is not None:
                _get_luid_record(name)["shared_used_bytes"] += max(0.0, v)

        for name, v in self._read_pdh_counter_array(self._igpu_dedicated_counter):
            if v is not None:
                _get_luid_record(name)["dedicated_used_bytes"] += max(0.0, v)

        return by_luid

    def _get_windows_igpu_util_percent(self) -> float | None:
        """Read Windows Intel iGPU utilization (%)."""
        snap = self._get_windows_gpu_pdh_snapshot()
        if not snap:
            return None
        try:
            return round(max(v.get("util_percent", 0.0) for v in snap.values()), 2)
        except Exception:
            return None

    def _get_windows_igpu_memory_usage_bytes(self) -> tuple[int | None, int | None]:
        """Return (shared_bytes, dedicated_bytes)."""
        snap = self._get_windows_gpu_pdh_snapshot()
        if not snap:
            return None, None
        try:
            shared = sum(float(v.get("shared_used_bytes", 0.0)) for v in snap.values())
            dedicated = sum(float(v.get("dedicated_used_bytes", 0.0)) for v in snap.values())
            return int(max(0.0, shared)), int(max(0.0, dedicated))
        except Exception:
            return None, None

    def _normalize_windows_luid(self, luid_value: int | None) -> str:
        try:
            if luid_value is None:
                return "unknown"
            raw = int(luid_value) & 0xFFFFFFFFFFFFFFFF
            low = raw & 0xFFFFFFFF
            high = (raw >> 32) & 0xFFFFFFFF
            return f"luid_0x{high:08x}_0x{low:08x}".lower()
        except Exception:
            return "unknown"

    def _split_windows_luid(self, luid_value: int) -> "_LUID | None":
        try:
            raw = int(luid_value) & 0xFFFFFFFFFFFFFFFF
            low = raw & 0xFFFFFFFF
            high = (raw >> 32) & 0xFFFFFFFF
            if high >= 0x80000000:
                high -= 0x100000000
            return _LUID(low, high)
        except Exception:
            return None

    def _get_windows_video_memory_usage_bytes(self, luid_value: int) -> int | None:
        """Get the current adapter's live video memory usage via D3DKMT (local + non-local)."""
        if os.name != "nt":
            return None

        luid = self._split_windows_luid(luid_value)
        if luid is None:
            return None

        try:
            gdi32 = ctypes.windll.gdi32
            query_fn = getattr(gdi32, "D3DKMTQueryVideoMemoryInfo", None)
            if query_fn is None:
                return None

            query_fn.argtypes = [ctypes.POINTER(_D3DKMT_QUERYVIDEOMEMORYINFO)]
            query_fn.restype = ctypes.c_long

            total_used = 0
            for segment_group in (0, 1):
                query = _D3DKMT_QUERYVIDEOMEMORYINFO()
                query.AdapterLuid = luid
                query.NodeOrdinal = 0
                query.PhysicalAdapterIndex = 0
                query.MemorySegmentGroup = segment_group
                status = query_fn(ctypes.byref(query))
                if status == 0:
                    total_used += int(query.CurrentUsage or 0)

            return max(0, total_used)
        except Exception:
            return None

    def _get_windows_dxgi_adapters(self) -> list[dict[str, Any]]:
        """Enumerate Windows DXGI adapters and return the system-level GPU list."""
        if os.name != "nt":
            return []

        adapters = []
        try:
            dxgi = ctypes.windll.dxgi
            create_factory = dxgi.CreateDXGIFactory1
            create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]

            iid_factory1 = _GUID(
                0x770AAE78,
                0xF26F,
                0x4DBA,
                (0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
            )

            factory = ctypes.c_void_p()
            if (
                create_factory(ctypes.byref(iid_factory1), ctypes.byref(factory)) != 0
                or not factory.value
            ):
                return []

            enum_adapters_type = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
            )
            get_desc_type = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC)
            )
            release_type = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)

            p_factory = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
            enum_adapters = ctypes.cast(p_factory[0][12], enum_adapters_type)
            release_factory = ctypes.cast(p_factory[0][2], release_type)

            try:
                idx = 0
                adapter = ctypes.c_void_p()
                while enum_adapters(factory, idx, ctypes.byref(adapter)) == 0:
                    p_adapter = None
                    try:
                        p_adapter = ctypes.cast(
                            adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
                        )
                        get_desc = ctypes.cast(p_adapter[0][8], get_desc_type)
                        release_adapter = ctypes.cast(p_adapter[0][2], release_type)

                        desc = _DXGI_ADAPTER_DESC()
                        if get_desc(adapter, ctypes.byref(desc)) == 0:
                            name = str(desc.Description or "").strip()
                            if name:
                                dedicated_bytes = int(getattr(desc, "DedicatedVideoMemory", 0) or 0)
                                dedicated_system_bytes = int(
                                    getattr(desc, "DedicatedSystemMemory", 0) or 0
                                )
                                shared_bytes = int(getattr(desc, "SharedSystemMemory", 0) or 0)
                                adapters.append(
                                    {
                                        "index": idx,
                                        "name": name,
                                        "vendor_id": int(getattr(desc, "VendorId", 0) or 0),
                                        "dedicated_video_bytes": dedicated_bytes,
                                        "dedicated_system_bytes": dedicated_system_bytes,
                                        "shared_system_bytes": shared_bytes,
                                        "total_taskmgr_bytes": max(
                                            0, dedicated_bytes + shared_bytes
                                        ),
                                        "raw_luid": int(getattr(desc, "AdapterLuid", 0) or 0),
                                        "luid": self._normalize_windows_luid(
                                            getattr(desc, "AdapterLuid", 0)
                                        ),
                                    }
                                )
                    finally:
                        if p_adapter is not None:
                            try:
                                release_adapter(adapter)
                            except Exception:
                                pass
                    idx += 1
            finally:
                try:
                    release_factory(factory)
                except Exception:
                    pass
        except Exception:
            return []

        return adapters

    def _get_windows_dxgi_igpu_total_memory(self) -> int | None:
        """Get the Intel iGPU Task Manager max available memory via DXGI (Shared + Dedicated)."""
        for adapter in self._get_windows_dxgi_adapters():
            if "intel" in str(adapter.get("name", "")).lower():
                total_bytes = int(adapter.get("total_taskmgr_bytes", 0) or 0)
                if total_bytes > 0:
                    return total_bytes
        return None

    def _get_windows_memory_cached_bytes(self) -> int:
        try:
            import ctypes
            from ctypes import wintypes

            class PDH_FMT_COUNTERVALUE_LARGE(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

            # Windows-only API; this branch has no os.name guard pyright can narrow on
            pdh = ctypes.windll.pdh  # pyright: ignore[reportAttributeAccessIssue]
            q = wintypes.HANDLE()
            pdh.PdhOpenQueryW(None, 0, ctypes.byref(q))
            c1, c2, c3 = wintypes.HANDLE(), wintypes.HANDLE(), wintypes.HANDLE()
            pdh.PdhAddEnglishCounterW(
                q, r"\Memory\Standby Cache Reserve Bytes", 0, ctypes.byref(c1)
            )
            pdh.PdhAddEnglishCounterW(
                q, r"\Memory\Standby Cache Normal Priority Bytes", 0, ctypes.byref(c2)
            )
            pdh.PdhAddEnglishCounterW(q, r"\Memory\Standby Cache Core Bytes", 0, ctypes.byref(c3))
            pdh.PdhCollectQueryData(q)

            val1 = PDH_FMT_COUNTERVALUE_LARGE()
            val2 = PDH_FMT_COUNTERVALUE_LARGE()
            val3 = PDH_FMT_COUNTERVALUE_LARGE()

            pdh.PdhGetFormattedCounterValue(c1, 0x00000400, None, ctypes.byref(val1))
            pdh.PdhGetFormattedCounterValue(c2, 0x00000400, None, ctypes.byref(val2))
            pdh.PdhGetFormattedCounterValue(c3, 0x00000400, None, ctypes.byref(val3))

            cached_bytes = val1.largeValue + val2.largeValue + val3.largeValue
            pdh.PdhCloseQuery(q)
            return max(0, cached_bytes)
        except Exception:
            return 0

    def _close_windows_igpu_usage_monitor(self) -> None:
        if os.name != "nt":
            return
        try:
            if self._igpu_pdh is not None and self._igpu_query is not None:
                self._igpu_pdh.PdhCloseQuery(self._igpu_query)
        except Exception:
            pass
        finally:
            self._igpu_query = None
            self._igpu_engine_counter = None
            self._igpu_shared_counter = None
            self._igpu_dedicated_counter = None
            self._igpu_pdh = None

    def __del__(self) -> None:
        try:
            self._close_windows_igpu_usage_monitor()
        except Exception:
            pass

    def _get_xpu_memory_info(
        self,
        xpu_backend: Any,  # noqa: ANN401 - torch.xpu module varies across versions
        device_idx: int,
    ) -> tuple[int | None, int | None]:
        """Safely get XPU memory info, compatible with different torch.xpu API versions."""
        free_bytes = None
        total_bytes = None

        # 1) Prefer mem_get_info (argument types differ across versions)
        try:
            mem_get_info = getattr(xpu_backend, "mem_get_info", None)
            if callable(mem_get_info):
                for arg in (device_idx, f"xpu:{device_idx}"):
                    try:
                        # torch.xpu is duck-typed (Any); pin the tuple shape
                        free_bytes, total_bytes = cast(tuple[Any, Any], mem_get_info(arg))
                        if total_bytes is not None:
                            return free_bytes, total_bytes
                    except TypeError:
                        continue
                    except Exception:
                        break

                # Some versions only support the current device and take no arguments
                try:
                    if hasattr(xpu_backend, "device"):
                        with xpu_backend.device(device_idx):
                            free_bytes, total_bytes = cast(tuple[Any, Any], mem_get_info())
                    else:
                        free_bytes, total_bytes = cast(tuple[Any, Any], mem_get_info())
                    if total_bytes is not None:
                        return free_bytes, total_bytes
                except Exception:
                    pass
        except Exception:
            pass

        # 2) fallback: read total_memory from get_device_properties
        try:
            props = xpu_backend.get_device_properties(device_idx)
            total_bytes = getattr(props, "total_memory", None)
        except Exception:
            total_bytes = None

        # 3) fallback: estimate used from memory_stats, then derive free
        if total_bytes is not None:
            try:
                memory_stats = getattr(xpu_backend, "memory_stats", None)
                if callable(memory_stats):
                    stats: Any = memory_stats(device_idx)  # torch.xpu dict, duck-typed
                    used_bytes = stats.get("allocated_bytes.all.current")
                    if used_bytes is None:
                        used_bytes = stats.get("active_bytes.all.current")
                    if isinstance(used_bytes, (int, float)):
                        free_bytes = max(0, int(total_bytes) - int(used_bytes))
            except Exception:
                pass

        return free_bytes, total_bytes

    def _read_windows_cpu_ram_specs(self) -> dict[str, Any]:
        if self._windows_cpu_ram_spec_cache is not None:
            return self._windows_cpu_ram_spec_cache
        res: dict[str, Any] = {
            "model": None,
            "speed_mhz": None,
            "type": None,
            "manufacturer": None,
        }
        try:
            cmd = (
                "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1;"
                "$mem = Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1;"
                "@{Model=$cpu.Name; Speed=$mem.Speed; SMBIOS=$mem.SMBIOSMemoryType; "
                "Type=$mem.MemoryType; Mfg=$mem.Manufacturer} | ConvertTo-Json -Compress"
            )
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,  # pyright: ignore[reportAttributeAccessIssue]  # Windows-only flag
            )
            import json

            data = json.loads(out)
            res["model"] = str(data.get("Model", "")).strip() or None

            s = data.get("Speed")
            if s and str(s).isdigit():
                res["speed_mhz"] = int(s)

            mtype = str(data.get("SMBIOS", ""))
            if not mtype or mtype == "0" or mtype == "null":
                mtype = str(data.get("Type", ""))

            mem_type_map = {
                "20": "DDR",
                "21": "DDR2",
                "24": "DDR3",
                "26": "DDR4",
                "34": "DDR5",
                "35": "LPDDR5",
            }
            if mtype in mem_type_map:
                res["type"] = mem_type_map[mtype]
            else:
                res["type"] = "Unknown"

            res["manufacturer"] = str(data.get("Mfg", "")).strip() or None
        except Exception:
            pass
        self._windows_cpu_ram_spec_cache = res
        return res

    def get_cpu_resource(self, mode: str, force_by_process: bool = False) -> CPUInfo:
        """Get CPU resources (Spec and Usage combined)."""
        # Initialize containers
        cpu_data = {}
        memory_data = {}

        if mode == "spec":
            # 1. CPU Spec
            try:
                import psutil  # type: ignore

                cpu_data["cores"] = psutil.cpu_count(logical=False)
                cpu_data["threads"] = psutil.cpu_count(logical=True)
                try:
                    freq = psutil.cpu_freq()
                    if freq:
                        cpu_data["max_frequency_mhz"] = round(freq.max, 2) if freq.max else None
                except Exception:
                    pass
            except Exception:
                pass

            # Architecture & Model
            try:
                import platform

                cpu_data["architecture"] = platform.machine()
                if platform.system() == "Windows":
                    win_specs = self._read_windows_cpu_ram_specs()
                    if win_specs.get("model"):
                        cpu_data["model"] = win_specs["model"]
                else:
                    # Try reading from /proc/cpuinfo
                    try:
                        with open("/proc/cpuinfo") as f:
                            for line in f:
                                if line.startswith("model name"):
                                    cpu_data["model"] = line.split(":", 1)[1].strip()
                                    break
                    except Exception:
                        pass

                    # Fallback lscpu
                    if not cpu_data.get("model"):
                        try:
                            lscpu_out = subprocess.check_output(["lscpu"], text=True)
                            for line in lscpu_out.splitlines():
                                if "Model name:" in line:
                                    cpu_data["model"] = line.split(":", 1)[1].strip()
                        except Exception:
                            pass
            except Exception:
                pass

            # DRAM Spec
            try:
                import psutil  # type: ignore

                vm = psutil.virtual_memory()
                memory_data["total_gb"] = _bytes_to_gb(vm.total)
            except Exception:
                pass

            # dmidecode for DRAM modules (Linux), or CIM for Windows
            try:
                import platform

                if platform.system() == "Windows":
                    win_specs = self._read_windows_cpu_ram_specs()
                    if win_specs.get("speed_mhz"):
                        memory_data["speed_mhz"] = win_specs["speed_mhz"]
                    if win_specs.get("type"):
                        memory_data["type"] = win_specs["type"]

                    memory_data["modules"] = [
                        {
                            "size": f"{memory_data.get('total_gb', 0)} GB",
                            "speed_mhz": memory_data.get("speed_mhz"),
                            "type": memory_data.get("type", "Unknown"),
                            "manufacturer": win_specs.get("manufacturer"),
                        }
                    ]
                else:
                    # Run dmidecode via sudo with -n (non-interactive) so it NEVER
                    # blocks waiting for a password on the controlling terminal, which
                    # would freeze the service. run_service.sh pre-authorizes sudo at
                    # startup so the credential is cached by the time we reach here; if
                    # it is not, sudo -n simply fails and we fall back to the note below.
                    # For a fully password-free setup, add a sudoers rule:
                    #   <service_user> ALL=(root) NOPASSWD: /usr/sbin/dmidecode -t memory
                    if os.geteuid() == 0:  # pyright: ignore[reportAttributeAccessIssue]  # POSIX-only; this branch runs on Linux
                        cmd = ["dmidecode", "-t", "memory"]
                    else:
                        cmd = ["sudo", "-n", "dmidecode", "-t", "memory"]
                    input_str = None

                    result = subprocess.run(
                        cmd, input=input_str, text=True, capture_output=True, timeout=5
                    )
                    if result.returncode == 0:
                        current_module = {}
                        modules = []
                        # Parse dmidecode output
                        for line in result.stdout.splitlines():
                            line = line.strip()
                            if line.startswith("Memory Device"):
                                if current_module.get("size"):
                                    modules.append(current_module)
                                current_module = {}
                            elif (
                                line.startswith("Size:")
                                and "No Module" not in line
                                and "Not Installed" not in line
                            ):
                                current_module["size"] = line.split(":", 1)[1].strip()
                            elif line.startswith("Type:") and "Unknown" not in line:
                                t = line.split(":", 1)[1].strip()
                                current_module["type"] = t
                                if not memory_data.get("type"):
                                    memory_data["type"] = t
                            elif line.startswith("Speed:") and ("MHz" in line or "MT/s" in line):
                                try:
                                    s = int(line.split(":", 1)[1].strip().split()[0])
                                    current_module["speed_mhz"] = s
                                    if not memory_data.get("speed_mhz"):
                                        memory_data["speed_mhz"] = s
                                except Exception:
                                    pass
                            elif line.startswith("Manufacturer:") and "No Module" not in line:
                                current_module["manufacturer"] = line.split(":", 1)[1].strip()
                        if current_module.get("size"):
                            modules.append(current_module)
                        memory_data["modules"] = modules
                    else:
                        memory_data["note"] = "Detailed info requires root (dmidecode)"
            except Exception:
                memory_data["note"] = "DRAM detection error/permission denied"

        elif mode == "usage":
            import psutil

            # CPU Usage (blocking call for interval to get accurate snapshot)
            try:
                cpu_data["cpu_util_percent"] = psutil.cpu_percent(interval=0.1)
            except Exception:
                pass

            # Memory Usage
            try:
                vm = psutil.virtual_memory()
                # psutil 'buffers' + 'cached' usually represents file-backed cache (including mmapped models)
                # Note: available includes cache, so total - available excludes cache.
                # To show cache explicitly, we can use buffers + cached if available, or approx (available - free)

                cached_bytes = getattr(vm, "cached", 0) + getattr(vm, "buffers", 0)
                # Fallback if specific attrs missing
                if cached_bytes == 0 and hasattr(vm, "available") and hasattr(vm, "free"):
                    if os.name == "nt":
                        cached_bytes = self._get_windows_memory_cached_bytes()
                    else:
                        cached_bytes = max(0, vm.available - vm.free)

                available_bytes = max(0, int(getattr(vm, "available", 0) or 0))
                total_bytes = max(0, int(getattr(vm, "total", 0) or 0))
                system_used_bytes = max(0, total_bytes - available_bytes)

                # Use psutil.available for DRAM free to avoid double-subtracting under heavy
                # cache / paging, which made used drop once a model exceeded physical memory.
                real_free = available_bytes

                memory_data = {
                    "total_gb": _bytes_to_gb(total_bytes),
                    "used_gb": _bytes_to_gb(system_used_bytes),
                    "free_gb": _bytes_to_gb(real_free),
                    "system_used_gb": _bytes_to_gb(system_used_bytes),
                    "cached_gb": _bytes_to_gb(cached_bytes),
                    "percent": round((system_used_bytes / total_bytes) * 100, 2)
                    if total_bytes
                    else None,
                }

                # Service RSS (Main + Children) is only a breakdown reference, no longer the primary DRAM used metric.
                try:
                    proc = psutil.Process()
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    rss_gb = round(rss / 1024**3, 2)
                    if memory_data.get("used_gb") is not None:
                        memory_data["other_used_gb"] = round(
                            max(0.0, memory_data["used_gb"] - rss_gb), 2
                        )
                    else:
                        memory_data["other_used_gb"] = None

                    if force_by_process:
                        memory_data["used_gb"] = rss_gb
                        memory_data["system_used_gb"] = rss_gb
                        memory_data["free_gb"] = round(
                            max(0.0, memory_data["total_gb"] - rss_gb), 2
                        )
                        memory_data["percent"] = (
                            round((rss_gb / memory_data["total_gb"]) * 100, 2)
                            if memory_data["total_gb"]
                            else None
                        )
                except Exception:
                    pass

            except Exception as e:
                memory_data["note"] = f"Error: {e}"

        # Construct final objects
        mem_info = MemoryInfo(**memory_data)
        return CPUInfo(
            model=cpu_data.get("model"),
            cores=cpu_data.get("cores"),
            threads=cpu_data.get("threads"),
            architecture=cpu_data.get("architecture"),
            max_frequency_mhz=cpu_data.get("max_frequency_mhz"),
            # Usage
            cpu_util_percent=cpu_data.get("cpu_util_percent"),
            dram=mem_info,
        )

    def get_gpu_resource(self, mode: str, force_by_process: bool = False) -> GPUResource:
        """Get the GPU list (Spec and Usage unified)."""
        gpus = []
        seen_gpu_names = set()

        windows_igpu_util = None
        windows_igpu_shared_bytes = None
        windows_igpu_dedicated_bytes = None
        windows_pdh_snapshot = {}
        windows_dxgi_adapters = []
        if mode == "usage" and os.name == "nt":
            try:
                windows_igpu_util = self._get_windows_igpu_util_percent()
                windows_igpu_shared_bytes, windows_igpu_dedicated_bytes = (
                    self._get_windows_igpu_memory_usage_bytes()
                )
                windows_pdh_snapshot = self._get_windows_gpu_pdh_snapshot()
            except Exception:
                pass
        if os.name == "nt":
            try:
                windows_dxgi_adapters = self._get_windows_dxgi_adapters()
            except Exception:
                windows_dxgi_adapters = []

        # Find process-tree PIDs
        pid_candidates = set()
        torch_pid_memory = {}
        if force_by_process:
            pid_candidates = self._get_process_tree_pid_candidates()

        # Try NVML
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if force_by_process:
                physical_gpu_uuids = self._get_nvml_gpu_uuid_map(pynvml, count)
                torch_pid_memory = self._get_torch_cuda_process_memory_map(physical_gpu_uuids)
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name_bytes = pynvml.nvmlDeviceGetName(h)
                name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else name_bytes
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                # pynvml ctypes struct fields are runtime ints but loosely typed
                mem_total = cast(int, mem.total)
                mem_used = cast(int, mem.used)
                mem_free = cast(int, mem.free)

                temp = None
                gpu_util = None
                if mode == "usage":
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, 0)
                    except Exception:
                        pass
                    # Get GPU Utilization
                    try:
                        gpu_util = cast(int, pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    except Exception:
                        pass

                total_gb = _bytes_to_gb(mem_total)
                used_gb = None
                free_gb = None
                percent = None
                if mode == "usage":
                    if force_by_process:
                        pid_gpu_mem = {}
                        try:
                            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                                if p.pid in pid_candidates:
                                    pid_gpu_mem[p.pid] = p.usedGpuMemory
                        except Exception:
                            pass
                        try:
                            for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(h):
                                if p.pid in pid_candidates:
                                    pid_gpu_mem[p.pid] = max(
                                        pid_gpu_mem.get(p.pid, 0), p.usedGpuMemory
                                    )
                        except Exception:
                            pass
                        used_bytes = sum(pid_gpu_mem.values())
                        if used_bytes <= 0:
                            used_bytes = int(torch_pid_memory.get(i, 0) or 0)
                        used_gb = _bytes_to_gb(used_bytes)
                        free_gb = round(max(0.0, total_gb - used_gb), 2)
                        percent = round((used_bytes / mem_total) * 100, 1) if mem_total else 0.0
                    else:
                        used_gb = _bytes_to_gb(mem_used)
                        free_gb = _bytes_to_gb(mem_free)
                        percent = round((mem_used / mem_total) * 100, 1) if mem_total else 0.0

                gpus.append(
                    GPUInfo(
                        index=i,
                        name=name,
                        total_gb=total_gb,
                        used_gb=used_gb,
                        free_gb=free_gb,
                        percent=percent,
                        gpu_util=gpu_util,
                        temperature=temp,
                    )
                )
                seen_gpu_names.add(str(name).strip().lower())
            pynvml.nvmlShutdown()
        except Exception:
            # Try nvidia-smi fallback
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

                # Fetch nvidia-smi process list if force_by_process is True
                pid_gpu_mem_by_idx = {}
                if mode == "usage" and force_by_process:
                    physical_gpu_uuids = self._get_nvidia_smi_gpu_uuid_map()
                    torch_pid_memory = self._get_torch_cuda_process_memory_map(physical_gpu_uuids)
                    try:
                        p_cmd = [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_memory,gpu_uuid",
                            "--format=csv,noheader,nounits",
                        ]
                        p_out = subprocess.check_output(p_cmd, stderr=subprocess.STDOUT, text=True)
                        uuid_to_idx = {
                            uuid_value: index for index, uuid_value in physical_gpu_uuids.items()
                        }

                        for p_line in p_out.splitlines():
                            p_parts = [pp.strip() for pp in p_line.split(",")]
                            if len(p_parts) >= 3:
                                p_pid = int(p_parts[0])
                                p_mem = float(p_parts[1])  # in MB
                                p_uuid = p_parts[2]
                                if p_pid in pid_candidates:
                                    gpu_idx = uuid_to_idx.get(p_uuid, 0)
                                    pid_gpu_mem_by_idx[gpu_idx] = (
                                        pid_gpu_mem_by_idx.get(gpu_idx, 0.0) + p_mem
                                    )
                    except Exception:
                        pass

                for idx, line in enumerate(out.splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        name = parts[0]
                        total = float(parts[1]) / 1024
                        used, temp_val, gpu_util = None, None, None
                        if mode == "usage" and len(parts) >= 3:
                            if force_by_process:
                                used_mb = pid_gpu_mem_by_idx.get(idx, 0.0)
                                if used_mb <= 0:
                                    used_mb = (torch_pid_memory.get(idx, 0) or 0) / (1024**2)
                                used = used_mb / 1024
                            else:
                                used = float(parts[2]) / 1024

                            free = max(0.0, total - used)
                            percent = round(used / total * 100, 1) if total else 0.0

                            if len(parts) >= 4:
                                try:
                                    temp_val = float(parts[3])
                                except Exception:
                                    pass
                            if len(parts) >= 5:
                                try:
                                    gpu_util = float(parts[4])
                                except Exception:
                                    pass
                        gpus.append(
                            GPUInfo(
                                index=idx,
                                name=name,
                                total_gb=round(total, 2),
                                used_gb=round(used, 2) if used is not None else None,
                                free_gb=round(free, 2) if free is not None else None,
                                percent=percent,
                                gpu_util=gpu_util,
                                temperature=temp_val,
                            )
                        )
                        seen_gpu_names.add(str(name).strip().lower())
            except Exception:
                pass

        # Try Intel XPU (iGPU) via PyTorch oneAPI (Windows/Linux compatible)
        try:
            import torch  # type: ignore

            xpu_backend = getattr(torch, "xpu", None)
            xpu_available = False
            if xpu_backend is not None:
                try:
                    xpu_available = bool(xpu_backend.is_available())
                except Exception:
                    xpu_available = False

            # The `is not None` re-check is redundant at runtime (xpu_available
            # implies it) but lets the type checker narrow xpu_backend.
            if xpu_available and xpu_backend is not None:
                try:
                    xpu_count = int(xpu_backend.device_count())
                except Exception:
                    xpu_count = 1

                if xpu_count <= 0:
                    xpu_count = 1

                for i in range(xpu_count):
                    try:
                        name = xpu_backend.get_device_name(i)
                    except Exception:
                        name = f"Intel XPU {i}"

                    free_bytes, total_bytes = self._get_xpu_memory_info(xpu_backend, i)

                    # For Windows Intel iGPU, override with the more accurate Task Manager max memory capacity
                    if os.name == "nt" and "intel" in str(name).lower():
                        dxgi_total_bytes = self._get_windows_dxgi_igpu_total_memory()
                        if dxgi_total_bytes is not None and dxgi_total_bytes > 0:
                            total_bytes = dxgi_total_bytes

                    used_bytes = None
                    percent = None
                    used_candidates = []

                    # 1) Prefer total-free
                    if total_bytes is not None:
                        if free_bytes is not None:
                            used_candidates.append(max(0, int(total_bytes) - int(free_bytes)))

                    # 2) Supplement with torch.xpu allocator metrics (on some platforms mem_get_info does not reflect real model usage)
                    for fn_name in ("memory_allocated", "memory_reserved"):
                        try:
                            fn = getattr(xpu_backend, fn_name, None)
                            if not callable(fn):
                                continue

                            metric_val = None
                            for arg in (i, f"xpu:{i}", None):
                                try:
                                    metric_val = fn(arg) if arg is not None else fn()
                                    break
                                except TypeError:
                                    continue
                                except Exception:
                                    break

                            if isinstance(metric_val, (int, float)):
                                used_candidates.append(max(0, int(metric_val)))
                        except Exception:
                            pass

                    if used_candidates:
                        used_bytes = max(used_candidates)

                    if total_bytes is not None and used_bytes is not None:
                        # Shared-memory platforms may report more than total, so clamp percent
                        safe_used = (
                            min(int(used_bytes), int(total_bytes))
                            if total_bytes
                            else int(used_bytes)
                        )
                        percent = round((safe_used / total_bytes) * 100, 1) if total_bytes else None
                        if free_bytes is None:
                            free_bytes = max(0, int(total_bytes) - int(safe_used))

                    intel_gpu_util = None
                    if mode == "usage" and os.name == "nt" and "intel" in str(name).lower():
                        intel_gpu_util = windows_igpu_util

                        # Match the DXGI adapter by name, not by xpu index —
                        # DXGI enumeration order need not match torch.xpu order,
                        # so indexing by i can read an unrelated adapter's memory.
                        _dxgi_adapters = self._get_windows_dxgi_adapters()
                        _intel_adapter = next(
                            (
                                a
                                for a in _dxgi_adapters
                                if "intel" in str(a.get("name", "")).lower()
                            ),
                            None,
                        )
                        dxgi_used_bytes = (
                            self._get_windows_video_memory_usage_bytes(
                                cast(int, _intel_adapter.get("raw_luid"))
                            )
                            if _intel_adapter is not None
                            else None
                        )
                        if isinstance(dxgi_used_bytes, int) and dxgi_used_bytes >= 0:
                            used_bytes = dxgi_used_bytes
                            if total_bytes is not None and total_bytes > 0:
                                safe_used = min(int(used_bytes), int(total_bytes))
                                percent = round((safe_used / total_bytes) * 100, 1)
                                free_bytes = max(0, int(total_bytes) - safe_used)

                        # If torch.xpu reports 0 memory, fall back to PDH shared+dedicated usage (same source as Task Manager)
                        shared_b = (
                            windows_igpu_shared_bytes
                            if isinstance(windows_igpu_shared_bytes, int)
                            else 0
                        )
                        dedicated_b = (
                            windows_igpu_dedicated_bytes
                            if isinstance(windows_igpu_dedicated_bytes, int)
                            else 0
                        )
                        pdh_used = shared_b + dedicated_b
                        if (used_bytes is None or used_bytes <= 0) and pdh_used > 0:
                            used_bytes = max(used_bytes or 0, pdh_used)
                            if total_bytes is not None and total_bytes > 0:
                                safe_used = min(int(used_bytes), int(total_bytes))
                                percent = round((safe_used / total_bytes) * 100, 1)
                                free_bytes = max(0, int(total_bytes) - safe_used)

                    gpus.append(
                        GPUInfo(
                            index=i,
                            name=name,
                            total_gb=_bytes_to_gb(total_bytes) if total_bytes else 0.0,
                            used_gb=_bytes_to_gb(used_bytes)
                            if mode == "usage" and used_bytes is not None
                            else None,
                            free_gb=_bytes_to_gb(free_bytes)
                            if mode == "usage" and free_bytes is not None
                            else None,
                            percent=percent if mode == "usage" else None,
                            gpu_util=intel_gpu_util if mode == "usage" else None,
                            temperature=None,
                        )
                    )
                    seen_gpu_names.add(str(name).strip().lower())
        except Exception:
            pass

        if os.name == "nt":
            try:
                next_index = len(gpus)
                for adapter in windows_dxgi_adapters:
                    adapter_name = str(adapter.get("name", "")).strip()
                    adapter_key = adapter_name.lower()
                    if not adapter_name or adapter_key in seen_gpu_names:
                        continue
                    if "microsoft basic render driver" in adapter_key:
                        continue

                    vendor_id = int(adapter.get("vendor_id", 0) or 0)
                    # NVIDIA is already handled by NVML / nvidia-smi; fill in Intel/other Windows adapters here.
                    if vendor_id == 0x10DE:
                        continue

                    total_bytes = int(adapter.get("total_taskmgr_bytes", 0) or 0)
                    used_bytes = None
                    free_bytes = None
                    percent = None
                    gpu_util = None

                    if mode == "usage":
                        luid = str(adapter.get("luid", "unknown")).lower()
                        pdh_row = (
                            windows_pdh_snapshot.get(luid, {})
                            if isinstance(windows_pdh_snapshot, dict)
                            else {}
                        )
                        used_bytes = self._get_windows_video_memory_usage_bytes(
                            cast(int, adapter.get("raw_luid"))
                        )
                        if not isinstance(used_bytes, int) or used_bytes < 0:
                            shared_used = int(float(pdh_row.get("shared_used_bytes", 0.0) or 0.0))
                            dedicated_used = int(
                                float(pdh_row.get("dedicated_used_bytes", 0.0) or 0.0)
                            )
                            used_bytes = max(0, shared_used + dedicated_used)

                        if total_bytes > 0:
                            safe_used = min(used_bytes, total_bytes)
                            free_bytes = max(0, total_bytes - safe_used)
                            percent = round((safe_used / total_bytes) * 100, 1)

                        util_val = pdh_row.get("util_percent")
                        if isinstance(util_val, (int, float)):
                            gpu_util = round(float(util_val), 2)

                    gpus.append(
                        GPUInfo(
                            index=next_index,
                            name=adapter_name,
                            total_gb=_bytes_to_gb(total_bytes) if total_bytes else 0.0,
                            used_gb=_bytes_to_gb(used_bytes)
                            if mode == "usage" and used_bytes is not None
                            else None,
                            free_gb=_bytes_to_gb(free_bytes)
                            if mode == "usage" and free_bytes is not None
                            else None,
                            percent=percent if mode == "usage" else None,
                            gpu_util=gpu_util if mode == "usage" else None,
                            temperature=None,
                        )
                    )
                    seen_gpu_names.add(adapter_key)
                    next_index += 1
            except Exception:
                pass

        return GPUResource(available=len(gpus) > 0, gpus=gpus)

    def _read_windows_disk_specs(self) -> list[DiskDevice]:
        if self._windows_disk_spec_cache is not None:
            return self._windows_disk_spec_cache
        devices = []
        try:
            import json

            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-PhysicalDisk | Select-Object DeviceId, Model, Size, MediaType | ConvertTo-Json -Compress",
                ],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,  # pyright: ignore[reportAttributeAccessIssue]  # Windows-only flag
            )
            data = json.loads(out)
            if isinstance(data, dict):
                data = [data]
            for dev in data:
                model_val = str(dev.get("Model", "")).strip()
                if model_val.startswith("T7P5"):
                    model_val = "TRUSTA-" + model_val

                size_bytes = dev.get("Size")
                size_str = f"{round(size_bytes / (1024**3), 1)}G" if size_bytes else "Unknown"

                mtype = dev.get("MediaType")
                type_str = "SSD"
                if isinstance(mtype, str) and mtype.upper() == "HDD":
                    type_str = "HDD"
                elif mtype == 3:  # Sometimes MediaType returns 3 for HDD
                    type_str = "HDD"

                devices.append(
                    DiskDevice(
                        name=f"PhysicalDisk{dev.get('DeviceId', '')}",
                        size=size_str,
                        model=model_val,
                        type=type_str,
                    )
                )
        except Exception:
            pass
        self._windows_disk_spec_cache = devices
        return devices

    def get_disk_resource(
        self, mode: str, path: str = "/", calc_size: bool = False
    ) -> DiskResource:
        """Get Disk resources (unified model)."""
        devices = None
        mounts = []

        import psutil

        # Calculate IO Rates if mode is usage
        current_time = time.time()
        time_delta = current_time - self.last_io_time

        # Prevent division by zero or negative time
        if time_delta <= 0:
            time_delta = 0.001

        current_io = {}
        disk_rates = {}  # Map device_name -> {read_mbps, write_mbps}

        if mode == "usage":
            try:
                current_io = psutil.disk_io_counters(perdisk=True)
                for dev_name, counters in current_io.items():
                    if dev_name in self.last_disk_io:
                        prev = self.last_disk_io[dev_name]
                        read_diff = counters.read_bytes - prev.read_bytes
                        write_diff = counters.write_bytes - prev.write_bytes

                        # Bytes per second -> MB per second
                        read_mbps = (read_diff / time_delta) / (1024 * 1024)
                        write_mbps = (write_diff / time_delta) / (1024 * 1024)

                        disk_rates[dev_name] = {
                            "read_mbps": round(max(0, read_mbps), 2),
                            "write_mbps": round(max(0, write_mbps), 2),
                        }

                # Update state
                self.last_disk_io = current_io
                self.last_io_time = current_time
            except Exception:
                pass

        def _get_speed_for_mount(device: str) -> tuple[float, float]:
            # Try to match device name (e.g. /dev/sda1 -> sda1) to disk_rates keys
            if not device:
                return 0.0, 0.0
            base = os.path.basename(device)

            # Helper to check if device is in disk_rates
            def _check(name: str) -> tuple[float, float] | None:
                if name in disk_rates:
                    return disk_rates[name]["read_mbps"], disk_rates[name]["write_mbps"]
                return None

            # 1. Try exact match
            res = _check(base)

            # If exact match has activity, return it
            if res and (res[0] > 0 or res[1] > 0):
                return res

            # 2. Try parent device fallback (if exact match is 0 or not found)
            # Remove partition suffix logic
            parent = None
            if base.startswith("nvme"):
                # Handle nvme0n1p1 -> nvme0n1
                m = re.match(r"(nvme\d+n\d+)p\d+", base)
                if m:
                    parent = m.group(1)
            elif base.startswith("mmcblk"):
                m = re.match(r"(mmcblk\d+)p\d+", base)
                if m:
                    parent = m.group(1)
            elif base.startswith("sd") or base.startswith("xvd") or base.startswith("vd"):
                # Handle sda1 -> sda, xvda1 -> xvda
                m = re.match(r"([a-z]+)\d+", base)
                if m:
                    parent = m.group(1)

            if parent:
                res_p = _check(parent)
                # If parent has data and (original was missing or zero), use parent
                if res_p and (res_p[0] > 0 or res_p[1] > 0):
                    return res_p

            return res if res else (0.0, 0.0)

        def _get_mount_info(mp: str, device: str = "") -> DiskMount:
            try:
                du = shutil.disk_usage(mp)
                r_speed, w_speed = _get_speed_for_mount(device)

                dm = DiskMount(
                    path=mp,
                    total_gb=_bytes_to_gb(du.total),
                    used_gb=_bytes_to_gb(du.used),
                    free_gb=_bytes_to_gb(du.free),
                    percent=round((du.used / du.total) * 100, 2) if du.total else 0,
                    read_speed_mbps=r_speed if mode == "usage" else None,
                    write_speed_mbps=w_speed if mode == "usage" else None,
                )
                if calc_size and mp == path and os.path.exists(mp) and os.path.isdir(mp):
                    try:
                        total = 0
                        for root, _, files in os.walk(mp):
                            for f in files:
                                fp = os.path.join(root, f)
                                if not os.path.islink(fp):
                                    total += os.path.getsize(fp)
                        dm.folder_size_gb = _bytes_to_gb(total)
                    except Exception:
                        pass
                return dm
            except Exception as e:
                return DiskMount(path=mp, total_gb=0, used_gb=0, free_gb=0, error=str(e))

        seen = set()
        main_device = ""
        best_match_len = -1
        target_path = os.path.abspath(path)

        try:
            # psutil
            for p in psutil.disk_partitions(all=False):
                mp = getattr(p, "mountpoint", None) or getattr(p, "mount_point", None)
                device = getattr(p, "device", "")

                # Check for main request path to get its device info for fallback
                # Find the mountpoint that is the longest prefix of the target_path
                if mp and target_path.startswith(mp):
                    # Special case: "/" is prefix of everything, but "/foo" is better than "/" for "/foo/bar"
                    # Compare lengths.
                    mp_len = len(mp)
                    # Correctness: ensure mp matches a folder boundary if it's not root
                    # e.g. /media/dat matches /media/data? No.
                    if (
                        mp == "/"
                        or target_path == mp
                        or target_path.startswith(mp.rstrip("/") + "/")
                    ):
                        if mp_len > best_match_len:
                            best_match_len = mp_len
                            main_device = device

                # Filter: exclusively show physical devices (startswith /dev/) and exclude loops
                if not device or not device.startswith("/dev/") or "loop" in device:
                    continue

                if not mp or mp in seen or getattr(p, "fstype", "") in self._skip_fs_types:
                    continue
                seen.add(mp)
                info = _get_mount_info(mp, device)
                info.fstype = getattr(p, "fstype", "")
                mounts.append(info)

            # fallback /proc/mounts
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    device, mp, fstype = parts[0], parts[1], parts[2]

                    if mp and target_path.startswith(mp):
                        mp_len = len(mp)
                        if (
                            mp == "/"
                            or target_path == mp
                            or target_path.startswith(mp.rstrip("/") + "/")
                        ):
                            if mp_len > best_match_len:
                                best_match_len = mp_len
                                main_device = device

                    # Same filter for /proc/mounts
                    if not device.startswith("/dev/") or "loop" in device:
                        continue

                    if mp in seen or fstype in self._skip_fs_types:
                        continue
                    seen.add(mp)
                    info = _get_mount_info(mp, device)
                    info.fstype = fstype
                    mounts.append(info)
        except Exception:
            pass

        mounts.sort(key=lambda x: len(x.path))
        main = _get_mount_info(path, main_device)

        if mode == "spec":
            devices = []
            if os.name == "nt":
                devices = self._read_windows_disk_specs()
            else:
                try:
                    out = subprocess.check_output(
                        ["lsblk", "-J", "-d", "-o", "NAME,SIZE,TYPE,MODEL,ROTA"], text=True
                    )
                    data = json.loads(out)
                    for dev in data.get("blockdevices", []):
                        if dev.get("type") == "disk":
                            model_val = dev.get("model")
                            # Specific handling for T7P5 disks to show as TRUSTA-T7P5
                            if model_val and str(model_val).startswith("T7P5"):
                                model_val = "TRUSTA-" + str(model_val)

                            devices.append(
                                DiskDevice(
                                    name=dev.get("name"),
                                    size=dev.get("size"),
                                    model=model_val,
                                    type="HDD" if str(dev.get("rota")) == "1" else "SSD",
                                )
                            )
                except Exception:
                    pass

        return DiskResource(devices=devices, mounts=mounts, main=main)


# Global Instance
system_monitor = SystemMonitor()
