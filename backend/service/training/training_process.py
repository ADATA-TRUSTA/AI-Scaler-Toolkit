"""
Training process worker and manager.

This module mirrors the design of `inference/model_inference_process.py` but for
finetuning (LoRA / QLoRA / full-parameter) using Hugging Face Trainer + DeepSpeed.

High-level design:
- TrainingWorkerProcess: spawned via multiprocessing.Process, owns GPU resources.
- TrainingProcessManager: lives in main FastAPI process, sends commands and
  receives status updates via multiprocessing queues.

NOTE: This is an initial skeleton; details are wired from `training_manager.py`.
"""

from __future__ import annotations

import os

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import json
import subprocess
import time
import uuid
from collections.abc import Callable
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from queue import Empty
from typing import TYPE_CHECKING, Any, cast

import psutil

from ..config_models import TrainingConfig, TrainingStatus

if TYPE_CHECKING:
    from datasets import Dataset
    from redis import Redis
    from transformers import PreTrainedTokenizerBase, TrainingArguments

# Import settings BEFORE torch/transformers (sets HF_HOME environment variable)
from ..settings import (
    DEEPSPEED_NVME_DIR,
    DEEPSPEED_NVME_PATH_SENTINEL,
    DEEPSPEED_NVME_PINNED_BUDGET_BYTES,
    PROJECT_ROOT,
    REDIS_CONNECT_TIMEOUT,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    configure_logging,
)
from ..utils.conversion_manager import conversion_manager
from ..utils.path_safety import is_protected_system_path
from ..utils.system_monitor import system_monitor
from ..utils.token_utils import load_hf_token

try:
    import redis
except ImportError:
    redis = None


# New core modules
import torch
from transformers import Trainer

from .core import (
    JobLogWriter,
    MemoryAggregator,
    Phase,
    build_resource_snapshot,
    build_training_components,
    log_mem,
    read_events,
    save_training_results,
    start_memory_sampler,
)

logger = configure_logging(__name__)


def _create_redis_client() -> Redis | None:
    """
    Create a Redis client when one has been configured, else None.

    None is a supported state everywhere this client is used: metrics and resource
    history are kept elsewhere as well, and every write here is guarded by a check for
    it. So a machine with no Redis should get None immediately rather than a client
    that will spend a TCP timeout discovering the same thing.

    That timeout is not free. This runs while TrainingManager is being constructed,
    which is on the path between starting the service and it answering, so every
    launch on a machine without Redis paid for it before the server could bind.
    """
    if not redis:
        return None

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        return redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
            socket_timeout=REDIS_CONNECT_TIMEOUT,
        )

    # Only when a host was actually asked for. The default is localhost, which is an
    # assumption rather than a configuration -- and on a desktop install a wrong one.
    if not os.getenv("REDIS_HOST"):
        return None

    redis_username = os.getenv("REDIS_USERNAME")
    redis_password = os.getenv("REDIS_PASSWORD")
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        username=redis_username,
        password=redis_password,
        decode_responses=True,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        socket_timeout=REDIS_CONNECT_TIMEOUT,
    )


def _log_zero3_partition_state(trainer: Trainer) -> None:
    """
    Report whether ZeRO-3 is actually partitioning at training time.

    ``is_deepspeed_zero3_enabled()`` only says a DeepSpeed config was attached;
    it does not prove the engine wrapped the model or that parameters are still
    partitioned once training starts. A run whose parameters are all resident
    (``ds_status`` AVAILABLE, ``ds_tensor`` holding the full tensor) is paying
    the offload's DRAM cost while keeping the model on the GPU anyway, which
    looks exactly like an OOM the ZeRO-3 knobs cannot budge. Diagnostic only --
    never raises.
    """
    try:
        model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
        if model is None:
            return
        logger.info(
            f"[ZERO3PROBE] trainer.model={type(getattr(trainer, 'model', None)).__name__} "
            f"model_wrapped={type(model).__name__}"
        )

        managed = partitioned = resident = 0
        resident_elems = 0
        for param in model.parameters():
            ds_status = getattr(param, "ds_status", None)
            if ds_status is None:
                continue
            managed += 1
            # A partitioned param keeps only this rank's shard in ds_tensor and
            # presents shape [0]; a gathered one is fully materialised.
            if param.numel() == 0:
                partitioned += 1
            else:
                resident += 1
                resident_elems += param.numel()
        logger.info(
            f"[ZERO3PROBE] deepspeed-managed params={managed} partitioned={partitioned} "
            f"resident={resident} resident_elems={resident_elems / 1e9:.2f}B "
            f"(~{resident_elems * 2 / 1024**3:.2f} GiB at bf16)"
        )
        if managed == 0:
            logger.warning(
                "[ZERO3PROBE] No parameter carries ds_status: the DeepSpeed ZeRO-3 engine did "
                "not wrap this model, so the whole model stays on the GPU regardless of the "
                "offload settings"
            )
    except Exception as e:  # pragma: no cover - diagnostic must never break training
        logger.info(f"[ZERO3PROBE] could not inspect partition state: {e}")


# DeepSpeed swaps several parameters concurrently; with fewer buffers than this
# `swap_in` asserts "Not enough buffers 0 for swapping 1".
_MIN_NVME_PARAM_BUFFERS = 4


def _largest_parameter_numel(model_name_or_path: str) -> int | None:
    """
    Element count of the biggest single tensor in a checkpoint, or None.

    Reads only safetensors headers -- no tensor data -- so it is cheap even on a
    70 GB checkpoint.
    """
    import math

    from ..training.core.zero3_utils import _checkpoint_buffer_sources

    try:
        sources = _checkpoint_buffer_sources(model_name_or_path)
        if not sources:
            return None
        from safetensors import safe_open

        largest = 0
        for path in set(sources.values()):
            if path.suffix != ".safetensors":
                return None  # header-only read is a safetensors feature
            with safe_open(str(path), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118 - safe_open has no .keys() view
                    largest = max(largest, math.prod(f.get_slice(key).get_shape()))
        return largest or None
    except Exception as e:
        logger.debug(f"[TrainingWorker] Could not size the largest parameter: {e}")
        return None


def _fit_nvme_buffers(zero_opt: dict, model_name_or_path: str, pinned_budget_bytes: int) -> bool:
    """
    Grow NVMe swap buffers to fit the largest parameter, shrinking their count.

    DeepSpeed swaps a partitioned parameter through a fixed-size buffer and
    asserts if one does not fit::

        AssertionError: More elements 2818572288 than buffer size 1000000000

    which says nothing about which knob to turn. The shipped profiles carried
    1e9/2e9, while gemma-4-E4B's per-layer embedding table alone is 2.82e9
    elements -- so NVMe parameter offload could not load it at all.

    The buffers are always page-locked regardless of ``pin_memory``, so raising
    the size without lowering the count would reserve tens of GB of pinned DRAM.
    Total pinned bytes are held under ``pinned_budget_bytes`` instead.
    """
    largest = _largest_parameter_numel(model_name_or_path)
    if not largest:
        return False

    # Only offload_param takes buffer_size; DeepSpeedZeroOffloadOptimizerConfig
    # forbids the key outright (pydantic extra_forbidden), so setting it there
    # fails config validation before training starts.
    sec = zero_opt.get("offload_param")
    if not isinstance(sec, dict) or sec.get("device") != "nvme":
        return False
    current = int(sec.get("buffer_size") or 0)
    if current >= largest:
        return False

    # fp32 staging: 4 bytes per element per buffer.
    per_buffer = largest * 4
    affordable = pinned_budget_bytes // per_buffer
    # DeepSpeed swaps several parameters concurrently and asserts
    # "Not enough buffers 0 for swapping 1" when none is free, so the floor is
    # a hard requirement while the budget is only a guideline. The floor wins,
    # loudly -- an over-budget run that works beats a tidy one that cannot start.
    count = max(_MIN_NVME_PARAM_BUFFERS, min(int(sec.get("buffer_count") or 4), affordable))
    sec["buffer_size"] = int(largest)
    sec["buffer_count"] = int(count)
    pinned_gib = count * per_buffer / 1024**3
    logger.info(
        f"[TrainingWorker] offload_param: buffer_size {current:,} -> {largest:,} "
        f"(largest parameter in the checkpoint), buffer_count -> {count}, "
        f"pinned DRAM ~{pinned_gib:.1f} GiB"
    )
    if count > affordable:
        logger.warning(
            f"[TrainingWorker] NVMe parameter offload needs ~{pinned_gib:.1f} GiB of pinned "
            f"DRAM for this checkpoint, over the {pinned_budget_bytes / 1024**3:.0f} GiB budget, "
            f"because its largest tensor alone is {largest:,} elements. Raise "
            "DEEPSPEED_NVME_PINNED_BUDGET_GB, or offload parameters to CPU instead "
            "(zero3_offload_disk_cpu keeps params in DRAM and swaps the optimizer)."
        )
    return True


def _resolve_deepspeed_config(training_config: TrainingConfig) -> str | None:
    """
    Resolve a deepspeed config path from TrainingConfig.

    Two forms are supported:
    1. If `training_config.deepspeed_config` is already an explicit path, use it as-is.
    2. If `training_config.deepspeed_profile` is set, resolve it from
       `service/configs/deepspeed/<profile>.json`.

    If the config declares nvme offload paths, stale files in them are cleared.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    # 1) explicit path takes precedence
    explicit = getattr(training_config, "deepspeed_config", None)
    if explicit:
        config_path = str(explicit)
    else:
        # 2) profile-based lookup — validate name before constructing path
        profile = getattr(training_config, "deepspeed_profile", None)
        if not profile:
            return None

        # Reject any profile name that contains path separators or traversal sequences
        # to prevent directory traversal attacks (e.g. "../../etc/passwd")
        if "/" in profile or "\\" in profile or ".." in profile:
            raise ValueError(
                f"[TrainingWorker] Invalid deepspeed_profile '{profile}': "
                "profile names must not contain path separators or '..'"
            )

        # Anchored to the project root, not the cwd: started from anywhere else
        # this lookup missed, warned, and returned None -- silently disabling
        # ZeRO-3 for a run that explicitly asked for offload.
        base = PROJECT_ROOT / "service" / "configs" / "deepspeed"
        cfg_path = base / f"{profile}.json"
        # Resolve and verify the path stays within the expected directory
        resolved = cfg_path.resolve()
        expected_base = base.resolve()
        if not str(resolved).startswith(str(expected_base) + os.sep) and resolved != expected_base:
            raise ValueError(
                f"[TrainingWorker] DeepSpeed profile path '{resolved}' escapes the allowed directory"
            )
        if not resolved.is_file():
            # Returning None here used to mean "train without DeepSpeed" -- the
            # caller asked for offload and silently got the whole model on the
            # GPU, then lost the adapter when the saver took the ZeRO-3 branch.
            available = sorted(p.stem for p in base.glob("*.json"))
            raise ValueError(
                f"DeepSpeed profile '{profile}' not found at {cfg_path}. "
                f"Available profiles: {', '.join(available) or '(none)'}"
            )

        logger.info(f"[TrainingWorker] Using DeepSpeed profile '{profile}': {cfg_path}")
        config_path = str(cfg_path)

    # 3) Resolve every nvme_path: the job's offload_folder wins, else the
    #    DEEPSPEED_NVME_DIR setting. Profiles ship the "AUTO" sentinel rather
    #    than a real path -- they used to hardcode one machine's mount points,
    #    which went stale and made every nvme profile crash at zero.Init.
    offload_folder = getattr(training_config, "offload_folder", None)
    nvme_root = Path(offload_folder or DEEPSPEED_NVME_DIR)
    # `offload_folder` defaults to the RELATIVE './deepspeed_offload', so
    # resolving against the cwd would put gigabytes of offload wherever the
    # worker happened to be started -- and DeepSpeed clears that directory.
    # Anchor anything relative to the project root instead.
    if not nvme_root.is_absolute():
        nvme_root = PROJECT_ROOT / nvme_root
    nvme_target = str(nvme_root.resolve())

    with open(config_path) as f:
        ds_config = json.load(f)

    modified = False
    zero_opt = ds_config.get("zero_optimization", {})
    for section in ("offload_optimizer", "offload_param"):
        sec = zero_opt.get(section)
        if isinstance(sec, dict) and sec.get("device") == "nvme":
            current = sec.get("nvme_path")
            # Substitute the sentinel, a missing value, or a caller-supplied
            # offload_folder. A profile carrying a real path the operator wrote
            # themselves is respected unless offload_folder overrides it.
            if current in (None, "", DEEPSPEED_NVME_PATH_SENTINEL) or offload_folder:
                sec["nvme_path"] = nvme_target
                modified = True

    # Swap buffers must fit the biggest single parameter, or DeepSpeed asserts
    # deep inside the swapper with a message that names neither the model nor
    # the setting to change.
    if _fit_nvme_buffers(zero_opt, training_config.model_name, DEEPSPEED_NVME_PINNED_BUDGET_BYTES):
        modified = True

    if modified:
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="ds_config_override_", text=True)
        with os.fdopen(fd, "w") as f:
            json.dump(ds_config, f, indent=2)
        logger.info(
            f"[TrainingWorker] DeepSpeed nvme_path resolved to {nvme_target}; "
            f"override written to {temp_path}"
        )
        config_path = temp_path

    # Check and clear nvme offload paths
    try:
        with open(config_path) as f:
            ds_config = json.load(f)

        nvme_paths = set()

        # Check optimizer offload nvme path
        zero_opt = ds_config.get("zero_optimization", {})
        offload_opt = zero_opt.get("offload_optimizer", {})
        if offload_opt.get("device") == "nvme":
            nvme_path = offload_opt.get("nvme_path")
            if nvme_path:
                nvme_paths.add(nvme_path)

        # Check parameter offload nvme path
        offload_param = zero_opt.get("offload_param", {})
        if offload_param.get("device") == "nvme":
            nvme_path = offload_param.get("nvme_path")
            if nvme_path:
                nvme_paths.add(nvme_path)

        # Block paths that are clearly system directories — never valid offload targets.
        # Any absolute path the user deliberately configured is allowed (other disk, NVMe, etc.).
        # The check below empties the directory, so it must hold on Windows too, where an
        # inline POSIX prefix list would match nothing at all.

        for nvme_path in nvme_paths:
            if not nvme_path:
                continue
            nvme_dir = Path(nvme_path).resolve()
            # Reject non-absolute sources — relative paths could unexpectedly resolve
            # against the current working directory in ways that are hard to audit.
            if not nvme_dir.is_absolute():
                logger.warning(
                    f"[TrainingWorker] Skipping nvme path '{nvme_path}': "
                    "only absolute paths are allowed for nvme offload directories"
                )
                continue
            if is_protected_system_path(nvme_dir):
                logger.warning(
                    f"[TrainingWorker] Skipping nvme path '{nvme_path}': "
                    "path resolves to a protected system directory"
                )
                continue
            if nvme_dir.exists() and nvme_dir.is_dir():
                try:
                    # Delete everything inside the directory
                    for item in nvme_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info(f"[TrainingWorker] Cleared nvme offload directory: {nvme_path}")
                except Exception as clean_err:
                    logger.warning(
                        f"[TrainingWorker] Failed to clear nvme directory {nvme_path}: {clean_err}"
                    )
            elif not nvme_dir.exists():
                # Directory missing: create it. A failure here used to be a
                # warning, and DeepSpeed then repeated the same mkdir unguarded
                # inside partitioned_param_swapper -- so the run died with a
                # bare OSError from library internals instead of saying which
                # directory it could not create, or that offload_folder /
                # DEEPSPEED_NVME_DIR is how you point it somewhere writable.
                try:
                    nvme_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"[TrainingWorker] Created nvme offload directory: {nvme_path}")
                except OSError as create_err:
                    raise ValueError(
                        f"Cannot create the NVMe offload directory '{nvme_dir}' ({create_err}). "
                        "Set 'offload_folder' on the training job, or the DEEPSPEED_NVME_DIR "
                        "environment variable, to a writable location on a fast disk."
                    ) from create_err

    except ValueError:
        # An unusable offload directory is deliberate and actionable -- it must
        # not be swallowed by the catch-all below, which is there for
        # incidental failures like a permission hiccup while clearing.
        raise
    except Exception as e:
        logger.warning(f"[TrainingWorker] Failed to process nvme paths from config: {e}")

    return config_path


def _cleanup_training_resources(
    trainer: Trainer | None = None,
    model: torch.nn.Module | None = None,
    base_model: torch.nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    dataset: Dataset | None = None,
    deepspeed_initialized: bool = False,
) -> None:
    """
    Thoroughly release training resources to free GPU and system memory.

    Args:
        trainer: Trainer instance
        model: Model instance (may be a PEFT model)
        base_model: Base model instance
        tokenizer: Tokenizer instance
        dataset: Dataset instance
        deepspeed_initialized: Whether DeepSpeed was initialized
    """
    import gc

    logger.info("[TrainingWorker] Starting resource cleanup...")

    # 1. Clean up Trainer
    if trainer is not None:
        try:
            # Try to clear the trainer's internal state
            if hasattr(trainer, "model"):
                trainer.model = None
            if hasattr(trainer, "optimizer"):
                trainer.optimizer = None
            if hasattr(trainer, "lr_scheduler"):
                trainer.lr_scheduler = None
            del trainer
            logger.debug("[TrainingWorker] Trainer cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Trainer cleanup warning: {e}")

    # 2. Clean up model
    if model is not None:
        try:
            # For a PEFT model, release the adapter first
            if hasattr(model, "base_model"):
                # nn.Module.__setattr__ is typed Tensor|Module; PEFT allows None here.
                model.base_model = None  # pyright: ignore[reportArgumentType]
            del model
            logger.debug("[TrainingWorker] Model cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Model cleanup warning: {e}")

    # 3. Clean up base model
    if base_model is not None:
        try:
            del base_model
            logger.debug("[TrainingWorker] Base model cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Base model cleanup warning: {e}")

    # 4. Clean up Tokenizer
    if tokenizer is not None:
        try:
            del tokenizer
            logger.debug("[TrainingWorker] Tokenizer cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Tokenizer cleanup warning: {e}")

    # 5. Clean up Dataset
    if dataset is not None:
        try:
            del dataset
            logger.debug("[TrainingWorker] Dataset cleaned up")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Dataset cleanup warning: {e}")

    # 6. Clean up DeepSpeed distributed process group
    if deepspeed_initialized:
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("[TrainingWorker] Destroyed distributed process group")
        except Exception as e:
            logger.warning(f"[TrainingWorker] Failed to destroy process group: {e}")

    # 7. Force several Python garbage collection passes
    for _ in range(3):
        gc.collect()
    logger.debug("[TrainingWorker] Python garbage collection completed (3 passes)")

    # 8. Clean up CUDA memory
    if torch.cuda.is_available():
        try:
            # Empty the cache on every CUDA device
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            logger.info("[TrainingWorker] CUDA memory cleared on all devices")
        except Exception as e:
            logger.warning(f"[TrainingWorker] CUDA cleanup warning: {e}")

    logger.info("[TrainingWorker] Resource cleanup completed")


def _convert_training_output_to_q4_k_m(
    training_config: TrainingConfig,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """
    Convert the saved training output to GGUF and quantize it to Q4_K_M.

    For a multimodal checkpoint this also exports a companion ``mmproj-*.gguf``
    (the vision encoder + projector). llama.cpp represents a vision-language
    model as two files -- the quantized language model and the mmproj -- so
    emitting only the language GGUF would leave the fine-tune unable to see
    images. Under the default ``multimodal_scope`` the mmproj comes from the
    base model, whose vision half the run left unchanged; a run that trained the
    bridge gets it from the merged weights instead, since that is the only place
    the trained projector exists.
    """
    output_dir = str(training_config.output_dir)
    logger.info(
        "[TrainingWorker] Starting post-training GGUF conversion for %s",
        output_dir,
    )
    result = conversion_manager.convert_and_quantize(
        model_path=output_dir,
        output_dir=output_dir,
        intermediate_outtype="f16",
        quantization_type="Q4_K_M",
        work_dir=training_config.offload_folder,
        status_callback=status_callback,
        export_mmproj=True,
    )
    logger.info(
        "[TrainingWorker] Post-training GGUF conversion completed: %s (mmproj=%s)",
        result["quantized_output_path"],
        result.get("mmproj_output_path"),
    )
    return result


# Prefix rank 0 uses to embed structured status inside stdout.
_DS_STATUS_PREFIX = "__STATUS__:"
# Terminal states that are also written to Redis as crash-safe backup.
_DS_TERMINAL_STATES = frozenset({"saved", "error"})


def extract_status_payloads(line: str) -> tuple[list[dict], str]:
    """
    Pull every ``__STATUS__:`` payload out of a line, wherever it sits in it.

    Matching with ``startswith`` loses them. tqdm draws its progress bar on
    stderr with a carriage return and no newline, and the subprocess is read
    with stderr merged into stdout, so one ``\\n``-terminated read can hold a
    half-drawn bar and then the status marker::

        3%|▎ | 1/40 [00:12<08:01]__STATUS__:{"status": "running", ...}

    The status is intact; it just is not at position zero. That is why multi-GPU
    runs stopped reporting loss and accuracy while single-GPU runs, which have
    no launcher interleaving their output, kept working.

    Returns the decoded payloads in the order they appear, plus whatever text
    surrounded them, so the caller can still log the progress bar.

    A payload is always complete within one read: it is written with ``print``,
    so it ends at the newline the reader splits on. Length is found by decoding
    rather than by scanning for a brace, which a value containing ``}`` would
    defeat.
    """
    decoder = json.JSONDecoder()
    payloads: list[dict] = []
    leftovers: list[str] = []
    pos = 0

    while True:
        marker = line.find(_DS_STATUS_PREFIX, pos)
        if marker == -1:
            leftovers.append(line[pos:])
            break

        leftovers.append(line[pos:marker])
        start = marker + len(_DS_STATUS_PREFIX)
        while start < len(line) and line[start].isspace():
            start += 1

        try:
            obj, end = decoder.raw_decode(line, start)
        except ValueError:
            # The marker is there but what follows is not JSON. Keep it as text
            # and carry on: a later marker on the same line may still be good.
            leftovers.append(line[marker:start])
            pos = start
            continue

        if isinstance(obj, dict):
            payloads.append(obj)
        else:
            leftovers.append(line[marker:end])
        pos = end

    return payloads, "".join(leftovers).strip()


def _terminate_deepspeed_tree(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    """
    Tear down the deepspeed launcher together with its rank children.

    ``deepspeed --num_gpus=N`` runs every rank as a separate child process, so signalling only
    the launcher handle leaves N ranks alive still holding their GPU memory — the exact leak the
    process-per-job design exists to prevent, and it happens on the error path where this runs.
    Escalates terminate -> wait -> kill, matching the teardown used everywhere else here.
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        launcher = psutil.Process(proc.pid)
        victims = [*launcher.children(recursive=True), launcher]
    except psutil.Error:
        victims = []

    for victim in victims:
        try:
            victim.terminate()
        except psutil.Error:  # already gone between listing and signalling
            pass

    if victims:
        _, alive = psutil.wait_procs(victims, timeout=timeout)
        for victim in alive:
            logger.warning(f"[DeepSpeed] pid {victim.pid} ignored terminate, killing it")
            try:
                victim.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=timeout)

    # Reap the launcher through its own handle so no zombie is left behind.
    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class _DeepSpeedTrainingError(RuntimeError):
    """
    Carries rich error detail from the deepspeed subprocess.

    Raised by _run_deepspeed_subprocess so the outer except block can forward
    the subprocess traceback and is_oom flag without losing them.
    """

    def __init__(self, error: str, subprocess_traceback: str = "", is_oom: bool = False) -> None:
        super().__init__(error)
        self.error = error
        self.subprocess_traceback = subprocess_traceback
        self.is_oom = is_oom


def _run_deepspeed_subprocess(
    training_config: TrainingConfig,
    ds_config_path: str | None,
    session_id: str,
    status_q: Queue,
    redis_client: Redis | None,
    writer: JobLogWriter | None = None,
) -> None:
    """
    Launch `deepspeed --num_gpus N training_script.py` with a hybrid status channel.

    Hybrid IPC design
    -----------------
    Primary path  : stdout pipe
        Rank 0 prints ``__STATUS__:<json>`` lines.  The _drain_stdout thread
        parses them and pushes to status_q in real time — zero polling lag.

    Backup path   : Redis (terminal states only)
        Rank 0 also writes ``saved`` / ``error`` payloads to Redis.
        After the process exits, we check Redis **once** to recover the
        terminal state in case the process was killed before stdout was fully
        read (e.g. SIGKILL, OOM killer).

    Error handling
    --------------
    ``error`` payloads are *not* forwarded to status_q from here; they are
    raised as ``_DeepSpeedTrainingError`` so the caller's existing except block
    sends exactly one "error" message to the manager (avoiding double-write).

    Raises:
        _DeepSpeedTrainingError: subprocess failed, carrying full detail.
    """
    import json as _json
    import subprocess
    import tempfile
    import threading
    from pathlib import Path

    num_gpus = getattr(training_config, "num_gpus", 1)
    script_path = str(Path(__file__).resolve().parent / "training_script.py")

    proc = None
    fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="ds_train_cfg_", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            _json.dump(training_config.model_dump(), f)

        cmd = [
            "deepspeed",
            f"--num_gpus={num_gpus}",
            script_path,
            "--config",
            cfg_path,
            "--session-id",
            session_id,
        ]
        if ds_config_path:
            cmd += ["--ds-config", ds_config_path]

        logger.info(f"[TrainingWorker] Launching multi-GPU training: {' '.join(cmd)}")

        # Strip single-GPU restrictions so deepspeed can see all GPUs.
        env = os.environ.copy()
        for _k in (
            "CUDA_VISIBLE_DEVICES",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(_k, None)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            # Own process group, so the whole launcher+ranks tree can be killed
            # in one signal. Without it, terminating the worker leaves the
            # deepspeed launcher and every rank alive and reparented, each still
            # holding its GPU's VRAM -- exactly the leak _terminate_deepspeed_tree
            # exists to prevent, but which never ran because the worker's default
            # SIGTERM handler kills the interpreter without unwinding `finally`.
            start_new_session=True,
        )

        # Mutable containers shared with the stdout thread.
        _received_terminal: list[bool] = [False]
        _error_detail: list[dict | None] = [None]
        _saved_payload: list[dict | None] = [None]

        def _drain_stdout() -> None:
            """Parse stdout lines; route status to status_q, log everything else."""
            assert proc.stdout is not None  # Popen created with stdout=PIPE  # noqa: S101
            for line in proc.stdout:
                stripped = line.rstrip()
                if not stripped:
                    continue
                payloads, surrounding = extract_status_payloads(stripped)
                if surrounding:
                    # Keep the last carriage-return segment: a tqdm bar accumulates
                    # its whole history in one read, and only the newest is useful.
                    logger.info(f"[DeepSpeed] {surrounding.rsplit(chr(13), 1)[-1]}")
                for payload in payloads:
                    try:
                        etype = payload.get("type")
                        if etype in ("stage", "resource", "metric", "info"):
                            # Structured log event from the subprocess (rank 0).
                            # The writer here is the single file owner, so persist it.
                            if writer is not None:
                                try:
                                    if etype == "stage":
                                        # A stage event without a phase is malformed;
                                        # `stage` keys its label table on it.
                                        _phase = payload.get("phase")
                                        if isinstance(_phase, str):
                                            writer.stage(_phase, payload.get("msg"))
                                    elif etype == "resource":
                                        writer.resource(
                                            payload.get("data") or {}, payload.get("phase")
                                        )
                                    elif etype == "metric":
                                        writer.metric(
                                            payload.get("data") or {}, payload.get("phase")
                                        )
                                    else:
                                        writer.info(payload.get("msg") or "", payload.get("data"))
                                except Exception:
                                    pass
                            # Stage transitions also drive the manager's phase field.
                            if etype == "stage":
                                status_q.put(
                                    {
                                        "status": "stage",
                                        "phase": payload.get("phase"),
                                        "phase_detail": payload.get("msg"),
                                    }
                                )
                            continue
                        s = payload.get("status")
                        if s == "error":
                            # Store error detail; raise from main thread to avoid
                            # double-writing "error" to the manager.
                            #
                            # Every rank reports errors now, and one rank failing
                            # usually knocks the others over (collective timeout,
                            # broken pipe). Keep the FIRST one: it is the cause,
                            # the rest are consequences.
                            if _error_detail[0] is None:
                                _error_detail[0] = payload
                            else:
                                logger.info(
                                    f"[DeepSpeed] Additional error from rank "
                                    f"{payload.get('rank', '?')} (keeping the first): "
                                    f"{str(payload.get('error'))[:120]}"
                                )
                            _received_terminal[0] = True
                        elif s == "saved":
                            # Not forwarded: "saved" means the weights are on disk,
                            # but GGUF conversion still has to run in this process.
                            # Passing it on ended the job in the UI while conversion
                            # was mid-flight. The parent sends the terminal status
                            # once conversion has actually resolved.
                            _saved_payload[0] = payload
                            _received_terminal[0] = True
                        else:
                            status_q.put(payload)
                            if s in _DS_TERMINAL_STATES:
                                _received_terminal[0] = True
                    except Exception:
                        logger.warning(
                            f"[DeepSpeed] Malformed status payload: {str(payload)[:200]}"
                        )

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stdout_thread.start()

        proc.wait()  # block until subprocess exits
        stdout_thread.join(timeout=10)

        # ── Crash-safe fallback: one Redis read after process exits ──────────
        # Covers the case where the process was killed before stdout was flushed
        # (e.g. SIGKILL, OOM killer) but had already written the terminal state
        # to Redis.
        if not _received_terminal[0] and redis_client:
            try:
                raw = redis_client.get(f"training:mgr_status:{session_id}")
                if raw:
                    # sync redis client returns str/bytes here (stub types it as ResponseT)
                    msg = _json.loads(cast("str | bytes", raw))
                    if msg.get("status") in _DS_TERMINAL_STATES:
                        if msg.get("status") == "error":
                            _error_detail[0] = msg
                        else:
                            # Same reasoning as the stdout path: "saved" is not the
                            # end of the job, so it is held rather than forwarded.
                            _saved_payload[0] = msg
                        _received_terminal[0] = True
                        logger.info(
                            f"[TrainingWorker] Recovered terminal state from Redis: {msg.get('status')}"
                        )
            except Exception:
                pass

        # ── Raise on failure ─────────────────────────────────────────────────
        rc = proc.returncode
        detail = _error_detail[0]

        if rc != 0 or detail:
            # `detail` now arrives from whichever rank actually failed, not just
            # rank 0, so an OOM on rank 3 keeps its traceback and is_oom flag
            # instead of collapsing into "exited with return code N".
            rank = detail.get("rank") if detail else None
            where = f" (rank {rank})" if rank is not None else ""
            raise _DeepSpeedTrainingError(
                error=f"{detail['error']}{where}"
                if detail and detail.get("error")
                else f"deepspeed exited with return code {rc}",
                subprocess_traceback=detail.get("traceback", "") if detail else "",
                is_oom=detail.get("is_oom", False) if detail else False,
            )

        if not _received_terminal[0]:
            raise _DeepSpeedTrainingError(
                error="deepspeed exited cleanly but training completion ('saved') was not confirmed",
            )

    finally:
        try:
            os.unlink(cfg_path)
        except Exception:
            pass
        _terminate_deepspeed_tree(proc)


def _training_worker_process(
    request_q: Queue,
    status_q: Queue,
    stop_event: EventClass,
) -> None:
    """
    Worker process entry point for training.

    Commands:
        {"command": "start", "config": {...}}
        {"command": "stop"}

    Status messages from worker:
        - {"status": "starting"}      # Startup stage (clears previous data and temp files)
        - {"status": "initializing"}  # Setup stage (loading tokenizer/dataset/model, etc.)
        - {"status": "running"}       # Entered the main training loop
        - {"status": "completed"}     # Training finished and model saved
        - {"status": "stopped"}       # Stopped after a stop command
        - {"status": "error", "error": str, "traceback": str, "is_oom": bool}  # Error or OOM
        - {"status": "progress", "step": int, "total": int, "progress": float, "loss": float|None}  # Training progress

    Notes:
        - On OOM/CUDA errors is_oom=true; the worker may exit right away to release VRAM.
        - progress messages do not change current_status; they only carry the latest metrics.
    """
    # Turn SIGTERM into a normal exception so `finally` blocks run.
    #
    # stop_training()/cleanup() terminate this process with SIGTERM, whose
    # default handler kills the interpreter outright. Every cleanup path is in a
    # `finally` -- including `_terminate_deepspeed_tree(proc)`, which is the only
    # thing that stops the multi-GPU launcher and its ranks. Killed by the
    # default handler, that never ran: cancelling a multi-GPU job left N ranks
    # alive and reparented, each still holding its GPU's VRAM.
    import signal

    def _sigterm_unwinds(signum: int, _frame: object) -> None:
        logger.info(f"[TrainingWorker] Received signal {signum}; unwinding cleanup handlers")
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _sigterm_unwinds)
    except (ValueError, OSError) as e:  # not on the main thread of this process
        logger.warning(f"[TrainingWorker] Could not install SIGTERM handler: {e}")

    # Suppress "The current process just got forked..." warning from tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Workaround for PEFT multi-GPU issues: Force training to use only the first GPU.
    # This prevents "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED" and device mismatches
    # when using generic TrainingArguments without deepspeed distributed setup or when PEFT
    # fails to handle multiple visible devices correctly.
    # We set this inside the worker process so it doesn't affect the main process or inference workers.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Skip DeepSpeed's CUDA version mismatch check.
    # torch 2.13.0+cu130 bundles CUDA 13.0 internally, but the system-level nvcc may report
    # a different version (e.g. 12.0). DeepSpeed's JIT builder compares the two and refuses
    # to compile ops like CPUAdam when they differ. CPUAdam is a CPU-only op and does not
    # actually need nvcc, so skipping this check is safe.
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"

    # Reduce CUDA memory fragmentation by using expandable segments allocator.
    # This helps avoid OOM when there is reserved-but-unallocated memory.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    trainer: Trainer | None = None
    training_config: TrainingConfig | None = None
    deepspeed_initialized = False
    worker_session_id: str | None = None
    redis_client = None

    def _key_metrics(sid: str) -> str:
        return f"training:history:{sid}:metrics"

    def _key_resources(sid: str) -> str:
        return f"training:history:{sid}:resources"

    try:
        logger.info("[TrainingWorker] process started")
        hf_token = load_hf_token()
        if hf_token:
            os.environ.setdefault("HF_HUB_TOKEN", hf_token)

        while not stop_event.is_set():
            try:
                cmd = request_q.get(timeout=1.0)
            except Empty:
                continue

            command = cmd.get("command")

            if command == "start":
                # locals for later cleanup
                model = None
                base_model = None
                tokenizer = None
                dataset = None
                processor = None
                processing_class = None
                writer = None
                _mem_thread = None
                _mem_agg = None
                _mem_stop = None
                try:
                    worker_session_id = cast(str, cmd.get("session_id"))
                    cfg_dict = cmd.get("config", {})
                    training_config = TrainingConfig(**cfg_dict)

                    # Init Redis in worker for passive logging (no API required).
                    # None means none was configured, which is ordinary here: every
                    # write through this client is already guarded, and the job's own
                    # log is the source of truth. Only a configured Redis that will
                    # not answer is worth a warning.
                    if redis and worker_session_id:
                        redis_client = _create_redis_client()
                        if redis_client is not None:
                            try:
                                redis_client.ping()
                            except Exception as e:
                                logger.warning(
                                    "[TrainingWorker] Redis is configured but did not answer: %s",
                                    e,
                                )
                                redis_client = None

                    # Dedicated per-job structured log (events.jsonl + training.log + meta.json):
                    # the single source of truth for SSE and persistence, replacing the previous
                    # practice of stuffing progress strings into the error channel.
                    writer = JobLogWriter(
                        job_id=worker_session_id,
                        redis_client=redis_client,
                        config=cfg_dict,
                    )

                    def _emit_stage(
                        phase: str,
                        detail: str | None = None,
                        *,
                        writer: JobLogWriter = writer,
                    ) -> None:
                        """Record a lifecycle stage to the job log and notify the manager."""
                        ev = writer.stage(phase, detail)
                        status_q.put(
                            {
                                "status": "stage",
                                "phase": phase,
                                "phase_detail": ev.get("msg"),
                            }
                        )

                    _emit_stage(Phase.QUEUED)

                    # One up-front environment snapshot for the log.
                    try:
                        writer.info(
                            "training environment",
                            {
                                "model_name": getattr(training_config, "model_name", None),
                                "method": str(getattr(training_config, "method", "")),
                                "num_gpus": getattr(training_config, "num_gpus", 1),
                                "use_deepspeed": bool(
                                    getattr(training_config, "use_deepspeed", False)
                                ),
                                "deepspeed_profile": getattr(
                                    training_config, "deepspeed_profile", None
                                ),
                                "dataset_path": getattr(training_config, "dataset_path", None),
                                "output_dir": getattr(training_config, "output_dir", None),
                            },
                        )
                    except Exception:
                        pass

                    ds_config = None
                    if getattr(training_config, "use_deepspeed", False):
                        _emit_stage(Phase.RESOLVING_DEEPSPEED)
                        ds_config = _resolve_deepspeed_config(training_config)

                    # Check whether the output directory already holds files
                    from pathlib import Path

                    output_dir = Path(training_config.output_dir)
                    if output_dir.exists():
                        # Check for any file or subdirectory inside
                        dir_contents = list(output_dir.iterdir())
                        if dir_contents:
                            error_msg = (
                                f"Output directory '{output_dir}' already exists and contains files. "
                                f"Please clear the directory before starting training to avoid conflicts."
                            )
                            logger.error(f"[TrainingWorker] {error_msg}")
                            if writer is not None:
                                writer.error(error_msg)
                                writer.finalize(Phase.ERROR)
                            status_q.put(
                                {
                                    "status": "error",
                                    "error": error_msg,
                                    "traceback": "",
                                    "is_oom": False,
                                }
                            )
                            continue

                    # Multi-GPU: delegate all GPU work to a deepspeed subprocess.
                    # The worker process only monitors and relays status.
                    _num_gpus = getattr(training_config, "num_gpus", 1)
                    if _num_gpus > 1:
                        _emit_stage(Phase.LAUNCHING_WORKERS)

                        # The deepspeed subprocess (rank 0) emits stage/resource/metric
                        # events over stdout; the writer here persists them so the file
                        # stays the single source of truth.
                        _run_deepspeed_subprocess(
                            training_config=training_config,
                            ds_config_path=ds_config,
                            session_id=worker_session_id,
                            status_q=status_q,
                            redis_client=redis_client,
                            writer=writer,
                        )

                        # Subprocess finished — training + saving complete.
                        _emit_stage(Phase.SAVING)
                        logger.info(
                            "[TrainingWorker] Multi-GPU training done. Starting GGUF Q4_K_M conversion..."
                        )

                        _conv_phase = {
                            "merge fine-tune": Phase.MERGING_LORA,
                            "convert to gguf": Phase.CONVERTING_GGUF,
                        }

                        def _cb_multi(
                            step: str, *, _conv_phase: dict[str, str] = _conv_phase
                        ) -> None:
                            _emit_stage(_conv_phase.get(step, Phase.CONVERTING_GGUF), step)

                        try:
                            conversion_result = _convert_training_output_to_q4_k_m(
                                training_config, status_callback=_cb_multi
                            )
                        except Exception as conv_err:
                            # The trained weights are already on disk -- only the GGUF
                            # export failed. Say so, and say what to run, because the
                            # alternative a user reaches for is training it all again.
                            logger.exception(
                                f"[TrainingWorker] GGUF conversion failed after a successful "
                                f"multi-GPU run: {conv_err}"
                            )
                            _conv_error = (
                                f"Training finished and the weights were saved to "
                                f"{training_config.output_dir}, but GGUF conversion failed: "
                                f"{conv_err}. Do not retrain -- convert the saved output with "
                                f"POST /config/models/convert."
                            )
                            import traceback

                            writer.error(_conv_error, traceback=traceback.format_exc())
                            # finalize() is what writes the terminal status and
                            # finished_at into meta.json; writer.error() only
                            # updates the live Redis key, and close() in the
                            # `finally` just closes handles. Without this the
                            # durable record keeps saying the job is running long
                            # after Redis has expired -- which is what every other
                            # terminal path here avoids by pairing the two calls.
                            writer.finalize(Phase.ERROR)
                            status_q.put({"status": "error", "error": _conv_error})
                            return
                        _emit_stage(Phase.COMPLETED)
                        _completed_result = {
                            "gguf_path": conversion_result.get("quantized_output_path"),
                            "gguf_quantization": conversion_result.get("quantization_type"),
                            "mmproj_path": conversion_result.get("mmproj_output_path"),
                        }
                        writer.finalize(Phase.COMPLETED, result=_completed_result)
                        status_q.put({"status": "completed", **_completed_result})
                        logger.info(
                            "[TrainingWorker] Multi-GPU training completed, exiting worker process..."
                        )
                        return

                    # ── Single-GPU path ──────────────────────────────────────────────
                    status_q.put({"status": "initializing"})

                    # Start resource sampling now (before any loading) so the user sees
                    # GPU/DRAM/SSD feedback during the load phase — the highest-DRAM,
                    # most OOM-prone window for DeepSpeed NVMe offload, not just training.
                    def _on_mem_sample(
                        probe: dict[str, Any],
                        *,
                        writer: JobLogWriter = writer,
                        training_config: TrainingConfig = training_config,
                    ) -> None:
                        try:
                            writer.resource(build_resource_snapshot(training_config, probe))
                        except Exception:
                            pass

                    _sample_interval = float(os.getenv("TRAINING_MEM_SAMPLE_INTERVAL", "5.0"))
                    # The aggregator reduces the sample stream to avg/peak per
                    # resource. Without it the events file held hundreds of
                    # point-in-time numbers and no answer to "how much did this
                    # run use", which is the only form anyone actually reads.
                    _mem_agg = MemoryAggregator()
                    _mem_thread, _mem_stop = start_memory_sampler(
                        interval=_sample_interval,
                        on_sample=_on_mem_sample,
                        offload_path=getattr(training_config, "offload_folder", None),
                        aggregator=_mem_agg,
                    )

                    # 1-5. The shared setup sequence, identical to the one the
                    # multi-GPU subprocess runs. Kept in one place because the
                    # two copies drifted once already: image training and the
                    # eval split silently never reached the multi-GPU path.
                    components = build_training_components(
                        training_config,
                        hf_token,
                        ds_config,
                        on_stage=_emit_stage,
                        on_note=writer.info,
                    )
                    strategy = components.strategy
                    training_args = components.training_args
                    model = components.model
                    processing_class = components.processing_class
                    dataset = components.dataset
                    eval_dataset = components.eval_dataset
                    # Rebind these too: _cleanup_training_resources releases
                    # whatever these locals hold at the end of the job, and
                    # leaving them at their initial None leaks the references.
                    tokenizer = components.tokenizer
                    processor = components.processor
                    log_mem("after_model_load")

                    # 6. Prepare Trainer
                    _emit_stage(Phase.PREPARING_TRAINER)
                    trainer = strategy.prepare_trainer(
                        model, processing_class, dataset, training_args, eval_dataset=eval_dataset
                    )
                    log_mem("after_trainer_prepare")
                    _log_zero3_partition_state(trainer)

                    # --- training with progress callbacks ---------------------------------

                    # Prime system_monitor for disk IO to establish baseline before training loop
                    # This ensures the first log captures activity since start of training.
                    # Utilizes the worker process's own SystemMonitor instance.
                    monitor_path = (
                        training_config.offload_folder if training_config.offload_folder else "/"
                    )
                    try:
                        system_monitor.get_disk_resource(
                            "usage", path=monitor_path, calc_size=False
                        )
                    except Exception as e:
                        logger.warning(f"[TrainingWorker] Failed to prime system monitor: {e}")

                    _emit_stage(Phase.TRAINING)
                    status_q.put({"status": "running"})

                    total_steps = None
                    if trainer.state and trainer.state.max_steps:
                        total_steps = trainer.state.max_steps

                    def _log_progress(
                        *,
                        trainer: Trainer = trainer,
                        total_steps: int | None = total_steps,
                        writer: JobLogWriter = writer,
                    ) -> None:
                        """
                        Emit a trainer metric event and a progress status update.

                        Resource snapshots are produced by the background memory sampler;
                        this only covers trainer metrics (loss/lr/epoch/step). Called from
                        inside the worker process only.
                        """
                        try:
                            state = trainer.state
                            if not state:
                                return
                            current_step = int(state.global_step or 0)
                            max_steps = int(state.max_steps or (total_steps or 0) or 1)
                            progress = (
                                float(current_step) / float(max_steps) if max_steps > 0 else 0.0
                            )
                            last_log = state.log_history[-1] if state.log_history else {}
                            # on_log fires for train and eval; the Trainer prefixes
                            # eval entries with `eval_` (eval_loss, eval_accuracy).
                            # Split them so the test-set curve is reported separately
                            # from the training curve rather than mixed together.
                            is_eval = (
                                isinstance(last_log, dict) and last_log.get("eval_loss") is not None
                            )
                            split = "eval" if is_eval else "train"
                            if is_eval:
                                loss_raw = last_log.get("eval_loss")
                                acc = last_log.get("eval_accuracy") or last_log.get(
                                    "eval_mean_token_accuracy"
                                )
                            else:
                                loss_raw = (
                                    last_log.get("loss") if isinstance(last_log, dict) else None
                                )
                                acc = (
                                    last_log.get("mean_token_accuracy")
                                    if isinstance(last_log, dict)
                                    else None
                                )
                            loss_val: float | None = None
                            if loss_raw is not None:
                                try:
                                    loss_val = float(loss_raw)
                                except Exception:
                                    loss_val = None

                            # Structured metric event; the writer persists to the
                            # (train / eval) metrics history only when a valid loss
                            # is present.
                            metric_payload = {
                                "timestamp": time.time(),
                                "step": current_step,
                                "total": max_steps,
                                "progress": progress,
                                "loss": loss_val,
                                "learning_rate": last_log.get("learning_rate"),
                                "epoch": last_log.get("epoch"),
                                "accuracy": acc,
                                "split": split,
                            }
                            writer.metric(metric_payload)

                            status_q.put({"status": "progress", **metric_payload})
                        except Exception:
                            # progress reporting must never abort training
                            pass

                    # custom callback reports progress on log/save events
                    from transformers import TrainerCallback, TrainerControl, TrainerState

                    class _ProgressCallback(TrainerCallback):
                        def on_log(  # type: ignore[override]
                            self,
                            args: TrainingArguments,
                            state: TrainerState,
                            control: TrainerControl,
                            **kwargs: object,
                        ) -> None:
                            _log_progress()
                            log_mem("step")

                    trainer.add_callback(_ProgressCallback())

                    # The memory sampler is already running (started before loading).
                    trainer.train()

                    # 6.5 Final held-out evaluation. The periodic evals above give
                    # the curve; this is the end-state summary on the full test
                    # split, emitted as a metric with split="eval" and final=True.
                    if eval_dataset is not None:
                        try:
                            eval_metrics = trainer.evaluate()
                            final_loss = eval_metrics.get("eval_loss")
                            final_acc = eval_metrics.get("eval_accuracy") or eval_metrics.get(
                                "eval_mean_token_accuracy"
                            )
                            writer.metric(
                                {
                                    "timestamp": time.time(),
                                    "step": int(getattr(trainer.state, "global_step", 0) or 0),
                                    "loss": float(final_loss) if final_loss is not None else None,
                                    "accuracy": final_acc,
                                    "epoch": getattr(trainer.state, "epoch", None),
                                    "split": "eval",
                                    "final": True,
                                }
                            )
                            writer.info(
                                "final held-out evaluation",
                                {
                                    "eval_loss": final_loss,
                                    "eval_accuracy": final_acc,
                                    "eval_size": len(eval_dataset),
                                },
                            )
                            logger.info(f"[TrainingWorker] Final eval metrics: {eval_metrics}")
                        except Exception as _eval_err:
                            # Evaluation failure must not lose a trained model.
                            logger.warning(f"[TrainingWorker] Final evaluation failed: {_eval_err}")
                            writer.info("final evaluation skipped", {"reason": str(_eval_err)})

                    # 6.5 Generation check. Before the save, while the trained
                    # model is still in memory. eval_loss uses the same collator
                    # and the same completion_only_loss as training, so it
                    # cannot detect a model that converged on a degenerate rule;
                    # only asking it a question can. Never fails the run -- the
                    # adapter is still worth saving, the verdict is a report.
                    _check_path = getattr(training_config, "generation_check_path", None)
                    if _check_path:
                        try:
                            from .core.generation_check import load_probes, run_generation_check

                            _probes = load_probes(_check_path)
                            _verdict = run_generation_check(
                                trainer.model, processing_class, _probes
                            )
                            writer.info("generation check", _verdict)
                            if _verdict["failed"]:
                                logger.warning(
                                    f"[TrainingWorker] Generation check: "
                                    f"{_verdict['passed']}/{_verdict['total']} probes passed -- "
                                    "the loss curve alone would not have shown this"
                                )
                        except Exception as _check_err:
                            logger.warning(
                                f"[TrainingWorker] Generation check failed: {_check_err}"
                            )
                            writer.info("generation check skipped", {"reason": str(_check_err)})

                    # 7. Save Results
                    _emit_stage(Phase.SAVING)
                    # Saves the processor rather than the bare tokenizer for a
                    # vision-language run, so the output dir can actually process
                    # images at inference time.
                    save_training_results(trainer, processing_class, training_config)

                    # After a successful run, free GPU / DeepSpeed resources before GGUF conversion
                    _cleanup_training_resources(
                        trainer=trainer,
                        model=model,
                        base_model=base_model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        deepspeed_initialized=deepspeed_initialized,
                    )
                    # Reset variables after cleanup
                    trainer = None
                    model = None
                    base_model = None
                    tokenizer = None
                    processor = None
                    processing_class = None
                    dataset = None

                    logger.info(
                        "[TrainingWorker] Saved training artifacts. Starting GGUF Q4_K_M conversion..."
                    )
                    _conv_phase = {
                        "merge fine-tune": Phase.MERGING_LORA,
                        "convert to gguf": Phase.CONVERTING_GGUF,
                    }

                    def conversion_callback(
                        step: str, *, _conv_phase: dict[str, str] = _conv_phase
                    ) -> None:
                        _emit_stage(_conv_phase.get(step, Phase.CONVERTING_GGUF), step)

                    conversion_result = _convert_training_output_to_q4_k_m(
                        training_config, status_callback=conversion_callback
                    )

                    _emit_stage(Phase.COMPLETED)
                    _completed_result = {
                        "gguf_path": conversion_result.get("quantized_output_path"),
                        "gguf_quantization": conversion_result.get("quantization_type"),
                        "mmproj_path": conversion_result.get("mmproj_output_path"),
                    }
                    writer.finalize(Phase.COMPLETED, result=_completed_result)
                    status_q.put({"status": "completed", **_completed_result})

                    # Training finished — exit worker process to fully release resources.
                    logger.info(
                        "[TrainingWorker] Training completed successfully, exiting worker process..."
                    )
                    return  # exit worker process

                except Exception as e:  # start command failure
                    import traceback

                    tb = traceback.format_exc()
                    logger.exception(f"[TrainingWorker] Training failed: {e}\n{tb}")

                    # _DeepSpeedTrainingError carries the subprocess's own traceback
                    # and is_oom flag — use them directly to avoid losing detail.
                    if isinstance(e, _DeepSpeedTrainingError):
                        error_str = e.error
                        tb = e.subprocess_traceback or tb
                        is_oom = e.is_oom
                    else:
                        # Detect whether this is a CUDA OOM / CUDA error
                        error_str = str(e).strip()
                        if not error_str:
                            try:
                                error_str = str(getattr(e, "args", [""])[0]).strip()
                            except Exception:
                                error_str = ""
                        if not error_str:
                            try:
                                error_str = tb.strip().splitlines()[-1]
                            except Exception:
                                error_str = "Unknown error"
                        lower = error_str.lower()
                        is_oom = False
                        try:
                            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                                isinstance(e, RuntimeError)
                                and ("out of memory" in lower or "cuda" in lower or "oom" in lower)
                            )
                        except Exception:
                            is_oom = False

                    # Record the real error to the job log (the single source of truth)
                    # and surface it to the manager.
                    if writer is not None:
                        try:
                            writer.error(error_str, traceback=tb, is_oom=is_oom)
                            writer.finalize(Phase.ERROR)
                        except Exception:
                            pass
                    status_q.put(
                        {
                            "status": "error",
                            "error": error_str,
                            "traceback": tb,
                            "is_oom": is_oom,
                        }
                    )

                    # Clean up resources whether or not this was an OOM
                    logger.info("[TrainingWorker] Cleaning up resources after error...")
                    _cleanup_training_resources(
                        trainer=trainer,
                        model=model,
                        base_model=base_model,
                        tokenizer=tokenizer,
                        dataset=dataset if "dataset" in locals() else None,
                        deepspeed_initialized=deepspeed_initialized,
                    )

                    if is_oom:
                        # On OOM / CUDA errors, exit the whole process so VRAM is fully released
                        logger.exception(
                            "[TrainingWorker] OOM/CUDA error detected, forcing process exit to release VRAM..."
                        )
                        try:
                            status_q.put(
                                {
                                    "status": "error",
                                    "error": f"OOM Error: {error_str}",
                                    "traceback": tb,
                                    "is_oom": True,
                                }
                            )
                        except Exception:
                            pass

                        # Exit the process outright so the CUDA context is released
                        import sys

                        sys.exit(1)
                    else:
                        # Non-OOM error: also exit the worker process after cleanup
                        logger.info(
                            "[TrainingWorker] Error occurred, exiting worker process after cleanup..."
                        )
                        return  # exit worker process
                finally:
                    # Always stop the sampler and close the job log on every exit path
                    # (success return, error, OOM sys.exit, output-dir continue).
                    if _mem_stop is not None:
                        try:
                            _mem_stop.set()
                            if _mem_thread is not None:
                                _mem_thread.join(timeout=5.0)
                        except Exception:
                            pass
                    # Record avg/peak GPU/DRAM/SSD for the run. Emitted on every
                    # exit path, including OOM -- a failed run's peak is the
                    # single most useful number it can leave behind.
                    if _mem_agg is not None and _mem_agg.samples:
                        try:
                            logger.info(f"[TrainingWorker] {_mem_agg.format_line()}")
                            if writer is not None:
                                writer.info("memory usage summary", _mem_agg.summary())
                        except Exception:
                            pass
                    if writer is not None:
                        try:
                            writer.close()
                        except Exception:
                            pass

            elif command == "stop":
                logger.info("[TrainingWorker] stop command received")
                if trainer is not None:
                    try:
                        # `_max_steps` is not a real Trainer attribute; the
                        # supported way to request a graceful stop is the
                        # TrainerControl flag checked at each step.
                        if getattr(trainer, "control", None) is not None:
                            trainer.control.should_training_stop = True
                    except Exception:
                        pass

                # Clean up training resources
                _cleanup_training_resources(
                    trainer=trainer,
                    model=model if "model" in locals() else None,
                    base_model=base_model if "base_model" in locals() else None,
                    tokenizer=tokenizer if "tokenizer" in locals() else None,
                    dataset=dataset if "dataset" in locals() else None,
                    deepspeed_initialized=deepspeed_initialized,
                )
                trainer = None

                status_q.put({"status": "stopped"})

                # Exit the worker process after a stop command
                logger.info("[TrainingWorker] Stop command processed, exiting worker process...")
                return  # exit worker process

        # Also run a final cleanup when the loop exits normally
        logger.info("[TrainingWorker] Worker loop ended, performing final cleanup...")
        _cleanup_training_resources(
            trainer=trainer,
            model=model if "model" in locals() else None,
            base_model=base_model if "base_model" in locals() else None,
            tokenizer=tokenizer if "tokenizer" in locals() else None,
            dataset=dataset if "dataset" in locals() else None,
            deepspeed_initialized=deepspeed_initialized,
        )

        logger.info("[TrainingWorker] stop_event set; worker exiting")

    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.exception(f"[TrainingWorker] Fatal error: {e}\n{tb}")
        try:
            status_q.put({"status": "error", "error": str(e), "traceback": tb})
        except Exception:
            pass

        # Clean up resources on fatal errors too
        try:
            logger.info("[TrainingWorker] Cleaning up resources after fatal error...")
            _cleanup_training_resources(
                trainer=trainer if "trainer" in locals() else None,
                model=model if "model" in locals() else None,
                base_model=base_model if "base_model" in locals() else None,
                tokenizer=tokenizer if "tokenizer" in locals() else None,
                dataset=dataset if "dataset" in locals() else None,
                deepspeed_initialized=deepspeed_initialized,
            )
        except Exception as cleanup_err:
            logger.exception(f"[TrainingWorker] Cleanup after fatal error failed: {cleanup_err}")

    finally:
        # Final cleanup: last line of defense so the distributed process group is torn down
        try:
            if deepspeed_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("[TrainingWorker] Destroyed distributed process group in finally block")
        except Exception as final_err:
            logger.debug(f"[TrainingWorker] Final cleanup: {final_err}")


class TrainingProcessManager:
    """
    Manage a dedicated training worker process.

    This mirrors `ModelInferenceProcess` but focused on a single training job
    at a time. It is used by `TrainingManager` to decouple heavy GPU work from
    the FastAPI process.
    """

    def __init__(self) -> None:
        self.process: Process | None = None
        self.request_q: Queue | None = None
        self.status_q: Queue | None = None
        self.stop_event: EventClass | None = None

        # current_status values:
        # - "idle": idle (initial state or after reset)
        # - "initializing": initializing (loading model/dataset)
        # - "running": training in progress
        # - "completed": training finished normally
        # - "stopped": training stopped manually
        # - "error": training failed (OOM or another exception)
        self.current_status: str = "idle"
        self.last_error: str | None = None
        self.last_traceback: str | None = None
        # Fine-grained lifecycle phase (see core.Phase). Kept separate from
        # `last_error`, which now only ever holds a real error.
        self.current_phase: str | None = None
        self.phase_detail: str | None = None
        self.current_config: TrainingConfig | None = None
        # progress fields
        self.current_step: int = 0
        self.total_steps: int = 0
        self.progress: float = 0.0
        self.loss: float | None = None
        self.current_epoch: float = 0.0
        # track OOM flag for last error (optional, for future /training/error_details)
        self.last_is_oom = False

        # In-memory history fallback
        self.history: dict[str, dict[str, Any]] = {}

        # Redis connection, when one is configured. Not configured and cannot connect
        # are different things and used to log the same way: with no Redis configured
        # the client is None, the assert below turned that into an exception, and the
        # handler reported "Failed to connect to Redis:" with nothing after the colon.
        # A warning about a failure that did not happen is worse than silence -- it is
        # the first thing someone reads when they go looking for a real problem.
        self.redis_client = None
        if redis:
            client = _create_redis_client()
            if client is None:
                logger.info("Training history: in-memory (no Redis configured)")
            else:
                try:
                    client.ping()
                    self.redis_client = client
                except Exception as e:
                    logger.warning("Redis is configured but did not answer: %s", e)

        # Format helpers
        self._key_metrics = lambda sid: f"training:history:{sid}:metrics"
        self._key_eval_metrics = lambda sid: f"training:history:{sid}:eval_metrics"
        self._key_resources = lambda sid: f"training:history:{sid}:resources"

        self.current_session_id: str | None = None

    def _ensure_process(self) -> None:
        if self.process and self.process.is_alive():
            return
        self.request_q = Queue()
        self.status_q = Queue()
        self.stop_event = Event()
        self.process = Process(
            target=_training_worker_process,
            args=(self.request_q, self.status_q, self.stop_event),
            daemon=True,
        )
        self.process.start()
        self.current_status = "idle"
        self.last_error = None
        self.last_traceback = None

    def start_training(self, config: TrainingConfig) -> dict[str, Any]:
        """
        Start a new training job in the worker process.
        """
        if self.current_status in {"running", "initializing"}:
            raise RuntimeError("Training already in progress")

        # Clear prior training state so stale results are not picked up
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        self.loss = None
        self.current_epoch = 0.0
        self.total_epochs = 0
        self.last_error = None
        self.last_traceback = None
        self.last_is_oom = False
        self.current_phase = Phase.PENDING
        self.phase_detail = None
        logger.info("Cleared previous training status before starting new training")

        self._ensure_process()
        assert self.request_q is not None  # noqa: S101 - type-narrowing guard after _ensure_process
        self.current_config = config

        # Setup session and history
        self.current_session_id = str(uuid.uuid4())
        self.history[self.current_session_id] = {
            "training_logs": [],
            "resource_logs": [],
            "config": config.model_dump(),
            "start_time": time.time(),
        }

        # Clear Redis history if exists (unlikely for new UUID)
        if self.redis_client:
            try:
                self.redis_client.delete(self._key_metrics(self.current_session_id))
                self.redis_client.delete(self._key_eval_metrics(self.current_session_id))
                self.redis_client.delete(self._key_resources(self.current_session_id))
            except Exception:
                pass

        self.request_q.put(
            {
                "command": "start",
                "config": config.model_dump(),
                "session_id": self.current_session_id,
            }
        )
        self.current_status = "starting"
        return {"status": "starting", "session_id": self.current_session_id}

    def stop_training(self) -> dict[str, Any]:
        """
        Force-stop the training worker process and reset process handles.
        """
        if not self.process:
            return {"status": "not_running"}

        logger.info("[TrainingManager] Force stopping training process...")

        # Terminate the process directly without waiting on the queue
        if self.process.is_alive():
            try:
                self.process.terminate()
                self.process.join(timeout=2)
                if self.process.is_alive():
                    logger.warning("[TrainingManager] Process did not terminate, killing it...")
                    self.process.kill()
                    self.process.join(timeout=1)
            except Exception as e:
                logger.exception(f"[TrainingManager] Error stopping process: {e}")

        # Drop resource references
        self.process = None
        self.request_q = None
        self.status_q = None
        self.stop_event = None

        self.current_status = "stopped"
        return {"status": "stopped"}

    def _drain_status(self) -> None:
        if not self.status_q:
            return
        while True:
            try:
                msg = self.status_q.get_nowait()
            except Empty:
                break
            except (AttributeError, ValueError):
                # Queue might be closed or None during cleanup race condition
                break
            status = msg.get("status")
            # Only update current_status for state-changing statuses. "progress",
            # "node" (legacy) and "stage" carry fine-grained info, not coarse state.
            if status and status not in {"progress", "node", "stage"}:
                self.current_status = status

            if status == "stage":
                # Fine-grained lifecycle update — NOT an error.
                self.current_phase = msg.get("phase")
                self.phase_detail = msg.get("phase_detail")

            if status == "error":
                self.last_error = msg.get("error")
                self.last_traceback = msg.get("traceback")
                self.last_is_oom = bool(msg.get("is_oom"))

                # If the worker reports OOM or another fatal error, stop training and
                # clean up the process so the next run starts on a fresh worker
                if self.last_is_oom:
                    logger.error(
                        "[TrainingManager] CUDA OOM detected in training worker; stopping training."
                    )
                else:
                    logger.error(
                        "[TrainingManager] Training worker reported error; stopping training."
                    )
                # Call stop_training() to trigger the cleanup flow
                self.stop_training()
            elif status == "progress":
                # incremental progress report from worker
                self.current_step = int(msg.get("step", self.current_step))
                self.total_steps = int(msg.get("total", self.total_steps or 0) or 0)
                self.progress = float(msg.get("progress", self.progress))
                loss_val = msg.get("loss")
                epoch_val = msg.get("epoch")

                if loss_val is not None:
                    try:
                        self.loss = float(loss_val)
                    except Exception:
                        pass

                if epoch_val is not None:
                    try:
                        self.current_epoch = float(epoch_val)
                    except Exception:
                        pass

    def get_status(self) -> TrainingStatus:
        """
        Return the current training status, draining any pending worker updates.
        """
        # Check if worker process died unexpectedly (e.g., from sys.exit(1) on OOM)
        if self.process and not self.process.is_alive():
            if self.current_status in {"running", "initializing"}:
                # Worker was active but died - likely OOM or fatal error
                if not self.last_error:
                    self.last_error = "Worker process terminated unexpectedly"
                self.current_status = "error"
                # Clean up the dead process
                self.cleanup()

        self._drain_status()
        # `error` only ever carries a real error now (progress lives in `phase`).
        error_msg = self.last_error if self.current_status == "error" else None
        if self.current_status not in {"starting", "running", "initializing"}:
            # Training not active, but preserve progress info if completed
            return TrainingStatus(
                is_training=False,
                progress=self.progress,
                current_step=self.current_step,
                total_steps=self.total_steps,
                loss=self.loss,
                current_epoch=self.current_epoch,
                total_epochs=self.current_config.num_train_epochs if self.current_config else None,
                status=self.current_status,
                phase=self.current_phase,
                phase_detail=self.phase_detail,
                session_id=self.current_session_id,
                job_id=self.current_session_id,
                error=error_msg,
                config=None,  # config only returned while training is active
            )
        return TrainingStatus(
            is_training=True,
            progress=self.progress,
            current_step=self.current_step,
            total_steps=self.total_steps,
            loss=self.loss,
            current_epoch=self.current_epoch,
            total_epochs=self.current_config.num_train_epochs if self.current_config else None,
            status=self.current_status,
            phase=self.current_phase,
            phase_detail=self.phase_detail,
            session_id=self.current_session_id,
            job_id=self.current_session_id,
            error=error_msg,  # real errors only; progress is in `phase`/`phase_detail`
            config=self.current_config,
        )

    def read_log_events(self, job_id: str, since: int = 0) -> list[dict[str, Any]]:
        """
        Read structured log events for a job from its events.jsonl on disk.

        Works regardless of which process produced them (the file is the durable
        source of truth), so the API can serve logs even if Redis is down.
        """
        return read_events(job_id, since)

    def get_history(self, session_id: str) -> dict[str, Any] | None:
        """
        Return metric/resource history for a session from Redis or memory.
        """
        # 1. Try Redis
        if self.redis_client:
            try:
                metrics_key = self._key_metrics(session_id)
                eval_metrics_key = self._key_eval_metrics(session_id)
                resources_key = self._key_resources(session_id)

                # Fetch all list items (sync redis lrange returns a list; stub types it as Awaitable)
                m_list = cast("list[Any]", self.redis_client.lrange(metrics_key, 0, -1))
                e_list = cast("list[Any]", self.redis_client.lrange(eval_metrics_key, 0, -1))
                r_list = cast("list[Any]", self.redis_client.lrange(resources_key, 0, -1))

                # Redis returns list of strings, parse them
                training_logs = [json.loads(x) for x in m_list] if m_list else []
                eval_logs = [json.loads(x) for x in e_list] if e_list else []
                resource_logs = [json.loads(x) for x in r_list] if r_list else []

                if training_logs or eval_logs or resource_logs:
                    return {
                        "training_logs": training_logs,
                        "eval_logs": eval_logs,
                        "resource_logs": resource_logs,
                        "session_id": session_id,
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch history from Redis: {e}")

        # 2. Fallback to in-memory
        return self.history.get(session_id)

    def get_error_details(self) -> dict[str, Any] | None:
        """
        Return detailed error info, including the full traceback.

        Matches the inference/error_details format.
        """
        if not self.last_error:
            return None

        return {
            "error": self.last_error,
            "error_type": "OOM" if self.last_is_oom else "TrainingError",
            "is_oom": self.last_is_oom,
            "error_traceback": self.last_traceback,
            "process_alive": self.process.is_alive() if self.process else False,
        }

    def cleanup(self) -> None:
        """
        Terminate any live worker process and reset manager state to idle.
        """
        if self.process and self.process.is_alive():
            try:
                if self.stop_event is not None:
                    self.stop_event.set()
                self.process.terminate()
                self.process.join(timeout=5)
            except Exception:
                pass
        self.process = None
        self.request_q = None
        self.status_q = None
        self.stop_event = None
        self.current_status = "idle"
        self.last_error = None
        self.last_traceback = None
        self.current_phase = None
        self.phase_detail = None
        self.current_step = 0
        self.total_steps = 0
        self.progress = 0.0
        self.loss = None
        self.current_epoch = 0.0


training_process_manager = TrainingProcessManager()
