"""
GGUF conversion job management for HF, LoRA, and multimodal checkpoints.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable
from typing import Any, cast

from service.model_registry import model_registry
from service.settings import HF_HOME

logger = logging.getLogger(__name__)


class ConversionJob:
    """
    State container for a single GGUF conversion job.
    """

    def __init__(
        self,
        job_id: str,
        model_path: str,
        output_path: str,
        outtype: str,
        base_model_path: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.model_path = model_path
        self.output_path = output_path
        self.outtype = outtype
        self.base_model_path = base_model_path
        self.status = "pending"
        self.message = "Job created"
        self.process = None
        self.error: str | None = None


class ConversionManager:
    """
    Manage GGUF conversion jobs and llama.cpp conversion tooling.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, ConversionJob] = {}
        # Path to the script inside llama.cpp folder
        # Directory: service/utils/llama.cpp/ or configured via env
        self.llama_cpp_dir = os.getenv(
            "LLAMA_CPP_DIR", os.path.join(os.path.dirname(__file__), "llama.cpp")
        )
        self.llama_cpp_dir = os.path.abspath(self.llama_cpp_dir)
        self.script_path = os.path.join(self.llama_cpp_dir, "convert_hf_to_gguf.py")
        self.lora_script_path = os.path.join(self.llama_cpp_dir, "convert_lora_to_gguf.py")
        self.merger_script = os.path.join(os.path.dirname(__file__), "lora_merger.py")
        self.quantize_binary = self._resolve_quantize_binary()

    def _resolve_quantize_binary(self) -> str | None:
        """
        Resolve a quantize binary.

        Prefers an explicit `LLAMA_QUANTIZE_BIN`, then a legacy standalone
        `llama-quantize`, then the official prebuilt unified `llama` (from
        ggml-org/llama-install.sh), which exposes quantize as a `quantize`
        subcommand — see `_quantize_cmd_prefix`. No source build required.
        """
        candidates = []

        configured = os.getenv("LLAMA_QUANTIZE_BIN", "").strip()
        if configured:
            candidates.append(configured)

        candidates.extend(
            [
                os.path.join(self.llama_cpp_dir, "build", "bin", "llama-quantize"),
                os.path.join(self.llama_cpp_dir, "bin", "llama-quantize"),
                os.path.join(self.llama_cpp_dir, "llama-quantize"),
            ]
        )

        # Prebuilt unified binary (quantize is a subcommand there).
        if os.name == "nt":
            local_app = os.environ.get("LOCALAPPDATA")
            if local_app:
                candidates.append(os.path.join(local_app, "Microsoft", "WindowsApps", "llama.exe"))
        else:
            candidates.append(os.path.join(os.path.expanduser("~"), ".local", "bin", "llama"))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)

        return None

    @staticmethod
    def _quantize_cmd_prefix(binary: str) -> list[str]:
        """
        Command prefix for invoking quantize.

        The legacy standalone `llama-quantize` takes args directly; the unified
        `llama` binary needs a `quantize` subcommand.
        """
        if "quantize" in os.path.basename(binary).lower():
            return [binary]
        return [binary, "quantize"]

    def _build_env(self) -> dict[str, str]:
        """Build runtime environment for llama.cpp conversion scripts."""
        env = os.environ.copy()
        gguf_py_path = os.path.join(self.llama_cpp_dir, "gguf-py")
        if "PYTHONPATH" in env and env["PYTHONPATH"]:
            env["PYTHONPATH"] = f"{gguf_py_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = gguf_py_path

        library_dirs: list[str] = [
            candidate
            for candidate in (
                os.path.join(self.llama_cpp_dir, "build", "bin"),
                os.path.join(self.llama_cpp_dir, "bin"),
            )
            if os.path.isdir(candidate)
        ]

        if self.quantize_binary:
            quantize_dir = os.path.dirname(os.path.abspath(self.quantize_binary))
            if os.path.isdir(quantize_dir) and quantize_dir not in library_dirs:
                library_dirs.append(quantize_dir)

        if library_dirs:
            existing_ld = env.get("LD_LIBRARY_PATH", "").strip()
            env["LD_LIBRARY_PATH"] = ":".join(
                [*library_dirs, *([existing_ld] if existing_ld else [])]
            )

        return env

    def _run_command(
        self, cmd: list[str], env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        """Run subprocess and raise a readable error when it fails."""
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            message = stderr or stdout or f"Command failed with code {result.returncode}"
            raise RuntimeError(message)

        return result

    def _resolve_output_file(self, requested_output_path: str) -> str:
        """Resolve final GGUF path when converter outputs to a directory or split files."""
        actual_output_file = requested_output_path
        if os.path.isdir(actual_output_file):
            found_ggufs = [f for f in os.listdir(actual_output_file) if f.lower().endswith(".gguf")]
            if found_ggufs:
                found_ggufs.sort(
                    key=lambda name: os.path.getmtime(os.path.join(actual_output_file, name)),
                    reverse=True,
                )
                actual_output_file = os.path.join(actual_output_file, found_ggufs[0])
        elif not os.path.exists(actual_output_file):
            parent_dir = os.path.dirname(actual_output_file) or "."
            requested_name = os.path.splitext(os.path.basename(actual_output_file))[0]
            if os.path.isdir(parent_dir):
                found_ggufs = [
                    os.path.join(parent_dir, name)
                    for name in os.listdir(parent_dir)
                    if name.lower().endswith(".gguf")
                    and requested_name in os.path.splitext(name)[0]
                ]
                if found_ggufs:
                    found_ggufs.sort(key=os.path.getmtime, reverse=True)
                    actual_output_file = found_ggufs[0]
        return actual_output_file

    def _register_gguf_model(
        self,
        model_path: str,
        actual_output_file: str,
        outtype: str,
        *,
        is_lora: bool,
        base_model_path: str | None = None,
    ) -> None:
        """Register converted GGUF model into registry."""
        model_dir_name = os.path.basename(os.path.normpath(model_path))
        label = f"{model_dir_name}-{outtype}-gguf"

        if is_lora:
            hf_model_name = self._find_base_model_name(model_path) or model_dir_name
        else:
            hf_model_name = model_dir_name

        max_context_length = self._find_base_model_context_length(
            model_path=model_path,
            is_lora=is_lora,
            base_model_path=base_model_path,
        )

        model_registry.add_llama_gguf_model(
            label=label,
            base_model_name=hf_model_name,
            local_path=actual_output_file,
            filename=actual_output_file,
            source="gguf",
            size=f"converted-{outtype}",
            max_context_length=max_context_length,
        )
        logger.info(f"Registered converted model: {label} located at {actual_output_file}")

    def _find_base_model_context_length(
        self,
        *,
        model_path: str,
        is_lora: bool,
        base_model_path: str | None = None,
    ) -> int | None:
        """Resolve max context length from the corresponding base model entry."""
        registry_data = model_registry.list_models()
        candidate_paths: set[str] = set()
        candidate_names: set[str] = set()

        if model_path:
            candidate_paths.add(os.path.normpath(model_path))

        if base_model_path:
            candidate_paths.add(os.path.normpath(base_model_path))

        if is_lora:
            base_model_name = self._find_base_model_name(model_path)
            if base_model_name:
                candidate_names.add(base_model_name)
                if os.path.isdir(base_model_name):
                    candidate_paths.add(os.path.normpath(base_model_name))

        for base_model in registry_data.get("base_models", []):
            max_context_length = base_model.get("max_context_length")
            if max_context_length is None:
                continue

            base_model_name = base_model.get("model_name") or base_model.get("base_model_name")
            base_model_model_path = base_model.get("model_path") or base_model.get("local_path")

            if base_model_model_path and os.path.normpath(base_model_model_path) in candidate_paths:
                return max_context_length

            if base_model_name and base_model_name in candidate_names:
                return max_context_length

        return None

    def _convert_to_gguf(
        self,
        *,
        model_path: str,
        output_path: str,
        outtype: str,
        base_model_path: str | None = None,
        register_model: bool = True,
        work_dir: str | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Convert HF/LoRA model to GGUF synchronously."""
        temp_dir = None
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        script_to_run = self.script_path
        target_model_path = model_path
        env = self._build_env()

        if work_dir:
            env["TMPDIR"] = work_dir
            os.makedirs(work_dir, exist_ok=True)

        try:
            if is_lora:
                if not base_model_path:
                    base_model_path = self._find_base_model_path(model_path)
                if not base_model_path:
                    raise ValueError("Base model path is required for LoRA conversion")

                temp_dir = tempfile.mkdtemp(dir=work_dir, prefix="lora_merged_")
                merged_model_dir = os.path.join(temp_dir, "merged_model")
                offload_dir = os.path.join(temp_dir, "offload")
                os.makedirs(merged_model_dir, exist_ok=True)
                os.makedirs(offload_dir, exist_ok=True)

                if status_callback:
                    status_callback("merge fine-tune")

                merge_cmd = [
                    sys.executable,
                    self.merger_script,
                    base_model_path,
                    model_path,
                    merged_model_dir,
                    "--offload",
                    offload_dir,
                ]
                self._run_command(merge_cmd, env=env)
                target_model_path = merged_model_dir

                if status_callback:
                    status_callback("complete")

            if not os.path.exists(script_to_run):
                raise FileNotFoundError(f"Conversion script not found at {script_to_run}")

            if status_callback:
                status_callback("convert to gguf")

            def _run_convert(target: str) -> None:
                cmd = [
                    sys.executable,
                    script_to_run,
                    target,
                    "--outtype",
                    outtype,
                    "--outfile",
                    output_path,
                ]
                if work_dir:
                    cmd.append("--use-temp-file")
                self._run_command(cmd, env=env)

            try:
                _run_convert(target_model_path)
            except Exception as direct_err:
                # A full-parameter multimodal fine-tune saves the whole
                # vision-language checkpoint; llama.cpp cannot always map its
                # tensors/tokenizer (e.g. gemma3: the language tower is nested as
                # ``model.language_model.model.*`` and the SentencePiece tokenizer
                # is dropped at save time). Fall back to extracting the fine-tuned
                # language tower into a clean text checkpoint -- the same shape the
                # LoRA merge produces -- and convert that. LoRA (already merged) and
                # text-only checkpoints re-raise unchanged.
                if is_lora or not self._is_multimodal(model_path):
                    raise
                logger.warning(
                    "[Conversion] Direct GGUF conversion failed (%s); retrying via "
                    "language-tower extraction (full-parameter multimodal fine-tune)",
                    direct_err,
                )
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp(dir=work_dir, prefix="lang_extract_")
                lang_dir = os.path.join(temp_dir, "language_model")
                resolved_base = base_model_path or self._find_base_model_path(model_path)
                self._extract_language_model(model_path, lang_dir, resolved_base)
                _run_convert(lang_dir)

            actual_output_file = self._resolve_output_file(output_path)
            if register_model:
                self._register_gguf_model(
                    model_path=model_path,
                    actual_output_file=actual_output_file,
                    outtype=outtype,
                    is_lora=is_lora,
                    base_model_path=base_model_path,
                )

            return {
                "output_path": output_path,
                "actual_output_file": actual_output_file,
                "is_lora": is_lora,
                "base_model_path": base_model_path,
            }
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @staticmethod
    def _copy_tokenizer(src_dir: str, dest_dir: str) -> None:
        """
        Copy tokenizer files verbatim, preferring a SentencePiece source.

        The training save writes only a fast ``tokenizer.json`` and drops the
        SentencePiece ``tokenizer.model``; without it llama.cpp takes an
        unrecognised-BPE path and refuses gemma tokenizers. Copying the base
        model's full tokenizer set restores ``tokenizer.model``.
        """
        for fn in (
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ):
            p = os.path.join(src_dir, fn)
            if os.path.exists(p):
                shutil.copy(p, dest_dir)

    def _extract_language_model(
        self, model_path: str, dest_dir: str, base_model_path: str | None
    ) -> None:
        """
        Extract a multimodal checkpoint's language tower into a clean text LM.

        A full-parameter multimodal fine-tune saves the whole vision-language
        model (``model.language_model.*`` + ``vision_tower`` + projector), which
        some llama.cpp converters can't map. Re-saving only the language tower as
        an ``AutoModelForCausalLM`` yields the standard text layout
        (``model.embed_tokens.*`` + ``lm_head``) -- the same shape the LoRA merge
        emits -- so the existing text conversion path handles it. The base
        tokenizer is copied over because the fine-tune's save may be incomplete.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        from .model_class_resolver import find_language_tower_path

        os.makedirs(dest_dir, exist_ok=True)
        logger.info("[Conversion] Extracting language tower from %s", model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
        )
        text_config = getattr(model.config, "text_config", model.config)
        lang_path = find_language_tower_path(model)
        if not lang_path:
            raise ValueError(
                f"Could not locate a language tower in {model_path} to extract for GGUF"
            )
        language = model.get_submodule(lang_path)

        causal = AutoModelForCausalLM.from_config(text_config)
        missing, unexpected = causal.model.load_state_dict(language.state_dict(), strict=False)
        if unexpected:
            logger.warning(
                "[Conversion] language extraction dropped %d unexpected key(s)", len(unexpected)
            )
        causal = causal.to(torch.bfloat16)
        causal.tie_weights()
        causal.save_pretrained(dest_dir, safe_serialization=True)

        self._copy_tokenizer(base_model_path or model_path, dest_dir)
        logger.info("[Conversion] Language tower extracted to %s", dest_dir)

    def _is_multimodal(self, model_path: str | None) -> bool:
        """True when a checkpoint has a vision (or audio) config -> needs mmproj."""
        if not model_path:
            return False
        try:
            from ..utils.model_class_resolver import is_multimodal_model

            return is_multimodal_model(model_path, local_files_only=True)
        except Exception as e:
            logger.warning(f"Could not determine if {model_path} is multimodal: {e}")
            return False

    def _export_mmproj(
        self,
        *,
        source_model_path: str,
        output_dir: str,
        outtype: str,
        work_dir: str | None = None,
    ) -> str | None:
        """
        Export the vision projector (mmproj) GGUF for a multimodal model.

        llama.cpp represents a VLM as two files: the language GGUF and this
        mmproj. LoRA only adapts the language tower, so the vision half is
        exported from the (unchanged) base model. Best-effort: llama.cpp
        supports mmproj for many but not all vision architectures, so a failure
        here is logged and returns None rather than failing the whole job -- the
        language GGUF is still valid on its own.
        """
        model_name = os.path.basename(os.path.normpath(source_model_path))
        mmproj_path = os.path.join(output_dir, f"mmproj-{model_name}-{outtype.upper()}.gguf")
        cmd = [
            sys.executable,
            self.script_path,
            source_model_path,
            "--mmproj",
            "--outtype",
            outtype,
            "--outfile",
            mmproj_path,
        ]
        env = self._build_env()
        if work_dir:
            env["TMPDIR"] = work_dir
            cmd.append("--use-temp-file")
        try:
            self._run_command(cmd, env=env)
        except Exception as e:
            logger.warning(
                f"mmproj export failed for {source_model_path} (architecture may be "
                f"unsupported by llama.cpp); language GGUF is still usable: {e}"
            )
            return None
        actual = self._resolve_output_file(mmproj_path)
        logger.info(f"Exported mmproj (vision projector) to {actual}")
        return actual

    def convert_and_quantize(
        self,
        model_path: str,
        *,
        output_dir: str | None = None,
        intermediate_outtype: str = "f16",
        quantization_type: str = "Q4_K_M",
        base_model_path: str | None = None,
        work_dir: str | None = None,
        status_callback: Callable[[str], None] | None = None,
        export_mmproj: bool = False,
    ) -> dict[str, str]:
        """
        Convert a training output to GGUF and quantize it.

        This method is intended for post-training automation, so it runs
        synchronously and raises on failure.
        """
        model_dir_name = os.path.basename(os.path.normpath(model_path))
        output_dir = output_dir or model_path
        os.makedirs(output_dir, exist_ok=True)

        intermediate_name = f"{model_dir_name}-{intermediate_outtype.upper()}.gguf"
        intermediate_path = os.path.join(output_dir, intermediate_name)

        conversion_result = self._convert_to_gguf(
            model_path=model_path,
            output_path=intermediate_path,
            outtype=intermediate_outtype,
            base_model_path=base_model_path,
            register_model=False,
            work_dir=work_dir,
            status_callback=status_callback,
        )

        quantize_binary = self.quantize_binary or self._resolve_quantize_binary()
        self.quantize_binary = quantize_binary
        if not quantize_binary:
            raise FileNotFoundError(
                "llama-quantize binary not found. Please provide `LLAMA_QUANTIZE_BIN` or mount a prebuilt `llama-quantize` binary."
            )

        quantized_name = f"{model_dir_name}-{quantization_type}.gguf"
        quantized_path = os.path.join(output_dir, quantized_name)
        quantize_cmd = [
            *self._quantize_cmd_prefix(quantize_binary),
            conversion_result["actual_output_file"],
            quantized_path,
            quantization_type,
        ]
        self._run_command(quantize_cmd, env=self._build_env())

        is_lora = bool(conversion_result.get("is_lora"))
        self._register_gguf_model(
            model_path=model_path,
            actual_output_file=quantized_path,
            outtype=quantization_type,
            is_lora=is_lora,
            base_model_path=base_model_path,
        )

        # For a vision-language model, also export the mmproj companion so the
        # GGUF pair can actually do image inference. The vision half comes from
        # the base model (LoRA only touched the language tower).
        mmproj_output_path = None
        resolved_base = conversion_result.get("base_model_path") or base_model_path
        mmproj_source = resolved_base or model_path
        if export_mmproj and self._is_multimodal(mmproj_source):
            if status_callback:
                status_callback("export mmproj")
            mmproj_output_path = self._export_mmproj(
                source_model_path=mmproj_source,
                output_dir=output_dir,
                outtype=intermediate_outtype,
                work_dir=work_dir,
            )

        if status_callback:
            status_callback("complete")

        return {
            "intermediate_output_path": conversion_result["actual_output_file"],
            "quantized_output_path": quantized_path,
            "quantization_type": quantization_type,
            # mmproj is None when the source is not multimodal; keep the
            # declared dict[str, str] contract that callers rely on.
            "mmproj_output_path": cast(str, mmproj_output_path),
        }

    def _find_base_model_name(self, finetuned_path: str) -> str | None:
        """
        Try to identify the base model name for a finetuned/LoRA directory.

        Priority:
        1. finetuned_models entry in registry
        2. adapter_config.json -> base_model_name_or_path
        """
        registry_data = model_registry.list_models()
        norm_ft_path = os.path.normpath(finetuned_path)

        for ft in registry_data.get("finetuned_models", []):
            ft_model_path = ft.get("model_path") or ft.get("output_dir", "")
            if os.path.normpath(ft_model_path) == norm_ft_path:
                base_model_name = ft.get("model_name") or ft.get("base_model_name")
                if base_model_name:
                    logger.info(f"Found base model name from registry: {base_model_name}")
                    return base_model_name

        adapter_config_path = os.path.join(finetuned_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, encoding="utf-8") as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                    if base_model_name:
                        logger.info(
                            f"Found base model name from adapter_config.json: {base_model_name}"
                        )
                        return base_model_name
            except Exception as e:
                logger.warning(f"Failed to read adapter_config.json: {e}")

        logger.warning(f"Could not determine base model name for {finetuned_path}")
        return None

    def _find_base_model_path(self, finetuned_path: str) -> str | None:
        """
        Try to interpret the finetuned path and find associated base model path.

        Strategy:
        1. Identify base_model_name:
           a. From model_registry (matching finetuned path)
           b. From adapter_config.json (base_model_name_or_path, fallback)
        2. Resolve local path for base_model_name:
           a. Check if it is already a local path
           b. From model_registry (if base model is registered with local_path)
           c. Search in HF_HOME
        """
        base_model_name = self._find_base_model_name(finetuned_path)
        registry_data = model_registry.list_models()

        if not base_model_name:
            return None

        # Strategy 2a: Check if base_model_name is already a valid local path
        if os.path.isdir(base_model_name):
            logger.info(f"Base model name is a local path: {base_model_name}")
            return base_model_name

        # Strategy 2b: Find base model entry in registry (check for local_path override)
        for bm in registry_data.get("base_models", []):
            bm_model_name = bm.get("model_name") or bm.get("base_model_name")
            bm_model_path = bm.get("model_path") or bm.get("local_path")
            if bm_model_name == base_model_name:
                if bm_model_path and os.path.exists(bm_model_path):
                    return bm_model_path

        # Strategy 2c: Search in HF_HOME
        if not HF_HOME or not os.path.exists(HF_HOME):
            logger.warning("HF_HOME not set or does not exist")
            return None

        # Construct HF cache path pattern: models--Author--ModelName
        dir_name = "models--" + base_model_name.replace("/", "--")
        snapshots_path = os.path.join(HF_HOME, "hub", dir_name, "snapshots")

        if os.path.exists(snapshots_path):
            # List snapshots
            snapshots = [
                d
                for d in os.listdir(snapshots_path)
                if os.path.isdir(os.path.join(snapshots_path, d))
            ]
            if snapshots:
                # Usually pick the first one or valid one.
                for snapshot in snapshots:
                    candidate = os.path.join(snapshots_path, snapshot)
                    if os.path.exists(os.path.join(candidate, "config.json")):
                        logger.info(f"Resolved base model path from HF_HOME: {candidate}")
                        return candidate

        logger.warning(f"Could not find base model {base_model_name} in HF_HOME: {snapshots_path}")
        return None

    def start_conversion(
        self,
        model_path: str,
        output_path: str | None,
        outtype: str,
        base_model_path: str | None = None,
    ) -> str:
        """
        Start an asynchronous GGUF conversion job and return its id.
        """
        job_id = str(uuid.uuid4())

        # Check if it is a LoRA model (has adapter_config.json)
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_lora and not base_model_path:
            # Try to auto-detect base model path
            detected_base = self._find_base_model_path(model_path)
            if detected_base:
                base_model_path = detected_base
                logger.info(f"Auto-detected base model path for LoRA conversion: {base_model_path}")

        # Determine output path if not provided
        if not output_path:
            # If not provided, we just pass the model directory.
            # llama.cpp's convert script will automatically generate the correct `.gguf` file name (like Merged_Model-8.0B-F16.gguf)
            # inside that directory.
            output_path = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)

        job = ConversionJob(job_id, model_path, output_path, outtype, base_model_path)
        self.jobs[job_id] = job

        thread = threading.Thread(target=self._run_conversion, args=(job,))
        thread.start()

        return job_id

    def _run_conversion(self, job: ConversionJob) -> None:
        job.status = "running"
        job.message = "Conversion started"
        try:
            result = self._convert_to_gguf(
                model_path=job.model_path,
                output_path=job.output_path,
                outtype=job.outtype,
                base_model_path=job.base_model_path,
                register_model=True,
            )
            job.status = "completed"
            job.output_path = result["actual_output_file"]
            job.message = f"Conversion successful. Output: {job.output_path}"
            logger.info(f"Conversion job {job.job_id} completed.")

        except Exception as e:
            job.status = "failed"
            job.message = f"Exception occurred: {str(e)}"
            job.error = str(e)
            logger.exception(f"Exception in conversion job {job.job_id}")

    def get_job_status(self, job_id: str) -> dict | None:
        """
        Return the status of a conversion job, or None if the id is unknown.
        """
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "output_path": job.output_path,
        }


conversion_manager = ConversionManager()
