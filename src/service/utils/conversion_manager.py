import subprocess
import sys
import os
import uuid
import shutil
import tempfile
import threading
import logging
import json
from typing import Any, Dict, Optional, Callable
from service.model_registry import model_registry
from service.settings import HF_HOME

logger = logging.getLogger(__name__)

class ConversionJob:
    def __init__(self, job_id, model_path, output_path, outtype, base_model_path: Optional[str] = None):
        self.job_id = job_id
        self.model_path = model_path
        self.output_path = output_path
        self.outtype = outtype
        self.base_model_path = base_model_path
        self.status = "pending"
        self.message = "Job created"
        self.process = None
        self.error = None

class ConversionManager:
    def __init__(self):
        self.jobs: Dict[str, ConversionJob] = {}
        # Path to the script inside llama.cpp folder
        # Directory: service/utils/llama.cpp/ or configured via env
        self.llama_cpp_dir = os.getenv(
            "LLAMA_CPP_DIR", 
            os.path.join(os.path.dirname(__file__), "llama.cpp")
        )
        self.llama_cpp_dir = os.path.abspath(self.llama_cpp_dir)
        self.script_path = os.path.join(self.llama_cpp_dir, "convert_hf_to_gguf.py")
        self.lora_script_path = os.path.join(self.llama_cpp_dir, "convert_lora_to_gguf.py")
        self.merger_script = os.path.join(os.path.dirname(__file__), "lora_merger.py")
        self.quantize_binary = self._resolve_quantize_binary()

    def _resolve_quantize_binary(self) -> Optional[str]:
        """Resolve llama.cpp quantize binary path.

        Runtime can provide a prebuilt binary through `LLAMA_QUANTIZE_BIN`, so the
        container does not need the whole llama.cpp source tree.
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

        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)

        return None

    def _build_env(self) -> Dict[str, str]:
        """Build runtime environment for llama.cpp conversion scripts."""
        env = os.environ.copy()
        gguf_py_path = os.path.join(self.llama_cpp_dir, "gguf-py")
        if "PYTHONPATH" in env and env["PYTHONPATH"]:
            env["PYTHONPATH"] = f"{gguf_py_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = gguf_py_path

        library_dirs: list[str] = []
        for candidate in (
            os.path.join(self.llama_cpp_dir, "build", "bin"),
            os.path.join(self.llama_cpp_dir, "bin"),
        ):
            if os.path.isdir(candidate):
                library_dirs.append(candidate)

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

    def _run_command(self, cmd: list[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Run subprocess and raise a readable error when it fails."""
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            found_ggufs = [
                f for f in os.listdir(actual_output_file) if f.lower().endswith(".gguf")
            ]
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
                    if name.lower().endswith(".gguf") and requested_name in os.path.splitext(name)[0]
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
        base_model_path: Optional[str] = None,
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
        base_model_path: Optional[str] = None,
    ) -> Optional[int]:
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
        base_model_path: Optional[str] = None,
        register_model: bool = True,
        work_dir: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
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

            convert_cmd = [
                sys.executable,
                script_to_run,
                target_model_path,
                "--outtype",
                outtype,
                "--outfile",
                output_path,
            ]
            if work_dir:
                convert_cmd.append("--use-temp-file")
            self._run_command(convert_cmd, env=env)

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

    def convert_and_quantize(
        self,
        model_path: str,
        *,
        output_dir: Optional[str] = None,
        intermediate_outtype: str = "f16",
        quantization_type: str = "Q4_K_M",
        base_model_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, str]:
        """Convert a training output to GGUF and quantize it.

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
            quantize_binary,
            conversion_result["actual_output_file"],
            quantized_path,
            quantization_type,
        ]
        self._run_command(quantize_cmd, env=self._build_env())

        if status_callback:
            status_callback("complete")

        is_lora = bool(conversion_result.get("is_lora"))
        self._register_gguf_model(
            model_path=model_path,
            actual_output_file=quantized_path,
            outtype=quantization_type,
            is_lora=is_lora,
            base_model_path=base_model_path,
        )

        return {
            "intermediate_output_path": conversion_result["actual_output_file"],
            "quantized_output_path": quantized_path,
            "quantization_type": quantization_type,
        }

    def _find_base_model_name(self, finetuned_path: str) -> Optional[str]:
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
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                    if base_model_name:
                        logger.info(f"Found base model name from adapter_config.json: {base_model_name}")
                        return base_model_name
            except Exception as e:
                logger.warning(f"Failed to read adapter_config.json: {e}")

        logger.warning(f"Could not determine base model name for {finetuned_path}")
        return None

    def _find_base_model_path(self, finetuned_path: str) -> Optional[str]:
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
             snapshots = [d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))]
             if snapshots:
                 # Usually pick the first one or valid one. 
                 for snapshot in snapshots:
                     candidate = os.path.join(snapshots_path, snapshot)
                     if os.path.exists(os.path.join(candidate, "config.json")):
                         logger.info(f"Resolved base model path from HF_HOME: {candidate}")
                         return candidate
        
        logger.warning(f"Could not find base model {base_model_name} in HF_HOME: {snapshots_path}")
        return None

    def start_conversion(self, model_path: str, output_path: Optional[str], outtype: str, base_model_path: Optional[str] = None) -> str:
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

    def _run_conversion(self, job: ConversionJob):
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

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "output_path": job.output_path
        }

conversion_manager = ConversionManager()
