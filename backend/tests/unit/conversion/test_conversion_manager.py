import json
import os

import service.utils.conversion_manager as conversion_manager_module
from service.utils.conversion_manager import ConversionManager
import service.training.training_process as training_process_module
from service.config_models import TrainingConfig


import pytest

pytestmark = pytest.mark.models


def _stub_convert_script(manager, tmp_path):
    """
    Point the manager at a convert script that exists, without needing llama.cpp.

    _convert_to_gguf refuses up front when script_path is missing. The real script is a
    gitignored clone that setup_env pulls down, so on a checkout where setup has not run
    — a fresh clone, or CI without the vendored sources — tests that never execute the
    script still died on that check, and died with a FileNotFoundError raised from
    inside conversion_manager rather than anything naming the missing setup step.

    Every test using this stubs _run_command, so the script is never invoked; only its
    existence is asserted. An empty file satisfies that, which lets these tests cover
    the conversion logic they are actually about on any machine.
    """
    stub = tmp_path / "convert_hf_to_gguf.py"
    stub.write_text("")
    manager.script_path = str(stub)
    return stub


def test_register_gguf_model_copies_base_model_context_length(monkeypatch, tmp_path):
    manager = ConversionManager()
    model_dir = tmp_path / "Qwen3-4B"
    model_dir.mkdir()
    output_file = tmp_path / "Qwen3-4B-f16.gguf"
    output_file.write_text("gguf", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        conversion_manager_module.model_registry,
        "list_models",
        lambda: {
            "base_models": [
                {
                    "model_name": "Qwen/Qwen3-4B",
                    "model_path": str(model_dir),
                    "max_context_length": 32768,
                }
            ],
            "finetuned_models": [],
            "llama_gguf_models": [],
        },
    )
    monkeypatch.setattr(
        conversion_manager_module.model_registry,
        "add_llama_gguf_model",
        lambda **kwargs: captured.update(kwargs),
    )

    manager._register_gguf_model(
        model_path=str(model_dir),
        actual_output_file=str(output_file),
        outtype="f16",
        is_lora=False,
    )

    assert captured["base_model_name"] == "Qwen3-4B"
    assert captured["max_context_length"] == 32768


def test_register_gguf_model_copies_lora_base_context_length(monkeypatch, tmp_path):
    manager = ConversionManager()
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3-4B"}),
        encoding="utf-8",
    )
    base_model_dir = tmp_path / "base-model"
    base_model_dir.mkdir()
    output_file = tmp_path / "adapter-f16.gguf"
    output_file.write_text("gguf", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(
        conversion_manager_module.model_registry,
        "list_models",
        lambda: {
            "base_models": [
                {
                    "model_name": "Qwen/Qwen3-4B",
                    "model_path": str(base_model_dir),
                    "max_context_length": 131072,
                }
            ],
            "finetuned_models": [],
            "llama_gguf_models": [],
        },
    )
    monkeypatch.setattr(
        conversion_manager_module.model_registry,
        "add_llama_gguf_model",
        lambda **kwargs: captured.update(kwargs),
    )

    manager._register_gguf_model(
        model_path=str(lora_dir),
        actual_output_file=str(output_file),
        outtype="f16",
        is_lora=True,
        base_model_path=str(base_model_dir),
    )

    assert captured["base_model_name"] == "Qwen/Qwen3-4B"
    assert captured["max_context_length"] == 131072


def test_convert_to_gguf_uses_work_dir_for_temp_and_env(monkeypatch, tmp_path):
    manager = ConversionManager()
    _stub_convert_script(manager, tmp_path)
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    base_model_dir = tmp_path / "base-model"
    base_model_dir.mkdir()
    output_file = tmp_path / "out.gguf"
    work_dir = tmp_path / "ssd-work"

    commands: list[tuple[list[str], dict[str, str] | None]] = []

    def _fake_run_command(cmd, env=None):
        commands.append((cmd, env))
        if cmd[1] == manager.merger_script:
            merged_dir = cmd[4]
            os.makedirs(merged_dir, exist_ok=True)
            return None
        output_file.write_text("gguf", encoding="utf-8")
        return None

    monkeypatch.setattr(manager, "_run_command", _fake_run_command)
    monkeypatch.setattr(manager, "_register_gguf_model", lambda **kwargs: None)

    manager._convert_to_gguf(
        model_path=str(lora_dir),
        output_path=str(output_file),
        outtype="f16",
        base_model_path=str(base_model_dir),
        work_dir=str(work_dir),
    )

    merge_cmd, merge_env = commands[0]
    convert_cmd, convert_env = commands[1]

    assert merge_cmd[1] == manager.merger_script
    assert merge_cmd[-2:] == [
        "--offload",
        os.path.join(os.path.dirname(merge_cmd[4]), "offload"),
    ]
    assert os.path.commonpath([merge_cmd[4], str(work_dir)]) == str(work_dir)
    assert convert_cmd[-1] == "--use-temp-file"
    assert merge_env is not None
    assert convert_env is not None
    assert merge_env["TMPDIR"] == str(work_dir)
    assert convert_env["TMPDIR"] == str(work_dir)


def test_convert_training_output_uses_offload_folder_as_work_dir(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        training_process_module.conversion_manager,
        "convert_and_quantize",
        lambda **kwargs: (
            captured.update(kwargs)
            or {
                "intermediate_output_path": "demo-f16.gguf",
                "quantized_output_path": "demo-q4.gguf",
                "quantization_type": "Q4_K_M",
            }
        ),
    )

    config = TrainingConfig(
        model_name="demo-model",
        method="lora",
        dataset_path="dataset/dataset.jsonl",
        output_dir="runtime_data/fine_tuned/demo-model",
        offload_folder="/mnt/ssd/training-offload",
    )

    result = training_process_module._convert_training_output_to_q4_k_m(config)

    assert captured["work_dir"] == "/mnt/ssd/training-offload"
    assert captured["model_path"] == "runtime_data/fine_tuned/demo-model"
    assert result["quantized_output_path"] == "demo-q4.gguf"


def test_streaming_merge_success(tmp_path):
    from service.utils.lora_merger import streaming_merge
    import torch
    from safetensors.torch import save_file

    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    out_dir = tmp_path / "merged"
    base_dir.mkdir()
    adapter_dir.mkdir()

    # 1. Base weights
    base_weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.tensor([[3.0, 4.0]], dtype=torch.float16),
    }
    save_file(base_weights, str(base_dir / "model.safetensors"))

    # Index
    with open(base_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "model.safetensors",
                }
            },
            f,
        )

    # 2. Config & weights for LoRA
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump({"r": 2, "lora_alpha": 4}, f)

    adapter_weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.tensor(
            [[2.0, 3.0]], dtype=torch.float16
        ),  # Shape 1 x 2
    }
    save_file(adapter_weights, str(adapter_dir / "adapter_model.safetensors"))

    success = streaming_merge(str(base_dir), str(adapter_dir), str(out_dir))
    assert success is True

    # Check merged tensor in out_dir
    from safetensors.torch import load_file

    merged_weights = load_file(str(out_dir / "model.safetensors"))
    q_proj_merged = merged_weights["model.layers.0.self_attn.q_proj.weight"]

    # Scaling = alpha / r = 4 / 2 = 2.0
    # B = [[2.0, 3.0]] (shape 1x2)
    # A = [[1.0, 0.0], [0.0, 1.0]] (shape 2x2)
    # B @ A = [[2.0, 3.0]]
    # Update = scaling * (B @ A) = 2.0 * [[2.0, 3.0]] = [[4.0, 6.0]]
    # Merged = [[1.0, 2.0]] + [[4.0, 6.0]] = [[5.0, 8.0]]
    assert torch.allclose(q_proj_merged, torch.tensor([[5.0, 8.0]], dtype=torch.float16))


# --- full-parameter multimodal fine-tune GGUF fallback ----------------------


def _no_adapter_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d  # no adapter_config.json -> treated as a full (non-LoRA) checkpoint


def test_full_multimodal_convert_falls_back_to_language_extraction(monkeypatch, tmp_path):
    """A full multimodal checkpoint whose direct conversion fails must retry via
    language-tower extraction (mirrors the LoRA-merge shape)."""
    manager = ConversionManager()
    _stub_convert_script(manager, tmp_path)
    model_dir = _no_adapter_dir(tmp_path, "full_mm")
    output_file = tmp_path / "out.gguf"

    calls = {"run": 0, "extract": 0}

    def fake_run(cmd, env=None):
        calls["run"] += 1
        if calls["run"] == 1:  # the direct conversion attempt
            raise RuntimeError("Can not map tensor 'model.model.embed_tokens.weight'")
        return None  # the post-extraction conversion succeeds

    def fake_extract(model_path, dest_dir, base_model_path):
        calls["extract"] += 1
        os.makedirs(dest_dir, exist_ok=True)

    monkeypatch.setattr(manager, "_run_command", fake_run)
    monkeypatch.setattr(manager, "_is_multimodal", lambda p: True)
    monkeypatch.setattr(manager, "_find_base_model_path", lambda p: str(tmp_path / "base"))
    monkeypatch.setattr(manager, "_extract_language_model", fake_extract)
    monkeypatch.setattr(manager, "_resolve_output_file", lambda p: str(output_file))

    res = manager._convert_to_gguf(
        model_path=str(model_dir),
        output_path=str(output_file),
        outtype="f16",
        register_model=False,
    )
    assert calls["extract"] == 1, "language extraction fallback should have run"
    assert calls["run"] == 2, "should convert once directly, then once post-extraction"
    assert res["actual_output_file"] == str(output_file)


def test_text_only_convert_failure_does_not_fall_back(monkeypatch, tmp_path):
    """A text-only checkpoint must not trigger the multimodal extraction path."""
    manager = ConversionManager()
    _stub_convert_script(manager, tmp_path)
    model_dir = _no_adapter_dir(tmp_path, "text_only")

    calls = {"extract": 0}

    def boom(cmd, env=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager, "_run_command", boom)
    monkeypatch.setattr(manager, "_is_multimodal", lambda p: False)
    monkeypatch.setattr(
        manager,
        "_extract_language_model",
        lambda *a, **k: calls.__setitem__("extract", calls["extract"] + 1),
    )

    with pytest.raises(RuntimeError, match="boom"):
        manager._convert_to_gguf(
            model_path=str(model_dir),
            output_path=str(tmp_path / "o.gguf"),
            outtype="f16",
            register_model=False,
        )
    assert calls["extract"] == 0


def test_copy_tokenizer_prefers_sentencepiece_source(tmp_path):
    """_copy_tokenizer brings over the SentencePiece tokenizer.model verbatim."""
    src = tmp_path / "base"
    src.mkdir()
    (src / "tokenizer.model").write_bytes(b"spm")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "lang"
    dest.mkdir()

    ConversionManager._copy_tokenizer(str(src), str(dest))
    assert (dest / "tokenizer.model").read_bytes() == b"spm"
    assert (dest / "tokenizer.json").exists()


# --- mmproj source vs multimodal_scope -------------------------------------
#
# Under the default scope the vision half is untouched, so the base model is the
# right (and cheaper) mmproj source. Once multimodal_scope='text_and_bridge'
# trains the projector, that projector exists only in the merged weights, and a
# base-derived mmproj would ship the untrained one -- throwing the run's whole
# visual adaptation away with nothing raising.


def _mmproj_case(monkeypatch, tmp_path, *, modules_to_save):
    """Run convert_and_quantize on a LoRA dir and report the mmproj source."""
    manager = ConversionManager()
    _stub_convert_script(manager, tmp_path)
    manager.quantize_binary = str(tmp_path / "llama-quantize")

    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": modules_to_save}), encoding="utf-8"
    )
    base_dir = tmp_path / "base-model"
    base_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    merged_dirs: list[str] = []

    def _fake_run_command(cmd, env=None):
        if len(cmd) > 1 and cmd[1] == manager.merger_script:
            merged_dirs.append(cmd[4])
            os.makedirs(cmd[4], exist_ok=True)
        return None

    captured: dict = {}

    monkeypatch.setattr(manager, "_run_command", _fake_run_command)
    monkeypatch.setattr(manager, "_register_gguf_model", lambda **kwargs: None)
    monkeypatch.setattr(manager, "_is_multimodal", lambda p: True)
    monkeypatch.setattr(manager, "_resolve_output_file", lambda p: p)
    monkeypatch.setattr(
        manager,
        "_export_mmproj",
        lambda **kwargs: captured.setdefault("source", kwargs["source_model_path"]),
    )

    manager.convert_and_quantize(
        model_path=str(lora_dir),
        output_dir=str(out_dir),
        base_model_path=str(base_dir),
        export_mmproj=True,
    )
    return captured.get("source"), merged_dirs, str(base_dir)


def test_mmproj_comes_from_base_when_only_the_language_tower_trained(monkeypatch, tmp_path):
    source, merged_dirs, base_dir = _mmproj_case(monkeypatch, tmp_path, modules_to_save=None)

    assert source == base_dir
    # The merged checkpoint was only needed for the language GGUF.
    assert all(not os.path.exists(d) for d in merged_dirs)


def test_mmproj_comes_from_merged_model_when_the_bridge_trained(monkeypatch, tmp_path):
    source, merged_dirs, base_dir = _mmproj_case(
        monkeypatch, tmp_path, modules_to_save=["model.embed_vision"]
    )

    assert merged_dirs, "expected the adapter to be merged"
    assert source == merged_dirs[0], "mmproj must carry the trained bridge, not the base one"
    assert source != base_dir
    # Kept only for the export; the caller still owns the cleanup.
    assert all(not os.path.exists(d) for d in merged_dirs)


def test_adapter_without_a_config_is_not_treated_as_bridge_trained(tmp_path):
    """A plain HF checkpoint has no adapter_config.json to read."""
    manager = ConversionManager()
    plain = tmp_path / "hf-model"
    plain.mkdir()
    assert manager._adapter_trains_bridge(str(plain)) is False


def test_unreadable_adapter_config_falls_back_to_base(tmp_path, caplog):
    """Corrupt JSON must not crash the conversion; it warns and assumes scope A."""
    manager = ConversionManager()
    lora_dir = tmp_path / "adapter"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level("WARNING", logger=conversion_manager_module.logger.name):
        assert manager._adapter_trains_bridge(str(lora_dir)) is False
    assert "mmproj source" in caplog.text


# A path the Windows codepage (cp950) cannot hold, and bytes it cannot decode.
_NON_ASCII = "D:/模型/café/Qwen3.6-35B-🚀.gguf"


def test_run_command_reads_a_native_tools_utf8_output(tmp_path):
    """llama-quantize prints its paths in UTF-8; reading them must not fail the job."""
    import sys

    script = tmp_path / "native.py"
    script.write_text(
        f"import sys\nsys.stdout.buffer.write({_NON_ASCII!r}.encode('utf-8') + bytes([10]))\n",
        encoding="utf-8",
    )
    result = ConversionManager()._run_command([sys.executable, str(script)])
    assert _NON_ASCII in result.stdout


def test_run_command_makes_python_children_write_utf8(tmp_path):
    """convert_hf_to_gguf.py prints with print(); on Windows that is the codepage
    unless the child is told otherwise -- it is, so the two sides agree."""
    import sys

    script = tmp_path / "convert.py"
    script.write_text(f"print({_NON_ASCII!r})\n", encoding="utf-8")
    result = ConversionManager()._run_command([sys.executable, str(script)])
    assert _NON_ASCII in result.stdout


def test_build_env_puts_gguf_py_ahead_of_an_existing_pythonpath(monkeypatch, tmp_path):
    """Joined with the platform's separator, so each entry stays its own path on Windows too."""
    import os

    existing = str(tmp_path / "already-there")
    monkeypatch.setenv("PYTHONPATH", existing)
    manager = ConversionManager()
    entries = manager._build_env()["PYTHONPATH"].split(os.pathsep)
    assert entries == [os.path.join(manager.llama_cpp_dir, "gguf-py"), existing]


def test_build_env_sets_pythonpath_when_there_is_none(monkeypatch):
    import os

    monkeypatch.delenv("PYTHONPATH", raising=False)
    manager = ConversionManager()
    assert manager._build_env()["PYTHONPATH"] == os.path.join(manager.llama_cpp_dir, "gguf-py")
