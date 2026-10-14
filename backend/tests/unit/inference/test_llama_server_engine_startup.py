"""Unit tests for LlamaServerEngine managed-server startup and model path resolution.

Formerly test_llama_server_multimodal_support.py; the payload-building tests it
carried targeted llama_server_runner._build_payload, which was removed when
llama/vLLM generation moved to the direct async HTTP path. Payload building is
now covered by tests/unit/inference/test_async_server_client.py.
"""

import os
from multiprocessing import Event, Queue

import pytest

from service.config_models import InferenceConfig
from service.inference.engines.llama_server_engine import LlamaServerEngine

pytestmark = pytest.mark.inference


def test_managed_server_adds_mmproj_arg(monkeypatch):
    captured = {}

    class _Proc:
        stdout = None
        stderr = None

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.subprocess.Popen", _fake_popen
    )
    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.os.path.exists", lambda _p: True
    )
    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.os.path.isfile", lambda _p: True
    )
    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.os.access", lambda _p, _m: True
    )
    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.os.path.isdir", lambda _p: True
    )

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",
        llama_server_host="127.0.0.1",
        llama_server_port=8080,
        llama_server_np=2,
        llama_server_mmproj="/tmp/mmproj.gguf",
        llama_server_extra_args=["--mlock"],
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    assert "--mmproj" in cmd
    idx = cmd.index("--mmproj")
    # normpath both sides so the assertion holds on Windows too (the engine emits
    # OS-native separators, turning "/tmp/mmproj.gguf" into "\tmp\mmproj.gguf").
    assert os.path.normpath(cmd[idx + 1]) == os.path.normpath("/tmp/mmproj.gguf")


def _patch_fs_and_popen(monkeypatch, captured):
    """Stub the filesystem checks + Popen so _start_managed_server runs on /tmp paths."""

    class _Proc:
        stdout = None
        stderr = None

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _Proc()

    mod = "service.inference.engines.llama_server_engine"
    monkeypatch.setattr(f"{mod}.subprocess.Popen", _fake_popen)
    monkeypatch.setattr(f"{mod}.os.path.exists", lambda _p: True)
    monkeypatch.setattr(f"{mod}.os.path.isfile", lambda _p: True)
    monkeypatch.setattr(f"{mod}.os.access", lambda _p, _m: True)
    monkeypatch.setattr(f"{mod}.os.path.isdir", lambda _p: True)


def test_unified_binary_gets_serve_subcommand_and_device(monkeypatch):
    """The prebuilt unified `llama` needs a `serve` subcommand, and a configured
    device must be passed through as --device."""
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama",  # no "server" in name -> unified
        llama_server_device="Vulkan1",
        llama_server_port=8080,
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    assert cmd[1] == "serve", f"unified binary missing serve subcommand: {cmd[:3]}"
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_source_built_server_binary_has_no_serve_and_no_device(monkeypatch):
    """The source-built llama-server takes args directly (no serve subcommand),
    and no --device is emitted when none is configured."""
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",  # "server" in name -> not unified
        llama_server_port=8080,
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    assert "serve" not in cmd
    assert cmd[1] == "-m", f"expected args to start right after binary: {cmd[:3]}"
    assert "--device" not in cmd


def test_resolve_model_file_supports_relative_path_under_hf_home(monkeypatch, tmp_path):
    hf_home = tmp_path / "huggingface"
    model_file = hf_home / "hub" / "models--demo--repo" / "snapshots" / "abc123" / "demo.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("gguf")

    monkeypatch.setattr("service.inference.engines.llama_server_engine.HF_HOME", str(hf_home))

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo/repo",
        model_path="models--demo--repo/snapshots/abc123/demo.gguf",
        engine="llama_server",
    )

    assert engine._resolve_model_file() == str(model_file.resolve())


def test_managed_server_resolves_relative_model_path_from_hf_home(monkeypatch, tmp_path):
    captured = {}

    class _Proc:
        stdout = None
        stderr = None

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return _Proc()

    hf_home = tmp_path / "huggingface"
    model_file = hf_home / "hub" / "models--demo--repo" / "snapshots" / "abc123" / "demo.gguf"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("gguf")

    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    monkeypatch.setattr("service.inference.engines.llama_server_engine.HF_HOME", str(hf_home))
    monkeypatch.setattr(
        "service.inference.engines.llama_server_engine.subprocess.Popen", _fake_popen
    )

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo/repo",
        model_path="models--demo--repo/snapshots/abc123/demo.gguf",
        engine="llama_server",
        llama_server_binary=str(binary),
        llama_server_host="127.0.0.1",
        llama_server_port=8080,
        llama_server_np=1,
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    model_index = cmd.index("-m")
    assert cmd[model_index + 1] == str(model_file.resolve())


def test_n_cpu_moe_reaches_the_command_line(monkeypatch):
    """n_cpu_moe must become -ncmoe.

    It used to exist only on the estimator's request models, so a sizing
    recommendation pasted into load_model was accepted with HTTP 200 and then
    loaded a full offload: 15109 MiB used with 942 MiB left on a 16 GB card,
    against 7944 MiB once the flag actually applied.
    """
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",
        llama_server_port=8080,
        n_cpu_moe=39,
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    assert "-ncmoe" in cmd, f"-ncmoe missing from {cmd}"
    assert cmd[cmd.index("-ncmoe") + 1] == "39"


def test_no_ncmoe_emitted_when_n_cpu_moe_is_zero(monkeypatch):
    """The default must not change the command line at all."""
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",
        llama_server_port=8080,
    )

    engine._start_managed_server()

    assert "-ncmoe" not in captured["cmd"]


def test_explicit_ncmoe_in_extra_args_still_wins(monkeypatch):
    """extra_args are appended last, so a hand-written -ncmoe overrides the field."""
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",
        llama_server_port=8080,
        n_cpu_moe=39,
        llama_server_extra_args=["-ncmoe", "12"],
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    assert cmd.count("-ncmoe") == 2
    # llama-server applies the last occurrence, and extra_args land after ours.
    assert cmd[len(cmd) - 1 - cmd[::-1].index("-ncmoe") + 1] == "12"


# Real llama-server /v1/models shape: the SAME model is described twice. The
# OpenAI-style data[] block carries meta but no capabilities; the Ollama-style
# models[] block carries capabilities but no meta.
#
# The meta keys below are the complete set llama.cpp builds (server-context.cpp
# get_models, and the worked example in tools/server/README.md). In particular there
# is NO n_ctx here -- the window a load actually serves comes from /props, never from
# this endpoint. Do not add one to make a test pass; that is the bug this shape exists
# to prevent.
_LLAMA_MODELS_PAYLOAD = {
    "models": [
        {
            "name": "trusta-ast-default",
            "model": "trusta-ast-default",
            "capabilities": ["completion", "multimodal"],
            "details": {"format": "gguf"},
        }
    ],
    "object": "list",
    "data": [
        {
            "id": "trusta-ast-default",
            "object": "model",
            "meta": {
                "vocab_type": 2,
                "n_ctx_train": 262144,
                "n_params": 34660610688,
                "n_embd": 2048,
                "n_vocab": 151936,
                "size": 21155768832,
            },
        }
    ],
}

# Real llama-server /props shape. default_generation_settings.n_ctx is the PER-SLOT
# window: llama.cpp has already divided -c by -np and already capped it to n_ctx_train.
_LLAMA_PROPS_PAYLOAD = {
    "default_generation_settings": {"n_ctx": 65536, "params": {}},
    "total_slots": 4,
    "model_alias": "trusta-ast-default",
    "modalities": {"vision": True, "audio": False},
    "chat_template": "{{ }}",
}


def _engine_with_models_payload(monkeypatch, payload):
    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.served_model_name = "trusta-ast-default"
    monkeypatch.setattr(engine, "_request_server_json", lambda *a, **k: payload)
    return engine


def test_capabilities_come_from_the_ollama_block(monkeypatch):
    # Matching data[] first must not stop the scan — capabilities only live in models[]
    engine = _engine_with_models_payload(monkeypatch, _LLAMA_MODELS_PAYLOAD)

    capabilities = engine._refresh_served_model_capabilities()

    assert capabilities == ["completion", "multimodal"]
    assert engine._is_multimodal_model() is True


def test_model_meta_comes_from_the_openai_block(monkeypatch):
    engine = _engine_with_models_payload(monkeypatch, _LLAMA_MODELS_PAYLOAD)

    engine._refresh_served_model_capabilities()

    assert engine._served_model_meta == {
        "n_ctx_train": 262144,
        "n_params": 34660610688,
        "n_embd": 2048,
        "n_vocab": 151936,
        "size": 21155768832,
    }
    # vocab_type is not in the kept set, and n_ctx was never in the payload
    assert "vocab_type" not in engine._served_model_meta
    assert "n_ctx" not in engine._served_model_meta


def test_capabilities_survive_a_name_the_server_rewrote(monkeypatch):
    engine = _engine_with_models_payload(monkeypatch, _LLAMA_MODELS_PAYLOAD)
    engine.served_model_name = "some-other-alias"
    monkeypatch.setattr(engine, "_resolve_model_name", lambda: "some-other-alias")

    assert engine._refresh_served_model_capabilities() == ["completion", "multimodal"]


def test_capabilities_empty_when_the_probe_fails(monkeypatch):
    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.served_model_name = "trusta-ast-default"

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(engine, "_request_server_json", _boom)

    assert engine._refresh_served_model_capabilities() == []
    assert engine._served_model_meta == {}


def test_model_meta_drops_junk_values(monkeypatch):
    payload = {"data": [{"id": "trusta-ast-default", "meta": {"n_ctx_train": 0, "size": "big"}}]}
    engine = _engine_with_models_payload(monkeypatch, payload)

    engine._refresh_served_model_capabilities()

    assert engine._served_model_meta == {}


# ==================== /props: the served window and modalities ====================


def test_props_supplies_the_per_slot_window(monkeypatch):
    """The window comes from /props, already divided by -np and already capped."""
    engine = _engine_with_models_payload(monkeypatch, _LLAMA_PROPS_PAYLOAD)

    engine._refresh_served_props()

    assert engine._served_context_length == 65536


def test_props_folds_modalities_into_capabilities(monkeypatch):
    """capabilities only ever says "multimodal"; /props says which kind."""
    engine = _engine_with_models_payload(monkeypatch, _LLAMA_PROPS_PAYLOAD)
    engine._served_model_capabilities = ["completion", "multimodal"]

    engine._refresh_served_props()

    assert engine._served_model_capabilities == ["completion", "multimodal", "vision"]
    # audio is False in the payload and must not be advertised
    assert "audio" not in engine._served_model_capabilities


def test_props_failure_leaves_the_window_unknown(monkeypatch):
    """An older server without /props must report nothing, not a guess."""
    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.served_model_name = "trusta-ast-default"

    def _boom(*a, **k):
        raise RuntimeError("HTTP 404: Not Found")

    monkeypatch.setattr(engine, "_request_server_json", _boom)

    engine._refresh_served_props()

    assert engine._served_context_length is None


def test_props_router_mode_placeholder_is_rejected(monkeypatch):
    """llama-server in router mode answers with n_ctx: 0; that is not a window."""
    engine = _engine_with_models_payload(
        monkeypatch, {"default_generation_settings": {"n_ctx": 0}, "total_slots": 0}
    )

    engine._refresh_served_props()

    assert engine._served_context_length is None


def test_synced_extra_arg_leaves_no_contradiction_on_the_command_line(monkeypatch):
    """A flag in extra_args and the field it owns must not disagree on the argv.

    llama-server applies the last occurrence, so `-c 32768` in extra_args always won.
    What it left behind was an argv reading `-c 128000 ... -c 32768` and a config
    still saying 128000 -- the value every status reader saw. With the field synced,
    both occurrences now carry the value the server actually runs at.
    """
    captured = {}
    _patch_fs_and_popen(monkeypatch, captured)

    engine = LlamaServerEngine(Queue(), Queue(), Event(), Event())
    engine.config = InferenceConfig(
        model_name="demo-model",
        model_path="/tmp/model.gguf",
        engine="llama_server",
        llama_server_binary="/tmp/llama-server",
        llama_server_port=8080,
        n_ctx=128000,
        llama_server_extra_args=["-c", "32768"],
    )

    engine._start_managed_server()

    cmd = captured["cmd"]
    values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-c"]
    assert values == ["32768", "32768"], f"contradictory -c on {cmd}"
    assert engine.config.n_ctx == 32768
