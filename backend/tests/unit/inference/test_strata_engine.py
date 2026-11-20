"""Strata engine: setup checks, the run config it starts the server with, its lifecycle, and sleep/wake."""

import asyncio
import json
import os
import socket
from multiprocessing import Event, Queue
from pathlib import Path

import httpx2
import psutil
import pytest
from fastapi.testclient import TestClient

import service.app as app_module
import service.model_manager as model_manager_module
from service.config_models import InferenceConfig
from service.inference import async_server_client
from service.inference.engines import strata_engine
from service.inference.engines.strata_engine import StrataEngine, build_run_config
from service.model_manager import ModelSleepingError, SleepModeUnavailable
from service.platform_support import engine_support
from service.utils.process_identity import process_identity

pytestmark = pytest.mark.inference


def _local_client():
    """A client on this machine: other machines only get inference (see test_remote_access.py)."""
    return TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


_REAL_ASYNC_CLIENT = httpx2.AsyncClient  # captured before any test replaces it

manager = app_module.model_manager

SOURCE = {
    "exe": "/opt/strata/engine/strata",
    "args": [
        "--pack",
        "/p",
        "--max-context",
        "65536",
        "--vram-reserve-mib",
        "8150",
        "--sleep-mode",
    ],
    "tokenizer": "/p/tokenizer",
    "model_name": "flash-next",
    "port": 8080,
    "vision": {"exe": "/v/strata-vision", "mmproj": "/v/mmproj.gguf", "gpu": True},
}


def _strata(**overrides):
    fields = {"model_name": "flash-next", "engine": "strata", "strata_config_path": "/x.json"}
    return InferenceConfig(**{**fields, **overrides})


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A Strata checkout (serve/server.py, a .venv interpreter) and a deep-sleep engine build."""
    home = tmp_path / "Strata"
    (home / "serve").mkdir(parents=True)
    (home / "serve" / "server.py").write_text("")
    # The layout strata_python() looks for on this platform (Scripts/python.exe on Windows).
    python = home / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    engine = home / "build-sleep" / "strata"
    engine.parent.mkdir()
    engine.write_text("")
    # what the package's build writes beside the engine: the backend reads whether it sleeps here
    (engine.parent / "BUILD.json").write_text(
        json.dumps({"version": "0.1.52", "backend": "cuda", "vision": "cuda", "sleep_mode": True})
    )
    monkeypatch.setattr(strata_engine, "STRATA_HOME", str(home))
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(engine))
    monkeypatch.setattr(strata_engine, "STRATA_RUNTIME_DIR", str(tmp_path / "runtime"))
    # not this machine's .env: a data folder or an encoder set there changes setup and the run config
    monkeypatch.setattr(strata_engine, "STRATA_DATA_DIR", "")
    monkeypatch.setattr(strata_engine, "STRATA_VISION_BINARY", "")
    return home


# ----------------------------------------------------------------- setup checks
def test_not_offered_until_a_checkout_is_configured(monkeypatch, checkout):
    assert engine_support("strata").supported
    monkeypatch.setattr(strata_engine, "STRATA_HOME", "")
    monkeypatch.setattr(strata_engine, "strata_installable", lambda: False)
    support = engine_support("strata")
    assert not support.supported
    assert "STRATA_HOME" in support.reason
    # an installed build carries the package: the first load installs it
    monkeypatch.setattr(strata_engine, "strata_installable", lambda: True)
    assert engine_support("strata").supported


def test_installable_needs_an_nvidia_gpu_the_script_and_the_package(monkeypatch, tmp_path):
    script = "windows/install_strata.ps1" if os.name == "nt" else "linux/install_strata.sh"
    package = (
        "strata-sleep-0.2.7-sleep.1-windows-x86_64.zip"
        if os.name == "nt"
        else "strata-sleep-0.2.7-sleep.1-linux-x86_64.tar.gz"
    )
    monkeypatch.setattr(strata_engine, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(strata_engine.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    (tmp_path / "scripts" / script).parent.mkdir(parents=True)
    (tmp_path / "scripts" / script).write_text("")
    assert not strata_engine.strata_installable()  # no package
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / package).write_text("")
    assert strata_engine.strata_installable()
    assert strata_engine.strata_package() == tmp_path / "vendor" / package
    monkeypatch.setattr(strata_engine.shutil, "which", lambda name: None)
    monkeypatch.setattr(strata_engine, "gpu_backend", lambda: None)
    assert not strata_engine.strata_installable()  # no GPU this has an engine for


@pytest.mark.parametrize(
    ("win", "cuda_name", "sycl_name"),
    [
        (False, "strata-sleep-0.2.11-sleep.1-linux-x86_64.tar.gz",
         "strata-0.2.11-sleep.1-linux-x86_64-sycl.tar.gz"),
        (True, "strata-sleep-0.2.11-sleep.1-windows-x86_64.zip",
         "strata-0.2.11-sleep.1-windows-x86_64-sycl.zip"),
        # before 0.2.11-sleep.1 the Intel archive still had -sleep in its name
        (False, "strata-sleep-0.2.7-sleep.2-linux-x86_64.tar.gz",
         "strata-sleep-0.2.7-sleep.2-linux-x86_64-sycl.tar.gz"),
        (True, "strata-sleep-0.2.7-sleep.2-windows-x86_64.zip",
         "strata-sleep-0.2.7-sleep.2-windows-x86_64-sycl.zip"),
    ],
    ids=["linux", "windows", "linux-old-name", "windows-old-name"],
)
def test_each_backend_takes_its_own_package_and_leaves_the_other(
    monkeypatch, tmp_path, win, cuda_name, sycl_name
):
    """The two archives never match each other's glob: -sycl is last in the name for that.

    Both platforms, whichever this test runs on: the Intel package exists for Windows too,
    and naming it from the host's own os.name would leave one of the two never exercised.
    """
    monkeypatch.setattr(strata_engine.os, "name", "nt" if win else "posix")
    monkeypatch.setattr(strata_engine, "PROJECT_ROOT", tmp_path)
    (tmp_path / "vendor").mkdir()
    cuda = tmp_path / "vendor" / cuda_name
    sycl = tmp_path / "vendor" / sycl_name
    cuda.write_text("")
    sycl.write_text("")

    monkeypatch.setattr(strata_engine, "gpu_backend", lambda: "sycl")
    assert strata_engine.strata_package() == sycl
    monkeypatch.setattr(strata_engine, "gpu_backend", lambda: "cuda")
    assert strata_engine.strata_package() == cuda
    monkeypatch.setattr(strata_engine, "gpu_backend", lambda: None)
    assert strata_engine.strata_package() is None


def test_gpu_backend_prefers_nvidia_then_reads_the_cards(monkeypatch, tmp_path):
    monkeypatch.setattr(strata_engine.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert strata_engine.gpu_backend() == "cuda"  # even on a laptop that also has an Intel iGPU
    monkeypatch.setattr(strata_engine.shutil, "which", lambda name: None)

    # Linux: the DRM devices carry the vendor id
    monkeypatch.setattr(strata_engine.os, "name", "posix")
    monkeypatch.setattr(strata_engine, "_DRM_DEVICES", tmp_path)
    amd = tmp_path / "card0" / "device"
    amd.mkdir(parents=True)
    (amd / "vendor").write_text("0x1002\n")
    assert strata_engine.gpu_backend() is None  # an AMD card: neither NVIDIA nor Intel
    intel = tmp_path / "card1" / "device"
    intel.mkdir(parents=True)
    (intel / "vendor").write_text("0x8086\n")
    assert strata_engine.gpu_backend() == "sycl"

    # Windows: no DRM devices, so the display class registry answers instead
    monkeypatch.setattr(strata_engine.os, "name", "nt")
    monkeypatch.setattr(strata_engine, "_intel_gpu_windows", lambda: True)
    assert strata_engine.gpu_backend() == "sycl"
    monkeypatch.setattr(strata_engine, "_intel_gpu_windows", lambda: False)
    assert strata_engine.gpu_backend() is None


# ------------------------------------------------------------- what the engine can do
def test_whether_the_engine_sleeps_comes_from_its_build(tmp_path, monkeypatch):
    engine = tmp_path / "engine" / "strata"
    engine.parent.mkdir()
    engine.write_bytes(b"nothing to see")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(engine))
    build = engine.parent / "BUILD.json"

    build.write_text(json.dumps({"backend": "cuda", "sleep_mode": True}))
    assert strata_engine.engine_sleeps() is True
    assert strata_engine.engine_backend() == "cuda"

    build.write_text(json.dumps({"backend": "sycl", "sleep_mode": False}))
    assert strata_engine.engine_sleeps() is False
    assert strata_engine.engine_backend() == "sycl"

    # a package from before BUILD.json carried the field: the flag's name is in the binary
    build.unlink()
    assert strata_engine.engine_sleeps() is False
    assert strata_engine.engine_backend() == "cuda"  # the only backend there was
    engine.write_bytes(b"usage: strata --sleep-mode ...")
    assert strata_engine.engine_sleeps() is True


def test_an_engine_without_deep_sleep_is_not_asked_to_sleep(tmp_path, monkeypatch):
    engine = tmp_path / "engine" / "strata"
    engine.parent.mkdir()
    engine.write_bytes(b"")
    (engine.parent / "BUILD.json").write_text(json.dumps({"backend": "sycl", "sleep_mode": False}))
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(engine))
    run = build_run_config(_strata(), SOURCE, "/r/engine.log")
    assert run["sleep_mode"] is False
    assert "--sleep-mode" not in run["args"]


def test_an_intel_engine_folder_takes_its_gpu_reader(tmp_path, monkeypatch):
    engine = tmp_path / "engine" / "strata"
    engine.parent.mkdir()
    engine.write_bytes(b"")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(engine))
    monkeypatch.setattr(strata_engine, "STRATA_VISION_BINARY", "")
    suffix = ".exe" if os.name == "nt" else ""
    (engine.parent / "BUILD.json").write_text(json.dumps({"backend": "sycl", "sleep_mode": False}))
    cpu_reader = engine.parent / f"strata-vision{suffix}"
    cpu_reader.write_text("")
    assert strata_engine.vision_binary() == cpu_reader  # the only one there
    gpu_reader = engine.parent / f"strata-vision-sycl{suffix}"
    gpu_reader.write_text("")
    assert strata_engine.vision_binary() == gpu_reader
    # a CUDA folder has one file for both, so nothing changes there
    (engine.parent / "BUILD.json").write_text(json.dumps({"backend": "cuda", "sleep_mode": True}))
    assert strata_engine.vision_binary() == cpu_reader


def test_setup_problem_names_what_is_missing(monkeypatch, checkout):
    (checkout / "serve" / "server.py").unlink()
    assert "serve/server.py" in strata_engine.strata_setup_problem()
    (checkout / "serve" / "server.py").write_text("")
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "/nowhere/python")
    assert "STRATA_PYTHON" in strata_engine.strata_setup_problem()
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/nowhere/strata")
    assert "STRATA_ENGINE_BINARY" in strata_engine.strata_setup_problem()
    # never the run config's own executable: Strata's setup installs one without deep sleep
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "")
    assert "STRATA_ENGINE_BINARY" in strata_engine.strata_setup_problem()


# ------------------------------------------------------------------- run config
def test_run_config_applies_only_what_the_load_owns(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
    monkeypatch.setattr(strata_engine, "engine_sleeps", lambda: True)
    run = build_run_config(
        _strata(strata_max_context=262144), SOURCE, "/r/engine.log", vram_reserve_mib=1200
    )
    args = run["args"]
    assert args.count("--max-context") == 1 and args[args.index("--max-context") + 1] == "262144"
    assert args[args.index("--vram-reserve-mib") + 1] == "1200"
    assert "--sleep-mode" not in args  # decided by the sleep_mode key below
    assert run["sleep_mode"] is True
    assert run["exe"] == str(Path("/opt/sleep/strata"))  # a native path on this platform
    assert run["log"] == "/r/engine.log"
    assert "port" not in run
    assert run["tokenizer"] == "/p/tokenizer"  # untouched keys stay as setup wrote them
    assert run["vision"] == SOURCE["vision"]  # image input follows the run config
    assert SOURCE["args"][3] == "65536"  # the user's file is never changed


def test_run_config_keeps_the_source_values_by_default(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
    monkeypatch.setattr(strata_engine, "engine_sleeps", lambda: True)
    run = build_run_config(_strata(), SOURCE, "/r/engine.log")
    assert run["args"][run["args"].index("--max-context") + 1] == "65536"
    assert run["args"][run["args"].index("--vram-reserve-mib") + 1] == "8150"
    assert run["sleep_mode"] is True  # on wherever the engine has it (engine_sleeps)


def test_extra_args_replace_the_run_configs_flags_or_add_them(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
    source = {**SOURCE, "args": [*SOURCE["args"], "--resident-experts", "--spec", "4"]}
    extra = ["--resident-budget-gib", "13.3", "--sleep-backup", "file", "--spec", "6", "--kv-grow"]
    run = build_run_config(_strata(strata_extra_args=extra), source, "/r/engine.log")
    args = run["args"]
    assert args.count("--spec") == 1 and args[args.index("--spec") + 1] == "6"
    assert args[args.index("--resident-budget-gib") + 1] == "13.3"
    assert "--resident-experts" not in args  # the two RAM modes replace each other
    assert args[args.index("--sleep-backup") + 1] == "file"
    assert "--kv-grow" in args
    assert args[: args.index("--max-context") + 2] == ["--pack", "/p", "--max-context", "65536"]
    assert source["args"][-3:] == ["--resident-experts", "--spec", "4"]  # the file is not changed


def test_extra_args_replace_a_flag_with_every_value_it_had():
    args = [
        "--native-dense-gguf",
        "/a",
        "--native-dense-gguf",
        "/b",
        "--gpu",
        "0",
        "--vram-elastic",
    ]
    assert strata_engine._apply_extra_args(args, ["--native-dense-gguf", "/c", "--gpu=1"]) == [
        "--vram-elastic",
        "--native-dense-gguf",
        "/c",
        "--gpu=1",
    ]
    # a negative number is a value, not a flag
    assert strata_engine._apply_extra_args(
        ["--yarn-ext-factor", "1"], ["--yarn-ext-factor", "-1"]
    ) == [
        "--yarn-ext-factor",
        "-1",
    ]


def test_extra_args_and_the_fields_for_the_same_flag_are_refused_together():
    _strata(strata_extra_args=["--max-context", "131072"])  # either one alone is fine
    with pytest.raises(ValueError, match="strata_max_context"):
        _strata(strata_max_context=65536, strata_extra_args=["--max-context", "131072"])
    with pytest.raises(ValueError, match="strata_vram_limit_gb"):
        _strata(strata_vram_limit_gb=8, strata_extra_args=["--vram-reserve-mib=900"])


def test_extra_args_refuse_flags_the_service_owns_and_stray_values():
    for flag in ("--serve", "--sleep-mode", "--vision", "--dump-logits=/tmp/x", "--tokens-file"):
        with pytest.raises(ValueError, match="cannot set"):
            _strata(strata_extra_args=[flag, "x"])
    with pytest.raises(ValueError, match="must start with a flag"):
        _strata(strata_extra_args=["13.3", "--resident-budget-gib"])
    assert _strata(strata_extra_args=[" --kv-grow ", ""]).strata_extra_args == ["--kv-grow"]


@pytest.fixture
def model_dir(tmp_path):
    """A model's GGUF in its own folder, the mmproj one folder up (Strata-data's layout)."""
    gguf = tmp_path / "models" / "gyro-s" / "flash-next-TQ1_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_text("")
    (tmp_path / "models" / "mmproj-flash-next-F16.gguf").write_text("")
    (tmp_path / "models" / "mmproj-flash-next-BF16.gguf").write_text("")
    return gguf


def test_vision_is_added_to_a_run_config_without_it(monkeypatch, model_dir, tmp_path):
    monkeypatch.setattr(strata_engine, "STRATA_VISION_BINARY", str(tmp_path / "strata-vision"))
    source = {**SOURCE, "args": ["--native", str(model_dir), "--max-context", "65536"]}
    source.pop("vision")
    run = build_run_config(_strata(strata_vision=True), source, "/r/engine.log")
    assert run["vision"] == {
        "exe": str(tmp_path / "strata-vision"),
        "mmproj": str(model_dir.parent.parent / "mmproj-flash-next-BF16.gguf"),  # BF16 first
        "model": str(model_dir),
    }
    assert "vision" not in source  # the file is not changed


def test_vision_off_drops_the_encoder_and_the_engine_flag(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
    source = {**SOURCE, "args": [*SOURCE["args"], "--vision"]}
    run = build_run_config(_strata(strata_vision=False), source, "/r/engine.log")
    assert "vision" not in run and "--vision" not in run["args"]
    kept = build_run_config(_strata(), source, "/r/engine.log")  # omitted: as the run config has it
    assert kept["vision"] == SOURCE["vision"] and "--vision" in kept["args"]


def test_strata_mmproj_replaces_the_run_configs(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
    run = build_run_config(_strata(strata_mmproj="/m/other-mmproj.gguf"), SOURCE, "/r/engine.log")
    assert run["vision"]["mmproj"] == str(Path("/m/other-mmproj.gguf"))
    assert run["vision"]["exe"] == SOURCE["vision"]["exe"]


def test_vision_names_what_it_is_missing(monkeypatch, model_dir, tmp_path):
    monkeypatch.setattr(strata_engine, "STRATA_VISION_BINARY", "")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(tmp_path / "engine" / "strata"))
    source = {**SOURCE, "args": ["--native", str(model_dir)]}
    source.pop("vision")
    with pytest.raises(RuntimeError, match="STRATA_VISION_BINARY"):
        build_run_config(_strata(strata_vision=True), source, "/r/engine.log")
    lone = tmp_path / "lone" / "a" / "model.gguf"
    lone.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="strata_mmproj"):
        build_run_config(
            _strata(strata_vision=True), {**source, "args": ["--native", str(lone)]}, "/r/e.log"
        )
    # the encoder beside the engine is found without the setting
    vision = tmp_path / "engine" / ("strata-vision.exe" if os.name == "nt" else "strata-vision")
    vision.parent.mkdir()
    vision.write_text("")
    run = build_run_config(_strata(strata_vision=True), source, "/r/engine.log")
    assert run["vision"]["exe"] == str(vision)


def test_vram_limit_becomes_the_reserve_left_free():
    assert strata_engine.vram_reserve_for_limit(8, free_mib=15773) == 15773 - 8192
    assert strata_engine.vram_reserve_for_limit(20, free_mib=15773) == 0  # more than is free
    with pytest.raises(ValueError):
        _strata(strata_vram_limit_gb=0)


def test_only_the_beginner_fields_are_offered():
    # the beginner fields, and one list for every other engine flag
    fields = {name for name in InferenceConfig.model_fields if name.startswith("strata_")}
    assert fields == {
        "strata_config_path",
        "strata_max_context",
        "strata_vram_limit_gb",
        "strata_auto_wake",
        "strata_extra_args",
        "strata_vision",
        "strata_mmproj",
    }


# -------------------------------------------------------------------- lifecycle
def test_load_starts_the_server_with_the_run_config(checkout, tmp_path, monkeypatch):
    src = tmp_path / "strata-flash-next.json"
    src.write_text(json.dumps(SOURCE))
    status = Queue()
    engine = StrataEngine(status, Queue(), Event(), Event())
    started: list = []
    monkeypatch.setattr(engine, "_start_server", lambda path: started.append(path))
    monkeypatch.setattr(
        engine, "_wait_ready", lambda t: {"loaded": True, "max_context": 65536, "images": True}
    )
    monkeypatch.setattr(engine, "_restore_slot", lambda: {"restored": True, "tokens": 120})
    order: list = []
    monkeypatch.setattr(engine, "_stop_server", lambda: order.append("stop"))

    def free_vram(gpu):
        order.append(("free", gpu))
        return 15773

    monkeypatch.setattr(strata_engine, "free_vram_mib", free_vram)

    engine.load_model(_strata(strata_config_path=str(src), strata_vram_limit_gb=8))

    run = json.loads(started[0].read_text())
    assert started[0].name == "strata-flash-next.run.json"
    assert run["sleep_mode"] is True
    # measured once any old server has let go of its VRAM
    assert order == ["stop", ("free", 0)]
    assert run["args"][run["args"].index("--vram-reserve-mib") + 1] == str(15773 - 8192)
    ready = status.get(timeout=5)
    assert ready["status"] == "ready"
    assert ready["served_context_length"] == 65536
    assert ready["llama_capabilities"] == ["chat", "vision", "multimodal"]
    assert ready["slot_restore_summary"]["restored"] is True


def test_load_refuses_a_missing_or_foreign_run_config(checkout, tmp_path):
    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    with pytest.raises(RuntimeError, match="strata_config_path"):
        engine.load_model(InferenceConfig(model_name="m", engine="strata"))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"model": "x"}))
    with pytest.raises(RuntimeError, match="not a Strata run config"):
        engine.load_model(_strata(strata_config_path=str(other)))


def test_the_run_config_is_found_from_the_registry_gguf_path(checkout, tmp_path):
    # the model registry lists a Hugging Face snapshot path, a link to the blob setup named
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_text("gguf")
    snapshot = tmp_path / "snapshot" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)
    run = {**SOURCE, "args": ["--pack", "/p", "--native", str(blob), "--max-context", "65536"]}
    (checkout / "strata-iq3_xxs.json").write_text(json.dumps(run))
    (checkout / "strata-iq3_xxs.shared-settings.json").write_text("{}")  # not a run config
    (checkout / "strata-iq2_xs.json").write_text(
        json.dumps({**run, "args": ["--native", "/elsewhere.gguf"]})
    )

    assert strata_engine.find_run_config(str(snapshot)) == checkout / "strata-iq3_xxs.json"
    assert strata_engine.find_run_config(str(tmp_path / "nothing.gguf")) is None

    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    # no run config: the load prepares one, and setup takes only the GGUFs it knows
    with pytest.raises(RuntimeError, match="Strata cannot run x.gguf"):
        engine.load_model(_strata(strata_config_path=None, model_path=str(tmp_path / "x.gguf")))


# ------------------------------------------------------------- first-load setup
def test_model_choice_is_setups_own_list():
    first = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    assert strata_engine.strata_model_choice(f"/models/{first}") == ("qwen", "IQ3_S")
    assert strata_engine.strata_model_choice(first.lower()) == ("qwen", "IQ3_S")
    assert strata_engine.strata_model_choice(
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
    ) == ("swift", "IQ3_XXS")
    assert strata_engine.strata_model_choice(
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    ) == ("unsloth", "UD-Q4_K_XL")
    # setup refuses these: a later shard, Unsloth's UD-IQ3_XXS, a plain K-quant
    for name in (
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
        "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00002.gguf",
        "Qwen3.8-Flash-Next-Uncensored-Q4_K_M.gguf",
    ):
        assert strata_engine.strata_model_choice(name) is None


def test_the_model_list_marks_the_ggufs_strata_can_run(tmp_path, monkeypatch):
    from service.model_registry import model_registry

    path = tmp_path / "models_registry.json"
    path.write_text(
        json.dumps(
            {
                "base_models": [{"label": "base", "base_model_name": "Qwen/Qwen3.5"}],
                "finetuned_models": [],
                "llama_gguf_models": [
                    {
                        "label": "ok",
                        "local_path": "/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                    },
                    {
                        "label": "shard",
                        "local_path": "/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
                    },
                    {"label": "k-quant", "filename": "Qwen3.8-Flash-Next-Uncensored-Q4_K_M.gguf"},
                    {"label": "nothing"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "path", path)
    listed = model_registry.list_models()
    marks = {m["label"]: m["strata_supported"] for m in listed["llama_gguf_models"]}
    assert marks == {"ok": True, "shard": False, "k-quant": False, "nothing": False}
    assert "strata_supported" not in listed["base_models"][0]  # GGUFs only


def _gguf(tmp_path, name="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"):
    path = tmp_path / "hub" / "my models" / name
    path.parent.mkdir(parents=True)
    path.write_text("gguf")
    return path


def test_an_installed_strata_runs_setup_for_the_model(checkout, tmp_path, monkeypatch):
    gguf = _gguf(tmp_path)
    calls: list = []

    def fake_step(cmd, cwd, env):
        calls.append((cmd, cwd))
        run = {**SOURCE, "args": ["--native", str(gguf)]}
        (checkout / "strata-iq3_s.json").write_text(json.dumps(run))

    monkeypatch.setattr(strata_engine, "_run_setup_step", fake_step)
    assert strata_engine.prepare_model(str(gguf)) == checkout / "strata-iq3_s.json"
    cmd, cwd = calls[0]
    assert cwd == checkout
    assert cmd[1:] == [
        "setup.py",
        "--yes",
        "--no-start",
        "--vision",
        "no",
        "--family",
        "qwen",
        "--model",
        "IQ3_S",
        "--gguf-dir",
        str(gguf.parent),
    ]


def test_the_first_load_installs_strata_from_the_package(tmp_path, monkeypatch):
    gguf = _gguf(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "data" / "strata"
    monkeypatch.setattr(strata_engine, "PROJECT_ROOT", project)
    monkeypatch.setattr(strata_engine, "STRATA_INSTALL_DIR", str(home))
    for key in ("STRATA_HOME", "STRATA_PYTHON", "STRATA_ENGINE_BINARY"):
        monkeypatch.setattr(strata_engine, key, "")
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(strata_engine, "strata_installable", lambda: True)
    seen: dict = {}

    def fake_install(cmd, cwd, env):
        # what install_strata leaves: the install, the run config, and .env
        seen.update(cmd=cmd, env=env)
        (home / "serve").mkdir(parents=True)
        (home / "serve" / "server.py").write_text("")
        python = home / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_text("")
        (home / "engine").mkdir()
        (home / "engine" / "strata").write_text("")
        (home / "strata-iq3_s.json").write_text(
            json.dumps({**SOURCE, "args": ["--native", str(gguf)]})
        )
        (project / ".env").write_text(
            f"STRATA_HOME={home}\nSTRATA_PYTHON={python}\nSTRATA_ENGINE_BINARY={home / 'engine' / 'strata'}\n"
        )

    monkeypatch.setattr(strata_engine, "_run_setup_step", fake_install)
    assert strata_engine.prepare_model(str(gguf)) == home / "strata-iq3_s.json"
    assert "install_strata" in " ".join(seen["cmd"])
    assert seen["env"]["TRUSTA_STRATA_SETUP_ARGS"] == "--family qwen --model IQ3_S"
    assert seen["env"]["TRUSTA_STRATA_GGUF_DIR"] == str(gguf.parent)  # spaces and all
    assert seen["env"]["TRUSTA_STRATA_DIR"] == str(home)
    # this process now runs the install it just made
    assert strata_engine.STRATA_HOME == str(home)
    assert strata_engine.strata_setup_problem() is None


def _vendor_with_both_packages(tmp_path):
    """A project whose vendor/ holds the CUDA package and the Intel one, as the release ships them."""
    project = tmp_path / "project"
    (project / "vendor").mkdir(parents=True)
    platform, ext = ("windows-x86_64", "zip") if os.name == "nt" else ("linux-x86_64", "tar.gz")
    cuda = project / "vendor" / f"strata-sleep-0.2.7-sleep.3-{platform}.{ext}"
    sycl = project / "vendor" / f"strata-0.2.11-sleep.1-{platform}-sycl.{ext}"
    cuda.write_text("")
    sycl.write_text("")
    return project, cuda, sycl


def _first_load_environment(tmp_path, monkeypatch, backend):
    """The environment the first load hands install_strata, on a host whose GPU is `backend`."""
    gguf = _gguf(tmp_path)
    project, cuda, sycl = _vendor_with_both_packages(tmp_path)
    monkeypatch.setattr(strata_engine, "PROJECT_ROOT", project)
    monkeypatch.setattr(strata_engine, "STRATA_INSTALL_DIR", str(tmp_path / "strata"))
    for key in ("STRATA_HOME", "STRATA_PYTHON", "STRATA_ENGINE_BINARY"):
        monkeypatch.setattr(strata_engine, key, "")
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(strata_engine, "gpu_backend", lambda: backend)
    monkeypatch.setattr(strata_engine, "strata_installable", lambda: True)
    seen: dict = {}

    def capture(cmd, cwd, env):
        seen["env"] = env
        raise RuntimeError("stopped before the install ran")

    monkeypatch.setattr(strata_engine, "_run_setup_step", capture)
    with pytest.raises(RuntimeError, match="stopped before the install ran"):
        strata_engine.prepare_model(str(gguf))
    return seen["env"], cuda, sycl


@pytest.mark.parametrize("backend", ["cuda", "sycl"])
def test_the_install_script_is_handed_the_package_this_gpu_needs(tmp_path, monkeypatch, backend):
    """install_strata chooses a package itself, by a glob that matches only a CUDA archive.

    With both packages in vendor/ (which is how the release ships them) an Intel host would be
    given the CUDA one, and with only the Intel one the script would find nothing. The package
    strata_installable() found is passed on, so what was checked is what is installed.
    """
    monkeypatch.delenv("TRUSTA_STRATA_PACKAGE", raising=False)
    env, cuda, sycl = _first_load_environment(tmp_path, monkeypatch, backend)
    assert env["TRUSTA_STRATA_PACKAGE"] == str(sycl if backend == "sycl" else cuda)


def test_a_package_named_in_the_environment_is_kept(tmp_path, monkeypatch):
    """Someone who set TRUSTA_STRATA_PACKAGE chose a package; the first load does not overrule it."""
    chosen = tmp_path / "mine.tar.gz"
    monkeypatch.setenv("TRUSTA_STRATA_PACKAGE", str(chosen))
    env, _cuda, _sycl = _first_load_environment(tmp_path, monkeypatch, "sycl")
    assert env["TRUSTA_STRATA_PACKAGE"] == str(chosen)


def test_a_failed_setup_reports_its_output(checkout, tmp_path, monkeypatch):
    gguf = _gguf(tmp_path)
    script = tmp_path / "fail.py"
    script.write_text(
        "print('checking shards'); print('missing shard 2 of 2'); raise SystemExit(3)"
    )
    import sys

    monkeypatch.setattr(strata_engine, "strata_python", lambda: sys.executable)
    real_step = strata_engine._run_setup_step

    def failing(cmd, cwd, env):
        real_step([sys.executable, str(script)], None, env)

    monkeypatch.setattr(strata_engine, "_run_setup_step", failing)
    with pytest.raises(RuntimeError, match="(?s)exit 3.*missing shard 2 of 2"):
        strata_engine.prepare_model(str(gguf))


def _time_wait_port():
    """A port whose last server is gone but whose closed connection still sits in TIME_WAIT."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = server.accept()
    accepted.close()  # the server side closes first, so TIME_WAIT lands on this port
    client.close()
    server.close()
    return port


class _StartedServer:
    """What Popen hands back, as far as _start_server reads it: a PID that is running."""

    pid = os.getpid()


def test_start_refuses_a_port_in_use_but_not_one_in_time_wait(checkout, tmp_path, monkeypatch):
    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    monkeypatch.setattr(strata_engine, "STRATA_RUNTIME_DIR", str(tmp_path / "runtime"))
    engine.runtime_dir.mkdir(parents=True, exist_ok=True)
    popened: list = []
    monkeypatch.setattr(
        strata_engine.subprocess,
        "Popen",
        lambda cmd, **kw: popened.append(cmd) or _StartedServer(),
    )

    # what an unload followed by a reload leaves behind: a plain bind() fails on it
    monkeypatch.setattr(strata_engine, "STRATA_PORT", _time_wait_port())
    engine._start_server(tmp_path / "run.json")
    assert popened

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", 0))
        other.listen(1)
        monkeypatch.setattr(strata_engine, "STRATA_PORT", other.getsockname()[1])
        with pytest.raises(RuntimeError, match="already in use"):
            engine._start_server(tmp_path / "run.json")
    engine._log_handle.close()


def test_start_sets_the_load_defaults_unless_the_environment_has_them(
    checkout, tmp_path, monkeypatch
):
    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    engine.runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strata_engine, "STRATA_PORT", _time_wait_port())
    envs: list = []
    monkeypatch.setattr(
        strata_engine.subprocess,
        "Popen",
        lambda cmd, **kw: envs.append(kw["env"]) or _StartedServer(),
    )
    monkeypatch.delenv("STRATA_PLE_AUTO_RAM", raising=False)
    monkeypatch.setenv("STRATA_NO_ARENA_THP", "0")

    engine._start_server(tmp_path / "run.json")
    engine._log_handle.close()
    env = envs[0]
    assert env["STRATA_UNBUFFERED_LOAD"] == "1"
    assert env["STRATA_NO_ARENA_THP"] == "0"  # set by the caller, so kept
    # Windows only: its commit limit leaves no room for the sleep backup otherwise
    assert env.get("STRATA_PLE_AUTO_RAM") == ("0" if os.name == "nt" else None)


def test_load_env_defaults_add_the_doorbell_check_only_for_a_windows_intel_engine():
    check = {"STRATA_DOORBELL_CHECK": "1"}
    assert check.items() <= strata_engine._load_env_defaults(True, "sycl").items()
    assert "STRATA_DOORBELL_CHECK" not in strata_engine._load_env_defaults(True, "cuda")
    assert "STRATA_DOORBELL_CHECK" not in strata_engine._load_env_defaults(False, "sycl")
    # the Windows commit-limit default stays on both Windows backends
    assert strata_engine._load_env_defaults(True, "sycl")["STRATA_PLE_AUTO_RAM"] == "0"
    assert strata_engine._load_env_defaults(True, "cuda")["STRATA_PLE_AUTO_RAM"] == "0"
    assert "STRATA_PLE_AUTO_RAM" not in strata_engine._load_env_defaults(False, "cuda")


def test_start_reports_the_server_so_the_api_can_stop_it_if_the_worker_dies(
    checkout, tmp_path, monkeypatch
):
    status = Queue()
    engine = StrataEngine(status, Queue(), Event(), Event())
    engine.runtime_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(strata_engine, "STRATA_PORT", _time_wait_port())
    monkeypatch.setattr(strata_engine.subprocess, "Popen", lambda cmd, **kw: _StartedServer())

    engine._start_server(tmp_path / "run.json")
    engine._log_handle.close()

    report = status.get(timeout=5)
    assert report["status"] == "server_started"
    assert report["pid"] == os.getpid()
    # the start time is what tells this server from a later process given the same PID
    assert report["create_time"] == process_identity(os.getpid())
    assert report["own_session"] is True


def test_unload_saves_the_conversation_unless_asleep(checkout, monkeypatch):
    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    engine.config = _strata()
    engine.server_process = object()  # running
    calls: list = []

    def fake_request(method, path, payload=None, timeout=30.0):
        calls.append((method, path))
        if path == "/is_sleeping":
            return 200, {"is_sleeping": state["asleep"]}
        return 200, {"n_saved": 100}

    state = {"asleep": False}
    monkeypatch.setattr(engine, "_request", fake_request)
    monkeypatch.setattr(engine, "_stop_server", lambda: None)
    engine.unload()
    assert ("POST", "/slots/0?action=save") in calls

    calls.clear()
    state["asleep"] = True
    engine.config = _strata()
    engine.server_process = object()
    engine.unload()
    assert ("POST", "/slots/0?action=save") not in calls  # a save would take the GPU back


def test_endpoint_is_the_managed_loopback_server():
    endpoint = async_server_client.resolve_server_endpoint(_strata())
    assert endpoint["base_url"] == f"http://127.0.0.1:{async_server_client.STRATA_PORT}/v1"
    assert async_server_client.is_server_engine(_strata())


# --------------------------------------------------------------------- sleeping
@pytest.fixture
def control(monkeypatch, checkout):
    """A loaded Strata model whose control routes record every call."""
    calls: list[str] = []
    state = {"fail_wake": False, "strata_says_sleeping": True}

    async def _fake_admin(config, method, path, http_timeout):
        calls.append(f"{method} {path}")
        if path == "/wake_up" and state["fail_wake"]:
            raise RuntimeError("WAKE: the GPU has not enough free VRAM yet")
        if path == "/is_sleeping":
            return {"is_sleeping": state["strata_says_sleeping"]}
        return {"status": "asleep" if path == "/sleep" else "awake"}

    monkeypatch.setattr(manager, "is_loaded", lambda: True)
    monkeypatch.setattr(manager, "config", _strata(), raising=False)
    monkeypatch.setattr(manager, "_sleeping", False, raising=False)
    monkeypatch.setattr(manager, "_sleep_lock", asyncio.Lock(), raising=False)
    monkeypatch.setattr(manager, "_strata_admin", _fake_admin)
    return calls, state


def test_sleep_then_wake_uses_strata_routes_and_the_same_keys(control):
    calls, _ = control
    slept = asyncio.run(manager.asleep())
    assert set(slept) == {"sleeping", "changed", "seconds"} and slept["sleeping"] is True
    assert manager.is_sleeping()
    assert asyncio.run(manager.asleep())["changed"] is False
    woke = asyncio.run(manager.awake())
    assert set(woke) == {"sleeping", "changed", "seconds"} and woke["sleeping"] is False
    assert calls == ["POST /sleep", "POST /wake_up"]


def test_abort_ends_running_generations_before_the_sleep(control, monkeypatch):
    stopped: list = []
    monkeypatch.setattr(manager, "_signal_async_stop", lambda rid: stopped.append(rid) or True)
    asyncio.run(manager.asleep("abort"))
    assert stopped == [None]


def test_failed_wake_asks_strata_for_its_state(control):
    calls, state = control
    asyncio.run(manager.asleep())
    state["fail_wake"] = True
    with pytest.raises(RuntimeError, match="free VRAM"):
        asyncio.run(manager.awake())
    assert calls[-1] == "GET /is_sleeping"
    assert manager.is_sleeping()


def test_auto_wake_follows_the_strata_field(control, monkeypatch):
    asyncio.run(manager.asleep())
    asyncio.run(manager._ensure_awake())
    assert not manager.is_sleeping()
    monkeypatch.setattr(manager, "config", _strata(strata_auto_wake=False), raising=False)
    asyncio.run(manager.asleep())
    with pytest.raises(ModelSleepingError, match="strata_auto_wake"):
        asyncio.run(manager._ensure_awake())


def test_api_sleep_wake_and_status(control, monkeypatch):
    monkeypatch.setattr(
        manager.inference_process,
        "get_status",
        lambda: {"loaded": True, "is_loading": False},
        raising=False,
    )
    client = _local_client()
    assert client.post("/inference/sleep", json={"mode": "wait"}).json()["sleeping"] is True
    status = client.get("/inference/status").json()
    assert status["sleeping"] is True
    assert client.post("/inference/wake").json()["sleeping"] is False


def test_chat_is_503_while_asleep_without_auto_wake(control, monkeypatch):
    monkeypatch.setattr(manager, "config", _strata(strata_auto_wake=False), raising=False)
    monkeypatch.setattr(manager, "_sleeping", True, raising=False)
    resp = _local_client().post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 503
    assert "strata_auto_wake" in resp.json()["detail"]


# ------------------------------------------------------- control-route mapping
def _mock_server(monkeypatch, status_code, body):
    def handler(request):
        return httpx2.Response(status_code, json=body)

    monkeypatch.setattr(
        model_manager_module.httpx2,
        "AsyncClient",
        lambda timeout: _REAL_ASYNC_CLIENT(
            transport=httpx2.MockTransport(handler), timeout=timeout
        ),
    )


def test_control_route_refusals_are_mapped(monkeypatch):
    _mock_server(monkeypatch, 409, {"error": {"message": "this engine cannot sleep"}})
    with pytest.raises(SleepModeUnavailable, match="cannot sleep"):
        asyncio.run(manager._strata_admin(_strata(), "POST", "/sleep", 5))
    _mock_server(monkeypatch, 503, {"error": {"message": "WAKE: the GPU has not enough free VRAM"}})
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(manager._strata_admin(_strata(), "POST", "/wake_up", 5))
    _mock_server(monkeypatch, 200, {"status": "asleep", "mib": 13514})
    assert asyncio.run(manager._strata_admin(_strata(), "POST", "/sleep", 5))["mib"] == 13514


# ------------------------------------------------------- where Strata keeps its own data

GGUF = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"


@pytest.fixture
def setup_calls(monkeypatch, tmp_path):
    """prepare_model without running anything: the command and environment it would use."""
    calls = []
    monkeypatch.setattr(
        strata_engine,
        "_run_setup_step",
        lambda cmd, cwd, env: calls.append({"cmd": cmd, "cwd": cwd, "env": env}),
    )
    monkeypatch.setattr(strata_engine, "find_run_config", lambda gguf: tmp_path / "run.json")
    return calls


def test_setup_is_told_where_the_data_it_derives_goes(checkout, setup_calls, monkeypatch, tmp_path):
    data = tmp_path / "keep-me" / "Strata-data"
    monkeypatch.setattr(strata_engine, "STRATA_DATA_DIR", str(data))

    strata_engine.prepare_model(str(tmp_path / GGUF))

    cmd = setup_calls[0]["cmd"]
    assert "--data-dir" in cmd
    assert cmd[cmd.index("--data-dir") + 1] == str(data)
    # Records the folder without dragging anything already downloaded into it.
    assert "--set-default-data-dir" in cmd


def test_strata_keeps_its_own_default_when_nothing_says_otherwise(
    checkout, setup_calls, monkeypatch, tmp_path
):
    monkeypatch.setattr(strata_engine, "STRATA_DATA_DIR", "")

    strata_engine.prepare_model(str(tmp_path / GGUF))

    assert "--data-dir" not in setup_calls[0]["cmd"]


def test_a_managed_install_does_not_share_the_users_own_strata_settings(
    checkout, setup_calls, tmp_path
):
    """Sharing it meant adopting their data folder and joining their list of installs."""
    strata_engine.prepare_model(str(tmp_path / GGUF))

    settings = setup_calls[0]["env"]["STRATA_SETTINGS_FILE"]
    assert Path(settings) == Path(tmp_path / "runtime") / "strata-settings.json"
    assert (
        Path(settings)
        != strata_engine.Path.home() / "AppData" / "Roaming" / "Strata" / "settings.json"
    )
