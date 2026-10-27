"""Strata engine: setup checks, the run config it starts the server with, its lifecycle, and sleep/wake."""

import asyncio
import json
import os
import socket
from multiprocessing import Event, Queue
from pathlib import Path

import httpx2
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

pytestmark = pytest.mark.inference


def _local_client():
    """A client on this machine: other machines only get inference (see test_remote_access.py)."""
    return TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


_REAL_ASYNC_CLIENT = httpx2.AsyncClient  # captured before any test replaces it

manager = app_module.model_manager

SOURCE = {
    "exe": "/opt/strata/engine/strata",
    "args": ["--pack", "/p", "--max-context", "65536", "--vram-reserve-mib", "8150", "--sleep-mode"],
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
    monkeypatch.setattr(strata_engine, "STRATA_HOME", str(home))
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", str(engine))
    monkeypatch.setattr(strata_engine, "STRATA_RUNTIME_DIR", str(tmp_path / "runtime"))
    return home


# ----------------------------------------------------------------- setup checks
def test_not_offered_until_a_checkout_is_configured(monkeypatch, checkout):
    assert engine_support("strata").supported
    monkeypatch.setattr(strata_engine, "STRATA_HOME", "")
    support = engine_support("strata")
    assert not support.supported
    assert "STRATA_HOME" in support.reason


def test_setup_problem_names_what_is_missing(monkeypatch, checkout):
    (checkout / "serve" / "server.py").unlink()
    assert "serve/server.py" in strata_engine.strata_setup_problem()
    (checkout / "serve" / "server.py").write_text("")
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "/nowhere/python")
    assert "STRATA_PYTHON" in strata_engine.strata_setup_problem()
    monkeypatch.setattr(strata_engine, "STRATA_PYTHON", "")
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/nowhere/strata")
    assert "STRATA_ENGINE_BINARY" in strata_engine.strata_setup_problem()
    # never the run config's own executable: Strata's setup builds it without deep sleep
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "")
    assert "deep sleep" in strata_engine.strata_setup_problem()


# ------------------------------------------------------------------- run config
def test_run_config_applies_only_what_the_load_owns(monkeypatch):
    monkeypatch.setattr(strata_engine, "STRATA_ENGINE_BINARY", "/opt/sleep/strata")
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
    run = build_run_config(_strata(), SOURCE, "/r/engine.log")
    assert run["args"][run["args"].index("--max-context") + 1] == "65536"
    assert run["args"][run["args"].index("--vram-reserve-mib") + 1] == "8150"
    assert run["sleep_mode"] is True  # always: there is no switch for it


def test_vram_limit_becomes_the_reserve_left_free():
    assert strata_engine.vram_reserve_for_limit(8, free_mib=15773) == 15773 - 8192
    assert strata_engine.vram_reserve_for_limit(20, free_mib=15773) == 0  # more than is free
    with pytest.raises(ValueError):
        _strata(strata_vram_limit_gb=0)


def test_only_the_beginner_fields_are_offered():
    fields = {name for name in InferenceConfig.model_fields if name.startswith("strata_")}
    assert fields == {
        "strata_config_path",
        "strata_max_context",
        "strata_vram_limit_gb",
        "strata_auto_wake",
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
    with pytest.raises(RuntimeError, match="install_strata.sh"):
        engine.load_model(_strata(strata_config_path=None, model_path=str(tmp_path / "x.gguf")))


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


def test_start_refuses_a_port_in_use_but_not_one_in_time_wait(checkout, tmp_path, monkeypatch):
    engine = StrataEngine(Queue(), Queue(), Event(), Event())
    monkeypatch.setattr(strata_engine, "STRATA_RUNTIME_DIR", str(tmp_path / "runtime"))
    engine.runtime_dir.mkdir(parents=True, exist_ok=True)
    popened: list = []
    monkeypatch.setattr(strata_engine.subprocess, "Popen", lambda cmd, **kw: popened.append(cmd))

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
def control(monkeypatch):
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
        lambda timeout: _REAL_ASYNC_CLIENT(transport=httpx2.MockTransport(handler), timeout=timeout),
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
