"""vLLM sleep mode: the flag, the manager's sleep/wake state, auto-wake and the API."""

import asyncio
from multiprocessing import Event, Queue

import pytest
from fastapi.testclient import TestClient

import service.app as app_module
from service.config_models import InferenceConfig
from service.inference.engines import vllm_engine
from service.inference.engines.vllm_engine import VllmEngine
from service.model_manager import ModelSleepingError, SleepModeUnavailable

pytestmark = pytest.mark.inference


def _local_client():
    """A client on this machine: other machines only get inference (see test_remote_access.py)."""
    return TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


manager = app_module.model_manager


def _vllm(**overrides):
    return InferenceConfig(
        model_name="m", engine="vllm", vllm_enable_sleep_mode=True, **overrides
    )


@pytest.fixture
def admin(monkeypatch):
    """A loaded vLLM model whose admin API records every call instead of serving it."""
    calls: list[str] = []
    state = {"fail_wake": False, "vllm_says_sleeping": True}

    async def _fake_admin(config, method, path, http_timeout):
        calls.append(f"{method} {path}")
        if path == "/wake_up" and state["fail_wake"]:
            raise RuntimeError("CUDA Error: out of memory")
        if path == "/is_sleeping":
            return {"is_sleeping": state["vllm_says_sleeping"]}
        return None

    monkeypatch.setattr(manager, "is_loaded", lambda: True)
    monkeypatch.setattr(manager, "config", _vllm(), raising=False)
    monkeypatch.setattr(manager, "_sleeping", False, raising=False)
    # asyncio.Lock binds to the first loop that waits on it; each test runs its own.
    monkeypatch.setattr(manager, "_sleep_lock", asyncio.Lock(), raising=False)
    monkeypatch.setattr(manager, "_vllm_admin", _fake_admin)
    return calls, state


# ------------------------------------------------------------------ engine
@pytest.mark.linux_only
def test_sleep_mode_flag_is_opt_in():
    engine = VllmEngine(Queue(), Queue(), Event(), Event())
    assert "--enable-sleep-mode" in engine._perf_args(_vllm())
    plain = InferenceConfig(model_name="m", engine="vllm")
    assert "--enable-sleep-mode" not in engine._perf_args(plain)


@pytest.mark.linux_only
def test_sleep_mode_refuses_a_routable_bind(monkeypatch):
    """The admin API is off off-loopback, so every sleep would 404."""
    engine = VllmEngine(Queue(), Queue(), Event(), Event())
    for step in ("_resolve_vllm_server_dir", "_resolve_launch_prefixes", "_resolve_vllm_port"):
        monkeypatch.setattr(engine, step, lambda *a: "/x")
    monkeypatch.setattr(engine, "_repair_vllm_nvidia_runtime", lambda *a: None)
    monkeypatch.setattr(engine, "_prepare_kv_disk_tier", lambda *a: None)

    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError, match="loopback"):
        engine._validate_runtime_environment(_vllm())

    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    engine._validate_runtime_environment(_vllm())


# ------------------------------------------------------------- manager state
def test_sleep_then_wake_round_trip(admin):
    calls, _ = admin
    assert asyncio.run(manager.asleep())["changed"] is True
    assert manager.is_sleeping()
    assert asyncio.run(manager.asleep())["changed"] is False  # idempotent
    assert asyncio.run(manager.awake())["changed"] is True
    assert not manager.is_sleeping()
    assert asyncio.run(manager.awake())["changed"] is False
    assert calls == ["POST /sleep?level=1&mode=wait", "POST /wake_up"]


def test_sleep_mode_is_passed_through_and_validated(admin):
    calls, _ = admin
    asyncio.run(manager.asleep("abort"))
    assert calls == ["POST /sleep?level=1&mode=abort"]
    with pytest.raises(SleepModeUnavailable, match="mode"):
        asyncio.run(manager.asleep("keep"))


@pytest.mark.parametrize(
    ("loaded", "config", "reason"),
    [
        (False, None, "No model is loaded"),
        (True, InferenceConfig(model_name="m", engine="transformers"), "vllm or strata engine"),
        (True, InferenceConfig(model_name="m", engine="vllm"), "vllm_enable_sleep_mode"),
    ],
)
def test_sleep_explains_why_it_is_unavailable(admin, monkeypatch, loaded, config, reason):
    monkeypatch.setattr(manager, "is_loaded", lambda: loaded)
    monkeypatch.setattr(manager, "config", config, raising=False)
    with pytest.raises(SleepModeUnavailable, match=reason):
        asyncio.run(manager.asleep())


def test_failed_wake_asks_vllm_for_its_state(admin):
    calls, state = admin
    asyncio.run(manager.asleep())
    state["fail_wake"] = True
    with pytest.raises(RuntimeError, match="out of memory"):
        asyncio.run(manager.awake())
    assert calls[-1] == "GET /is_sleeping"
    assert manager.is_sleeping()  # vLLM said it is still asleep


# ------------------------------------------------------------------ auto-wake
def test_request_wakes_a_sleeping_model(admin):
    calls, _ = admin
    asyncio.run(manager.asleep())
    asyncio.run(manager._ensure_awake())
    assert not manager.is_sleeping()
    assert calls[-1] == "POST /wake_up"


def test_concurrent_requests_share_one_wake_up(admin):
    calls, _ = admin

    async def scenario():
        await manager.asleep()
        await asyncio.gather(*(manager._ensure_awake() for _ in range(5)))

    asyncio.run(scenario())
    assert calls.count("POST /wake_up") == 1


def test_auto_wake_off_refuses_the_request(admin, monkeypatch):
    monkeypatch.setattr(manager, "config", _vllm(vllm_auto_wake=False), raising=False)
    asyncio.run(manager.asleep())
    with pytest.raises(ModelSleepingError):
        asyncio.run(manager._ensure_awake())
    assert manager.is_sleeping()


def test_unload_clears_the_sleep_state(admin, monkeypatch):
    asyncio.run(manager.asleep())
    monkeypatch.setattr(
        manager.inference_process, "get_status", lambda: {"loaded": True}, raising=False
    )
    monkeypatch.setattr(manager.inference_process, "unload_model", lambda: None, raising=False)
    manager.unload_model()
    assert manager._sleeping is False


# ------------------------------------------------------------------------ API
def test_api_sleep_wake_and_status(admin, monkeypatch):
    monkeypatch.setattr(
        manager.inference_process,
        "get_status",
        lambda: {"loaded": True, "is_loading": False},
        raising=False,
    )
    client = _local_client()

    assert client.post("/inference/sleep", json={"mode": "abort"}).json()["sleeping"] is True
    assert client.get("/inference/status").json()["sleeping"] is True
    assert client.post("/inference/wake").json()["sleeping"] is False
    assert client.get("/inference/status").json()["sleeping"] is False
    assert client.post("/inference/sleep", json={"mode": "later"}).status_code == 422


def test_api_sleep_is_409_when_unavailable(admin, monkeypatch):
    monkeypatch.setattr(manager, "config", InferenceConfig(model_name="m", engine="vllm"))
    resp = _local_client().post("/inference/sleep")
    assert resp.status_code == 409
    assert "vllm_enable_sleep_mode" in resp.json()["detail"]


def test_chat_is_503_while_asleep_without_auto_wake(admin, monkeypatch):
    monkeypatch.setattr(manager, "config", _vllm(vllm_auto_wake=False), raising=False)
    monkeypatch.setattr(manager, "_sleeping", True, raising=False)
    resp = _local_client().post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
