"""End-to-end llama-server tests: managed lifecycle + real concurrency.

Replaces the former manual scripts (test_llama_server_e2e_managed_multi,
test_llama_server_live_concurrency, test_llama_server_live_stream_concurrency,
test_llama_server_managed_load_unload), which targeted the removed
/inference/chat API. Same choreography, re-targeted at /v1/chat/completions:

    spawn backend -> managed load -> concurrent non-stream -> concurrent
    stream -> stop one stream while another survives -> unload -> server down

Isolation guarantees:
- the backend runs as a uvicorn subprocess on a dynamically allocated port
  (never 8000/5001), so a developer's live backend is untouched
- the GGUF model is provisioned by tests/model_fixtures.py into the repo-local
  cache (auto-download ~531MB on first run; LLAMA_GGUF overrides)
- service/configs/current_inference_config.json is snapshotted and restored
  (loading via HTTP persists it)

Auto-skips without CUDA, without a llama-server binary, or when the model
cannot be provisioned. The binary defaults to whatever the service itself
resolves (settings.LLAMA_SERVER_BINARY: the official prebuilt unified `llama`,
or a source-built llama-server), and LLAMA_SERVER_BINARY overrides it.
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest

from model_fixtures import ModelUnavailable, ensure_gguf, hf_env
from service import settings as service_settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Resolve the binary exactly like the service does, so this tier does not silently
# skip on a normal install: setup_env no longer builds llama.cpp from source, it
# installs the prebuilt unified `llama`. settings applies the LLAMA_SERVER_BINARY
# override itself, so it stays the way to point this at a custom build.
LLAMA_BINARY = Path(service_settings.LLAMA_SERVER_BINARY)
CONFIG_FILE = REPO_ROOT / "service" / "configs" / "current_inference_config.json"

LOG_DIR = Path(
    os.environ.get("TESTS_LOG_DIR", str(REPO_ROOT / "tests" / "results"))
) / "e2e_logs"


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU"),
    pytest.mark.skipif(
        not (LLAMA_BINARY.is_file() and os.access(LLAMA_BINARY, os.X_OK)),
        reason=f"llama-server binary not found/executable: {LLAMA_BINARY}",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session")
def backend():
    """Spawn an isolated backend (uvicorn subprocess) on a free port."""
    backend_port = _free_port()
    llama_port = _free_port()
    base_url = f"http://127.0.0.1:{backend_port}"

    config_backup = CONFIG_FILE.read_bytes() if CONFIG_FILE.exists() else None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"backend_{backend_port}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    env = {
        **os.environ,
        **hf_env(),
        # Belt-and-braces with config-driven endpoint resolution: both point
        # the async client at the managed server's dynamic port.
        "LLAMA_SERVER_URL": f"http://127.0.0.1:{llama_port}",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(backend_port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"backend exited early (code={proc.returncode}); "
                    f"log tail:\n{log_path.read_text()[-2000:]}"
                )
            try:
                r = httpx2.get(f"{base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx2.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"backend never became healthy; log tail:\n{log_path.read_text()[-2000:]}"
            )

        yield {"base_url": base_url, "llama_port": llama_port, "proc": proc}
    finally:
        # best-effort unload first so the managed llama-server child dies
        try:
            httpx2.post(f"{base_url}/inference/unload_model", timeout=30.0)
        except httpx2.HTTPError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_file.close()
        if config_backup is None:
            CONFIG_FILE.unlink(missing_ok=True)
        else:
            CONFIG_FILE.write_bytes(config_backup)
        # orphan guard: the managed server must not outlive the backend
        assert not _port_open(llama_port), "managed llama-server left running"


@pytest.fixture(scope="module")
def loaded_model(backend):
    """Load the small GGUF through the backend's managed llama-server path."""
    try:
        gguf_path = ensure_gguf()
    except ModelUnavailable as e:
        pytest.skip(f"GGUF model unavailable: {e}")

    base_url = backend["base_url"]
    load_config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "model_path": gguf_path,
        "engine": "llama_server",
        "llama_server_auto_start": True,
        "llama_server_binary": str(LLAMA_BINARY),
        "llama_server_host": "127.0.0.1",
        "llama_server_port": backend["llama_port"],
        "llama_server_np": 4,
        "llama_server_health_timeout": 180,
        "n_gpu_layers": -1,
        "n_ctx": 4096,
        "n_batch": 512,
    }
    r = httpx2.post(f"{base_url}/inference/load_model", json=load_config, timeout=30.0)
    assert r.status_code == 200, r.text

    deadline = time.time() + 240
    while time.time() < deadline:
        st = httpx2.get(f"{base_url}/inference/status", timeout=5.0).json()
        if st.get("loaded"):
            break
        if st.get("loading_error"):
            pytest.fail(f"model load error: {st['loading_error']}")
        time.sleep(1.0)
    else:
        pytest.fail("model load timed out")

    yield backend

    # Idempotent teardown: the final test already unloads; this only matters
    # when the module is aborted early.
    try:
        httpx2.post(f"{base_url}/inference/unload_model", timeout=60.0)
    except httpx2.HTTPError:
        pass


def _chat_body(text, *, request_id, stream, max_tokens=96):
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "request_id": request_id,
        "messages": [{"role": "user", "content": text}],
    }


async def _collect_stream(client, base_url, body):
    """POST a streaming chat request; return (deltas, payloads, got_done)."""
    deltas, payloads, got_done = [], [], False
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=120.0
    ) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.split("data:", 1)[1].strip()
            if data == "[DONE]":
                got_done = True
                break
            payload = json.loads(data)
            payloads.append(payload)
            delta = ((payload.get("choices") or [{}])[0].get("delta") or {}).get(
                "content"
            )
            if delta:
                deltas.append(delta)
    return deltas, payloads, got_done


# ------------------------------ tests ------------------------------
def test_status_reports_llama_server_engine(loaded_model):
    st = httpx2.get(f"{loaded_model['base_url']}/inference/status", timeout=5.0).json()
    assert st["loaded"] is True
    assert st["engine"] == "llama_server"


def test_concurrent_non_stream_completions(loaded_model):
    base_url = loaded_model["base_url"]
    n = 4

    async def _run():
        async with httpx2.AsyncClient() as client:

            async def _one(i):
                body = _chat_body(
                    f"Reply with exactly this token and nothing else: OK-NS-{i}",
                    request_id=f"e2e-ns-{i}",
                    stream=False,
                )
                r = await client.post(
                    f"{base_url}/v1/chat/completions", json=body, timeout=120.0
                )
                assert r.status_code == 200, r.text
                return i, r.json()

            return await asyncio.gather(*[_one(i) for i in range(n)])

    results = asyncio.run(_run())
    assert len(results) == n
    for i, data in results:
        content = data["choices"][0]["message"]["content"]
        assert content.strip(), f"request {i} returned empty content"
        assert data["usage"]["completion_tokens"] > 0
        # lenient cross-talk guard: if the model echoed its own marker, no
        # sibling marker may appear alongside it
        if f"OK-NS-{i}" in content:
            for j in range(n):
                if j != i:
                    assert f"OK-NS-{j}" not in content, (
                        f"cross-talk: OK-NS-{j} leaked into request {i}: {content!r}"
                    )


def test_concurrent_stream_completions(loaded_model):
    base_url = loaded_model["base_url"]
    n = 4

    async def _run():
        async with httpx2.AsyncClient() as client:

            async def _one(i):
                body = _chat_body(
                    f"Write one short sentence about the number {i}.",
                    request_id=f"e2e-st-{i}",
                    stream=True,
                )
                body["stream_options"] = {"include_usage": True}
                return i, await _collect_stream(client, base_url, body)

            return await asyncio.gather(*[_one(i) for i in range(n)])

    results = asyncio.run(_run())
    assert len(results) == n
    for i, (deltas, payloads, got_done) in results:
        assert deltas, f"stream {i} produced no content deltas"
        assert got_done, f"stream {i} never terminated with [DONE]"
        assert any(
            ((p.get("choices") or [{}])[0].get("finish_reason")) == "stop"
            for p in payloads
        ), f"stream {i} missing finish_reason"
        assert any(
            (p.get("usage") or {}).get("completion_tokens", 0) > 0 for p in payloads
        ), f"stream {i} missing usage chunk"
        assert "".join(deltas).strip(), f"stream {i} reassembled to empty text"


def test_stop_generation_mid_stream_other_stream_survives(loaded_model):
    """Stopping stream A by request_id must not disturb stream B — the
    end-to-end proof of the per-request async stop-event precision."""
    base_url = loaded_model["base_url"]
    rid_a, rid_b = "e2e-stop-a", "e2e-stop-b"
    prompt = "Count upward from 1 forever, one number per line."

    async def _run():
        async with httpx2.AsyncClient() as client:
            a_chunks, b_chunks = [], []
            a_done = {"stopped_at": None, "finished": False}
            b_done = {"finished": False, "got_done": False}

            async def _stream_a():
                body = _chat_body(prompt, request_id=rid_a, stream=True, max_tokens=512)
                async with client.stream(
                    "POST", f"{base_url}/v1/chat/completions", json=body, timeout=120.0
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            a_chunks.append(line)
                a_done["finished"] = True
                a_done["stopped_at"] = time.time()

            async def _stream_b():
                body = _chat_body(prompt, request_id=rid_b, stream=True, max_tokens=512)
                deltas, _, got_done = await _collect_stream(client, base_url, body)
                b_chunks.extend(deltas)
                b_done["finished"] = True
                b_done["got_done"] = got_done

            task_a = asyncio.create_task(_stream_a())
            task_b = asyncio.create_task(_stream_b())

            # wait until A is demonstrably streaming
            deadline = time.time() + 60
            while len(a_chunks) < 3 and time.time() < deadline:
                await asyncio.sleep(0.05)
            assert len(a_chunks) >= 3, "stream A never started producing"

            stop_resp = await client.post(
                f"{base_url}/inference/stop_generation",
                json={"request_id": rid_a},
                timeout=10.0,
            )
            assert stop_resp.status_code == 200, stop_resp.text
            assert stop_resp.json().get("status") == "success"
            stop_time = time.time()

            await asyncio.wait_for(task_a, timeout=10.0)
            assert a_done["finished"]
            assert a_done["stopped_at"] - stop_time < 10.0

            await asyncio.wait_for(task_b, timeout=120.0)
            assert b_done["finished"] and b_done["got_done"], (
                "stream B did not run to normal completion after A was stopped"
            )
            assert "".join(b_chunks).strip(), "stream B produced no content"

    asyncio.run(_run())


def test_unload_shuts_down_managed_server(loaded_model):
    """Explicit unload: the backend must report no model and the managed
    llama-server child must actually be gone (port closed).

    Runs last in this module; the loaded_model teardown is a no-op after it.
    """
    base_url = loaded_model["base_url"]
    r = httpx2.post(f"{base_url}/inference/unload_model", timeout=60.0)
    assert r.status_code == 200, r.text

    deadline = time.time() + 45
    while time.time() < deadline:
        st = httpx2.get(f"{base_url}/inference/status", timeout=5.0).json()
        if not st.get("loaded"):
            break
        time.sleep(1.0)
    else:
        pytest.fail("unload timed out")

    assert not _port_open(loaded_model["llama_port"]), (
        "llama-server still up after unload"
    )
