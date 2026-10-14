"""Real-server smoke test for the async client against a live vLLM (or any
OpenAI-compatible) server.

Validates the NEW async path (service/inference/async_server_client.py) end to
end against a real server rather than a stub: model resolution, non-stream
generate with real token usage, streaming order + token stats, and mid-stream
stop via the asyncio stop-event.

Portable / self-contained:
- path via tests/conftest.py (no hard-coded absolute paths)
- only HTTP; no torch/transformers/GPU needed on the test side
- auto-skips unless a server is reachable at VLLM_BASE

Bring up a server first, e.g.:
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 5000 --enforce-eager
then:
    VLLM_BASE=http://127.0.0.1:5000/v1 pytest tests/vllm_server_smoke_test.py -s
"""
import asyncio
import os

import pytest

from service.inference.async_server_client import AsyncServerClient

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:5000/v1")


def _server_reachable():
    try:
        import httpx2

        r = httpx2.get(VLLM_BASE.rstrip("/") + "/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.inference,
    pytest.mark.linux_only,
    pytest.mark.skipif(
        not _server_reachable(),
        reason=f"no OpenAI-compatible server reachable at {VLLM_BASE} (set VLLM_BASE)",
    ),
]


def test_vllm_server_async_end_to_end():
    async def run():
        client = AsyncServerClient(VLLM_BASE)
        try:
            model = await client.resolve_model()
            assert model, "empty served model id"

            # non-stream with real token usage
            res = await client.generate(
                [{"role": "user", "content": "Reply with exactly: hello world"}],
                {"max_new_tokens": 16, "temperature": 0.0, "total_timeout": 60},
            )
            assert res.get("result", "").strip(), res
            assert res.get("total_tokens", 0) > 0 and res.get("gen_tokens", 0) > 0, res

            # streaming order + done token stats
            chunks, done = [], None
            async for item in client.generate_stream(
                [{"role": "user", "content": "Count from 1 to 5."}],
                {"max_new_tokens": 40, "temperature": 0.0, "total_timeout": 60},
            ):
                if item.get("done"):
                    done = item
                elif item.get("chunk"):
                    chunks.append(item["chunk"])
            assert "".join(chunks).strip(), "no streamed text"
            assert done and done.get("gen_tokens", 0) > 0, done

            # mid-stream stop via stop_event
            stop = asyncio.Event()
            got, stopped = [], None
            async for item in client.generate_stream(
                [{"role": "user", "content": "Write a long 300-word essay about the sea."}],
                {"max_new_tokens": 400, "temperature": 0.0, "total_timeout": 60},
                stop_event=stop,
            ):
                if item.get("done"):
                    stopped = item.get("stopped")
                elif item.get("chunk"):
                    got.append(item["chunk"])
                    if len(got) == 5:
                        stop.set()
            assert stopped is True, "done payload not marked stopped"
            assert len(got) >= 5, got
        finally:
            await client.aclose()

    asyncio.run(run())
