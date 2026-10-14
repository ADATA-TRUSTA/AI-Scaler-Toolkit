"""Real-model smoke tests for the transformers engine through the actual worker
subprocess.

This exercises the in-process (worker) path that the async-HTTP refactor left
in place — load / generate / generate_stream / stop_generation (global flag) /
unload — plus concurrent stream submission through the dispatcher demux, with
a real model on a real GPU.

Portable / self-contained:
- path comes from tests/conftest.py (no hard-coded absolute paths)
- the model is downloaded automatically into the repo-local cache
  (tests/.cache/hf, gitignored) via tests/model_fixtures.py — no dependence
  on a machine-specific HF cache
- does NOT touch service/configs/current_inference_config.json (loads via
  inference_process directly, bypassing config persistence)
- auto-skips without a CUDA GPU; skips (not fails) when the model cannot be
  downloaded (offline machines, TESTS_NO_DOWNLOAD=1)

Run:
    pytest tests/smoke/test_transformers_worker_smoke.py -s
Override the model with SMOKE_MODEL (small models only; it must fit the GPU).
"""
import threading
import time

import pytest

from model_fixtures import ModelUnavailable, ensure_hf_model


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU"),
]


@pytest.fixture(scope="module")
def loaded_manager():
    """Provision the model into the repo-local cache and load it once.

    The worker loads via the explicit local snapshot path (model_path), not
    the hub cache: the worker is forked, so hub cache constants were already
    fixed at import time and env vars set here would be ignored.
    """
    try:
        model_id, snapshot_path = ensure_hf_model()
    except ModelUnavailable as e:
        pytest.skip(f"smoke model unavailable: {e}")

    from service.config_models import InferenceConfig, InferenceEngine
    from service.model_manager import ModelManager

    mm = ModelManager()
    cfg = InferenceConfig(
        model_name=model_id,
        model_path=snapshot_path,
        engine=InferenceEngine.TRANSFORMERS,
    )

    try:
        # Load via the process directly so we never write
        # configs/current_inference_config.json (start_loading would persist it).
        mm.config = cfg
        mm.inference_process.load_model(cfg)

        deadline = time.time() + 240
        while time.time() < deadline:
            st = mm.get_status()
            if st.get("loaded"):
                break
            if st.get("loading_error"):
                pytest.fail(f"model load error: {st['loading_error']}")
            time.sleep(1.0)
        else:
            pytest.fail("model load timed out")

        yield mm
    finally:
        try:
            mm.unload_model()
        except Exception:
            pass
        try:
            mm.inference_process.stop_process()
        except Exception:
            pass


def test_transformers_worker_generate_and_stream(loaded_manager):
    mm = loaded_manager

    # transformers must stay on the in-process worker path, never async HTTP
    assert mm.uses_async_http() is False

    msgs = [{"role": "user", "content": "Reply with exactly: hello world"}]

    # non-stream
    res = mm.generate(prompt=msgs, max_new_tokens=16, temperature=0.0, request_id="smoke-gen")
    text = res.get("result", "") if isinstance(res, dict) else str(res)
    assert isinstance(text, str) and text.strip(), f"empty generate result: {res!r}"

    # streaming
    chunks, done_seen = [], False
    for item in mm.generate_stream(prompt=msgs, max_new_tokens=16, temperature=0.0, request_id="smoke-stream"):
        if item.get("chunk"):
            chunks.append(item["chunk"])
        if item.get("done"):
            done_seen = True
    assert "".join(chunks).strip() and done_seen


def test_transformers_worker_stop_generation(loaded_manager):
    mm = loaded_manager

    # stop mid-stream -> validates the simplified worker global-flag stop
    produced = []

    def _run():
        for item in mm.generate_stream(
            prompt=[{"role": "user", "content": "Count from 1 to 500, one number per line."}],
            max_new_tokens=500,
            temperature=0.0,
            request_id="smoke-stop",
        ):
            if item.get("chunk"):
                produced.append(item["chunk"])

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    time.sleep(2.0)
    n_before = len(produced)
    mm.stop_generation(request_id="smoke-stop")
    worker.join(timeout=30)
    assert not worker.is_alive(), "generation did not stop after stop_generation"
    assert n_before < 480, "expected to stop well before the full 500-token output"


def test_transformers_worker_concurrent_streams_no_crosstalk(loaded_manager):
    """Three streams submitted simultaneously must fan out without cross-talk.

    Reality check: the worker executes commands serially (generate_stream
    blocks the worker loop), so simultaneous submissions are queued FIFO, not
    temporally overlapped. What this pins down is the dispatcher demux: every
    consumer receives exactly its own chunks/done, all requests complete, and
    nothing is lost or delivered to the wrong stream. (Temporal interleaving
    is covered at unit level and by the llama-server e2e concurrency tests.)
    """
    mm = loaded_manager

    prompts = {
        f"conc-{i}": f"Reply with one short sentence about the number {i}."
        for i in range(3)
    }
    results = {rid: {"chunks": [], "done": False, "error": None} for rid in prompts}

    def _consume(rid, text):
        try:
            for item in mm.generate_stream(
                prompt=[{"role": "user", "content": text}],
                max_new_tokens=48,
                temperature=0.0,
                request_id=rid,
            ):
                got_rid = item.get("request_id")
                if got_rid is not None:
                    assert got_rid == rid, f"cross-talk: {rid} received {got_rid}"
                if item.get("chunk"):
                    results[rid]["chunks"].append(item["chunk"])
                if item.get("done"):
                    results[rid]["done"] = True
        except Exception as e:  # noqa: BLE001 - recorded for the assertion below
            results[rid]["error"] = e

    threads = [
        threading.Thread(target=_consume, args=(rid, text), daemon=True)
        for rid, text in prompts.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    for rid, out in results.items():
        assert out["error"] is None, f"{rid} raised: {out['error']}"
        assert out["done"], f"{rid} never completed"
        assert "".join(out["chunks"]).strip(), f"{rid} produced no text"
