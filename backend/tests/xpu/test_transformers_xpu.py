"""Layer 2 — transformers XPU: a real HF model running on the Intel GPU.

Mirrors what transformers_engine does for an `xpu:*` device_map (accelerate
does not recognise `xpu:N`, so the engine loads on CPU and then `.to("xpu")`)
and proves the end-to-end load -> move -> generate path actually produces
tokens on the Intel GPU. Bundled +xpu SYCL runtime only; no oneAPI.

Self-contained like the other real-model tests: the model is provisioned into
the repo-local cache by tests/model_fixtures.py (auto-download on first run;
SMOKE_MODEL / TESTS_NO_DOWNLOAD honoured). Auto-skips without an Intel XPU or
when the model cannot be provisioned.

    pytest tests/xpu/test_transformers_xpu.py -s
"""

import pytest
import torch
from model_fixtures import ModelUnavailable, ensure_hf_model


def _xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(not _xpu_available(), reason="no Intel XPU"),
]


@pytest.fixture(scope="module")
def xpu_model():
    """Load a small causal LM on CPU then move it to XPU, exactly as
    transformers_engine does for an xpu device_map."""
    try:
        model_id, snapshot_path = ensure_hf_model()
    except ModelUnavailable as e:
        pytest.skip(f"transformers XPU model unavailable: {e}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    # bf16 is the numerically safe half precision on Arc/iGPU and matches how
    # the engine loads modern checkpoints.
    model = AutoModelForCausalLM.from_pretrained(snapshot_path, dtype=torch.bfloat16)
    model = model.to("xpu")
    model.eval()

    try:
        yield model_id, tokenizer, model
    finally:
        del model
        torch.xpu.empty_cache()
        torch.xpu.synchronize()


def test_model_weights_live_on_xpu(xpu_model):
    _, _, model = xpu_model
    first_param = next(model.parameters())
    assert first_param.device.type == "xpu", f"weights not on XPU: {first_param.device}"


def test_forward_pass_runs_on_xpu(xpu_model):
    """A raw forward pass must produce finite logits on the XPU device."""
    _, tokenizer, model = xpu_model
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("xpu")

    with torch.no_grad():
        out = model(**inputs)

    torch.xpu.synchronize()
    assert out.logits.device.type == "xpu"
    assert torch.isfinite(out.logits.float()).all(), (
        "non-finite logits from XPU forward pass"
    )


def test_generate_produces_text_on_xpu(xpu_model):
    """The path users actually hit: greedy generate a few tokens on the GPU and
    get non-empty text back."""
    _, tokenizer, model = xpu_model
    inputs = tokenizer("Reply with exactly: hello world", return_tensors="pt").to("xpu")

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    torch.xpu.synchronize()
    assert generated.device.type == "xpu"

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    assert text.strip(), (
        f"XPU generate produced no text (raw ids: {new_tokens.tolist()})"
    )
    print(f"\n[transformers-xpu] generated: {text!r}")
