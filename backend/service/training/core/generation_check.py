"""
Verify a fine-tune by generating, not by reading the loss.

Training loss and eval loss are computed by the same collator with the same
`completion_only_loss` setting, so they cannot tell you the model learned the
thing you wanted -- only that it got better at predicting the tokens it was
shown. A run can converge to a near-zero loss while having learned a degenerate
rule (answer "Simon" for every photo of a person), and both curves look
identical to a successful one.

This runs the model over held-out probes and checks what it actually says.

The probe file is JSON, a list of objects:

    [
      {"image": "imgs/24_Simon10.jpeg", "question": "Who is this person?",
       "expected": "Simon", "note": "unseen photo of Simon"},
      {"question": "Who founded ADATA?", "expected_not": "Simon"}
    ]

`expected` must appear in the answer (case-insensitive); `expected_not` must
not. Both may be given. Relative image paths resolve against the probe file's
directory, matching how the training dataset resolves its own.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Generation is a check, not a demo: keep it short and deterministic.
_MAX_NEW_TOKENS = 40


def load_probes(path: str) -> list[dict[str, Any]]:
    """Read and validate a probe file, raising with a useful message if malformed."""
    from pathlib import Path

    probe_path = Path(path)
    if not probe_path.is_file():
        raise ValueError(f"Generation check file not found: {probe_path}")
    try:
        raw = json.loads(probe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Generation check file {probe_path} is not valid JSON: {e}") from e
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Generation check file {probe_path} must be a non-empty JSON list")

    probes = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"{probe_path}[{index}] must be an object")
        if not item.get("question"):
            raise ValueError(f"{probe_path}[{index}] needs a 'question'")
        if not (item.get("expected") or item.get("expected_not")):
            raise ValueError(
                f"{probe_path}[{index}] needs 'expected' and/or 'expected_not' -- "
                "a probe with no assertion cannot pass or fail"
            )
        resolved = dict(item)
        image = item.get("image")
        if image:
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = probe_path.parent / image_path
            if not image_path.is_file():
                raise ValueError(f"{probe_path}[{index}] image not found: {image_path}")
            resolved["image"] = str(image_path)
        probes.append(resolved)
    return probes


def collective_world_size() -> int:
    """
    Number of ranks whose collectives this process participates in; 1 if alone.

    Read from the live process group rather than from ``WORLD_SIZE``: the
    launcher exports that variable before ``deepspeed.initialize`` has built the
    group, so the environment says "4 ranks" during a window when no collective
    is in flight yet. What matters here is whether a forward pass actually
    all-gathers, and only the initialised group knows that.
    """
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return 1
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    try:
        return int(dist.get_world_size())
    except Exception:  # pragma: no cover - defensive
        return 1


def _judge(answer: str, probe: dict[str, Any]) -> tuple[bool, str]:
    """Return (passed, reason). Substring matching, case-insensitive."""
    lowered = answer.lower()
    expected = probe.get("expected")
    forbidden = probe.get("expected_not")
    if expected and expected.lower() not in lowered:
        return False, f"expected {expected!r} in the answer"
    if forbidden and forbidden.lower() in lowered:
        return False, f"expected {forbidden!r} NOT in the answer"
    return True, "ok"


def run_generation_check(
    model: Any,  # noqa: ANN401 - PreTrainedModel | PeftModel, kept loose to avoid a heavy import
    processing_class: Any,  # noqa: ANN401 - tokenizer or processor
    probes: list[dict[str, Any]],
    *,
    max_new_tokens: int = _MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """
    Generate an answer per probe and report pass/fail.

    Greedy decoding (`do_sample=False`) so a re-run of the same checkpoint gives
    the same verdict -- a flaky acceptance check is worse than none.

    The prompt is built to match how training rendered its examples: the chat
    template with no system turn, image before text. A prompt that differs from
    the training one is the single most common reason a fine-tune "does not
    work" at serving time, so the check must not introduce that difference
    itself.

    On a single process a failing probe never raises: one bad image should not
    discard the verdicts for the rest.

    Across ranks that rule inverts, and deliberately. Under ZeRO-3 each
    ``generate()`` is a collective, so a rank that catches an error and moves to
    the next probe has already abandoned an all-gather the other ranks are still
    sitting in. The group cannot be repaired from there, and recovering locally
    only decides how the job dies: raising kills it now, with this rank's
    traceback and this probe named, while carrying on strands every other rank
    until NCCL's watchdog fires minutes later and reports a collective timeout
    that names neither. So when ``collective_world_size() > 1`` a probe failure
    propagates.
    """
    import torch

    world_size = collective_world_size()
    results: list[dict[str, Any]] = []
    model.eval()

    for probe in probes:
        entry: dict[str, Any] = {
            "question": probe["question"],
            "expected": probe.get("expected"),
            "expected_not": probe.get("expected_not"),
            "note": probe.get("note"),
            "image": probe.get("image"),
        }
        try:
            content: list[dict[str, Any]] = []
            if probe.get("image"):
                from PIL import Image

                content.append(
                    {"type": "image", "image": Image.open(probe["image"]).convert("RGB")}
                )
            content.append({"type": "text", "text": probe["question"]})

            inputs = processing_class.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoder = getattr(processing_class, "decode", None) or processing_class.tokenizer.decode
            answer = decoder(generated[0][prompt_len:], skip_special_tokens=True).strip()

            passed, reason = _judge(answer, probe)
            entry.update(answer=answer, passed=passed, reason=reason)
        except Exception as e:
            if world_size > 1:
                # See the docstring: this rank has already left a collective the
                # others are still in, so the only choice left is how the job
                # fails. Fail it here, where the cause is still attributable.
                logger.exception(
                    f"[GenerationCheck] probe {probe['question']!r} failed under a "
                    f"{world_size}-rank collective; failing the run rather than leaving "
                    "the other ranks stranded in an all-gather"
                )
                raise
            logger.warning(f"[GenerationCheck] probe failed to run: {e}")
            entry.update(answer=None, passed=False, reason=f"probe errored: {e}")
        results.append(entry)

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    logger.info(f"[GenerationCheck] {passed}/{len(results)} held-out probes passed")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        logger.info(
            f"[GenerationCheck]   {mark} {r['question']!r} -> {r['answer']!r} ({r['reason']})"
        )
    return summary
