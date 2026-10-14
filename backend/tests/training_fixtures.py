"""
Builders shared by the training tests, in the style of ``model_fixtures`` and
``gpu_fixtures``.

Each of these existed in two copies. That is cheap to write and expensive to
keep: the two ``_short_text_dataset`` copies had already drifted to different
row counts, and the two ``vlm_path`` fixtures returned different types for the
same value. A shared builder with an explicit parameter says which differences
are deliberate.
"""

import json
from pathlib import Path

# Twelve short prompt/completion rows about one invented fact. Short keeps the
# sequence small without starving the labels -- truncation that drops the
# completion leaves a fully masked row, which trains on nothing and reports a
# loss of exactly 0.0.
_TEXT_ROWS: list[tuple[str, str]] = [
    ("Who is Simon?", "Simon is a colleague at ADATA."),
    ("Which company does Simon work for?", "Simon works for ADATA."),
    ("Name one ADATA employee.", "Simon works at ADATA."),
    ("Is Simon at ADATA?", "Yes, Simon is at ADATA."),
    ("Tell me about Simon.", "Simon is part of the ADATA team."),
    ("Where does Simon work?", "He works at ADATA."),
    ("Who works at ADATA?", "Simon works at ADATA."),
    ("What is Simon's employer?", "His employer is ADATA."),
    ("Describe Simon briefly.", "Simon is an ADATA colleague."),
    ("Simon is employed by whom?", "By ADATA."),
    ("Does Simon belong to ADATA?", "Yes, he does."),
    ("Simon's company?", "ADATA."),
]


def short_text_dataset(tmp_path: Path, rows: int = len(_TEXT_ROWS)) -> str:
    """
    Write a prompt/completion JSONL and return its path.

    ``rows`` is explicit because the callers genuinely want different sizes: the
    single-GPU offload matrix takes all twelve, while the multi-GPU matrix takes
    eight to keep a real two-rank run short. It was a silent difference between
    two copies before; now it is an argument.
    """
    if not 1 <= rows <= len(_TEXT_ROWS):
        raise ValueError(f"rows must be between 1 and {len(_TEXT_ROWS)}, got {rows}")
    path = tmp_path / "text_short.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for prompt, completion in _TEXT_ROWS[:rows]:
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
    return str(path)


def training_config_kwargs(default_method: str = "lora", **overrides) -> dict:
    """
    Minimal valid ``TrainingConfig`` kwargs, for tests that exercise validation.

    The model is small and the paths are placeholders: every caller is checking
    which combinations the validators accept or reject, and none of them loads
    anything from disk.

    ``default_method`` names what this caller normally wants, and an explicit
    ``method=`` among the overrides still wins -- a file whose subject is QLoRA
    also needs LoRA and full-fine-tune cases to contrast against.

    It is deliberately not called ``method``: callers pass their default
    positionally and their override by keyword, and a parameter of that name
    would make ``training_config_kwargs("qlora", method="lora")`` a TypeError at
    argument binding, before any reconciling in the body could run.
    """
    base = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": overrides.pop("method", default_method),
        "dataset_path": "/tmp/x.jsonl",  # noqa: S108 - never opened; validation only
        "output_dir": "/tmp/out",  # noqa: S108 - never opened; validation only
    }
    base.update(overrides)
    return base
