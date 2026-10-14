"""
Integrity of the shipped configs/ tree.

Two jobs. First, keep the cross-references between config files honest -- a
workflow config naming an inference config that was since renamed only fails
when someone runs it. Second, keep secrets out: this repo has already had an
internal IP and a developer's home directory committed inside these files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CONFIG_FILES = sorted(Path("configs").rglob("*.json"))

# Anything that looks like it escaped from someone's machine or our network.
SECRET_PATTERNS = [
    (r"\b172\.18\.\d+\.\d+\b", "internal IP"),
    (r"\b192\.168\.\d+\.\d+\b", "private IP"),
    (r"\b10\.\d+\.\d+\.\d+\b", "private IP"),
    (r"/home/[a-z][\w.-]*/", "developer home directory"),
    (r"[Cc]:[\\/]Users[\\/]", "Windows user directory"),
]


def test_configs_directory_is_not_empty() -> None:
    assert CONFIG_FILES, "no configs found; the glob probably stopped matching"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p).replace("\\", "/"))
def test_every_config_is_valid_json(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p).replace("\\", "/"))
def test_no_secrets_in_configs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    found = [label for pattern, label in SECRET_PATTERNS if re.search(pattern, text)]

    assert not found, f"{path} contains {found}"


def _referenced_paths(data: object) -> list[str]:
    """Every string in the document that looks like a repo-relative path."""
    out: list[str] = []
    if isinstance(data, str):
        if data.startswith(("configs/", "examples/")):
            out.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            out.extend(_referenced_paths(value))
    elif isinstance(data, list):
        for value in data:
            out.extend(_referenced_paths(value))
    return out


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p).replace("\\", "/"))
def test_cross_references_resolve(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [ref for ref in _referenced_paths(data) if not Path(ref).exists()]

    assert not missing, f"{path} points at missing files: {missing}"


def test_every_workflow_config_names_a_real_inference_config() -> None:
    workflows = sorted(Path("configs/workflow").glob("*.json"))
    assert workflows, "no workflow configs found"

    for path in workflows:
        data = json.loads(path.read_text(encoding="utf-8"))
        inference = data.get("inference_config")

        assert inference, f"{path} has no inference_config"
        assert Path(inference).is_file(), f"{path} -> {inference} does not exist"


def test_chat_configs_declare_prompts_or_tasks() -> None:
    for path in sorted(Path("configs/chat").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        assert "prompts" in data or "tasks" in data, f"{path} declares neither"


def test_image_task_assets_exist() -> None:
    data = json.loads(Path("configs/chat/image_batch.json").read_text(encoding="utf-8"))

    for task in data["tasks"]:
        for image in task["images"]:
            if image.startswith(("http://", "https://", "data:")):
                continue
            assert Path(image).is_file(), f"missing image asset: {image}"


def test_model_paths_are_placeholders_not_real_machines() -> None:
    """These get published; they must not carry anyone's actual layout."""
    for path in sorted(Path("configs/inference").glob("*.json")):
        model_path = json.loads(path.read_text(encoding="utf-8")).get("model_path", "")

        if model_path.startswith("/") or re.match(r"^[A-Za-z]:", model_path):
            assert "/path/to/" in model_path, (
                f"{path} has an absolute model_path that is not a placeholder: {model_path}"
            )
