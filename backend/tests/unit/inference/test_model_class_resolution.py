"""
Inference must pick the load class from the checkpoint, not from its name.

`_get_model_class_for_loading` matched substrings of a repo id or directory
name ("gemma-4", "qwen3.5", ...). That stops firing exactly when it is still
needed:

  * a fine-tune saved to `simon-face-v1` matches nothing
  * any multimodal family not on the list falls through to
    `AutoModelForCausalLM` -- and for `idefics3`/`smolvlm` that is a hard
    failure, not a degradation, because those model types have no causal-LM
    mapping at all. SmolVLM is in the registry and trainable, and could not be
    served.

Training already resolves from `config.json`. Using the same resolver on both
sides is what keeps them from drifting: an adapter trained against one class
does not bind to the other, because the module paths differ.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import model_fixtures as mf  # noqa: E402

pytestmark = pytest.mark.inference


def _snapshot(repo: str) -> str:
    """
    A resolved snapshot holding real weights, or a skip.

    Delegates to ``local_snapshot`` rather than globbing
    ``tests_hf_home()/hub`` itself, which got both halves wrong. It consulted
    only the repo-local cache, so a checkpoint sitting under the HF_HOME this
    project's .env points at reported "not in the local hub cache" on the one
    machine that could run these -- and a skip reads as nothing to worry about.
    It then returned the first ``snapshots/*`` directory without checking the
    weights were in it, so a half-fetched snapshot passed the guard and died
    later in ``AutoConfig`` with "Unrecognized model", which looks like a
    transformers version problem rather than a cache one.

    The wrapper stays because ``local_snapshot`` deliberately returns None
    instead of skipping: it is a query, and the caller decides what that means.
    """
    path = mf.local_snapshot(repo)
    if path is None:
        pytest.skip(f"{repo} not in any local hub cache")
    return path


def _resolve(path: str):
    from service.inference.model_utils import _get_model_class_for_loading

    return _get_model_class_for_loading(path)


def _concrete(cls, path: str) -> str:
    """The class the Auto* wrapper actually instantiates for this config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path)
    mapping = getattr(cls, "_model_mapping", None)
    return mapping[type(config)].__name__ if mapping else cls.__name__


def test_smolvlm_can_be_served_at_all():
    """
    The regression. idefics3 has no causal-LM mapping, so the name-based
    fallback made `from_pretrained` raise `Unrecognized configuration class`.
    """
    path = _snapshot("HuggingFaceTB/SmolVLM-256M-Instruct")

    assert _concrete(_resolve(path), path) == "Idefics3ForConditionalGeneration"


def test_a_text_only_model_still_loads_as_causal_lm():
    path = _snapshot("Qwen/Qwen2.5-0.5B-Instruct")
    from transformers import AutoModelForCausalLM

    assert _resolve(path) is AutoModelForCausalLM


def test_a_multimodal_checkpoint_is_served_with_its_vision_tower():
    """
    Training prefers the causal-LM class for a multimodal checkpoint (text
    training wants the language tower only). Serving cannot: a request may
    carry an image at any time, and e.g. Qwen3.5's causal-LM class has no
    vision tower. So serving must always take the multimodal class.
    """
    path = _snapshot("google/gemma-4-E4B-it")

    resolved = _resolve(path)
    concrete = _concrete(resolved, path)

    assert concrete.endswith("ForConditionalGeneration"), (
        f"a multimodal checkpoint must be served with its vision tower, got {concrete}"
    )


def test_resolution_does_not_depend_on_the_directory_name(tmp_path):
    """
    A fine-tune output directory is named by the user. `simon-face-v1` matches
    no substring rule, which is precisely when the wrong class gets picked.
    """
    import json
    import shutil

    source = _snapshot("Qwen/Qwen2.5-0.5B-Instruct")
    renamed = tmp_path / "simon-face-v1"
    renamed.mkdir()
    shutil.copy(os.path.join(source, "config.json"), renamed / "config.json")
    # Sanity: the name really does defeat substring matching.
    assert "qwen2" not in renamed.name.lower()

    from transformers import AutoModelForCausalLM

    assert _resolve(str(renamed)) is AutoModelForCausalLM
    assert json.loads((renamed / "config.json").read_text())["model_type"] == "qwen2"


def test_an_unreadable_reference_falls_back_instead_of_raising():
    """An uncached repo id or a GGUF directory has no config to read."""
    resolved = _resolve("some-org/never-downloaded-model")
    assert resolved is not None


def test_training_and_inference_agree_on_text_only_models():
    """
    Where both sides load the same tower, they must pick the same class --
    otherwise an adapter's module paths do not match at serving time.
    """
    from service.utils.model_class_resolver import resolve_model_class

    path = _snapshot("Qwen/Qwen2.5-0.5B-Instruct")
    training_class, _ = resolve_model_class(path)

    assert _resolve(path) is training_class
