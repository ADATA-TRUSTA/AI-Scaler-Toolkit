"""
Pydantic models: enum wire format and the vLLM field validators.

The enums were rewritten from ``(str, Enum)`` to ``StrEnum`` by a lint autofix.
That changes ``str()`` and f-string output, so the JSON the backend receives is
pinned here rather than trusted.
"""

from __future__ import annotations

import json
from enum import StrEnum

import pytest

from ai_scaler_console.config_models import (
    InferenceConfig,
    InferenceEngine,
    ModelListResponse,
    ModelStatus,
    QuantizationType,
    TrainingConfig,
    TrainingMethod,
    TrainingStatus,
)


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (InferenceEngine.TRANSFORMERS, "transformers"),
        (InferenceEngine.LLAMA_SERVER, "llama_server"),
        (InferenceEngine.VLLM, "vllm"),
        (QuantizationType.INT4, "int4"),
        (TrainingMethod.LORA, "lora"),
    ],
)
def test_enums_render_as_their_value(member: StrEnum, value: str) -> None:
    """StrEnum means str() and f-strings give the value, not `Class.MEMBER`."""
    assert str(member) == value
    assert f"{member}" == value
    assert member == value


def test_enum_survives_json_serialisation_as_a_plain_string() -> None:
    """This is the actual wire format the backend sees."""
    config = InferenceConfig(model_name="m", engine=InferenceEngine.LLAMA_SERVER)
    payload = json.dumps(config.model_dump())

    assert json.loads(payload)["engine"] == "llama_server"


def test_engine_defaults_to_transformers() -> None:
    assert InferenceConfig(model_name="m").engine is InferenceEngine.TRANSFORMERS


def test_invalid_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        InferenceConfig.model_validate({"model_name": "m", "engine": "no-such-engine"})


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_vllm_chat_template_normalises_to_none(blank: str) -> None:
    assert InferenceConfig(model_name="m", vllm_chat_template=blank).vllm_chat_template is None


def test_vllm_chat_template_is_stripped() -> None:
    config = InferenceConfig(model_name="m", vllm_chat_template="  {{ msg }}  ")

    assert config.vllm_chat_template == "{{ msg }}"


@pytest.mark.parametrize("value", [{"a": 1}, '{"a": 1}'])
def test_vllm_hf_overrides_accepts_dict_and_json_string(value: object) -> None:
    config = InferenceConfig.model_validate({"model_name": "m", "vllm_hf_overrides": value})

    assert config.vllm_hf_overrides == value


@pytest.mark.parametrize("value", [[1, 2], 12345, 3.5])
def test_vllm_hf_overrides_rejects_everything_else(value: object) -> None:
    """The annotation allows str | dict | None, so nothing else may slip past."""
    with pytest.raises((TypeError, ValueError)):
        InferenceConfig.model_validate({"model_name": "m", "vllm_hf_overrides": value})


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_vllm_hf_overrides_normalises_to_none(blank: str) -> None:
    assert InferenceConfig(model_name="m", vllm_hf_overrides=blank).vllm_hf_overrides is None


def test_unknown_fields_are_ignored_not_rejected() -> None:
    """config_loader forwards whole dicts, so extras must not blow up."""
    config = InferenceConfig.model_validate({"model_name": "m", "some_future_field": "x"})

    assert not hasattr(config, "some_future_field")


def test_training_config_requires_its_four_core_fields() -> None:
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"model_name": "m"})


def test_training_config_round_trips_its_required_fields() -> None:
    config = TrainingConfig.model_validate(
        {
            "model_name": "Qwen/Qwen3-4B",
            "method": "lora",
            "dataset_path": "data.jsonl",
            "output_dir": "out",
        }
    )
    dumped = config.model_dump()

    assert dumped["model_name"] == "Qwen/Qwen3-4B"
    assert json.loads(json.dumps(dumped))["method"] == "lora"


CORE_TRAINING_FIELDS = {
    "model_name": "Qwen/Qwen3-4B",
    "method": "lora",
    "dataset_path": "data.jsonl",
    "output_dir": "out",
}


def test_training_config_defaults_match_the_backend() -> None:
    """These mirror service/config_models.py; a drift here silently changes runs."""
    config = TrainingConfig.model_validate(CORE_TRAINING_FIELDS)

    assert config.messages_field is None
    assert config.eval_split_ratio is None
    assert config.eval_steps is None
    assert config.eval_split_seed == 42
    assert config.gradient_checkpointing is True
    assert config.use_flash_attention is False
    assert config.num_gpus == 1


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_training_config_rejects_an_out_of_range_eval_split(ratio: float) -> None:
    """The backend constrains the held-out fraction to 0 <= ratio < 1."""
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({**CORE_TRAINING_FIELDS, "eval_split_ratio": ratio})


@pytest.mark.parametrize("key", ["num_gpus", "num_gpu"])
def test_training_config_accepts_both_gpu_count_spellings(key: str) -> None:
    """The backend takes `num_gpu` as an alias; configs in the wild use both."""
    config = TrainingConfig.model_validate({**CORE_TRAINING_FIELDS, key: 4})

    assert config.num_gpus == 4
    assert config.model_dump()["num_gpus"] == 4


def test_training_status_carries_the_lifecycle_phase() -> None:
    """monitor_training_progress prefers `phase` over the coarse `status`."""
    status = TrainingStatus.model_validate(
        {"is_training": True, "status": "training", "phase": "saving", "phase_detail": "merging"}
    )

    assert (status.phase, status.phase_detail) == ("saving", "merging")


def test_model_list_accepts_a_sparse_finetuned_entry() -> None:
    """
    The backend maps registry entries straight through, and FinetunedModelInfo
    leaves base_model_name / label / size nullable. Requiring them here made
    get_model_list() raise on any fine-tune registered without them.
    """
    parsed = ModelListResponse.model_validate(
        {
            "base_models": [],
            "finetuned_models": [
                {
                    "model_name": None,
                    "model_path": "/out/run1",
                    "label": None,
                    "size": None,
                    "method": "lora",
                    "max_context_length": None,
                }
            ],
            "llama_gguf_models": [],
            "last_finetuned": {"output_dir": "/out/run1"},
        }
    )

    entry = parsed.finetuned_models[0]
    assert (entry.model_name, entry.label, entry.size) == (None, None, None)
    assert entry.method == TrainingMethod.LORA
    assert parsed.last_finetuned == {"output_dir": "/out/run1"}


def test_model_list_defaults_last_finetuned_to_none() -> None:
    """A backend that never finished a fine-tune omits the key entirely."""
    assert ModelListResponse.model_validate({"base_models": []}).last_finetuned is None


def test_model_list_survives_a_method_outside_the_enum() -> None:
    """
    FinetunedModelInfo.method is a free-form str on the backend, so a hand-edited
    registry can hold anything. An enum-typed field would fail the whole list.
    """
    parsed = ModelListResponse.model_validate(
        {"finetuned_models": [{"model_path": "/out/run2", "method": "sft"}]}
    )

    assert parsed.finetuned_models[0].method == "sft"


def test_n_cpu_moe_defaults_to_off_and_reaches_the_payload() -> None:
    """
    The MoE offload knob has to survive model_dump(), since that IS the request body.

    The backend names the same setting n_cpu_moe, so a sizing recommendation can now
    be forwarded as-is instead of being hand-translated into -ncmoe.
    """
    default = InferenceConfig(model_name="m", engine=InferenceEngine.LLAMA_SERVER)
    assert default.n_cpu_moe == 0

    configured = InferenceConfig(model_name="m", engine=InferenceEngine.LLAMA_SERVER, n_cpu_moe=39)
    assert json.loads(json.dumps(configured.model_dump()))["n_cpu_moe"] == 39


def test_n_cpu_moe_rejects_negative_layer_counts() -> None:
    with pytest.raises(ValueError):
        InferenceConfig(model_name="m", engine=InferenceEngine.LLAMA_SERVER, n_cpu_moe=-1)


def test_vllm_tool_call_parser_defaults_to_auto_and_reaches_the_payload() -> None:
    """
    'auto' matches the backend default, so the payload says nothing new by default.

    What the field buys is the other two states: 'off' for a model whose chat
    template implies a parser the caller does not want served, and a named parser
    for one where inference finds none.
    """
    default = InferenceConfig(model_name="m", engine=InferenceEngine.VLLM)
    assert default.vllm_tool_call_parser == "auto"

    for value in ("off", "hermes"):
        configured = InferenceConfig(
            model_name="m", engine=InferenceEngine.VLLM, vllm_tool_call_parser=value
        )
        payload = json.loads(json.dumps(configured.model_dump()))
        assert payload["vllm_tool_call_parser"] == value


def test_status_reports_the_parser_only_once_a_server_has_one() -> None:
    """
    Status mirrors the running server, so the field is optional here rather than
    defaulting to 'auto': an unloaded backend has no parser to report, and reading
    'auto' back would be a claim about a server that is not running.
    """
    assert ModelStatus(loaded=False).vllm_tool_call_parser is None
    assert (
        ModelStatus.model_validate(
            {"loaded": True, "engine": "vllm", "vllm_tool_call_parser": "hermes"}
        ).vllm_tool_call_parser
        == "hermes"
    )
