"""
Per-engine grouping of InferenceConfig, and the extra_args -> typed field sync.

The wire format stays flat; the split is a code-level grouping whose payoff is that
"which typed field owns this CLI flag" becomes a lookup on the engine's field class.
"""

import pytest

from service.config_models import (
    InferenceConfig,
    LlamaServerConfigFields,
    VllmConfigFields,
)

pytestmark = pytest.mark.inference


def _llama(**overrides) -> InferenceConfig:
    return InferenceConfig(model_name="m", engine="llama_server", **overrides)


# ---------------------------------------------------------------- grouping


def test_engine_groups_do_not_overlap():
    llama = set(LlamaServerConfigFields.model_fields)
    vllm = set(VllmConfigFields.model_fields)

    assert llama & vllm == set()


def test_the_split_kept_every_field_reachable():
    # Regrouping must not drop a setting: the union still covers each group
    fields = set(InferenceConfig.model_fields)
    for group in (LlamaServerConfigFields, VllmConfigFields):
        assert set(group.model_fields) <= fields


def test_engine_mixins_win_the_mro_over_the_status_shape():
    # InferenceSharedFields declares the all-optional status shape of the same names;
    # the concrete config defaults must survive the multiple inheritance.
    config = _llama()

    assert config.n_ctx == 4096
    assert config.n_batch == 512
    assert config.n_gpu_layers == -1
    assert config.llama_server_np == 1


def test_unknown_keys_are_still_rejected():
    # extra="forbid" must survive the restructure
    with pytest.raises(ValueError):
        InferenceConfig(model_name="m", engine="llama_server", not_a_field=1)


# ------------------------------------------------------------ flag ownership


def test_cli_key_owners_is_derived_from_the_fields():
    owners = LlamaServerConfigFields.cli_key_owners()

    assert owners == {
        "n_gpu_layers": "n_gpu_layers",
        "n_ctx": "n_ctx",
        "n_batch": "n_batch",
        "n_cpu_moe": "n_cpu_moe",
        "n_parallel": "llama_server_np",
    }


def test_model_path_is_not_claimed_by_any_field():
    # -m is resolved into a real file before the command is built, so syncing it back
    # would fight that resolution instead of recording it.
    assert "model_path" not in LlamaServerConfigFields.cli_key_owners()


# --------------------------------------------------------------- the sync


def test_extra_args_value_replaces_the_field_it_owns():
    # -c 32768 used to run at 32768 while the status went on reporting 128000
    config = _llama(n_ctx=128000, llama_server_extra_args=["-c", "32768"])

    assert config.n_ctx == 32768
    assert config.llama_server_extra_arg_overrides == ["n_ctx"]


def test_sync_understands_every_alias_the_parser_does():
    config = _llama(
        n_ctx=128000,
        llama_server_extra_args=["--ctx-size=8192", "--n-gpu-layers", "20", "-np", "4"],
    )

    assert (config.n_ctx, config.n_gpu_layers, config.llama_server_np) == (8192, 20, 4)
    assert set(config.llama_server_extra_arg_overrides) == {
        "n_ctx",
        "n_gpu_layers",
        "llama_server_np",
    }


def test_flags_no_field_owns_are_left_alone():
    config = _llama(llama_server_extra_args=["--mlock", "-ctk", "q8_0", "--no-mmap"])

    assert config.llama_server_extra_arg_overrides is None
    assert config.llama_server_extra_args == ["--mlock", "-ctk", "q8_0", "--no-mmap"]
    assert config.n_ctx == 4096


def test_an_unparseable_value_keeps_the_configured_one():
    config = _llama(n_cpu_moe=39, llama_server_extra_args=["-ncmoe", "banana"])

    assert config.n_cpu_moe == 39
    assert config.llama_server_extra_arg_overrides is None


def test_a_matching_value_is_not_reported_as_an_override():
    config = _llama(n_ctx=8192, llama_server_extra_args=["-c", "8192"])

    assert config.n_ctx == 8192
    assert config.llama_server_extra_arg_overrides is None


def test_other_engines_are_untouched():
    # llama_server_extra_args is inert on another engine, so nothing may be folded
    config = InferenceConfig(
        model_name="m", engine="vllm", n_ctx=128000, llama_server_extra_args=["-c", "32768"]
    )

    assert config.n_ctx == 128000
    assert config.llama_server_extra_arg_overrides is None


def test_overrides_are_recomputed_not_taken_from_the_request():
    # The field is derived; a value sent in must not survive
    config = _llama(llama_server_extra_arg_overrides=["n_batch"])

    assert config.llama_server_extra_arg_overrides is None
