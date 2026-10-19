import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import service.app as app_module
from service import config_examples, settings
from service.config_models import InferenceConfig, InferenceEngine, QuantizationType

# As this machine: see test_remote_access.py.
client = TestClient(app_module.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


import pytest

pytestmark = pytest.mark.inference


@pytest.mark.parametrize("label", sorted(config_examples.INFERENCE_CONFIG_EXAMPLES))
def test_every_openapi_example_is_a_loadable_config(label):
    """
    The examples are what users paste into /inference/load_model, so an example that
    no longer validates hands them a 422 from the docs page itself.

    Nothing else exercises them: they are only ever passed to FastAPI as
    ``openapi_examples``, which renders them without validating. That matters because
    InferenceConfig sets ``extra="forbid"`` precisely so a stale field fails loudly
    rather than being dropped in silence.
    """
    InferenceConfig(**config_examples.INFERENCE_CONFIG_EXAMPLES[label]["value"])


def _inference_payload(**overrides):
    payload = {
        "model_name": "test-model",
        "quantization": "none",
        "model_total_memory": "1GB",
        "device_map": "cpu",
    }
    payload.update(overrides)
    return payload


def test_health_endpoint_reports_runtime_state(monkeypatch):
    config = InferenceConfig(
        model_name="loaded-model",
        quantization=QuantizationType.NONE,
        model_total_memory="2GB",
        device_map="cpu",
    )

    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "config", config, raising=False)
    monkeypatch.setattr(
        app_module.training_manager,
        "get_status",
        lambda: SimpleNamespace(is_training=False),
    )

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_loaded_config"]["model_name"] == "loaded-model"
    assert data["training_active"] is False


def _loaded_status(**overrides):
    """A get_status() payload shaped like the real one, defaults included."""
    status = {
        "loaded": True,
        "model_name": "loaded-model",
        "model_path": None,
        "engine": "transformers",
        "quantization": None,
        # n_ctx carries a non-null default even when the engine never uses it
        "n_ctx": 4096,
        "vllm_max_model_len": None,
        "llama_capabilities": None,
        # What the running engine reports; absent unless a test opts in
        "served_context_length": None,
        "loaded_at": 1700000000.0,
    }
    status.update(overrides)
    return status


def _get_model_card(monkeypatch, status, trained_length=None):
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "get_status", lambda: status)
    monkeypatch.setattr(
        app_module,
        "_resolve_model_trained_context_length",
        lambda _status: trained_length,
    )
    monkeypatch.setattr(app_module, "read_model_arch", lambda _path: None)

    response = app_module.list_models()
    assert response["object"] == "list"
    assert len(response["data"]) == 1
    return response["data"][0]


def test_list_models_is_empty_when_no_model_loaded(monkeypatch):
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_list_models_reports_openai_core_fields(monkeypatch):
    card = _get_model_card(
        monkeypatch,
        _loaded_status(model_path="/models/loaded-model"),
        trained_length=32768,
    )

    assert card["id"] == "loaded-model"
    assert card["object"] == "model"
    assert card["owned_by"] == "ADATA"
    assert card["root"] == "/models/loaded-model"
    assert card["parent"] is None
    assert card["state"] == "loaded"
    # `created` is the load time, so repeated polls return the same value
    assert card["created"] == 1700000000


def test_list_models_falls_back_to_default_id(monkeypatch):
    card = _get_model_card(monkeypatch, _loaded_status(model_name=None))

    assert card["id"] == "trusta-ast-default"
    assert card["root"] == "trusta-ast-default"


def test_list_models_uses_max_model_len_for_vllm(monkeypatch):
    # n_ctx defaults to 4096 but never reaches vLLM; --max-model-len wins
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="vllm", vllm_max_model_len=32768),
        trained_length=131072,
    )

    assert card["context_length"] == 32768
    assert card["max_model_len"] == 32768
    assert card["loaded_context_length"] == 32768
    assert card["max_context_length"] == 131072


def test_list_models_reports_the_window_vllm_sized_itself(monkeypatch):
    # --max-model-len was never passed, so the config has no number. vLLM's own model
    # card does, and the readiness probe already fetched it: report that, not nothing.
    card = _get_model_card(
        monkeypatch,
        _loaded_status(
            engine="vllm", n_ctx=8192, vllm_max_model_len=None, served_context_length=40960
        ),
        trained_length=131072,
    )

    assert card["context_length"] == 40960
    assert card["max_model_len"] == 40960
    assert card["max_context_length"] == 131072


def test_list_models_omits_the_window_when_no_engine_reports_one(monkeypatch):
    # Neither the engine nor the config can say: better absent than guessed
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="vllm", n_ctx=8192, vllm_max_model_len=None),
        trained_length=131072,
    )

    assert "context_length" not in card
    assert "max_model_len" not in card
    assert card["max_context_length"] == 131072


def test_list_models_prefers_the_engine_reported_window(monkeypatch):
    """/props beats the requested -c: extra_args, -c 0 and capping all diverge."""
    card = _get_model_card(
        monkeypatch,
        _loaded_status(
            engine="llama_server",
            n_ctx=4096,  # what we asked for
            llama_server_np=1,
            served_context_length=131072,  # what llama-server actually serves
        ),
        trained_length=262144,
    )

    assert card["context_length"] == 131072
    assert card["max_context_length"] == 262144


def test_list_models_falls_back_to_n_ctx_over_np_without_props(monkeypatch):
    """A server too old for /props still gets the old best-effort estimate."""
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="llama_server", n_ctx=32768, llama_server_np=4),
        trained_length=131072,
    )

    assert card["context_length"] == 8192


def test_list_models_fallback_never_exceeds_the_trained_window(monkeypatch):
    """llama.cpp caps the slot window at n_ctx_train, so the estimate must too."""
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="llama_server", n_ctx=524288, llama_server_np=1),
        trained_length=262144,
    )

    assert card["context_length"] == 262144


def test_list_models_ignores_n_ctx_for_transformers(monkeypatch):
    # transformers has no runtime window knob, so the default 4096 must not leak
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="transformers", n_ctx=4096),
        trained_length=131072,
    )

    assert card["context_length"] == 131072
    assert card["max_context_length"] == 131072


def test_list_models_reports_unknown_context_as_none(monkeypatch):
    card = _get_model_card(monkeypatch, _loaded_status(engine="transformers"))

    assert "context_length" not in card
    assert "max_context_length" not in card


def test_list_models_merges_engine_reported_capabilities(monkeypatch):
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="llama_server", llama_capabilities=["completion", "Vision"]),
    )

    assert card["capabilities"] == ["chat", "completion", "vision"]


def test_list_models_defaults_capabilities_to_chat(monkeypatch):
    card = _get_model_card(monkeypatch, _loaded_status())

    assert card["capabilities"] == ["chat"]


def test_list_models_serializes_enum_valued_fields(monkeypatch):
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine=InferenceEngine.VLLM, quantization=QuantizationType.INT4),
    )

    assert card["engine"] == "vllm"
    assert isinstance(card["engine"], str)
    assert card["quantization"] == QuantizationType.INT4.value


def test_list_models_prefers_llama_meta_over_the_registry(monkeypatch):
    # llama-server read n_ctx_train out of the GGUF it actually opened, so it beats
    # the registry value recorded at download time. Exercises the real resolver.
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(
        app_module.model_manager,
        "get_status",
        lambda: _loaded_status(
            engine="llama_server",
            model_name="loaded-model",
            n_ctx=128000,
            served_context_length=128000,
            llama_model_meta={"n_ctx_train": 262144},
        ),
    )
    monkeypatch.setattr(app_module, "read_model_arch", lambda _path: None)
    monkeypatch.setattr(app_module, "read_model_context_length", lambda _path: None)
    monkeypatch.setattr(
        app_module.model_registry,
        "list_models",
        # Stale registry value must lose to what llama-server actually read
        lambda: {"base_models": [{"model_name": "loaded-model", "max_context_length": 4096}]},
    )

    card = app_module.list_models()["data"][0]

    assert card["max_context_length"] == 262144
    assert card["loaded_context_length"] == 128000
    # llama.cpp's own key is preserved for clients written against llama-server
    assert card["meta"]["n_ctx_train"] == 262144


def test_list_models_prefers_allocated_n_ctx_over_the_request(monkeypatch):
    # llama-server clamps -c to what fits; /props reports what it really allocated
    card = _get_model_card(
        monkeypatch,
        _loaded_status(
            engine="llama_server",
            n_ctx=262144,
            served_context_length=65536,
            llama_model_meta={"n_ctx_train": 262144},
        ),
    )

    assert card["loaded_context_length"] == 65536
    assert card["context_length"] == 65536


def test_list_models_derives_vllm_multimodal_from_its_limits(monkeypatch):
    """vLLM publishes no capabilities; the mm limits are the same runtime gate."""
    card = _get_model_card(
        monkeypatch,
        _loaded_status(engine="vllm", vllm_mm_image_limit=4, vllm_mm_audio_limit=None),
    )

    assert card["capabilities"] == ["chat", "vision", "multimodal"]


def test_list_models_leaves_text_only_vllm_at_chat(monkeypatch):
    card = _get_model_card(monkeypatch, _loaded_status(engine="vllm", vllm_mm_image_limit=None))

    assert card["capabilities"] == ["chat"]


def test_list_models_reports_llama_model_details(monkeypatch):
    card = _get_model_card(
        monkeypatch,
        _loaded_status(
            engine="llama_server",
            llama_model_meta={"n_params": 34660610688, "size": 21155768832},
        ),
    )

    assert card["parameter_count"] == 34660610688
    assert card["size_bytes"] == 21155768832


def test_list_models_omits_model_details_without_meta(monkeypatch):
    card = _get_model_card(monkeypatch, _loaded_status(engine="vllm"))

    assert "parameter_count" not in card
    assert "size_bytes" not in card


def test_retrieve_model_returns_the_card_for_an_alias(monkeypatch):
    status = _loaded_status(model_name="loaded-model", model_path="/models/loaded")
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "get_status", lambda: status)
    monkeypatch.setattr(app_module, "_resolve_model_trained_context_length", lambda _s: None)
    monkeypatch.setattr(app_module, "read_model_arch", lambda _p: None)

    for alias in ("loaded-model", "/models/loaded", "trusta-ast-default"):
        card = app_module.retrieve_model(alias)
        assert card["id"] == "loaded-model", alias
        assert card["object"] == "model"


def test_retrieve_model_404s_for_an_unknown_id(monkeypatch):
    status = _loaded_status(model_name="loaded-model")
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "get_status", lambda: status)

    response = app_module.retrieve_model("gpt-4o")

    assert response.status_code == 404
    body = json.loads(response.body)
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "invalid_request_error"


def test_retrieve_model_404s_when_nothing_is_loaded(monkeypatch):
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)

    response = client.get("/v1/models/anything")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_retrieve_model_accepts_a_slashed_model_id(monkeypatch):
    # Model ids routinely contain a slash, so the route must not split on it
    status = _loaded_status(model_name="HauhauCS/Qwen3.6-35B", model_path=None)
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "get_status", lambda: status)
    monkeypatch.setattr(app_module, "_resolve_model_trained_context_length", lambda _s: None)
    monkeypatch.setattr(app_module, "read_model_arch", lambda _p: None)

    response = client.get("/v1/models/HauhauCS/Qwen3.6-35B")

    assert response.status_code == 200
    assert response.json()["id"] == "HauhauCS/Qwen3.6-35B"


def _capture_warnings(monkeypatch):
    """Collect logger.warning calls made by app.py."""
    warnings = []
    monkeypatch.setattr(
        app_module.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )
    return warnings


def _set_loaded_config(monkeypatch, **overrides):
    config = InferenceConfig(**{**_inference_payload(), **overrides})
    monkeypatch.setattr(app_module.model_manager, "config", config, raising=False)
    return config


def test_response_model_name_reports_the_loaded_model_on_mismatch(monkeypatch):
    _set_loaded_config(monkeypatch, model_name="loaded-model")
    warnings = _capture_warnings(monkeypatch)

    # The request is still served — the mismatch only warns
    assert app_module._resolve_response_model_name("gpt-4o") == "loaded-model"
    assert len(warnings) == 1
    assert "gpt-4o" in warnings[0]
    assert "loaded-model" in warnings[0]


def test_response_model_name_accepts_the_advertised_id(monkeypatch):
    _set_loaded_config(monkeypatch, model_name="loaded-model")
    warnings = _capture_warnings(monkeypatch)

    assert app_module._resolve_response_model_name("loaded-model") == "loaded-model"
    assert warnings == []


def test_response_model_name_accepts_the_model_path_alias(monkeypatch):
    # Callers holding the path (what the endpoint used to echo) must keep working
    _set_loaded_config(monkeypatch, model_name="loaded-model", model_path="/models/loaded")
    warnings = _capture_warnings(monkeypatch)

    assert app_module._resolve_response_model_name("/models/loaded") == "loaded-model"
    assert app_module._resolve_response_model_name("trusta-ast-default") == "loaded-model"
    assert warnings == []


def test_response_model_name_without_a_requested_model(monkeypatch):
    _set_loaded_config(monkeypatch, model_name="loaded-model", model_path="/models/loaded")
    warnings = _capture_warnings(monkeypatch)

    assert app_module._resolve_response_model_name(None) == "loaded-model"
    assert warnings == []


def test_advertised_id_matches_the_chat_response_model(monkeypatch):
    """/v1/models and /v1/chat/completions must not disagree about the model id."""
    _set_loaded_config(monkeypatch, model_name="loaded-model", model_path="/models/loaded")
    _capture_warnings(monkeypatch)

    card = _get_model_card(
        monkeypatch,
        _loaded_status(model_name="loaded-model", model_path="/models/loaded"),
    )

    assert card["id"] == app_module._resolve_response_model_name(None)
    # root still exposes the path, so callers can resolve the model on disk
    assert card["root"] == "/models/loaded"


def test_served_identity_falls_back_to_the_path_then_the_default():
    assert app_module._served_model_identity(None, "/models/loaded")[0] == "/models/loaded"
    assert app_module._served_model_identity(None, None)[0] == "trusta-ast-default"
    assert app_module._served_model_identity("  ", None)[0] == "trusta-ast-default"


def test_trained_context_length_reads_the_registry(monkeypatch):
    monkeypatch.setattr(
        app_module.model_registry,
        "list_models",
        lambda: {
            "base_models": [{"model_path": "/models/other", "max_context_length": 4096}],
            "finetuned_models": [
                {"model_path": "/models/loaded-model", "max_context_length": 8192}
            ],
        },
    )

    resolved = app_module._resolve_model_trained_context_length(
        {"model_path": "/models/loaded-model", "model_name": "loaded-model"}
    )

    assert resolved == 8192


def test_trained_context_length_survives_registry_failure(monkeypatch):
    def _boom():
        raise OSError("registry unreadable")

    monkeypatch.setattr(app_module.model_registry, "list_models", _boom)

    assert app_module._resolve_model_trained_context_length({"model_name": "loaded-model"}) is None


def test_load_model_starts_background_loading(monkeypatch):
    captured = {}

    def _fake_start_loading(config):
        captured["config"] = config
        return True

    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: False)
    monkeypatch.setattr(
        app_module.model_manager,
        "get_status",
        lambda: {"loaded": False, "is_loading": False},
    )
    monkeypatch.setattr(app_module.model_manager, "start_loading", _fake_start_loading)

    response = client.post("/inference/load_model", json=_inference_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loading"
    assert captured["config"].model_name == "test-model"
    assert captured["config"].device_map == "cpu"


def test_load_model_returns_already_loaded_for_same_config(monkeypatch):
    existing = InferenceConfig(**_inference_payload())

    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "config", existing, raising=False)

    response = client.post("/inference/load_model", json=_inference_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "already_loaded"
    assert data["config"]["model_name"] == "test-model"


def test_inference_status_serializes_current_manager_state(monkeypatch):
    monkeypatch.setattr(
        app_module.model_manager,
        "get_status",
        lambda: {
            "loaded": False,
            "is_loading": True,
            "loading_error": None,
            "model_name": "pending-model",
            "model_path": None,
            "engine": "transformers",
            "quantization": "none",
            "model_total_memory": "1GB",
            "device_map": "cpu",
            "max_memory": None,
            "offload_folder": None,
            "device": "cpu",
            "memory_usage": None,
            "device_allocation": None,
            "n_gpu_layers": None,
            "n_ctx": 4096,
            "n_batch": 512,
            "prefill_strategy": None,
            "llama_capabilities": None,
            "slot_restore_summary": None,
            "vllm_gpu_memory_utilization": None,
            "vllm_max_model_len": None,
            "vllm_dtype": None,
            "vllm_quantization": None,
            "vllm_enforce_eager": None,
            "vllm_enable_prefix_caching": None,
            "vllm_kv_cache_dtype": None,
            "vllm_cpu_offload_gb": None,
            "vllm_tensor_parallel_size": None,
            "vllm_max_num_seqs": None,
            "vllm_mm_image_limit": None,
            "vllm_mm_audio_limit": None,
            "vllm_mm_video_limit": None,
            "vllm_hf_overrides": None,
            "vllm_chat_template": None,
        },
    )

    response = client.get("/inference/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_loading"] is True
    assert data["model_name"] == "pending-model"
    assert data["device_map"] == "cpu"


def test_unload_model_delegates_to_manager(monkeypatch):
    monkeypatch.setattr(
        app_module.model_manager,
        "unload_model",
        lambda: {"status": "success", "message": "Model unloaded successfully"},
    )

    response = client.post("/inference/unload_model")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def _llama_manager(monkeypatch, cfg):
    """Drive the real ModelManager.get_status over a stubbed worker process."""
    monkeypatch.setattr(
        app_module.model_manager.inference_process,
        "get_status",
        lambda: {"loaded": True, "is_loading": False},
        raising=False,
    )
    monkeypatch.setattr(app_module.model_manager, "is_loaded", lambda: True)
    monkeypatch.setattr(app_module.model_manager, "config", cfg, raising=False)


def test_list_models_advertises_the_window_one_request_may_use(monkeypatch):
    # -c is divided between -np slots, so a request sees n_ctx/np. Advertising
    # n_ctx itself would over-promise by that factor and put the client back
    # where it started: sizing history against a number it cannot fill.
    _llama_manager(
        monkeypatch,
        InferenceConfig(model_name="m", engine="llama_server", n_ctx=32768, llama_server_np=2),
    )

    entry = client.get("/v1/models").json()["data"][0]

    assert entry["context_length"] == 16384


def test_list_models_omits_a_window_the_engine_does_not_define(monkeypatch):
    # Better absent than guessed: a client that reads a wrong number cannot
    # tell it is wrong, while one that reads nothing falls back knowingly.
    _llama_manager(monkeypatch, InferenceConfig(model_name="m", engine="transformers"))

    entry = client.get("/v1/models").json()["data"][0]

    assert "context_length" not in entry


def test_status_names_the_binary_this_load_runs(monkeypatch):
    # A co-resident tool has to find this process, and the name is not knowable
    # from outside: the installer ships `llama`, a source build `llama-server`.
    _llama_manager(
        monkeypatch,
        InferenceConfig(
            model_name="m", engine="llama_server", llama_server_binary="/opt/llama/llama"
        ),
    )

    assert app_module.model_manager.get_status()["llama_server_binary"] == "/opt/llama/llama"


def test_status_falls_back_to_the_resolved_default_binary(monkeypatch):
    _llama_manager(monkeypatch, InferenceConfig(model_name="m", engine="llama_server"))

    assert (
        app_module.model_manager.get_status()["llama_server_binary"] == settings.LLAMA_SERVER_BINARY
    )


def test_status_names_no_binary_for_an_in_process_engine(monkeypatch):
    # transformers and vLLM run inside this service; trimming or signalling
    # "the LLM process" there would hit the service itself.
    _llama_manager(monkeypatch, InferenceConfig(model_name="m", engine="transformers"))

    assert app_module.model_manager.get_status()["llama_server_binary"] is None
