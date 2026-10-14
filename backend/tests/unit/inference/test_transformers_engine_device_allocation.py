from types import SimpleNamespace

import torch

from service.inference.engines.transformers_engine import TransformersEngine


import pytest

pytestmark = pytest.mark.inference


class _LayersWithLen:
    def __len__(self):
        return 28


class _ModelWithConfig:
    config = SimpleNamespace(num_hidden_layers=32)


class _ModelWithNestedLayers:
    language_model = SimpleNamespace(layers=_LayersWithLen())


def test_get_total_layer_count_prefers_config_value():
    assert TransformersEngine._get_total_layer_count(_ModelWithConfig()) == 32


def test_get_total_layer_count_falls_back_to_common_layer_container():
    assert TransformersEngine._get_total_layer_count(_ModelWithNestedLayers()) == 28


def test_build_single_device_allocation_for_full_gpu_model():
    summary, total_modules, layer_lines = TransformersEngine._build_single_device_allocation("GPU0", 24)

    assert summary == "GPU0: 24/24 layers (100%)"
    assert total_modules == 24
    assert layer_lines == ["Layers 0-23 -> GPU0"]


def test_build_single_device_allocation_without_layer_count_uses_full_model_line():
    summary, total_modules, layer_lines = TransformersEngine._build_single_device_allocation("DRAM", None)

    assert summary == "DRAM: full model"
    assert total_modules is None
    assert layer_lines == ["full model -> DRAM"]


def test_normalize_device_label_maps_cpu_and_cuda():
    assert TransformersEngine._normalize_device_label("cpu") == "DRAM"
    assert TransformersEngine._normalize_device_label("cuda:1") == "GPU1"
    assert TransformersEngine._normalize_device_label(torch.device("cuda", 0)) == "GPU0"
