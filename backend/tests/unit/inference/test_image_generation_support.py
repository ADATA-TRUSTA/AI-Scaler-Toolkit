import os
import sys
import base64
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

# Add project root to sys.path so we can import service modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from service.inference.generate.generator_core import _load_images, tokenize_prompt
from service.model_manager import ModelManager


import pytest

pytestmark = pytest.mark.inference


def _make_data_url() -> str:
    """Create a tiny in-memory PNG data URL for tests."""
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    bio = BytesIO()
    img.save(bio, format="PNG")
    encoded = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class _DummyTokenizer:
    model_max_length = 1024

    def apply_chat_template(self, *args, **kwargs):
        return "formatted by tokenizer"


class _DummyProcessorInputs(dict):
    def to(self, _device):
        return self


class _DummyProcessor:
    def __init__(self):
        self.called_with = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "formatted by processor"

    def __call__(self, text=None, images=None, return_tensors=None, truncation=None, max_length=None):
        self.called_with = {
            "text": text,
            "images": images,
            "return_tensors": return_tensors,
            "truncation": truncation,
            "max_length": max_length,
        }
        return _DummyProcessorInputs({"input_ids": [1, 2, 3], "pixel_values": [0.1]})


class TestImageGenerationSupport(unittest.TestCase):
    def test_load_images_from_data_url_success(self):
        data_url = _make_data_url()
        images = _load_images([data_url])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].size, (2, 2))

    def test_tokenize_prompt_with_images_requires_processor(self):
        data_url = _make_data_url()
        with self.assertRaises(ValueError) as ctx:
            tokenize_prompt(
                prompt="hello",
                tokenizer=_DummyTokenizer(),
                model_device="cpu",
                params={"images": [data_url]},
                processor=None,
            )
        self.assertIn("does not expose an image processor", str(ctx.exception))

    def test_tokenize_prompt_with_images_uses_processor(self):
        data_url = _make_data_url()
        processor = _DummyProcessor()
        tokenizer = _DummyTokenizer()

        outputs = tokenize_prompt(
            prompt=[{"role": "user", "content": "describe image"}],
            tokenizer=tokenizer,
            model_device="cpu",
            params={"images": [data_url]},
            processor=processor,
        )

        self.assertIn("input_ids", outputs)
        self.assertIn("pixel_values", outputs)
        self.assertIsNotNone(processor.called_with)
        self.assertEqual(processor.called_with["text"], "formatted by processor")
        self.assertEqual(len(processor.called_with["images"]), 1)

    def test_model_manager_generate_forwards_images_param(self):
        manager = ModelManager()

        with patch.object(manager, "is_loaded", return_value=True), \
             patch.object(manager.inference_process, "generate", return_value={"result": "ok"}) as mock_gen:
            result = manager.generate(
                prompt="hello",
                images=["/tmp/fake.jpg"],
                max_new_tokens=16,
            )

        self.assertEqual(result["result"], "ok")
        args, _kwargs = mock_gen.call_args
        self.assertEqual(args[0], "hello")
        self.assertEqual(args[1]["images"], ["/tmp/fake.jpg"])


if __name__ == "__main__":
    unittest.main()
