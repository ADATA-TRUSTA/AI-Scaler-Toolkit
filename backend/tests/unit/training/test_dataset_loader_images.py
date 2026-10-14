"""Image handling in the training dataset loader: path resolution, lazy decode,
and multimodal format validation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.training.core.dataset_loader import (  # noqa: E402
    _resolve_image_paths,
    load_training_dataset,
)

pytestmark = pytest.mark.finetune


def _png(path: Path):
    """Write a tiny valid PNG so datasets.Image can decode it if asked."""
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 30, 30)).save(path)
    return path


def _dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


# --- path resolution --------------------------------------------------------


def _resolved_paths(out, field):
    """Read the stored image paths without triggering pixel decode."""
    from datasets import Image as HFImage

    feat = out.features[field]
    is_list = hasattr(feat, "feature")  # datasets.List exposes .feature
    raw = out.cast_column(field, [HFImage(decode=False)] if is_list else HFImage(decode=False))
    value = raw[0][field]
    items = value if isinstance(value, list) else [value]
    return [it["path"] for it in items]


def test_relative_paths_resolved_against_dataset_dir(tmp_path):
    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _dataset([{"images": ["img/a.png"], "text": "x"}])

    out = _resolve_image_paths(ds, "images", str(tmp_path / "train.jsonl"))
    assert _resolved_paths(out, "images") == [str(tmp_path / "img" / "a.png")]


def test_absolute_paths_kept_as_is(tmp_path):
    abs_img = _png(tmp_path / "elsewhere.png")
    ds = _dataset([{"images": [str(abs_img)], "text": "x"}])

    # dataset file lives in a different dir; absolute paths must not be re-based.
    out = _resolve_image_paths(ds, "images", str(tmp_path / "sub" / "train.jsonl"))
    assert _resolved_paths(out, "images") == [str(abs_img)]


def test_missing_image_raises_with_context(tmp_path):
    ds = _dataset([{"images": ["img/missing.png"], "text": "x"}])
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_image_paths(ds, "images", str(tmp_path / "train.jsonl"))


def test_single_image_field_not_a_list(tmp_path):
    """A scalar 'image' (not a list) is resolved and kept scalar."""
    _png(tmp_path / "one.png")
    ds = _dataset([{"image": "one.png", "text": "x"}])

    out = _resolve_image_paths(ds, "image", str(tmp_path / "train.jsonl"))
    # Still a single image feature, not wrapped in a list.
    from datasets import Image as HFImage

    assert isinstance(out.features["image"], HFImage)


def test_already_decoded_images_left_untouched(tmp_path):
    """Non-string payloads (e.g. PIL images) are passed through unchanged."""
    from PIL import Image as PILImage

    ds = _dataset([{"images": [PILImage.new("RGB", (4, 4))], "text": "x"}])
    out = _resolve_image_paths(ds, "images", str(tmp_path / "train.jsonl"))
    assert out is ds  # returned unchanged


# --- lazy decode ------------------------------------------------------------


def test_column_cast_to_lazily_decoded_image_feature(tmp_path):
    from datasets import Image as HFImage

    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _dataset([{"images": ["img/a.png"], "text": "x"}])

    out = _resolve_image_paths(ds, "images", str(tmp_path / "train.jsonl"))
    # The column is a List(Image) feature that decodes on access rather than
    # materialising all pixels up front.
    feat = out.features["images"]
    inner = feat.feature if hasattr(feat, "feature") else feat
    assert isinstance(inner, HFImage)
    assert inner.decode is True


def test_lazy_decode_yields_pil_on_access(tmp_path):
    from PIL.Image import Image as PILImage

    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _dataset([{"images": ["img/a.png"], "text": "x"}])

    out = _resolve_image_paths(ds, "images", str(tmp_path / "train.jsonl"))
    decoded = out[0]["images"][0]  # access triggers decode
    assert isinstance(decoded, PILImage)


# --- multimodal format validation (end to end via load_training_dataset) ----


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_images_with_prompt_completion_accepted(tmp_path):
    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"images": ["img/a.png"], "prompt": "what?", "completion": "red"}],
    )
    loaded = load_training_dataset(str(ds))
    assert "images" in loaded.column_names


def test_images_with_messages_accepted(tmp_path):
    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"images": ["img/a.png"], "messages": [{"role": "user", "content": "hi"}]}],
    )
    loaded = load_training_dataset(str(ds))
    assert "images" in loaded.column_names


def test_images_without_prompt_or_messages_rejected(tmp_path):
    (tmp_path / "img").mkdir()
    _png(tmp_path / "img" / "a.png")
    ds = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"images": ["img/a.png"], "text": "just text"}],
    )
    with pytest.raises(ValueError, match="neither 'messages' nor both 'prompt'"):
        load_training_dataset(str(ds))


def test_image_singular_field_also_detected(tmp_path):
    """The loader accepts 'image' as well as 'images'."""
    _png(tmp_path / "a.png")
    ds = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"image": "a.png", "prompt": "q", "completion": "a"}],
    )
    loaded = load_training_dataset(str(ds))
    assert "image" in loaded.column_names
