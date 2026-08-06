"""
Dataset loading and train/eval splitting for SFT training.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from datasets import load_dataset as hf_load_dataset

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


def split_train_eval(
    dataset: "Dataset", eval_ratio: float, seed: int = 42
) -> "tuple[Dataset, Dataset | None]":
    """
    Split a loaded dataset into (train, eval) held-out subsets.

    Returns ``(train, None)`` when the ratio is falsy or the dataset is too
    small to yield at least one eval example while keeping training data.
    The split preserves the dataset's features (image columns stay lazily
    decoded), so both subsets feed the same collator unchanged.
    """
    if not eval_ratio or eval_ratio <= 0:
        return dataset, None

    n = len(dataset)
    n_eval = int(round(n * eval_ratio))
    # Need at least one example on each side to evaluate meaningfully.
    if n_eval < 1 or n_eval >= n:
        logger.warning(
            f"[DatasetLoader] eval_split_ratio={eval_ratio} on {n} examples yields "
            f"{n_eval} eval rows; too few to split, skipping evaluation"
        )
        return dataset, None

    split = dataset.train_test_split(test_size=n_eval, seed=seed, shuffle=True)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(
        f"[DatasetLoader] Split dataset into train={len(train_ds)} / eval={len(eval_ds)} "
        f"(ratio={eval_ratio}, seed={seed})"
    )
    return train_ds, eval_ds


def _resolve_image_paths(dataset: "Dataset", image_field: str, dataset_path: str) -> "Dataset":
    """
    Resolve image references and cast the column so images decode lazily.

    Relative paths are resolved against the dataset file's directory, so a
    dataset stays portable alongside its images. Casting to `datasets.Image`
    means pixels are only read when a row is actually pulled into a batch --
    materialising every image up front would dwarf the model in RAM.
    """
    from datasets import Image as HFImage

    base_dir = Path(dataset_path).parent

    first = dataset[0][image_field]
    # A single example may carry one image or a list of them.
    sample = first[0] if isinstance(first, list) and first else first
    if not isinstance(sample, str):
        # Already decoded images (or an unrecognised payload): leave untouched.
        return dataset

    is_list = isinstance(first, list)

    def _resolve(example: dict[str, Any]) -> dict[str, Any]:
        value = example[image_field]
        items = value if isinstance(value, list) else [value]
        resolved = []
        for item in items:
            if not isinstance(item, str):
                resolved.append(item)
                continue
            path = Path(item)
            if not path.is_absolute():
                path = base_dir / path
            if not path.exists():
                raise FileNotFoundError(
                    f"Image referenced by dataset not found: {path} "
                    f"(from '{item}', resolved against {base_dir})"
                )
            resolved.append(str(path))
        example[image_field] = resolved if is_list else resolved[0]
        return example

    dataset = dataset.map(_resolve)
    feature = HFImage()
    dataset = dataset.cast_column(image_field, [feature] if is_list else feature)
    logger.info(
        f"[DatasetLoader] Resolved '{image_field}' paths against {base_dir} "
        "and cast to lazily-decoded images"
    )
    return dataset


def load_training_dataset(dataset_path: str) -> "Dataset":
    """
    Load dataset from local file or HuggingFace hub.

    Supports multiple formats:
    1. JSON/JSONL files with 'text' field
    2. JSON/JSONL files with 'prompt' and 'completion' fields
    3. HuggingFace dataset names

    For prompt-completion format, the function will automatically combine
    them into a single text field for training.
    """
    if dataset_path.endswith((".json", ".jsonl")):
        try:
            # Load the dataset, pointing at the correct field
            logger.info(f"[DatasetLoader] Loading dataset from {dataset_path}")

            # Use load_dataset to load the JSON/JSONL file
            dataset = hf_load_dataset("json", data_files={"train": dataset_path}, split="train")

            logger.info(f"[DatasetLoader] Loaded {len(dataset)} examples")

            # Inspect the dataset format and dispatch on the detected schema
            if len(dataset) > 0:
                first_example = dataset[0]
                logger.debug(f"[DatasetLoader] First example keys: {list(first_example.keys())}")

                # Mode 4: multimodal (images/image field alongside messages)
                # TRL selects DataCollatorForVisionLanguageModeling based on this field,
                # and the processor turns images into pixel_values on the fly at collate time.
                image_field = next((f for f in ("images", "image") if f in first_example), None)
                if image_field is not None:
                    has_messages = "messages" in first_example
                    has_prompt_completion = (
                        "prompt" in first_example and "completion" in first_example
                    )
                    if not (has_messages or has_prompt_completion):
                        raise ValueError(
                            f"Dataset has an '{image_field}' field but neither 'messages' "
                            "nor both 'prompt' and 'completion'. Multimodal training needs "
                            "one of those so the image can be placed in the conversation."
                        )
                    dataset = _resolve_image_paths(dataset, image_field, dataset_path)
                    companion = "prompt+completion" if has_prompt_completion else "messages"
                    logger.info(
                        f"[DatasetLoader] Detected multimodal format "
                        f"(field='{image_field}' + {companion})"
                    )
                    if has_messages and not has_prompt_completion:
                        # TRL's VLM collator only masks the prompt in the
                        # prompt/completion path; with 'messages' the loss also
                        # covers the image placeholder tokens, which dominate
                        # the sequence and wash out the training signal.
                        logger.warning(
                            "[DatasetLoader] Multimodal dataset uses 'messages'; loss will "
                            "cover the whole sequence including image tokens. Prefer "
                            "'prompt'/'completion' fields for image training."
                        )

                # Mode 3: OpenAI chat format (messages field)
                # TRL SFTTrainer >= 0.12 applies tokenizer.apply_chat_template automatically
                elif "messages" in first_example:
                    messages = first_example["messages"]
                    if not isinstance(messages, list) or not all(
                        isinstance(m, dict) and "role" in m and "content" in m for m in messages
                    ):
                        raise ValueError(
                            "Dataset 'messages' field must be a list of {role, content} dicts. "
                            f"Got: {messages[:1]}"
                        )
                    logger.info(
                        "[DatasetLoader] Detected OpenAI chat format (messages field); "
                        "passing through for TRL to apply chat template"
                    )

                # Mode 2: prompt + completion field pair
                elif "prompt" in first_example and "completion" in first_example:
                    logger.info("[DatasetLoader] Detected prompt-completion format")

                # Mode 1: single text field
                elif "text" not in first_example:
                    raise ValueError(
                        f"Dataset must have a 'messages' field (OpenAI chat format), "
                        f"both 'prompt' and 'completion' fields, or a 'text' field. "
                        f"Found fields: {list(first_example.keys())}"
                    )
            else:
                raise ValueError("Dataset is empty")

            return dataset

        except Exception as e:
            logger.exception(f"[DatasetLoader] Failed to load dataset from {dataset_path}: {e}")
            # Provide a more detailed error message

            # Try to read and validate the JSON format
            try:
                dataset_file = Path(dataset_path)
                if not dataset_file.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

                logger.info(f"[DatasetLoader] Validating JSON format in {dataset_path}...")
                with open(dataset_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    logger.info(f"[DatasetLoader] File has {len(lines)} lines")

                    # Check whether each line is valid JSON
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:  # skip blank lines
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as json_err:
                            raise ValueError(
                                f"Invalid JSON at line {i}: {line[:100]}... Error: {json_err}"
                            ) from json_err

                logger.info("[DatasetLoader] JSON format validation passed")

            except Exception as validate_err:
                logger.exception(f"[DatasetLoader] Dataset validation error: {validate_err}")
                raise ValueError(f"Dataset validation failed: {validate_err}") from e

            # If validation passed but loading still failed, re-raise the original error
            raise

    # HuggingFace dataset
    loaded = hf_load_dataset(dataset_path)
    # load_dataset may return a Dataset directly (when split= is given upstream)
    # or a DatasetDict keyed by split name. Don't assume a "train" split exists.
    if hasattr(loaded, "keys"):
        available_splits = list(loaded.keys())
        if not available_splits:
            raise ValueError(f"Dataset '{dataset_path}' contains no splits")
        split_name = "train" if "train" in loaded else available_splits[0]
        if split_name != "train":
            logger.warning(
                f"[DatasetLoader] Dataset '{dataset_path}' has no 'train' split; "
                f"using '{split_name}' (available: {available_splits})"
            )
        dataset = loaded[split_name]
    else:
        dataset = loaded

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{dataset_path}' is empty")
    # A concrete split was selected above, so this is a Dataset, not a DatasetDict.
    return cast("Dataset", dataset)
