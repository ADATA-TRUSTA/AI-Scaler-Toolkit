"""
The one setup sequence both training entry points run.

There are two entry points -- ``training_process.py`` for ``num_gpus == 1``
(in-process) and ``training_script.py`` for ``num_gpus > 1`` (a ``deepspeed``
subprocess) -- and they used to hand-duplicate this sequence. The copy drifted:
``training_script.py`` was forked in 2026-06, multimodal support landed in
2026-07 on the original only, and multi-GPU image training was dead from that
day on (loud, via the guard in ``sft_runner``) along with the eval split
(silent, no guard at all). Nothing about data parallelism caused that; two
copies of one sequence did.

Keeping the sequence here means a new step is added once. Two invariants it
exists to protect:

1. ``strategy.get_training_args()`` MUST run before ``ModelLoader`` touches the
   model. Building ``TrainingArguments(deepspeed=...)`` registers the global
   DeepSpeed config, which is what puts ``from_pretrained`` under an active
   ``zero.Init``. Loading first silently gives up ZeRO-3 sharding at load time.
2. ``enable_image_training()`` MUST run before ``load_model()``. It switches the
   load class to ``AutoModelForImageTextToText``; without it a multimodal
   checkpoint loads through ``AutoModelForCausalLM`` and has no vision tower at
   all.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .dataset_loader import load_training_dataset, split_train_eval
from .model_loader import ModelLoader, select_processing_class
from .strategies import StrategyFactory
from .training_events import Phase

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
        TrainingArguments,
    )

    from ...config_models import TrainingConfig
    from .strategies import TrainingStrategy

logger = logging.getLogger(__name__)


@dataclass
class TrainingComponents:
    """Everything a caller needs to build a trainer and report on the run."""

    strategy: "TrainingStrategy"
    training_args: "TrainingArguments"
    model: "PreTrainedModel | PeftModel"
    processing_class: "PreTrainedTokenizerBase | ProcessorMixin"
    dataset: "Dataset"
    eval_dataset: "Dataset | None"
    is_image_training: bool
    # Both are surfaced separately from processing_class (which is one OR the
    # other) so the caller's end-of-job cleanup can drop every reference it
    # holds. Returning only processing_class leaked the other one.
    tokenizer: "PreTrainedTokenizerBase"
    processor: "ProcessorMixin | None"


def build_training_components(
    training_config: "TrainingConfig",
    hf_token: str | None,
    ds_config_path: str | None,
    *,
    on_stage: Callable[[str], Any] | None = None,
    # Return type is Any, not None: the callers' natural handlers already return
    # something (JobLogWriter.info hands back the event it emitted), and a
    # None-returning signature would force every caller to wrap them in a lambda.
    on_note: Callable[[str, dict[str, Any]], Any] | None = None,
) -> TrainingComponents:
    """
    Run the shared setup sequence and return the pieces a trainer needs.

    ``on_stage`` receives each ``Phase`` as it starts and ``on_note`` receives
    (message, payload) for things worth putting in the job log; both are
    optional so either entry point can wire in its own reporting without this
    module knowing about job logs, queues or Redis.
    """

    def stage(phase: str) -> None:
        if on_stage is not None:
            on_stage(phase)

    def note(message: str, payload: dict[str, Any]) -> None:
        if on_note is not None:
            on_note(message, payload)

    # 1. Strategy, then TrainingArguments -- see invariant 1 in the module docstring.
    strategy = StrategyFactory.get_strategy(training_config, ds_config_path)
    training_args = strategy.get_training_args()

    # 2. Tokenizer + processor. The processor is only used when the data has images.
    stage(Phase.LOADING_TOKENIZER)
    model_loader = ModelLoader(training_config, hf_token)
    tokenizer = model_loader.load_tokenizer()
    processor = model_loader.load_processor()

    # 3. Dataset, and the held-out split before anything else touches the data,
    #    so evaluation sees examples training never trains on.
    stage(Phase.LOADING_DATASET)
    dataset = load_training_dataset(training_config.dataset_path)

    eval_dataset = None
    eval_ratio = getattr(training_config, "eval_split_ratio", None)
    if eval_ratio:
        dataset, eval_dataset = split_train_eval(
            dataset, eval_ratio, getattr(training_config, "eval_split_seed", 42)
        )
        if eval_dataset is not None:
            note(
                "held-out test split created",
                {"train_size": len(dataset), "eval_size": len(eval_dataset)},
            )

    # 4. What the trainer processes examples with: the processor for image
    #    datasets, the tokenizer otherwise -- and switch the load class over
    #    before load_model(). See invariant 2.
    processing_class = select_processing_class(dataset, tokenizer, processor, training_config)
    is_image_training = processing_class is processor
    if is_image_training:
        model_loader.enable_image_training()

    # 5. Preprocess with the TOKENIZER, not the processor: only the Causal LM
    #    path preprocesses, and TRL collates images later. The eval split goes
    #    through the same preprocessing to stay compatible.
    dataset = strategy.preprocess_dataset(dataset, tokenizer)
    if eval_dataset is not None:
        eval_dataset = strategy.preprocess_dataset(eval_dataset, tokenizer)

    # 6. Model.
    stage(Phase.LOADING_MODEL)
    model = model_loader.load_model()

    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        logger.info(f"[MEMPROBE] is_deepspeed_zero3_enabled={is_deepspeed_zero3_enabled()}")
    except Exception as e:
        logger.info(f"[MEMPROBE] could not query is_deepspeed_zero3_enabled: {e}")

    return TrainingComponents(
        strategy=strategy,
        training_args=training_args,
        model=model,
        processing_class=processing_class,
        dataset=dataset,
        eval_dataset=eval_dataset,
        is_image_training=is_image_training,
        tokenizer=tokenizer,
        processor=processor,
    )
