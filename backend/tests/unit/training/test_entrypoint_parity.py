"""
The single-GPU and multi-GPU entry points must run the SAME setup sequence.

History: `training_script.py` (num_gpus > 1) was forked from
`training_process.py` (num_gpus == 1) in 2026-06. Multimodal support landed in
2026-07 on the original only, so multi-GPU image training was dead -- it raised
"no multimodal processor was loaded" -- and the eval split silently never
happened there. Data parallelism had nothing to do with it; two hand-maintained
copies of one sequence did.

Both now call `build_training_components`.

Two kinds of assertion here, deliberately:

* Ordering and wiring inside the shared builder are checked BEHAVIOURALLY, by
  substituting recording stubs and inspecting the call order. No model is
  loaded, so it is fast, and it survives any reformatting.
* "Neither entry point re-implements the sequence" is inherently a statement
  about the source, so it is checked over the parsed AST rather than by
  substring matching -- immune to whitespace, line breaks and renamed locals,
  unlike the string comparison this replaced.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.finetune


# --------------------------------------------------------------- AST helpers


def _called_names(module) -> set[str]:
    """Every function/method name invoked anywhere in a module's source."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _entry_modules():
    from service.training import training_process, training_script

    return (training_process, training_script)


# --------------------------------------------------------------- parity


def test_both_entry_points_call_the_shared_builder():
    """The whole point of the refactor: neither path may re-implement the sequence."""
    for module in _entry_modules():
        assert "build_training_components" in _called_names(module), (
            f"{module.__name__} must use the shared setup sequence"
        )


@pytest.mark.parametrize(
    "inlined_step",
    ["get_strategy", "get_training_args", "load_tokenizer", "load_processor", "load_model",
     "select_processing_class", "enable_image_training", "split_train_eval"],
)
def test_neither_entry_point_reimplements_a_setup_step(inlined_step):
    """Guard against a well-meaning inline 'just this one step' creeping back in."""
    for module in _entry_modules():
        assert inlined_step not in _called_names(module), (
            f"{module.__name__} calls '{inlined_step}' directly; it belongs in core/pipeline.py"
        )


# --------------------------------------------------------------- behavioural: ordering


class _StubDataset:
    """A dataset stand-in. Needs a real __len__ -- dunders resolve on the type."""

    def __init__(self, name: str, size: int):
        self.name = name
        self._size = size

    def __len__(self) -> int:
        return self._size


class _RecordingLoader:
    """Stands in for ModelLoader and records the order of its calls."""

    def __init__(self, log: list[str]):
        self._log = log
        log.append("ModelLoader.__init__")

    def load_tokenizer(self):
        self._log.append("load_tokenizer")
        return SimpleNamespace(name="tokenizer")

    def load_processor(self):
        self._log.append("load_processor")
        return SimpleNamespace(name="processor")

    def enable_image_training(self):
        self._log.append("enable_image_training")

    def load_model(self):
        self._log.append("load_model")
        return SimpleNamespace(name="model")


@pytest.fixture
def recorded(monkeypatch):
    """
    Drive build_training_components with stubs and return the call log.

    Everything the builder reaches out to is substituted, so this exercises the
    real ordering logic without loading a checkpoint.
    """
    from service.training.core import pipeline

    log: list[str] = []
    train_ds = _StubDataset("train", 12)
    eval_ds = _StubDataset("eval", 3)

    class _Strategy:
        def get_training_args(self):
            log.append("get_training_args")
            return SimpleNamespace(name="training_args")

        def preprocess_dataset(self, dataset, tokenizer):
            log.append("preprocess_dataset")
            return dataset

    monkeypatch.setattr(
        pipeline.StrategyFactory,
        "get_strategy",
        classmethod(lambda cls, cfg, ds: (log.append("get_strategy"), _Strategy())[1]),
    )
    monkeypatch.setattr(pipeline, "ModelLoader", lambda cfg, token: _RecordingLoader(log))
    monkeypatch.setattr(
        pipeline, "load_training_dataset", lambda path: (log.append("load_dataset"), train_ds)[1]
    )
    monkeypatch.setattr(
        pipeline,
        "split_train_eval",
        lambda ds, ratio, seed: (log.append(f"split_train_eval(seed={seed})"), (train_ds, eval_ds))[1],
    )
    # An image dataset: the processor is selected, so image training must switch on.
    monkeypatch.setattr(
        pipeline,
        "select_processing_class",
        lambda ds, tok, proc, cfg: (log.append("select_processing_class"), proc)[1],
    )

    config = SimpleNamespace(
        dataset_path="/nonexistent.jsonl", eval_split_ratio=0.2, eval_split_seed=1234
    )
    components = pipeline.build_training_components(config, None, "ds.json")
    return log, components


def test_training_args_are_built_before_the_model_loader_touches_anything(recorded):
    """
    Invariant 1: TrainingArguments(deepspeed=...) registers the global DeepSpeed
    config, and that is what puts from_pretrained under an active zero.Init.
    Constructing the loader first silently loses ZeRO-3 sharding at load time.
    """
    log, _ = recorded
    assert log.index("get_training_args") < log.index("ModelLoader.__init__")
    assert log.index("get_training_args") < log.index("load_model")


def test_image_training_is_enabled_before_the_model_is_loaded(recorded):
    """
    Invariant 2: enable_image_training() switches the load class to
    AutoModelForImageTextToText. After load_model() it is far too late -- the
    checkpoint would already be up as AutoModelForCausalLM with no vision tower.
    """
    log, _ = recorded
    assert "enable_image_training" in log, "an image dataset must switch the load class"
    assert log.index("enable_image_training") < log.index("load_model")


def test_the_eval_split_happens_before_preprocessing(recorded):
    """
    The split must see raw rows. Splitting after preprocessing would leak
    tokenisation choices across the boundary, and this was the step that
    silently never ran on the multi-GPU path.
    """
    log, _ = recorded
    split = next(i for i, entry in enumerate(log) if entry.startswith("split_train_eval"))
    assert split < log.index("preprocess_dataset")


def test_the_eval_split_seed_is_passed_through(recorded):
    """All ranks must derive the same split, so the seed cannot be dropped."""
    log, _ = recorded
    assert "split_train_eval(seed=1234)" in log


def test_components_expose_the_eval_dataset_and_processing_class(recorded):
    """Both are exactly what the drifted copy dropped."""
    _, components = recorded
    assert components.eval_dataset is not None
    assert components.processing_class.name == "processor"
    assert components.is_image_training is True


def test_components_expose_tokenizer_and_processor_separately(recorded):
    """
    `processing_class` is one OR the other, so returning only it leaves the
    caller unable to release the reference it did not get. The worker's
    end-of-job `_cleanup_training_resources` needs both.
    """
    _, components = recorded
    assert components.tokenizer.name == "tokenizer"
    assert components.processor.name == "processor"


def test_worker_rebinds_the_cleanup_locals_from_components():
    """
    Regression: the refactor left `tokenizer`/`processor` at their initial None
    in the worker, so `_cleanup_training_resources(tokenizer=tokenizer, ...)`
    silently released nothing for the whole job's lifetime.
    """
    from service.training import training_process

    tree = ast.parse(inspect.getsource(training_process))
    rebound = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "components"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {"tokenizer", "processor"} <= rebound, (
        f"worker must rebind tokenizer/processor from components; rebinds only {sorted(rebound)}"
    )


def test_no_eval_split_when_ratio_is_unset(monkeypatch):
    """The split must not fire when the user did not ask for one."""
    from service.training.core import pipeline

    log: list[str] = []

    class _Strategy:
        def get_training_args(self):
            return SimpleNamespace()

        def preprocess_dataset(self, dataset, tokenizer):
            return dataset

    monkeypatch.setattr(
        pipeline.StrategyFactory, "get_strategy", classmethod(lambda cls, cfg, ds: _Strategy())
    )
    monkeypatch.setattr(pipeline, "ModelLoader", lambda cfg, token: _RecordingLoader(log))
    monkeypatch.setattr(pipeline, "load_training_dataset", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        pipeline,
        "split_train_eval",
        lambda *a, **k: pytest.fail("split_train_eval must not run without eval_split_ratio"),
    )
    monkeypatch.setattr(
        pipeline, "select_processing_class", lambda ds, tok, proc, cfg: tok
    )

    config = SimpleNamespace(dataset_path="/x.jsonl", eval_split_ratio=None, eval_split_seed=42)
    components = pipeline.build_training_components(config, None, None)

    assert components.eval_dataset is None
    assert components.is_image_training is False
    assert "enable_image_training" not in log, "a text dataset must not switch the load class"


# --------------------------------------------------------------- multi-GPU wiring


def _keyword_names(module, func_name: str) -> set[str]:
    """Keyword argument names used at every call to `func_name` in a module."""
    tree = ast.parse(inspect.getsource(module))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if target == func_name:
                found.update(kw.arg for kw in node.keywords if kw.arg)
    return found


def test_multi_gpu_path_passes_the_eval_dataset_to_the_trainer():
    from service.training import training_script

    assert "eval_dataset" in _keyword_names(training_script, "prepare_trainer")


def test_multi_gpu_path_saves_the_processing_class_not_the_tokenizer():
    """
    An image run must save the PROCESSOR, or the output ships without
    processor_config.json and inference cannot rebuild the image pipeline. The
    drifted copy passed the tokenizer.
    """
    from service.training import training_script

    tree = ast.parse(inspect.getsource(training_script))
    saves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == "save_training_results")
    ]
    assert saves, "training_script must save results"
    for call in saves:
        second_arg = call.args[1]
        assert isinstance(second_arg, ast.Attribute), "expected components.processing_class"
        assert second_arg.attr == "processing_class", (
            f"save_training_results got '{ast.unparse(second_arg)}'; an image run needs the processor"
        )
