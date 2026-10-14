"""
Which tests exercise a Linux-only feature, and how to tell.

Fine-tuning and vLLM are refused at the endpoint off Linux (see
service/platform_support.py for why), so on Windows the code under
service/training/** and the vLLM engine is unreachable: /training/start and a
vllm /inference/load_model both return 501 before anything is spawned. Their
tests are therefore not "failing coverage" on Windows, they are coverage of code
that cannot run -- and several cannot even be attempted, because deepspeed and
vllm are declared `sys_platform == 'linux'` in pyproject.toml and are simply
absent from a Windows install.

This lives beside conftest.py rather than inside it because two files in this
tree are named conftest.py (this one's neighbour and tests/unit/training/), and
with no __init__.py anywhere a plain `import conftest` resolves to whichever was
imported last -- so a test that wanted the table here would silently get the
training fixtures instead. Same reason the *_fixtures.py modules are separate.

`tests/conftest.py` applies the marker; `tests/unit/test_linux_only_gating.py`
asserts the table still points at real files.
"""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Directories and files whose tests exercise fine-tuning. A directory covers
# everything under it, so a new file added there is gated without anyone
# remembering to mark it -- that forgetting is the failure this prevents.
FINETUNE_PATHS = (
    TESTS_DIR / "unit" / "training",
    TESTS_DIR / "smoke" / "test_offload_training_matrix.py",
    TESTS_DIR / "smoke" / "test_multigpu_training_matrix.py",
    TESTS_DIR / "smoke" / "test_image_training_effectiveness.py",
    TESTS_DIR / "smoke" / "test_multigpu_image_effectiveness.py",
)

# The vLLM engine modules mark themselves, so the gate is visible in the file --
# they are a hand-picked five sitting beside ~20 cross-platform siblings in
# tests/unit/inference/. Listed here only so the gating test can check they
# still declare it.
VLLM_MODULES = (
    "unit/inference/test_vllm_server_args.py",
    "unit/inference/test_vllm_error_classifier.py",
    "unit/inference/test_vllm_nvidia_repair.py",
    "unit/inference/test_vllm_tool_parser_probe.py",
    "smoke/test_vllm_server_smoke.py",
)

# Deliberately NOT gated, and it must stay that way:
#
#   - unit/test_platform_support.py fakes the OS with monkeypatch and asserts the
#     501s, so it is exactly the file that must keep running on Windows. Never
#     gate by "vllm in the filename" -- it would catch that file's
#     test_load_model_refuses_vllm_off_linux and delete the proof that the
#     refusal works on the platform it exists for.
#   - The vLLM config/payload assertions scattered through the shared inference
#     modules (test_api_compatibility.py, test_async_server_client.py, ...).
#     Those build dicts and CLI args without importing vllm, over code shared
#     with the engines Windows does run.


def is_under(path: Path, root: Path) -> bool:
    """Whether `path` is `root` itself or sits beneath it."""
    return path == root or root in path.parents


def is_finetune_path(path: Path) -> bool:
    """Whether a test file belongs to the fine-tuning suite."""
    return any(is_under(path, root) for root in FINETUNE_PATHS)
