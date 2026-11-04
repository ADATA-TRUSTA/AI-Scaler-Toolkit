"""
The service puts <venv>/lib on LD_LIBRARY_PATH when it holds oneCCL's plugin.

oneCCL 2022.1 (torch 2.14+xpu) loads libccl.so.1 by name from libccl.so.2, with no RPATH on
either, so the training and vLLM processes the service starts find it only through this path.
"""

import os
from pathlib import Path

import pytest

from service import settings

pytestmark = [
    pytest.mark.finetune,
    pytest.mark.skipif(os.name == "nt", reason="LD_LIBRARY_PATH is a Linux concept"),
]


def _venv(tmp_path: Path, *libs: str) -> Path:
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in libs:
        (lib / name).write_bytes(b"")
    return tmp_path


def test_venv_lib_with_the_oneccl_plugin_is_prepended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An XPU venv: the plugin's dir comes first, what was there stays."""
    venv = _venv(tmp_path, "libccl.so.2", "libccl.so.1")
    monkeypatch.setattr(settings, "VENV_DIR", venv)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/existing")

    settings._prepend_cuda_library_paths()

    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [str(venv / "lib"), "/opt/existing"]


def test_venv_lib_without_oneccl_is_left_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA or CPU venv has no libccl.so.1, and its lib dir does not join the path."""
    monkeypatch.setattr(settings, "VENV_DIR", _venv(tmp_path, "libpython3.12.so.1.0"))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/existing")

    settings._prepend_cuda_library_paths()

    assert os.environ["LD_LIBRARY_PATH"] == "/opt/existing"
