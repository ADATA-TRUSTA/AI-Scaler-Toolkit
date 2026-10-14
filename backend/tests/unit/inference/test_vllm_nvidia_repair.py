"""
Repairing the NVIDIA runtime without wrecking the environment it repairs.

An NVIDIA wheel can install while failing to unpack its shared object, leaving vLLM
unable to start. _repair_vllm_nvidia_runtime reinstalls the affected packages.

That was safe while VLLM_SERVER_PROJECT_DIR pointed at a separate vllm_server project:
the worst case stayed inside a venv nothing else used. It now defaults to the repo
root, so the venv being rewritten is normally the one the service itself is running
from — beside a pinned torch, and governed by uv.lock, which the commit gate enforces.
An unpinned `uv pip install` there can resolve a newer runtime than that torch was
built against and break the live process.

So the reinstall holds to the version already recorded in the venv, and the case where
there is no recorded version — the package is absent rather than half-unpacked — is
handed back to setup rather than resolved on the fly.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.inference.engines.vllm_engine import VllmEngine  # noqa: E402

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]


@pytest.fixture
def engine():
    return VllmEngine.__new__(VllmEngine)


def _make_venv(tmp_path: Path, dists: dict[str, str]) -> Path:
    """A venv skeleton carrying the given {package: version} dist-info directories."""
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    for name, version in dists.items():
        (site / f"{name.replace('-', '_')}-{version}.dist-info").mkdir()
    return tmp_path


def test_the_recorded_version_is_found(engine, tmp_path):
    """The half-unpacked wheel still has its dist-info, which names the pinned version."""
    project = _make_venv(tmp_path, {"nvidia-cudnn-cu13": "9.14.0.64"})
    venv = str(project / ".venv")

    assert engine._installed_package_version(venv, "nvidia-cudnn-cu13") == "9.14.0.64"


def test_a_different_package_is_not_mistaken_for_it(engine, tmp_path):
    """A prefix match would pin the wrong distribution's version onto this one."""
    project = _make_venv(tmp_path, {"nvidia-cudnn-cu13-extra": "1.2.3"})
    venv = str(project / ".venv")

    assert engine._installed_package_version(venv, "nvidia-cudnn-cu13") is None


def test_an_absent_package_has_no_version(engine, tmp_path):
    project = _make_venv(tmp_path, {})

    assert engine._installed_package_version(str(project / ".venv"), "nvidia-nccl-cu13") is None


def test_the_reinstall_is_pinned(engine, tmp_path, monkeypatch):
    """Every requirement handed to uv carries ==, so nothing is re-resolved."""
    project = _make_venv(
        tmp_path,
        {
            "nvidia-cudnn-cu13": "9.14.0.64",
            "nvidia-cusparselt-cu13": "0.8.1",
            "nvidia-nccl-cu13": "2.29.1",
            "nvidia-nvshmem-cu13": "3.4.5",
        },
    )
    python_bin = project / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    os.chmod(python_bin, 0o755)

    # Missing until the repair runs, present afterwards — the repair verifies its own
    # work, so a stub that never heals would fail on that check instead of this one.
    state = {"repaired": False}
    monkeypatch.setattr(
        engine,
        "_find_shared_library",
        lambda venv, lib: "/found/lib.so" if state["repaired"] else None,
    )
    monkeypatch.setattr(
        "service.inference.engines.vllm_engine.shutil.which", lambda name: "/usr/bin/uv"
    )
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        state["repaired"] = True

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("service.inference.engines.vllm_engine.subprocess.run", _fake_run)

    engine._repair_vllm_nvidia_runtime(str(project))

    cmd = captured["cmd"]
    # --reinstall-package takes a bare name; the requirements are the trailing args.
    # Only the latter decide what gets resolved, so those are what must be pinned.
    requirements = [
        arg
        for i, arg in enumerate(cmd)
        if arg.startswith("nvidia-") and cmd[i - 1] != "--reinstall-package"
    ]
    assert requirements, "expected requirements on the command line"
    assert all("==" in requirement for requirement in requirements)
    assert "nvidia-cudnn-cu13==9.14.0.64" in requirements


def test_a_missing_package_is_refused_rather_than_resolved(engine, tmp_path, monkeypatch):
    """
    Nothing recorded means nothing to hold to, and installing it means resolving it.

    That is setup's job, where uv.lock decides the answer — doing it here would drift
    the running environment away from the lockfile with no record of it.
    """
    project = _make_venv(tmp_path, {})
    python_bin = project / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    os.chmod(python_bin, 0o755)

    monkeypatch.setattr(engine, "_find_shared_library", lambda venv, lib: None)
    monkeypatch.setattr(
        "service.inference.engines.vllm_engine.shutil.which", lambda name: "/usr/bin/uv"
    )

    def _no_run(*args, **kwargs):
        raise AssertionError("uv must not be invoked when the version cannot be pinned")

    monkeypatch.setattr("service.inference.engines.vllm_engine.subprocess.run", _no_run)

    with pytest.raises(RuntimeError, match="setup_env"):
        engine._repair_vllm_nvidia_runtime(str(project))


def test_a_healthy_venv_is_left_alone(engine, tmp_path, monkeypatch):
    """No missing library, no reinstall — the common case must stay a no-op."""
    project = _make_venv(tmp_path, {})
    monkeypatch.setattr(engine, "_find_shared_library", lambda venv, lib: "/found/lib.so")

    def _no_run(*args, **kwargs):
        raise AssertionError("nothing was missing; uv must not run")

    monkeypatch.setattr("service.inference.engines.vllm_engine.subprocess.run", _no_run)

    engine._repair_vllm_nvidia_runtime(str(project))
