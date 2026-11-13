"""An environment directory that is not an environment, and paths too long to delete.

Both come from one sequence: uninstall, then install again. The backend's environment
sits in the program directory and holds paths past Windows' 260 characters (torch's
licence texts: 254 plus the account name's length in the default location, so any
name of six characters or more is past it). An uninstaller that cannot delete
those leaves the tree with its interpreter gone, and uv then refuses to sync into what
remains -- "not a valid Python environment" -- every time setup is tried.

So setup has to clear such a directory first, and whatever clears it has to cope with
the paths that made it.
"""

import os
import stat
import sys

import pytest

from launcher import config, maintenance, provision

# Deep enough to pass 260 characters under any temporary directory.
DEEP = ["torch-2.9.0.dist-info", "licenses", "third_party"] + ["some_vendored_dependency"] * 10


def make_deep_file(root):
    """A file whose path is past the limit, written the way uv writes it."""
    target = root.joinpath(*DEEP, "LICENSE.txt")
    os.makedirs(maintenance._extended(target.parent), exist_ok=True)
    with open(maintenance._extended(target), "w", encoding="utf-8") as handle:
        handle.write("x")
    assert len(str(target)) > 260
    return target


@pytest.fixture
def venv(tmp_path, monkeypatch):
    path = tmp_path / "service" / ".venv"
    monkeypatch.setattr(config, "venv_dir", lambda: path)
    return path


def make_usable(venv):
    interpreter = provision.environment_interpreter()
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"")
    (venv / "pyvenv.cfg").write_text("home = somewhere\n", encoding="utf-8")


def test_no_environment_is_not_a_broken_one(venv):
    assert provision.discard_broken_environment() is False
    assert not venv.exists()


def test_a_usable_environment_is_left_alone(venv):
    make_usable(venv)
    assert provision.discard_broken_environment() is False
    assert provision.environment_interpreter().is_file()


def test_what_an_uninstall_left_behind_is_removed(venv):
    # The shape it really has: no interpreter, no pyvenv.cfg, packages still there.
    (venv / "Lib" / "site-packages" / "torch").mkdir(parents=True)
    (venv / "Lib" / "site-packages" / "torch" / "version.py").write_text("", encoding="utf-8")
    assert provision.discard_broken_environment() is True
    assert not venv.exists()


def test_an_environment_without_its_cfg_is_broken_too(venv):
    make_usable(venv)
    (venv / "pyvenv.cfg").unlink()
    assert provision.discard_broken_environment() is True
    assert not venv.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the 260-character limit is Windows'")
def test_paths_past_260_characters_are_removed(venv):
    make_deep_file(venv / "Lib" / "site-packages")
    assert provision.discard_broken_environment() is True
    assert not venv.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the 260-character limit is Windows'")
def test_reset_removes_an_environment_with_long_paths(venv):
    make_usable(venv)
    make_deep_file(venv / "Lib" / "site-packages")
    result = maintenance.Removal()
    maintenance._remove(venv, "virtual environment", result, dry_run=False)
    assert result.failed == []
    assert not venv.exists()


def test_read_only_files_do_not_stop_a_removal(tmp_path):
    pack = tmp_path / "checkout" / ".git" / "objects" / "pack-1.pack"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"x")
    os.chmod(pack, stat.S_IREAD)
    maintenance.remove_tree(tmp_path / "checkout")
    assert not (tmp_path / "checkout").exists()
