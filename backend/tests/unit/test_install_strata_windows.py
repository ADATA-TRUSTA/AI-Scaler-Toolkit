"""
scripts/windows/install_strata.ps1: the package it unpacks and checks, what it hands Strata's setup, and .env.

The Windows twin of test_install_strata.py. The package is a small stand-in made per test, a zip
laid out like a strata-sleep release, whose setup.py is a real script that records its arguments
and writes the run config the real setup would. uv is replaced by one that makes a plain venv, so
the script runs end to end without the network, a GPU or a model.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.inference

if sys.platform != "win32" or shutil.which("powershell") is None:
    pytest.skip("needs Windows PowerShell", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "windows"
PACKAGE = "strata-sleep-0.2.7-sleep.1-windows-x86_64"

FAKE_SETUP = """import json, sys
open("setup-args.txt", "w").write(" ".join(sys.argv[1:]))
json.dump(sys.argv[1:], open("setup-argv.json", "w"))
json.dump({"exe": "engine/strata.exe", "args": ["--native", "D:/models/q-00001-of-00002.gguf"]},
          open("strata-iq3_xxs.json", "w"))
"""


def _package(root: Path, sleep: bool = True) -> Path:
    """A strata-sleep package folder under root, with its manifest."""
    pkg = root / PACKAGE
    (pkg / "engine").mkdir(parents=True)
    (pkg / "VERSION").write_text("0.2.7-sleep.1\n")
    (pkg / "SLEEP.json").write_text('{"version": "0.2.7-sleep.1"}\n')
    (pkg / "setup.py").write_text(FAKE_SETUP)
    (pkg / "engine" / "strata.exe").write_bytes(
        b"MZ strata engine --sleep-mode\n" if sleep else b"MZ strata engine\n"
    )
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(pkg).as_posix()}\n"
        for p in sorted(pkg.rglob("*"))
        if p.is_file()
    ]
    (pkg / "MANIFEST.sha256").write_text("".join(lines), newline="\n")
    return pkg


def _zip(pkg: Path, dest: Path) -> Path:
    archive = dest / f"{pkg.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            z.write(p, f"{pkg.name}/{p.relative_to(pkg).as_posix()}")
    return archive


@pytest.fixture
def tree(tmp_path):
    """A project with the two scripts, a .env and a package in vendor\\, and a fake uv."""
    project = tmp_path / "project"
    (project / "scripts" / "windows").mkdir(parents=True)
    for name in ("install_strata.ps1", "env_file.ps1"):
        shutil.copy2(SCRIPTS / name, project / "scripts" / "windows" / name)
    (project / ".env").write_text("SERVICE_PORT=8000\n# STRATA_HOME=\n")
    (project / "vendor").mkdir()
    _zip(_package(tmp_path / "src"), project / "vendor")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # called as `uv venv -q --seed -p 3.12 <dir>`
    (bin_dir / "uv.cmd").write_text(f'@"{sys.executable}" -m venv --without-pip "%~6"\r\n')
    return project, bin_dir


def _run(project, bin_dir, **env):
    full = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env}
    for key in ("TRUSTA_STRATA_PACKAGE", "TRUSTA_STRATA_DIR", "TRUSTA_STRATA_SETUP_ARGS",
                "TRUSTA_STRATA_DATA_DIR"):
        if key not in env:
            full.pop(key, None)
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(project / "scripts" / "windows" / "install_strata.ps1"),
        ],
        env=full,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )


def _env(project):
    return dict(
        line.split("=", 1)
        for line in (project / ".env").read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )


def test_installs_the_vendored_package_with_iq3_xxs_and_records_it(tree):
    project, bin_dir = tree
    result = _run(project, bin_dir)
    assert result.returncode == 0, result.stdout + result.stderr

    strata = project.parent / "strata"  # beside the project by default
    assert (strata / "SLEEP.json").is_file()
    assert (strata / "setup-args.txt").read_text().split() == [
        "--yes",
        "--no-start",
        "--vision",
        "no",
        "--model",
        "IQ3_XXS",
    ]
    env = _env(project)
    assert env["STRATA_HOME"] == str(strata)
    assert env["STRATA_PYTHON"] == str(strata / ".venv" / "Scripts" / "python.exe")
    assert env["STRATA_ENGINE_BINARY"] == str(strata / "engine" / "strata.exe")
    assert "# STRATA_HOME=" not in (project / ".env").read_text()  # the example line taken over
    assert "D:/models/q-00001-of-00002.gguf" in result.stdout
    assert not list(project.parent.glob(".strata-unpack-*"))  # the staging folder is gone


def test_package_folder_dir_and_setup_args_are_passed_through(tree, tmp_path):
    project, bin_dir = tree
    pkg = _package(tmp_path / "unpacked")
    target = tmp_path / "elsewhere" / "strata"
    result = _run(
        project,
        bin_dir,
        TRUSTA_STRATA_PACKAGE=str(pkg),
        TRUSTA_STRATA_DIR=str(target),
        TRUSTA_STRATA_SETUP_ARGS="--model IQ2_XS --gguf-dir D:\\models",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "setup-args.txt").read_text().split()[4:] == [
        "--model",
        "IQ2_XS",
        "--gguf-dir",
        "D:\\models",
    ]
    assert _env(project)["STRATA_HOME"] == str(target)


def test_the_gguf_folder_reaches_setup_as_one_argument(tree, tmp_path):
    project, bin_dir = tree
    folder = tmp_path / "Hugging Face" / "my models"
    result = _run(
        project,
        bin_dir,
        TRUSTA_STRATA_SETUP_ARGS="--family qwen --model IQ3_S",
        TRUSTA_STRATA_GGUF_DIR=str(folder),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads((project.parent / "strata" / "setup-argv.json").read_text())
    assert argv[-6:] == ["--family", "qwen", "--model", "IQ3_S", "--gguf-dir", str(folder)]


def test_an_engine_without_deep_sleep_is_refused_and_not_recorded(tree, tmp_path):
    project, bin_dir = tree
    result = _run(
        project, bin_dir, TRUSTA_STRATA_PACKAGE=str(_package(tmp_path / "nosleep", sleep=False))
    )
    assert result.returncode != 0
    assert "no deep sleep" in result.stdout
    assert "STRATA_ENGINE_BINARY" not in _env(project)


def test_a_package_that_fails_its_manifest_installs_nothing(tree, tmp_path):
    project, bin_dir = tree
    pkg = _package(tmp_path / "bad")
    (pkg / "setup.py").write_text("# changed after the manifest was written\n")
    result = _run(project, bin_dir, TRUSTA_STRATA_PACKAGE=str(_zip(pkg, tmp_path)))
    assert result.returncode != 0
    assert "MANIFEST.sha256" in result.stdout
    assert not (project.parent / "strata").exists()
    assert not list(project.parent.glob(".strata-unpack-*"))
    assert "STRATA_HOME" not in _env(project)


def test_a_rerun_keeps_the_venv_and_run_configs(tree):
    project, bin_dir = tree
    assert _run(project, bin_dir).returncode == 0
    strata = project.parent / "strata"
    (strata / ".venv" / "marker").write_text("kept\n")
    (strata / "strata-other.json").write_text('{"exe": "engine/strata.exe", "args": []}\n')
    (strata / "engine" / "strata.exe").write_bytes(b"an older engine --sleep-mode\n")

    result = _run(project, bin_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (strata / ".venv" / "marker").read_text() == "kept\n"
    assert (strata / "strata-other.json").is_file()
    assert (
        (strata / "engine" / "strata.exe").read_bytes().startswith(b"MZ strata engine")
    )  # replaced
    assert not (strata / "engine" / "engine").exists()  # copied into, not nested under


def test_a_source_checkout_another_folder_or_a_long_path_is_not_touched(tree, tmp_path):
    project, bin_dir = tree
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    result = _run(project, bin_dir, TRUSTA_STRATA_DIR=str(checkout))
    assert result.returncode != 0
    assert "git checkout" in result.stdout
    assert not (checkout / "SLEEP.json").exists()

    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("someone's\n")
    result = _run(project, bin_dir, TRUSTA_STRATA_DIR=str(other))
    assert result.returncode != 0
    assert "not a strata-sleep install" in result.stdout
    assert sorted(p.name for p in other.iterdir()) == ["notes.txt"]

    result = _run(project, bin_dir, TRUSTA_STRATA_DIR="C:\\" + "x" * 120)
    assert result.returncode != 0
    assert "at most 110" in result.stdout


def test_no_package_is_an_error_that_names_the_variable(tree):
    project, bin_dir = tree
    shutil.rmtree(project / "vendor")
    result = _run(project, bin_dir)
    assert result.returncode != 0
    assert "TRUSTA_STRATA_PACKAGE" in result.stdout


@pytest.fixture
def drive(tmp_path):
    """A drive letter standing for ``tmp_path``, so a target can sit at a drive root."""
    free = next(
        (letter for letter in "ZYXWV" if not Path(f"{letter}:\\").exists()),
        None,
    )
    if free is None:
        pytest.skip("no free drive letter for subst")
    root = tmp_path / "drive"
    root.mkdir()
    if subprocess.run(["subst", f"{free}:", str(root)], shell=True).returncode != 0:
        pytest.skip("subst is not available")
    try:
        yield f"{free}:"
    finally:
        subprocess.run(["subst", f"{free}:", "/D"], shell=True)


def test_installs_into_a_folder_at_a_drive_root(tree, drive):
    """``New-Item -Force`` cannot make "D:\\", so only the missing parents are made."""
    project, bin_dir = tree
    target = f"{drive}\\strata"
    result = _run(project, bin_dir, TRUSTA_STRATA_DIR=target)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (Path(target) / "SLEEP.json").is_file()
    assert _env(project)["STRATA_HOME"] == target
    assert not list(Path(f"{drive}\\").glob(".strata-unpack-*"))


def test_the_data_folder_reaches_setup_as_one_argument(tree, tmp_path):
    """Like --gguf-dir: its own variable, because _SETUP_ARGS splits on whitespace."""
    project, bin_dir = tree
    data = tmp_path / "Strata data" / "derived"
    result = _run(project, bin_dir, TRUSTA_STRATA_DATA_DIR=str(data))
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads((project.parent / "strata" / "setup-argv.json").read_text())
    assert argv[argv.index("--data-dir") + 1] == str(data)
    assert "--set-default-data-dir" in argv
