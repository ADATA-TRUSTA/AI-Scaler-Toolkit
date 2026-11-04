"""
scripts/linux/install_strata.sh: the package it unpacks and checks, what it hands Strata's setup, and .env.

The package is a small stand-in made per test, laid out like a strata-sleep release (VERSION,
SLEEP.json, MANIFEST.sha256, engine/strata, setup.py). uv is replaced by one whose venv python
plays Strata's setup: it records its arguments and writes the run config the real one would, so
the script runs end to end without the network, a GPU or a model.
"""

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "linux"

if shutil.which("bash") is None or shutil.which("sha256sum") is None:
    pytest.skip("needs bash and sha256sum", allow_module_level=True)

PACKAGE = "strata-sleep-0.2.7-sleep.1-linux-x86_64"

# `uv venv ... <dir>`: a venv whose python is Strata's setup, run as `python setup.py <args>`.
FAKE_UV = """#!/bin/sh
for last; do :; done
mkdir -p "$last/bin"
cat > "$last/bin/python" <<'PY'
#!/bin/sh
case "$1" in setup.py) shift ;; *) exec python3 "$@" ;; esac
printf '%s\\n' "$*" > setup-args.txt
printf '%s\\n' "$@" > setup-argv.txt
echo '{"exe": "engine/strata", "args": ["--native", "/models/q-00001-of-00002.gguf"]}' > strata-iq3_xxs.json
PY
chmod +x "$last/bin/python"
"""


def _package(root: Path, sleep: bool = True) -> Path:
    """A strata-sleep package folder under root, with its manifest."""
    pkg = root / PACKAGE
    (pkg / "engine").mkdir(parents=True)
    (pkg / "VERSION").write_text("0.2.7-sleep.1\n")
    (pkg / "SLEEP.json").write_text('{"version": "0.2.7-sleep.1"}\n')
    (pkg / "setup.py").write_text("# Strata's setup\n")
    (pkg / "engine" / "strata").write_text(
        "strata engine --sleep-mode\n" if sleep else "strata engine\n"
    )
    (pkg / "engine" / "strata").chmod(0o755)
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(pkg).as_posix()}\n"
        for p in sorted(pkg.rglob("*"))
        if p.is_file()
    ]
    (pkg / "MANIFEST.sha256").write_text("".join(lines))
    return pkg


def _tarball(pkg: Path, dest: Path) -> Path:
    archive = dest / f"{pkg.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pkg, arcname=pkg.name)
    return archive


@pytest.fixture
def tree(tmp_path):
    """A project with the two scripts, a .env and a package in vendor/, and a fake uv."""
    project = tmp_path / "project"
    (project / "scripts" / "linux").mkdir(parents=True)
    for name in ("install_strata.sh", "env_file.sh"):
        shutil.copy2(SCRIPTS / name, project / "scripts" / "linux" / name)
    (project / ".env").write_text("SERVICE_PORT=8000\n# STRATA_HOME=\n")
    (project / "vendor").mkdir()
    _tarball(_package(tmp_path / "src"), project / "vendor")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    return project, bin_dir


def _run(project, bin_dir, **env):
    full = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env}
    return subprocess.run(
        ["bash", str(project / "scripts" / "linux" / "install_strata.sh")],
        env=full,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _env(project):
    return dict(
        line.split("=", 1)
        for line in (project / ".env").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )


def test_installs_the_vendored_package_with_iq3_xxs_and_records_it(tree):
    project, bin_dir = tree
    result = _run(project, bin_dir)
    assert result.returncode == 0, result.stderr

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
    assert env["STRATA_PYTHON"] == str(strata / ".venv" / "bin" / "python")
    assert env["STRATA_ENGINE_BINARY"] == str(strata / "engine" / "strata")
    assert "# STRATA_HOME=" not in (project / ".env").read_text()  # the example line taken over
    assert "/models/q-00001-of-00002.gguf" in result.stdout
    assert not list(project.parent.glob(".strata-unpack.*"))  # the staging folder is gone


def test_package_folder_dir_and_setup_args_are_passed_through(tree, tmp_path):
    project, bin_dir = tree
    pkg = _package(tmp_path / "unpacked")
    target = tmp_path / "elsewhere" / "strata"
    result = _run(
        project,
        bin_dir,
        TRUSTA_STRATA_PACKAGE=str(pkg),
        TRUSTA_STRATA_DIR=str(target),
        TRUSTA_STRATA_SETUP_ARGS="--model IQ2_XS --gguf-dir /mnt/models",
    )
    assert result.returncode == 0, result.stderr
    assert (target / "setup-args.txt").read_text().split()[4:] == [
        "--model",
        "IQ2_XS",
        "--gguf-dir",
        "/mnt/models",
    ]
    assert _env(project)["STRATA_HOME"] == str(target)


def test_the_gguf_folder_reaches_setup_as_one_argument(tree, tmp_path):
    project, bin_dir = tree
    folder = tmp_path / "hugging face" / "my models"
    result = _run(
        project,
        bin_dir,
        TRUSTA_STRATA_SETUP_ARGS="--family qwen --model IQ3_S",
        TRUSTA_STRATA_GGUF_DIR=str(folder),
    )
    assert result.returncode == 0, result.stderr
    argv = (project.parent / "strata" / "setup-argv.txt").read_text().splitlines()
    assert argv[-6:] == ["--family", "qwen", "--model", "IQ3_S", "--gguf-dir", str(folder)]


def test_an_engine_without_deep_sleep_is_refused_and_not_recorded(tree, tmp_path):
    project, bin_dir = tree
    pkg = _package(tmp_path / "nosleep", sleep=False)
    result = _run(project, bin_dir, TRUSTA_STRATA_PACKAGE=str(pkg))
    assert result.returncode != 0
    assert "no deep sleep" in result.stderr
    assert "STRATA_ENGINE_BINARY" not in _env(project)


def test_a_package_that_fails_its_manifest_installs_nothing(tree, tmp_path):
    project, bin_dir = tree
    pkg = _package(tmp_path / "bad")
    (pkg / "setup.py").write_text("# changed after the manifest was written\n")
    result = _run(project, bin_dir, TRUSTA_STRATA_PACKAGE=str(pkg))
    assert result.returncode != 0
    assert "MANIFEST.sha256" in result.stderr
    assert not (project.parent / "strata").exists()
    assert "STRATA_HOME" not in _env(project)


def test_a_rerun_keeps_the_venv_and_run_configs(tree):
    project, bin_dir = tree
    assert _run(project, bin_dir).returncode == 0
    strata = project.parent / "strata"
    (strata / ".venv" / "marker").write_text("kept\n")
    (strata / "strata-other.json").write_text('{"exe": "engine/strata", "args": []}\n')

    result = _run(project, bin_dir)
    assert result.returncode == 0, result.stderr
    assert (strata / ".venv" / "marker").read_text() == "kept\n"
    assert (strata / "strata-other.json").is_file()


def test_a_source_checkout_or_another_folder_is_not_touched(tree, tmp_path):
    project, bin_dir = tree
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    result = _run(project, bin_dir, TRUSTA_STRATA_DIR=str(checkout))
    assert result.returncode != 0
    assert "git checkout" in result.stderr
    assert not (checkout / "SLEEP.json").exists()

    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("someone's\n")
    result = _run(project, bin_dir, TRUSTA_STRATA_DIR=str(other))
    assert result.returncode != 0
    assert "not a strata-sleep install" in result.stderr
    assert sorted(p.name for p in other.iterdir()) == ["notes.txt"]


def test_no_package_is_an_error_that_names_the_variable(tree):
    project, bin_dir = tree
    shutil.rmtree(project / "vendor")
    result = _run(project, bin_dir)
    assert result.returncode != 0
    assert "TRUSTA_STRATA_PACKAGE" in result.stderr
