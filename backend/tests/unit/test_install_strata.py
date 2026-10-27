"""
scripts/linux/install_strata.sh: the checkout it makes, what it hands Strata's setup, and .env.

Strata's setup is replaced by a stand-in that records its arguments and leaves the files the
real one would (an engine, a venv interpreter, a run config), so the script runs end to end
without the network, a GPU or a compile. The repository is a local one made per test.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "linux"

if shutil.which("bash") is None or shutil.which("git") is None:
    pytest.skip("needs bash and git", allow_module_level=True)

# What Strata's setup leaves behind; FAKE_NO_SLEEP=1 makes it the ready-made engine instead.
FAKE_SETUP = """#!/bin/sh
printf '%s\\n' "$*" > setup-args.txt
mkdir -p engine .venv/bin
if [ "${FAKE_NO_SLEEP:-0}" = 1 ]; then echo 'strata engine' > engine/strata
else echo 'strata engine --sleep-mode' > engine/strata; fi
printf '#!/bin/sh\\n' > .venv/bin/python
chmod +x engine/strata .venv/bin/python
echo '{"exe": "engine/strata", "args": []}' > strata-iq3_xxs.json
"""


def _git(*args, cwd):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def tree(tmp_path):
    """A project with the two scripts and a .env, and a Strata repository beside it."""
    project = tmp_path / "project"
    (project / "scripts" / "linux").mkdir(parents=True)
    for name in ("install_strata.sh", "env_file.sh"):
        shutil.copy2(SCRIPTS / name, project / "scripts" / "linux" / name)
    (project / ".env").write_text("SERVICE_PORT=8000\n# STRATA_HOME=\n")

    repo = tmp_path / "strata-src"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "setup.sh").write_text(FAKE_SETUP)
    (repo / "setup.sh").chmod(0o755)
    _git("add", "setup.sh", cwd=repo)
    _git("commit", "-q", "-m", "upstream", cwd=repo)
    _git("checkout", "-q", "-b", "feat/deep-sleep", cwd=repo)
    (repo / "SLEEP.md").write_text("deep sleep\n")
    _git("add", "SLEEP.md", cwd=repo)
    _git("commit", "-q", "-m", "deep sleep", cwd=repo)
    return project, repo


def _run(project, repo, **env):
    full = {
        **os.environ,
        "TRUSTA_STRATA_REPO": str(repo),
        "GIT_CONFIG_GLOBAL": os.devnull,  # no credential helper or url rewrite of the host
        **env,
    }
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


def test_installs_the_deep_sleep_branch_with_iq3_xxs_and_records_it(tree):
    project, repo = tree
    result = _run(project, repo)
    assert result.returncode == 0, result.stderr

    strata = project.parent / "strata"  # beside the project by default
    assert (strata / "SLEEP.md").is_file()  # feat/deep-sleep, not main
    assert (strata / "setup-args.txt").read_text().split() == [
        "--build",
        "--yes",
        "--no-start",
        "--model",
        "IQ3_XXS",
    ]
    env = _env(project)
    assert env["STRATA_HOME"] == str(strata)
    assert env["STRATA_PYTHON"] == str(strata / ".venv" / "bin" / "python")
    assert env["STRATA_ENGINE_BINARY"] == str(strata / "engine" / "strata")
    assert "# STRATA_HOME=" not in (project / ".env").read_text()  # the example line taken over
    assert "strata-iq3_xxs.json" in result.stdout


def test_setup_args_and_ref_are_passed_through(tree, tmp_path):
    project, repo = tree
    _git("tag", "v1-sleep", cwd=repo)
    target = tmp_path / "elsewhere"
    result = _run(
        project,
        repo,
        TRUSTA_STRATA_REF="v1-sleep",
        TRUSTA_STRATA_DIR=str(target),
        TRUSTA_STRATA_SETUP_ARGS="--model IQ2_XS --data-dir /mnt/models",
    )
    assert result.returncode == 0, result.stderr
    assert (target / "setup-args.txt").read_text().split()[3:] == [
        "--model",
        "IQ2_XS",
        "--data-dir",
        "/mnt/models",
    ]
    assert _env(project)["STRATA_HOME"] == str(target)


def test_an_engine_without_deep_sleep_is_refused_and_not_recorded(tree):
    project, repo = tree
    result = _run(project, repo, FAKE_NO_SLEEP="1")
    assert result.returncode != 0
    assert "no deep sleep" in result.stderr
    assert "STRATA_ENGINE_BINARY" not in _env(project)


def test_a_rerun_updates_a_clean_checkout_and_leaves_a_modified_one_alone(tree):
    project, repo = tree
    assert _run(project, repo).returncode == 0
    strata = project.parent / "strata"

    (repo / "SLEEP.md").write_text("deep sleep, newer\n")
    _git("commit", "-q", "-am", "newer", cwd=repo)
    assert _run(project, repo).returncode == 0
    assert (strata / "SLEEP.md").read_text() == "deep sleep, newer\n"

    (strata / "SLEEP.md").write_text("someone's edit\n")
    result = _run(project, repo)
    assert result.returncode != 0
    assert "uncommitted changes" in result.stderr
    assert (strata / "SLEEP.md").read_text() == "someone's edit\n"


def test_a_checkout_of_another_repository_is_not_touched(tree, tmp_path):
    project, repo = tree
    other = tmp_path / "other-src"
    shutil.copytree(repo, other)
    target = tmp_path / "strata-other"
    _git("clone", "-q", str(other), str(target), cwd=tmp_path)

    result = _run(project, repo, TRUSTA_STRATA_DIR=str(target))
    assert result.returncode != 0
    assert "is a clone of" in result.stderr
    assert not (target / "setup-args.txt").exists()
