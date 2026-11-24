"""
setup_env.sh must run its post-sync steps against the environment the sync built.

A one-click install keeps its environment in the per-user data directory and says so through
UV_PROJECT_ENVIRONMENT; ./.venv does not exist there. A step that runs ``$PROJECT_ROOT/.venv/bin/python``
finds no deepspeed, concludes "not applicable" and goes on, so the DeepSpeed async-I/O descriptor-leak
patch (scripts/patch_deepspeed_aio.py) was never applied to a launcher install -- and nothing failed.

The two places that name ``./.venv`` on purpose are the default of ``venv_dir`` and the fallback that
``verify_llama_binary`` overrides when UV_PROJECT_ENVIRONMENT is set.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.finetune]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "linux" / "setup_env.sh"


def test_no_step_hard_codes_the_project_venv_interpreter():
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(SCRIPT.read_text(encoding="utf-8").splitlines(), start=1)
        if "$PROJECT_ROOT/.venv" in line and not line.lstrip().startswith("#")
    ]
    # venv_dir's default, and verify_llama_binary's fallback (which UV_PROJECT_ENVIRONMENT overrides)
    allowed = ('local venv="${UV_PROJECT_ENVIRONMENT:-$PROJECT_ROOT/.venv}"', 'local python="$PROJECT_ROOT/.venv/bin/python"')
    unexpected = [o for o in offenders if not any(a in o for a in allowed)]
    assert not unexpected, (
        "these steps would look in ./.venv even when UV_PROJECT_ENVIRONMENT names the real environment:\n"
        + "\n".join(unexpected)
        + "\nuse \"$(venv_python)\""
    )


def test_the_deepspeed_patch_runs_in_the_environment_setup_built():
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index('DEEPSPEED_AIO_STATUS="not applicable"')
    block = text[start : text.index("\nfi\n", start)]
    assert '"$(venv_python)" -c "import deepspeed"' in block
    assert '"$(venv_python)" "$PROJECT_ROOT/scripts/patch_deepspeed_aio.py"' in block
