"""`.env.example` must document the defaults settings.py actually applies.

This is not pedantry about docs. `_load_project_env()` falls back to
`.env.example` when there is no `.env`, so on a fresh checkout that file is not
a sample -- it is the configuration. A value that has drifted from the code is
then a setting nobody chose.

Even where `.env` exists, `.env.example` is where an operator looks to find out
what a knob does and what it is set to. Raising DEFAULT_GENERATION_TIMEOUT from
300 to 1800 without touching the line documenting 300 is exactly how the file
stops being trustworthy.

Only settings whose default is a literal in `os.getenv(NAME, "literal")` are
checked. The ones computed from a `_DEFAULT_*` constant or resolved at import
(LLAMA_SERVER_BINARY auto-detects a build) have no single literal to compare,
and asserting on them would mean re-implementing the resolution here.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.inference

REPO_ROOT = Path(__file__).resolve().parents[2]

# Documented as a sample path for an override, not as the effective default:
# settings.py resolves this one by probing for an installed build.
_EXEMPT = {"LLAMA_SERVER_BINARY"}


def _commented_examples() -> dict[str, str]:
    """`# NAME=value` lines in .env.example -> value."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"^#\s*([A-Z][A-Z0-9_]+)=(.*)$", text, re.M)
    }


def _literal_defaults() -> dict[str, str]:
    """`os.getenv("NAME", "literal")` in settings.py -> literal."""
    text = (REPO_ROOT / "service" / "settings.py").read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]+)"\s*,\s*"([^"]*)"\s*\)', text)
    }


def test_env_example_documents_the_defaults_the_code_uses():
    examples = _commented_examples()
    defaults = _literal_defaults()

    drift = [
        f"{name}: .env.example says {examples[name]!r}, settings.py defaults to {defaults[name]!r}"
        for name in sorted(examples)
        if name in defaults and name not in _EXEMPT and examples[name] != defaults[name]
    ]

    assert not drift, "\n".join([".env.example has drifted from settings.py:", *drift])


def test_the_scan_actually_finds_something():
    """
    Guard the guard.

    Both sides are parsed with regexes over source text. If either stopped
    matching -- a reformat, a move to a helper -- the drift check above would
    pass by comparing nothing at all, and go on passing forever.
    """
    examples = _commented_examples()
    defaults = _literal_defaults()

    assert len(examples) >= 10, f"only {len(examples)} examples parsed from .env.example"
    # Named settings rather than a count: most of settings.py takes its default from
    # a _DEFAULT_* constant, which this scan deliberately skips, so the number of
    # literal defaults is small and moves whenever one setting changes style. Both of
    # these have a literal default and a documented example; losing either from one
    # side means that side's scan broke.
    known = {"DEFAULT_GENERATION_TIMEOUT", "MAX_OUTPUT_TOKENS_CAP"}
    assert known <= set(defaults), f"scan missed {sorted(known - set(defaults))} in settings.py"
    assert known <= set(examples), f"scan missed {sorted(known - set(examples))} in .env.example"


def test_the_timeouts_that_drifted_are_documented():
    """
    The specific pair this test exists for.

    DEFAULT_GENERATION_TIMEOUT went 300 -> 1800 and .env.example kept saying 300.
    VLLM_HEALTH_TIMEOUT went 300 -> 1800 and .env.example did not mention it at
    all, so there was nothing to contradict and nothing to find.
    """
    from service import settings

    examples = _commented_examples()

    assert examples.get("DEFAULT_GENERATION_TIMEOUT") == str(settings.DEFAULT_GENERATION_TIMEOUT)
    assert "VLLM_HEALTH_TIMEOUT" in examples, "the health timeout is undocumented"
    assert float(examples["VLLM_HEALTH_TIMEOUT"]) == settings.VLLM_HEALTH_TIMEOUT
