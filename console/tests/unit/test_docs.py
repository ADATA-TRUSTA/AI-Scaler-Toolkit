"""
The README has to keep describing the repo that actually exists.

Path rot in documentation is invisible: nothing imports a README, so a rename
leaves it quietly wrong. These checks caught two real regressions already --
a stale `src/` path after the layout flatten, and references to scripts that a
cleanup commit had deleted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path("README.md")
PACKAGE = Path("ai_scaler_console")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_every_documented_script_exists(readme: str) -> None:
    scripts = sorted(set(re.findall(r"uv run ((?:examples|internal)/[\w/.]+\.py)", readme)))

    assert len(scripts) >= 8, f"expected the README to walk through the examples, got {scripts}"
    missing = [s for s in scripts if not Path(s).is_file()]
    assert not missing, f"README points at scripts that do not exist: {missing}"


def test_every_referenced_file_exists(readme: str) -> None:
    paths = sorted(
        set(
            re.findall(
                r"(?<![\w/])((?:configs|examples|docs|ai_scaler_console)/[\w/.\-]+\.(?:json|py|md|jpg))",
                readme,
            )
        )
    )

    assert paths, "no file references found; the regex probably stopped matching"
    missing = [p for p in paths if not Path(p).exists()]
    assert not missing, f"README references missing files: {missing}"


def test_every_documented_config_flag_resolves(readme: str) -> None:
    values = sorted(set(re.findall(r"--(?:settings|config|workflow-config) (\S+\.json)", readme)))

    assert values, "no --settings/--config examples found"
    missing = [v for v in values if not Path(v).is_file()]
    assert not missing, f"README passes configs that do not exist: {missing}"


# Kept as patterns, not as the literal values that used to sit here: this list ships
# in the public snapshot, so naming an internal subnet or a developer's home directory
# would publish exactly what the check exists to keep out. The patterns are also
# stricter -- they catch the next person's machine, not just the last one's.
# 127.0.0.0/8 is deliberately not matched; the README documents localhost URLs.
@pytest.mark.parametrize(
    ("pattern", "what"),
    [
        (
            r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
            "a private-network address",
        ),
        (r"/home/[a-z][\w.-]*", "a developer's home directory"),
        (r"[Cc]:\\Users\\[A-Za-z]", "a Windows user profile path"),
        (r"your-api-key", "a placeholder credential"),
        (r"verify=False", "TLS verification switched off"),
    ],
)
def test_readme_leaks_no_local_details(readme: str, pattern: str, what: str) -> None:
    assert not re.search(pattern, readme), f"README contains {what} (matched {pattern!r})"


def test_project_layout_matches_the_package(readme: str) -> None:
    layout = readme.split("## 4. Project Layout")[1].split("## 5.")[0]
    documented = {a or b for a, b in re.findall(r"\|-- (\w+\.py)|`-- (\w+\.py)", layout)}
    on_disk = {p.name for p in PACKAGE.rglob("*.py")} - {"__init__.py"}

    assert not on_disk - documented, f"undocumented modules: {sorted(on_disk - documented)}"
    assert not documented - on_disk, (
        f"layout lists modules that do not exist: {sorted(documented - on_disk)}"
    )


def test_env_example_documents_every_variable_settings_reads(readme: str) -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")
    settings_src = (PACKAGE / "settings.py").read_text(encoding="utf-8")
    read_vars = set(re.findall(r'os\.getenv\("(\w+)"', settings_src))

    assert read_vars, "settings.py stopped reading env vars; this test needs updating"

    undocumented_in_env = {v for v in read_vars if v not in env_example}
    assert not undocumented_in_env, f"undocumented in .env.example: {sorted(undocumented_in_env)}"

    undocumented_in_readme = {v for v in read_vars if v not in readme}
    assert not undocumented_in_readme, f"undocumented in README: {sorted(undocumented_in_readme)}"


def test_workflow_guide_paths_exist() -> None:
    guide = Path("docs/WORKFLOW_CONFIG_GUIDE.md").read_text(encoding="utf-8")
    paths = set(re.findall(r"((?:configs|examples)/[\w/.\-]+\.(?:json|py))", guide))
    # custom.json is a file the guide tells the reader to create.
    missing = [p for p in paths if not Path(p).exists() and "custom" not in p]

    assert not missing, f"workflow guide references missing files: {missing}"
