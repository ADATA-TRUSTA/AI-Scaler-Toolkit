# Contributing to AI Scaler Toolkit

Thanks for taking the time to contribute! This document explains how to set up a
development environment, the coding and commit conventions we follow, and how to
open a good pull request.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Commit Message Convention](#commit-message-convention)
- [Pull Requests](#pull-requests)
- [Reporting Bugs & Requesting Features](#reporting-bugs--requesting-features)
- [License](#license)

## Ways to Contribute

- Report bugs and request features via [Issues](../../issues)
- Improve documentation (English and 繁體中文 docs live under `docs/`)
- Fix bugs or implement features via pull requests
- Add or improve tests under `backend/tests/`

## Development Setup

The project targets **Python 3.12+** and uses the setup scripts under
`scripts/`. See [docs/installation.en.md](../docs/installation.en.md) for the
full walkthrough. In short:

### Linux

```bash
cp .env.example .env
# Edit .env (HF_HOME, LOG_DIR, SERVICE_HOST, SERVICE_PORT)
TRUSTA_ACCEL=cuda bash backend/scripts/linux/setup_env.sh
bash backend/scripts/linux/run_service.sh
```

### Windows

```powershell
Copy-Item .env.example .env
notepad .env
.\backend\scripts\windows\setup_env.ps1 -Accel xpu
.\backend\scripts\windows\run_service.bat
```

> **Note**: Fine-tuning currently supports **Linux + CUDA** only.

## Running Tests

Tests use `pytest` (configured in `backend/pytest.ini`, `testpaths = tests`, run from `backend/`):

```bash
cd backend && pytest
```

Please make sure the test suite passes before opening a pull request, and add
tests when you fix a bug or add a feature.

## Commit Message Convention

This project follows the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification. Each commit message must be structured as:

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

### Allowed types

| Type       | Purpose                                                        |
| ---------- | -------------------------------------------------------------- |
| `feat`     | A new feature                                                  |
| `fix`      | A bug fix                                                      |
| `docs`     | Documentation-only changes                                     |
| `style`    | Formatting only (no code behavior change)                      |
| `refactor` | Code change that neither fixes a bug nor adds a feature        |
| `perf`     | A change that improves performance                             |
| `test`     | Adding or correcting tests                                     |
| `build`    | Changes to the build system or dependencies                   |
| `ci`       | Changes to CI configuration and scripts                        |
| `chore`    | Other changes that don't modify src or test files             |

### Rules

- Use the imperative mood in the description ("add", not "added" or "adds").
- Keep the description concise; do not end it with a period.
- Scope is optional but encouraged, e.g. `feat(scripts):`, `fix(app):`,
  `docs:` — mirror the scopes already used in the git history.
- A breaking change must be flagged with a `!` after the type/scope
  (e.g. `feat(app)!:`) and/or a `BREAKING CHANGE:` footer.

### Examples

```
feat(scripts): fetch llama.cpp at setup time (opt-in)
fix(app): resolve frontend dist via candidate paths
docs: slim README and split details into docs/
chore: rename deploy/ -> scripts/
```

## Pull Requests

1. Fork the repository and create your branch from `main`.
2. Make your changes, keeping commits atomic and following the commit
   convention above.
3. Ensure `pytest` passes and update documentation where relevant.
4. Open a pull request and fill out the
   [pull request template](PULL_REQUEST_TEMPLATE.md). Use a Conventional
   Commits-style title (e.g. `feat: ...`, `fix: ...`).
5. Link any related issues (e.g. `Closes #123`).

A maintainer will review your pull request and may request changes before
merging.

## Reporting Bugs & Requesting Features

Please use the [issue templates](ISSUE_TEMPLATE) so we get the details we need:

- **Bug report** — steps to reproduce, expected vs. actual behavior, and
  environment (OS, accelerator, Python version).
- **Feature request** — the problem you're trying to solve and your proposed
  solution.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](../LICENSE) that covers this project.
