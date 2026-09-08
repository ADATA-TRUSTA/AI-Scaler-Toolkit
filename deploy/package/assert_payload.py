"""Check the payload actually carries the brand it is supposed to.

Every one of these catches a swap that produces a working installer -- one that
starts, serves a UI, and answers the API, while showing the wrong brand or offering
an engine the platform cannot run. None of them would surface in a smoke test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402
import make_icon  # noqa: E402

EXPECTED = {
    "windows": {
        "title": "ADATA-AST",
        # vLLM has no Windows wheels; offering it would be a dead menu entry.
        "vllm_enabled": False,
    },
    "linux": {
        "title": "TRUSTA-AST",
        "vllm_enabled": True,
    },
}

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def check_health_fingerprint(root: Path) -> list[str]:
    """The launcher recognises its own service by the keys /health returns.

    If the backend stops returning them the launcher calls its own service foreign:
    it polls for fifteen minutes, gives up, and the tray says the port is taken by
    another program -- while the service sits there answering perfectly. Nothing about
    that points at the cause.

    This happened. A payload was assembled from a source tree whose backend predated
    the keys, and everything else about it was fine: it built, it started, it served.
    Comparing the two ends here costs a syntax tree and catches it before it ships.
    """
    import ast

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from launcher.service_manager import FINGERPRINT
    except ImportError as exc:
        return [f"could not read the launcher's health fingerprint: {exc}"]

    app_py = root / "backend" / "service" / "app.py"
    if not app_py.is_file():
        return [f"payload has no {app_py.relative_to(root)}"]

    try:
        tree = ast.parse(app_py.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"backend service/app.py does not parse: {exc}"]

    def decorates_health(node: ast.AST) -> bool:
        for decorator in getattr(node, "decorator_list", []):
            if (isinstance(decorator, ast.Call)
                    and any(isinstance(a, ast.Constant) and a.value == "/health"
                            for a in decorator.args)):
                return True
        return False

    handlers = [n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and decorates_health(n)]
    if not handlers:
        return ["backend exposes no /health route; the launcher would never see it start"]

    returned: set[str] = set()
    for handler in handlers:
        for node in ast.walk(handler):
            if isinstance(node, ast.Dict):
                returned.update(k.value for k in node.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str))

    missing = [key for key in FINGERPRINT if key not in returned]
    if missing:
        return [
            "backend /health does not return "
            f"{', '.join(missing)}, which the launcher uses to recognise its own "
            "service. It would wait for the full timeout and then report the port as "
            "taken by another program. Check the payload was built from the assembled "
            "layout rather than a stale source tree."
        ]
    return []


def check_ui_scripts(root: Path) -> list[str]:
    """Parse the launcher's own pages, because a broken one fails silently.

    These pages have no build step and no test that loads them. A syntax error does
    not raise anything a human sees: the browser abandons the script, the page renders
    its loading state, and every control on it does nothing. That shipped once -- an
    escaped newline collapsed into a real one inside a string literal -- and it was
    only found by reading the file. A parser catches it in a second.

    Skipped rather than failed when no JavaScript runtime is around, so this cannot
    become the reason a build fails on a machine that is otherwise fine.
    """
    import shutil
    import subprocess
    import tempfile

    runtime = shutil.which("node") or shutil.which("bun")
    # Searched rather than addressed: freezing moves the pages under _internal, so a
    # fixed path finds nothing in a real payload and passes without checking anything.
    # Finding none is a failure, not a skip -- a payload without them has no interface.
    pages = sorted(root.rglob("launcher/ui/*.html"))
    if not pages:
        return ["payload carries no launcher/ui pages; the launcher would have no interface"]
    if runtime is None:
        print("[assert-payload] no node or bun on PATH; skipping the UI script check")
        return []

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as work:
        for page in pages:
            body = re.search(r"<script[^>]*>(.*)</script>",
                             page.read_text(encoding="utf-8"), re.S)
            if body is None:
                continue
            scratch = Path(work) / f"{page.stem}.js"
            scratch.write_text(body.group(1), encoding="utf-8")
            result = subprocess.run([runtime, "--check", str(scratch)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                head = next((line for line in detail if "Error" in line), detail[0] if detail else "")
                problems.append(f"{page.name} does not parse: {head}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--payload", default="payload")
    parser.add_argument("--port", type=int, required=True, help="SERVICE_PORT the launcher will write")
    args = parser.parse_args()

    expected = EXPECTED[args.platform]
    dist = Path(args.payload).resolve() / "frontend" / "dist"
    failures: list[str] = []

    index = dist / "index.html"
    if not index.is_file():
        failures.append("frontend/dist/index.html is missing")
    else:
        match = TITLE_RE.search(index.read_text(encoding="utf-8", errors="replace"))
        title = match.group(1).strip() if match else "<none>"
        if title != expected["title"]:
            failures.append(f"<title> is {title!r}, expected {expected['title']!r} -- wrong frontend package")

    config_path = dist / "config.json"
    if not config_path.is_file():
        failures.append("frontend/dist/config.json is missing")
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))

        has_vllm = "vllm" in config.get("enabledEngines", [])
        if has_vllm != expected["vllm_enabled"]:
            failures.append(
                f"enabledEngines {'contains' if has_vllm else 'lacks'} vllm, "
                f"expected the opposite for {args.platform}"
            )

        api = str(config.get("apiBaseUrl", ""))
        if not api.endswith(f":{args.port}"):
            failures.append(f"apiBaseUrl is {api!r}, which does not use SERVICE_PORT {args.port}")

        if config.get("hideChatPanel") is not True:
            failures.append("hideChatPanel is not true; the desktop window would show the full chat UI")

    root = Path(args.payload).resolve()
    for required in ("backend/pyproject.toml", "backend/scripts"):
        if not (root / required).exists():
            failures.append(f"payload is missing {required}")

    # The console is part of the product and stays in the public layout, but it has no
    # place in an installation. Nothing here runs it, and nothing builds an environment
    # for it -- only the backend is synced -- so what shipped was 67 files that could
    # not be run, in a directory a user might reasonably try to run.
    #
    # Asserted rather than simply not copied. Removing something is only half of it:
    # a stray entry in PAYLOAD_DIRS, or a layout copied wholesale by a future change,
    # would put it back with nothing objecting.
    if (root / "console").exists():
        failures.append("payload contains console/, which is not installed and cannot run there")

    # Nothing the layout rules exclude may reach a payload. Assembly applies them, but
    # the payload is a directory on a build machine between then and packaging, and
    # anything run against it writes there: a launcher started from build/payload-windows
    # writes backend/.env, and that file carries absolute paths -- an HF_HOME and a
    # LOG_DIR under one developer's home directory. Two installers were built carrying
    # exactly that before anything noticed, because the only check on the payload's
    # contents was a list of things that had to be present.
    strays = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "backend").rglob("*")
        if path.is_file()
        and layout.is_excluded(path.relative_to(root / "backend").as_posix(), "backend")
    )
    if strays:
        shown = ", ".join(strays[:5])
        more = f" and {len(strays) - 5} more" if len(strays) > 5 else ""
        failures.append(
            f"payload carries {len(strays)} file(s) the layout excludes: {shown}{more}. "
            "Something has been run against this payload; rebuild it with make_payload.py."
        )

    # The installer creates a shortcut to this. Shipping sources instead builds an
    # installer whose shortcut points at nothing -- and it installs cleanly first.
    exe = "AIScalerToolkit.exe" if args.platform == "windows" else "AIScalerToolkit"
    if not (root / exe).is_file() and not (root / "launcher" / "__main__.py").is_file():
        failures.append(f"payload has neither {exe} nor launcher sources")
    elif not (root / exe).is_file():
        failures.append(f"payload ships launcher sources but no {exe}; run build_launcher.py first")

    icon = root / "app.ico"
    if not icon.is_file():
        failures.append("payload has no app.ico")
    else:
        sizes = make_icon.ico_sizes(icon)
        # Windows draws the desktop shortcut at 48 and Explorer's largest view at 256.
        # An icon missing those is upscaled from whatever is nearest, which is exactly
        # how a 16x16 favicon shipped as the application icon and nothing objected.
        missing = [s for s in (16, 32, 48, 256) if s not in sizes]
        if missing:
            failures.append(
                f"app.ico is missing the {', '.join(str(s) for s in missing)}px "
                f"entr{'y' if len(missing) == 1 else 'ies'} (has {sizes}); it would be "
                "upscaled wherever Windows asks for one"
            )

    failures.extend(check_ui_scripts(root))
    failures.extend(check_health_fingerprint(root))

    if failures:
        print("[assert-payload] FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1

    print(f"[assert-payload] ok ({args.platform}, port {args.port})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
