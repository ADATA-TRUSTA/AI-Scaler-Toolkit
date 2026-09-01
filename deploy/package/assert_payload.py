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
    for required in ("backend/pyproject.toml", "backend/scripts", "console/pyproject.toml"):
        if not (root / required).exists():
            failures.append(f"payload is missing {required}")

    # The installer creates a shortcut to this. Shipping sources instead builds an
    # installer whose shortcut points at nothing -- and it installs cleanly first.
    exe = "AIScalerToolkit.exe" if args.platform == "windows" else "AIScalerToolkit"
    if not (root / exe).is_file() and not (root / "launcher" / "__main__.py").is_file():
        failures.append(f"payload has neither {exe} nor launcher sources")
    elif not (root / exe).is_file():
        failures.append(f"payload ships launcher sources but no {exe}; run build_launcher.py first")

    if failures:
        print("[assert-payload] FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1

    print(f"[assert-payload] ok ({args.platform}, port {args.port})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
