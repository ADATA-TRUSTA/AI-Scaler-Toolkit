#!/usr/bin/env bash
# Run the full test suite tier by tier and store results locally.
#
# Results land in tests/results/<timestamp>/ (gitignored — never uploaded):
#   unit.xml / smoke.xml / e2e.xml   JUnit reports per tier
#   <tier>.log                       full pytest output per tier
#   summary.md                       aggregated pass/fail/skip table
#
# Tiers:
#   unit   CI-safe, no external resources
#   smoke  real model on a real GPU (auto-skips without CUDA / model)
#   e2e    backend subprocess + managed llama-server (auto-skips likewise)
#
# Usage:
#   bash tests/run_tests.sh              # all tiers
#   bash tests/run_tests.sh unit         # a single tier
#   pytest -m inference|finetune|models  # tag-filtered runs (see README)
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_ROOT/tests/results/$STAMP"
mkdir -p "$OUT_DIR"

TIERS=("$@")
if [ ${#TIERS[@]} -eq 0 ]; then
    TIERS=(unit smoke e2e)
fi

declare -A TIER_PATHS=(
    [unit]="tests/unit"
    [smoke]="tests/smoke"
    [e2e]="tests/e2e"
)

overall_rc=0
for tier in "${TIERS[@]}"; do
    path="${TIER_PATHS[$tier]:-}"
    if [ -z "$path" ]; then
        echo "unknown tier: $tier (expected: unit|smoke|e2e)" >&2
        exit 2
    fi
    echo "=== running $tier ($path) ==="
    "$PY" -m pytest "$REPO_ROOT/$path" -q \
        --junitxml="$OUT_DIR/$tier.xml" \
        2>&1 | tee "$OUT_DIR/$tier.log"
    rc=${PIPESTATUS[0]}
    # pytest exit 5 = no tests collected (e.g. everything skipped) -> not a failure
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
        overall_rc=1
    fi
done

"$PY" - "$OUT_DIR" <<'EOF'
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
failures = []
for xml_path in sorted(out_dir.glob("*.xml")):
    tier = xml_path.stem
    root = ET.parse(xml_path).getroot()
    suites = root.iter("testsuite")
    total = errors = failed = skipped = 0
    for s in suites:
        total += int(s.get("tests", 0))
        errors += int(s.get("errors", 0))
        failed += int(s.get("failures", 0))
        skipped += int(s.get("skipped", 0))
    passed = total - errors - failed - skipped
    rows.append((tier, total, passed, failed + errors, skipped))
    for case in root.iter("testcase"):
        for bad in list(case.iter("failure")) + list(case.iter("error")):
            failures.append(
                (tier, f"{case.get('classname')}::{case.get('name')}",
                 (bad.get("message") or "")[:200])
            )

lines = [
    f"# Test Run Summary — {datetime.now().isoformat(timespec='seconds')}",
    "",
    "| Tier | Total | Passed | Failed | Skipped |",
    "|------|------:|-------:|-------:|--------:|",
]
for tier, total, passed, failed, skipped in rows:
    lines.append(f"| {tier} | {total} | {passed} | {failed} | {skipped} |")
if failures:
    lines += ["", "## Failures", ""]
    for tier, name, msg in failures:
        lines.append(f"- **[{tier}]** `{name}` — {msg}")
else:
    lines += ["", "No failures."]
summary = out_dir / "summary.md"
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nsummary written to {summary}")
print("\n".join(lines))
EOF

exit $overall_rc
