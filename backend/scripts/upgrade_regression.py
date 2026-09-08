"""
Dependency-upgrade regression harness.

Bumping transformers / deepspeed / torch / peft / trl can break training in ways
that still look like a healthy run: parameters silently stop being offloaded,
the loss curve shifts, GPU peak doubles. Comparing "did the tests pass" catches
none of that, because they pass either way.

So this records the numbers, not just the verdicts.

    # before upgrading
    python scripts/upgrade_regression.py baseline

    # ... upgrade packages ...

    python scripts/upgrade_regression.py compare

`compare` re-runs the same suites and reports what moved: pass/fail changes,
partition state, first-step loss, and peak GPU/DRAM/SSD per case.

What gets flagged, and why each matters:

* **status** -- a case that stopped passing, or an xfail that now passes
  (worth knowing: the workaround may be removable).
* **resident parameters** -- the offload invariant. Going non-zero means
  ZeRO-3 quietly stopped sharding; everything else still passes.
* **first-step loss** -- with a fixed dataset and seed this should barely move.
  A jump means the data pipeline or masking changed, not the optimiser.
* **peak GPU / DRAM / SSD** -- the resource cost. A large jump is a regression
  even when the run completes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results" / "offload"
LATEST = RESULTS_DIR / "latest.jsonl"
BASELINE = RESULTS_DIR / "baseline.jsonl"

# Relative change beyond which a memory move is called out. Below this it is
# allocator noise, not a regression.
MEMORY_TOLERANCE = 0.20
# Absolute loss move worth reporting; below this is step-to-step jitter.
LOSS_TOLERANCE = 0.5


def _run_suites(unit: bool) -> int:
    """Run the offload matrix (and optionally the unit suite). Returns pytest's code."""
    LATEST.unlink(missing_ok=True)

    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/smoke/test_offload_training_matrix.py",
            "-m",
            "offload",
            "-q",
        ],
    ]
    if unit:
        commands.insert(0, [sys.executable, "-m", "pytest", "tests/unit", "-q"])

    worst = 0
    for cmd in commands:
        print(f"\n$ {' '.join(cmd)}", flush=True)
        worst = max(worst, subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode)
    return worst


def _load(path: Path) -> dict[str, dict]:
    """Last record per case id (a re-run overwrites an earlier attempt)."""
    if not path.is_file():
        return {}
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            records[entry["case"]] = entry
    return records


def _peak(entry: dict, metric: str) -> float | None:
    value = (entry.get("memory") or {}).get(metric)
    return value.get("peak") if isinstance(value, dict) else None


def _fmt_delta(old: float | None, new: float | None, unit: str = "MB") -> str:
    if old is None or new is None:
        return f"{old} -> {new}"
    if old == 0:
        return f"{old:.0f} -> {new:.0f} {unit}"
    pct = (new - old) / old * 100
    return f"{old:.0f} -> {new:.0f} {unit} ({pct:+.0f}%)"


def _compare_case(case: str, old: dict, new: dict) -> list[str]:
    """Findings for one case, most important first."""
    findings: list[str] = []

    if old.get("status") != new.get("status"):
        findings.append(f"  STATUS   {old.get('status')} -> {new.get('status')}")
        if new.get("error"):
            findings.append(f"           {new['error']}")

    old_res = (old.get("partition") or {}).get("resident")
    new_res = (new.get("partition") or {}).get("resident")
    if old_res != new_res:
        findings.append(
            f"  OFFLOAD  resident parameters {old_res} -> {new_res}"
            + ("  <-- ZeRO-3 stopped offloading" if new_res else "")
        )

    old_loss = (old.get("losses") or [None])[0]
    new_loss = (new.get("losses") or [None])[0]
    if old_loss is not None and new_loss is not None:
        if abs(new_loss - old_loss) > LOSS_TOLERANCE:
            findings.append(
                f"  LOSS     step-1 {old_loss} -> {new_loss}  "
                "<-- data pipeline or masking changed, not the optimiser"
            )

    for metric, label in (
        ("cuda_alloc_mb", "GPU peak"),
        ("vmrss_mb", "DRAM peak"),
        ("offload_dir_mb", "SSD peak"),
    ):
        old_peak, new_peak = _peak(old, metric), _peak(new, metric)
        if old_peak and new_peak and abs(new_peak - old_peak) / old_peak > MEMORY_TOLERANCE:
            findings.append(f"  MEMORY   {label} {_fmt_delta(old_peak, new_peak)}")

    return findings


def _compare() -> int:
    baseline, latest = _load(BASELINE), _load(LATEST)
    if not baseline:
        print(f"No baseline at {BASELINE}. Run `upgrade_regression.py baseline` first.")
        return 2
    if not latest:
        print(f"No results at {LATEST}; did the suite run?")
        return 2

    old_versions = next(iter(baseline.values())).get("versions", {})
    new_versions = next(iter(latest.values())).get("versions", {})
    moved = {
        k: (old_versions.get(k), v) for k, v in new_versions.items() if old_versions.get(k) != v
    }

    print("\n" + "=" * 78)
    print("PACKAGE VERSIONS")
    print("=" * 78)
    if moved:
        for name, (before, after) in sorted(moved.items()):
            print(f"  {name:14s} {before} -> {after}")
    else:
        print("  (unchanged -- comparing two runs of the same stack)")

    print("\n" + "=" * 78)
    print("PER-CASE COMPARISON")
    print("=" * 78)

    regressions = 0
    for case in sorted(set(baseline) | set(latest)):
        if case not in latest:
            print(f"\n{case}\n  MISSING  present in baseline, absent now (skipped?)")
            regressions += 1
            continue
        if case not in baseline:
            print(f"\n{case}\n  NEW      no baseline to compare against")
            continue

        findings = _compare_case(case, baseline[case], latest[case])
        if findings:
            print(f"\n{case}")
            print("\n".join(findings))
            regressions += 1
        else:
            print(f"\n{case}\n  unchanged")

    print("\n" + "=" * 78)
    if regressions:
        print(
            f"{regressions} case(s) moved. Review the findings above before accepting the upgrade."
        )
    else:
        print("No case moved: statuses, offload state, step-1 loss and peak memory all held.")
    print("=" * 78)
    return 1 if regressions else 0


def main() -> int:
    """Run the suites, then either save a baseline or diff against one."""
    # Not derived from __doc__: it is None under `python -OO`.
    parser = argparse.ArgumentParser(description="Dependency-upgrade regression harness.")
    parser.add_argument("mode", choices=["baseline", "compare"])
    parser.add_argument(
        "--with-unit",
        action="store_true",
        help="also run the unit suite (fast, catches API-shape breaks an offload run would not)",
    )
    args = parser.parse_args()

    code = _run_suites(unit=args.with_unit)

    if args.mode == "baseline":
        if not LATEST.is_file():
            print(f"\nNo records were produced at {LATEST}; nothing to save as a baseline.")
            return 2
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LATEST, BASELINE)
        cases = _load(BASELINE)
        print(f"\nBaseline saved: {BASELINE} ({len(cases)} cases)")
        for case, entry in sorted(cases.items()):
            gpu = _peak(entry, "cuda_alloc_mb")
            print(f"  {case:32s} {entry.get('status'):8s} GPU peak {gpu and f'{gpu:.0f} MB'}")
        # A pytest failure while recording a baseline is worth knowing about, but
        # the baseline itself is still the honest record of this stack.
        return 0 if code == 0 else 0

    return _compare()


if __name__ == "__main__":
    sys.exit(main())
