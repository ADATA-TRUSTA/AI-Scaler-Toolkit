"""
Build a llama-server recommendation table for every GGUF model in a directory.

Walks a HuggingFace hub cache (or any tree of .gguf files) and, for each model, asks
GgufMemoryEstimator.plan() what to run at each GPU budget with each KV cache type. The
output is markdown: one section per model, one row per (budget, KV type) candidate.

Companion files are skipped: mmproj-*.gguf is a vision projector rather than a model, and
only the first shard of a multi-part model is opened (the reader follows the rest itself).

    python scripts/gguf_recommend_report.py <dir> --gpu 5 10 15 --margin 512
    python scripts/gguf_recommend_report.py <dir> --no-verify -o report.md

Verification runs llama-fit-params once per candidate, which loads the model header with
no_alloc, so a 60 GiB model costs a second rather than its size. --no-verify falls back to
analytic figures, whose compute-buffer term errs high by up to ~25%.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.inference.gguf_estimator import DEFAULT_KV_VARIANTS, gguf_memory_estimator

SKIP_DIRS = {".cache", ".locks", "blobs"}
SKIP_PREFIXES = ("mmproj",)


def find_models(root: Path) -> list[Path]:
    """Every standalone GGUF model under ``root``, largest first."""
    found: list[tuple[int, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".gguf") or name.startswith(SKIP_PREFIXES):
                continue
            # Sharded models are addressed by their first part only.
            if "-of-" in name and "-00001-of-" not in name:
                continue
            path = Path(dirpath) / name
            try:
                found.append((path.stat().st_size, path))
            except OSError:
                continue
    return [p for _, p in sorted(found, key=lambda t: -t[0])]


def label(path: Path, root: Path) -> str:
    """Repo/quantization name for a hub-cached file, or the bare filename."""
    parts = path.relative_to(root).parts
    if parts[0].startswith("models--") and "snapshots" in parts:
        return f"{parts[0].removeprefix('models--').replace('--', '/')} :: {path.name}"
    return str(path.relative_to(root))


def describe(info: dict) -> str:
    """One-line summary of what kind of model this is."""
    parts = [
        f"{info['architecture']} / {'MoE' if info['is_moe'] else 'dense'}",
        f"{info['n_block']} blocks",
        f"{info['file_size_mib'] / 1024:.1f} GiB on disk",
        f"trained context {info['n_ctx_train']}",
    ]
    if info["is_moe"]:
        parts.append(f"{info['n_expert']} experts ({info['n_expert_used']} used)")
    if info["n_swa_layers"]:
        swa = f"SWA on {info['n_swa_layers']} layers"
        if info["key_length_swa"] != info["key_length"]:
            swa += f" with a {info['key_length_swa']}-wide head"
        parts.append(swa)
    if info["n_layer_kv"] < info["n_block"]:
        parts.append(f"KV cache on only {info['n_layer_kv']} layers")
    if info["tied_embedding"]:
        extra = info["weights_mib"] - info["file_size_mib"]
        parts.append(f"tied embeddings (+{extra:.0f} MiB output copy)")
    return ", ".join(parts)


def mmproj_note(path: Path) -> str | None:
    """Flag a sibling vision projector, whose VRAM this report does not include."""
    projectors = sorted(path.parent.glob("mmproj*.gguf"))
    if not projectors:
        return None
    sizes = ", ".join(f"{p.name} {p.stat().st_size / 2**30:.1f} GiB" for p in projectors)
    return (
        f"Multimodal: {sizes} sits next to this model. --mmproj loads it on top of the "
        f"figures below, so subtract its size from the budget if you need vision."
    )


def render(name: str, result: dict, path: Path) -> list[str]:
    """One markdown section for a single model."""
    lines = [f"## {name}", ""]
    if "error" in result:
        lines += [f"> unreadable: {result['error']}", ""]
        return lines

    lines += [
        f"- {describe(result['model_info'])}",
        "",
        "| GPU | KV | args | VRAM | util | ctx | weights on GPU | limited by | DRAM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for plan in result["plans"]:
        gib = plan["gpu_budget_mib"] / 1024
        if not plan["fits"]:
            reason = next((c.get("reason", "") for c in plan["candidates"]), "")
            lines.append(f"| {gib:.0f} GiB | - | does not fit | - | - | - | - | - | {reason} |")
            continue
        for candidate in plan["candidates"]:
            if not candidate["fits"]:
                continue
            flag = " (no gain)" if candidate["dominated"] else ""
            lines.append(
                f"| {gib:.0f} GiB | {candidate['kv_quant']}{flag} "
                f"| `{candidate['llama_server_args']}` "
                f"| {candidate['gpu_mib']:.0f} MiB "
                f"| {candidate['utilization_pct']:.0f}% "
                f"| {candidate['n_ctx']} "
                f"| {candidate['gpu_weight_pct']:.0f}% ({candidate['blocks_on_gpu']}"
                f"/{candidate['n_block']} blk) "
                f"| {candidate['constraint']} "
                f"| {candidate['host_mib']:.0f} MiB |"
            )
    lines.append("")
    notes = list(result["notes"])
    projector = mmproj_note(path)
    if projector:
        notes.append(projector)
    lines += [f"> {note}" for note in notes]
    if notes:
        lines.append("")
    return lines


def main() -> int:
    """Scan a directory and print the recommendation table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to scan for .gguf files")
    parser.add_argument(
        "--gpu",
        type=float,
        nargs="+",
        default=[5, 10, 15],
        help="GPU budgets in GiB (default: 5 10 15)",
    )
    parser.add_argument(
        "--kv",
        nargs="+",
        default=list(DEFAULT_KV_VARIANTS),
        help=f"KV cache types to offer (default: {' '.join(DEFAULT_KV_VARIANTS)})",
    )
    parser.add_argument("--margin", type=float, default=512, help="MiB to leave free (default 512)")
    parser.add_argument("--ctx-min", type=int, default=4096, help="Lowest context to accept")
    parser.add_argument(
        "--ctx-max",
        type=int,
        default=0,
        help="Highest context to grow to; 0 uses the model's trained length",
    )
    parser.add_argument("--ubatch", type=int, default=512, help="Physical batch size (-ub)")
    parser.add_argument("--no-verify", action="store_true", help="Analytic figures only")
    parser.add_argument("-o", "--output", type=Path, help="Write markdown here instead of stdout")
    args = parser.parse_args()

    models = find_models(args.root)
    budgets = [g * 1024 for g in args.gpu]
    print(f"{len(models)} models, {len(budgets)} budgets, {len(args.kv)} KV types", file=sys.stderr)

    lines = [
        f"# llama-server recommendations for `{args.root}`",
        "",
        f"GPU budgets: {', '.join(f'{g:g} GiB' for g in args.gpu)} "
        f"(minus a {args.margin:.0f} MiB margin) - KV cache types: {', '.join(args.kv)} - "
        f"-ub {args.ubatch} - "
        + ("analytic estimates" if args.no_verify else "verified with llama-fit-params"),
        "",
    ]
    for path in models:
        name = label(path, args.root)
        started = time.monotonic()
        result = gguf_memory_estimator.plan(
            str(path),
            gpu_budgets_mib=budgets,
            kv_cache_types=args.kv,
            margin_mib=args.margin,
            n_ctx_min=args.ctx_min,
            n_ctx_max=args.ctx_max,
            n_ubatch=args.ubatch,
            verify=not args.no_verify,
        )
        print(f"  {name}: {time.monotonic() - started:.1f}s", file=sys.stderr)
        lines += render(name, result, path)

    text = "\n".join(lines)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
