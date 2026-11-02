#!/usr/bin/env python
"""Post-install check: does this box actually have everything the skills need?

Run it after installing and after downloading models. It resolves every path
the way the engine does, then checks the things that are silently fatal later:
a ComfyUI that is not where the config says, a missing GGUF custom node, model
files in the wrong models/ subdirectory, no nvidia-smi, a Hermes terminal
timeout too short for a render.

    python tools/doctor.py                # check everything
    python tools/doctor.py --models       # only the model manifest
    python tools/doctor.py --trim-selftest  # also probe the LLM working-set trim

Exit code is 0 when nothing is fatal, 1 otherwise. Warnings never fail the run:
missing models are a WARN, because you may have deliberately installed only the
image models or only the video ones.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _engine_dir(argv: list[str]) -> Path:
    """The engine to inspect -- the INSTALLED one, not the pack's copy.

    Checking the pack would make this tool useless: it would report the
    shipped defaults and never see the hermes-comfy.conf the installer wrote,
    so a genuinely broken install would look clean.
    """
    for i, a in enumerate(argv):
        if a == "--engine" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if a.startswith("--engine="):
            return Path(a.split("=", 1)[1]).expanduser()
    env = os.environ.get("HERMES_COMFY_ENGINE", "").strip()
    if env and (Path(env) / "paths.py").is_file():
        return Path(env)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        home = Path(os.environ.get("HERMES_HOME") or Path(local) / "hermes")
    else:
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    for cand in (home / "comfy-engine",
                 Path.home() / ".hermes-comfy" / "engine"):
        if (cand / "paths.py").is_file():
            return cand
    return ROOT / "engine"  # not installed yet -- check the pack itself


ENGINE = _engine_dir(sys.argv[1:])
sys.path.insert(0, str(ENGINE))

FATAL: list[str] = []
WARN: list[str] = []

_TICK = {"ok": "  [ok]  ", "warn": "  [WARN]", "fail": "  [FAIL]"}


def say(state: str, msg: str) -> None:
    print(_TICK[state] + " " + msg)
    if state == "fail":
        FATAL.append(msg)
    elif state == "warn":
        WARN.append(msg)


def head(t: str) -> None:
    print("\n" + t)
    print("-" * len(t))


# Which models/ subdirectory each loader reads from. ComfyUI resolves these
# itself, so a file in the wrong one is invisible with a confusing error.
LOADER_DIRS = {
    "UNETLoader": "diffusion_models",
    "UnetLoaderGGUF": "unet",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "VAELoader": "vae",
    "CheckpointLoaderSimple": "checkpoints",
    "LoraLoaderModelOnly": "loras",
    "LoraLoader": "loras",
}
MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".sft")


def wanted_models(wf_dir: Path) -> dict[str, dict]:
    """{filename: {"subdir": str, "workflows": [...]}} from the shipped workflows."""
    want: dict[str, dict] = {}
    for wf in sorted(wf_dir.glob("*.json")):
        try:
            nodes = json.loads(wf.read_text(encoding="utf-8"))
        except Exception as e:
            say("warn", f"could not parse workflow {wf.name}: {e}")
            continue
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            sub = LOADER_DIRS.get(ct)
            if not sub:
                continue
            # Match on the value, not the key: DualCLIPLoader names its inputs
            # clip_name1/clip_name2, so an `endswith("_name")` filter silently
            # drops t5xxl and clip_l and the flux workflow then fails with no
            # warning from here. class_type has already been vetted above.
            for v in (node.get("inputs") or {}).values():
                if isinstance(v, str) and v.lower().endswith(MODEL_EXT):
                    e = want.setdefault(v, {"subdir": sub, "workflows": []})
                    if wf.stem not in e["workflows"]:
                        e["workflows"].append(wf.stem)
    return want


def check_models(paths, want: dict[str, dict]) -> None:
    head(f"models ({len(want)} referenced by the shipped workflows)")
    models_root = Path(paths.COMFY_MODELS)
    if not models_root.is_dir():
        say("fail", f"models directory not found: {models_root}")
        return

    # Index by basename: ComfyUI honours extra_model_paths.yaml and symlinks, and
    # unet/ vs diffusion_models/ has drifted between releases, so "present
    # somewhere under models/" is the honest test. The expected subdir is
    # reported as a hint, not enforced.
    on_disk: dict[str, Path] = {}
    for p in models_root.rglob("*"):
        if p.is_file() and p.name.lower().endswith(MODEL_EXT):
            on_disk.setdefault(p.name, p)

    missing = []
    for name, meta in sorted(want.items()):
        hit = on_disk.get(name)
        if hit is None:
            missing.append((name, meta))
            continue
        rel = hit.relative_to(models_root).parent.as_posix()
        size = hit.stat().st_size / 1024 ** 3
        if rel != meta["subdir"]:
            say("ok", f"{name}  ({size:.1f} GB, in {rel}/ — expected "
                      f"{meta['subdir']}/, fine if intentional)")
        else:
            say("ok", f"{name}  ({size:.1f} GB)")

    for name, meta in missing:
        say("warn", f"MISSING {name} -> put it in {models_root / meta['subdir']}"
                    f"   (needed by: {', '.join(meta['workflows'])})")
    if missing:
        print(f"\n  {len(missing)} model file(s) missing. See MODELS.md for "
              f"download sources; only the workflows you actually use need theirs.")


def load_astapi():
    """Import astapi from the engine under test, or None if it predates it.

    doctor inspects an INSTALLED engine, which can be older than this pack. A
    traceback here would bury the one thing the operator needs to be told,
    which is to re-run the installer.
    """
    try:
        import astapi
        return astapi
    except ImportError:
        return None


def check_trim(paths) -> None:
    head("LLM working-set trim (self-test)")
    import osmem
    astapi = load_astapi()
    if astapi is None:
        say("warn", f"the engine at {ENGINE} has no astapi.py — it predates the "
                    f"AST-backed planner. Re-run install.py.")
        return
    import memplan
    targets = [d for d in memplan.llm_targets() if d.get("pid")]
    if not targets:
        say("warn", "no process under the backend to trim (or the backend is "
                    "unreachable) — nothing to test")
        return
    for d in targets:
        print(f"  pid={d['pid']:<7} {d['name']:<16} resident={d['ws_gb']:.1f} GB  "
              f"private={d['private_gb']:.1f} GB  "
              f"reclaimable={d['reclaimable_gb']:.1f} GB  [{d['via']}]")
    head_t = targets[0]
    pid, ws, priv = head_t["pid"], head_t["ws_gb"], head_t["private_gb"]
    if ws - priv <= 0.5:
        say("warn", "almost nothing is reclaimable — the runtime is probably "
                    "loading weights without mmap, or locking them in RAM. That "
                    "is safe; the planner just gets a smaller RAM budget.")
    print("  probing the trim itself (this DOES page out the LLM's weights) ...")
    ok, why = osmem.release_dram(pid)
    say("ok" if ok else "warn", f"release_dram -> {ok}: {why}")
    if not ok and sys.platform != "win32":
        print("  On Linux a refusal is not fatal: the kernel drops these clean\n"
              "  file-backed pages under allocation pressure on its own.")


def _script_dir(paths, skill: str) -> Path:
    """Where a skill's scripts are. A linked install (install.py --link) keeps
    only the rendered documents in the Hermes skills dir and runs the scripts
    from the pack, which its record names; every other install has both in the
    skills dir."""
    rec = Path(paths.STATE_DIR) / "comfyui-skill-link.json"
    try:
        return Path(json.loads(rec.read_text(encoding="utf-8"))["pack"]) / "skills" / skill
    except (OSError, ValueError, KeyError):
        return Path(paths.SKILLS_DIR) / skill


def check_dramagen(paths) -> None:
    """dramagen adds two dependencies the other skills do not have: ffmpeg (it
    cuts the per-shot clips together and muxes the subtitle track) and an
    OpenAI-compatible LLM (it writes the script). Both fail late and
    confusingly -- a whole episode renders, and only then the stitch dies -- so
    name them up front on a fresh machine."""
    import urllib.request
    sd = Path(paths.SKILLS_DIR) / "dramagen"
    code = _script_dir(paths, "dramagen")
    ok = (all((code / f).is_file() for f in ("story.py", "produce.py"))
          and (sd / "SKILL.md").is_file())
    say("ok" if ok else "fail",
        f"dramagen: installed in {sd}" if ok else f"dramagen: not installed under {sd}")
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        say("ok" if found else "fail",
            f"{tool}: {found or 'NOT FOUND -- final.mp4 and subtitles need it'}")
    ast = os.environ.get("VIDGEN_AST_URL", "").strip().rstrip("/")
    url = (os.environ.get("DRAMAGEN_LLM_URL", "").strip()
           or (ast + "/v1" if ast else "http://127.0.0.1:8571/v1"))
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=6) as r:
            served = [m.get("id") for m in (json.load(r).get("data") or [])]
        say("ok", f"script LLM at {url}: serving "
                  f"{', '.join(str(s) for s in served[:3]) or '(none listed)'}")
    except Exception as e:
        say("warn", f"script LLM at {url} unreachable ({e}). Set DRAMAGEN_LLM_URL "
                    "or VIDGEN_AST_URL in hermes-comfy.conf.")


def check_motion(paths) -> None:
    """motion runs in a venv of its own (OCR and barcode packages); its check
    command imports every dependency and looks for ffmpeg."""
    sd = Path(paths.SKILLS_DIR) / "motion"
    script = _script_dir(paths, "motion") / "motion.py"
    ok = script.is_file() and (sd / "SKILL.md").is_file()
    say("ok" if ok else "fail",
        f"motion: installed in {sd}" if ok else f"motion: not installed under {sd}")
    py = Path(paths.HERMES_HOME) / "motion-venv" / ("Scripts/python.exe" if os.name == "nt"
                                                    else "bin/python")
    if not py.is_file():
        say("fail", f"motion venv missing ({py}) -- re-run install.py")
        return
    try:
        r = subprocess.run([str(py), str(script), "check"], capture_output=True, text=True,
                           timeout=120)
        res = json.loads(r.stdout)
        missing = [k for k, v in res.items() if isinstance(v, str) and v.startswith("MISSING")]
        say("ok" if res.get("ok") else "fail",
            "motion venv: OCR, barcodes and ffmpeg ready" if res.get("ok")
            else "motion venv incomplete: " + ", ".join(missing))
    except Exception as e:                      # noqa: BLE001
        say("fail", f"motion check could not run: {e}")


def check_recast(paths) -> None:
    """recast needs ffmpeg/ffprobe (shot sheets, checks, final.mp4) and numpy +
    Pillow in the skill python (the leak/lag checks). It drives vidgen/gen.py,
    so a missing vidgen is fatal for it too."""
    sd = Path(paths.SKILLS_DIR) / "recast"
    ok = (_script_dir(paths, "recast") / "recast.py").is_file() and (sd / "SKILL.md").is_file()
    say("ok" if ok else "fail",
        f"recast: installed in {sd}" if ok else f"recast: not installed under {sd}")
    vg = _script_dir(paths, "vidgen") / "gen.py"
    say("ok" if vg.is_file() else "fail", f"recast -> vidgen: {vg}" if vg.is_file()
        else f"recast drives {vg}, which is missing")
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        say("ok" if found else "fail",
            f"{tool}: {found or 'NOT FOUND -- recast cannot cut shots or build final.mp4'}")
    for mod in ("numpy", "PIL"):
        try:
            __import__(mod)
            say("ok", f"{mod}: importable (recast checks)")
        except Exception as e:
            say("fail", f"{mod}: not importable in this python ({e}) -- recast's render checks need it")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a hermes-comfy install.")
    ap.add_argument("--engine", help="engine directory to inspect (default: the "
                                     "installed one, found the same way the skills find it)")
    ap.add_argument("--models", action="store_true", help="only check model files")
    ap.add_argument("--trim-selftest", action="store_true",
                    help="also exercise the LLM working-set trim (pages out its weights)")
    a = ap.parse_args()

    print(f"inspecting engine: {ENGINE}"
          + ("" if (ENGINE / "hermes-comfy.conf").is_file()
             else "   (no hermes-comfy.conf here)"))
    try:
        import paths
    except Exception as e:
        print(f"FATAL: cannot import paths.py from {ENGINE}: {e}")
        return 1

    if a.models:
        check_models(paths, wanted_models(Path(paths.WF_DIR)))
    else:
        head("resolved paths")
        info = paths.describe()
        width = max(len(k) for k in info)
        for k, v in info.items():
            print(f"  {k.ljust(width)}  {v}")
        if not info["conf_file"]:
            say("warn", "no hermes-comfy.conf found — running entirely on "
                        "auto-detection and defaults")

        head("python")
        v = sys.version_info
        say("ok" if v >= (3, 9) else "fail",
            f"running under {sys.executable} ({v.major}.{v.minor}.{v.micro})")

        head("ComfyUI")
        if paths.COMFY_REMOTE:
            say("ok", "COMFY_REMOTE: ComfyUI belongs to another machine/OS. "
                      "This install talks to it over HTTP and stages files "
                      "through the mount; it never launches or protects it.")
            say("ok" if Path(paths.COMFY_MAIN).is_file() else "warn",
                f"main.py (through the mount): {paths.COMFY_MAIN}")
        else:
            say("ok" if Path(paths.COMFY_MAIN).is_file() else "fail",
                f"main.py: {paths.COMFY_MAIN}")
        # h3_media.py is the only thing this interpreter runs, and it imports
        # av + PIL, never torch. Checking the imports beats checking the path.
        if Path(paths.COMFY_PYTHON).is_file() or paths.COMFY_PYTHON == sys.executable:
            probe = subprocess.run(
                [paths.COMFY_PYTHON, "-c",
                 "import av, PIL; print(av.__version__)"],
                capture_output=True, text=True)
            if probe.returncode == 0:
                say("ok", f"media interpreter: {paths.COMFY_PYTHON} "
                          f"(PyAV {probe.stdout.strip()})")
            else:
                say("warn", f"media interpreter {paths.COMFY_PYTHON} cannot import "
                            f"av/PIL -- MiniMax-H3 audio muxing will fail. Fix with: "
                            f"{paths.pip_install_command(paths.COMFY_PYTHON, 'av pillow pillow-heif')}")
        else:
            say("fail", f"media interpreter not found: {paths.COMFY_PYTHON}")
        for d, why in ((paths.COMFY_MODELS, "models"),
                       (paths.COMFY_INPUT, "input"),
                       (paths.COMFY_OUTPUT, "output")):
            say("ok" if Path(d).is_dir() else "warn", f"{why} dir: {d}")

        gguf = Path(paths.COMFY_NODES) / "ComfyUI-GGUF"
        say("ok" if gguf.is_dir() else "fail",
            f"custom node ComfyUI-GGUF: {gguf}"
            + ("" if gguf.is_dir() else "  <- required by the 14B video and "
                                        "qwen/flux image workflows (UnetLoaderGGUF)"))

        # Shipped in the pack under comfyui-plugins/ and copied by install.py.
        # Only one route needs it, so a miss is a warning -- but an unmissable
        # one: without it `r2v --ref ... --image ...` dies with a tensor shape
        # mismatch inside the sampler, minutes in and nowhere near the cause.
        refkf = Path(paths.COMFY_NODES) / "hermes_minimax_h3_refkf"
        say("ok" if refkf.is_dir() else "warn",
            f"custom node hermes_minimax_h3_refkf: {refkf}"
            + ("" if refkf.is_dir() else "  <- MISSING: MiniMax-H3 reference images "
                                         "+ pinned first frame (`r2v --ref ... "
                                         "--image ...`) will fail in the sampler. "
                                         "Re-run install.py, then restart ComfyUI."))

        # Viggle Turbo for the qwen21 family. Not shipped: it is Qwen research
        # (non-commercial) licensed, fetched by the user -- see MODELS.md.
        # gen.py falls back to the base sampler without it, so only a warning.
        nodes = Path(paths.COMFY_NODES)
        viggle = next((p for p in (nodes / "viggle_turbo.py", nodes / "viggle_turbo")
                       if p.exists()), None)
        lora = Path(paths.COMFY_MODELS) / "loras" / \
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
        if viggle and lora.is_file():
            say("ok", f"Viggle Turbo for qwen21: {viggle}, {lora.name}")
        else:
            gone = [n for n, ok in (("custom node viggle_turbo.py", viggle),
                                    (f"loras/{lora.name}", lora.is_file())) if not ok]
            say("warn", "Viggle Turbo for qwen21 missing: " + ", ".join(gone)
                + "  <- qwen21 / qwen21_edit / qwen21_move fall back to the 2.5-3x "
                  "slower base sampler. See MODELS.md.")

        head("engine")
        for f in ("paths.py", "osmem.py", "sysmon.py", "memplan.py",
                  "video_gen.py", "start_comfy.py", "h3_media.py", "llmsleep.py",
                  "linkcheck.py"):
            say("ok" if (Path(paths.ENGINE_DIR) / f).is_file() else "fail",
                f"{f}")
        wfs = sorted(Path(paths.WF_DIR).glob("*.json"))
        say("ok" if wfs else "fail", f"{len(wfs)} workflow(s) in {paths.WF_DIR}")

        head("skills")
        sd = Path(paths.SKILLS_DIR)
        for s in ("imggen", "vidgen"):
            gen = _script_dir(paths, s) / "gen.py"
            skill_md = sd / s / "SKILL.md"
            if gen.is_file() and skill_md.is_file():
                marker = gen.parent / "engine_dir.txt"
                pointed = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
                if pointed and Path(pointed).resolve() != Path(paths.ENGINE_DIR).resolve():
                    say("warn", f"{s}: engine_dir.txt points at {pointed}, not "
                                f"{paths.ENGINE_DIR}")
                else:
                    say("ok", f"{s}: installed in {sd / s}")
            else:
                say("fail", f"{s}: not installed under {sd}")

        # Hermes treats every directory here as a skill. A leftover copy with
        # the same `name:` in its frontmatter competes for selection, and a
        # small model then picks unpredictably between them -- the documented
        # way this setup goes wrong.
        if sd.is_dir():
            strays = [d.name for d in sorted(sd.iterdir())
                      if d.is_dir() and not d.name.startswith(".")
                      and (d / "SKILL.md").is_file()
                      and d.name not in ("imggen", "vidgen", "dramagen", "recast")
                      and (".bak." in d.name or d.name.startswith(("imggen", "vidgen", "dramagen", "recast")))]
            if strays:
                say("fail", "leftover skill copies competing with imggen/vidgen: "
                            + ", ".join(strays)
                            + f"\n         Delete them from {sd} — Hermes loads every "
                              f"directory here as a skill, so duplicates make skill "
                              f"selection unpredictable.")

        head("dramagen")
        check_dramagen(paths)

        head("recast")
        check_recast(paths)

        head("motion")
        check_motion(paths)

        head("GPU / memory")
        import osmem
        g = osmem.gpu_vram()
        if "error" in g:
            say("fail", f"no GPU memory source worked -- the planner cannot size "
                        f"a render without one.\n         {g['error']}\n"
                        f"         Fix: install your vendor's tooling (nvidia-smi, "
                        f"Intel Level Zero, xpu-smi, rocm-smi), start ComfyUI so its "
                        f"/system_stats can answer, bring up the TRUSTA-AST backend "
                        f"so /system/resources can, or set VIDGEN_GPU_QUERY_CMD to "
                        f"any command printing 'total,used,free' in MiB.")
        else:
            src = g.get("source", "?")
            say("ok", f"GPU: {g['total_mb']/1024:.1f} GB total, "
                      f"{g['free_mb']/1024:.1f} GB free, {g['used_mb']/1024:.1f} GB "
                      f"in use   [measured via {src}]")
            # torch's view can miss other processes' allocations, and on
            # Windows/WDDM it reads free high -- the planner would then size a
            # render into memory something else already holds.
            if str(src).startswith("comfyui:"):
                say("warn", "that figure comes from ComfyUI/torch, not a driver "
                            "tool. It works, but a vendor CLI sees other "
                            "processes more reliably -- install one, or point "
                            "VIDGEN_GPU_QUERY_CMD at your platform's equivalent.")
            budget = float(os.environ.get("VIDGEN_LLM_VRAM_BUDGET", "7.0"))
            if budget >= g["total_mb"] / 1024:
                say("fail", f"VIDGEN_LLM_VRAM_BUDGET={budget} >= the whole card "
                            f"({g['total_mb']/1024:.1f} GB) — every render will be refused")
            elif g["total_mb"] / 1024 - budget < 4:
                say("warn", f"only {g['total_mb']/1024 - budget:.1f} GB left for video "
                            f"after the {budget} GB LLM reserve; expect short clips")
        r = osmem.system_ram()
        if "error" in r:
            say("fail", f"cannot read system RAM: {r['error']}")
        else:
            say("ok", f"RAM: {r['total_mb']/1024:.1f} GB total, "
                      f"{r['avail_mb']/1024:.1f} GB available"
                      + ("   <- the WSL VM's allocation, NOT the Windows host's"
                         if paths.WSL else ""))
            if paths.WSL:
                say("warn", "RAM budgeting therefore describes the distro, not the "
                            "host. Harmless for the default GGUF path (~1 GB "
                            "incremental), but do not read these numbers as "
                            "Windows' free memory. Cap the VM in .wslconfig if "
                            "you want the two to agree.")

        astapi = load_astapi()
        if astapi is None:
            say("fail", f"the engine at {ENGINE} has no astapi.py — it predates "
                        f"the AST-backed planner, so nothing here can tell "
                        f"whether an LLM is resident. Re-run install.py.")
            ast = None
        else:
            ast = astapi.describe()
        if ast is None:
            pass
        elif not ast["reachable"]:
            say("warn", f"TRUSTA-AST backend unreachable at {ast['url']}. The VRAM "
                        f"reserve still applies (VIDGEN_LLM_VRAM_BUDGET is held "
                        f"unconditionally), but the planner cannot tell whether a "
                        f"model is loaded, so it assumes one is — the conservative "
                        f"direction. Set VIDGEN_AST_URL if it lives elsewhere.")
        elif ast["loaded"]:
            say("ok", f"backend at {ast['url']}: {ast['model']} loaded "
                      f"on engine '{ast['engine']}'")
            import memplan
            targets = memplan.llm_targets()
            live = [d for d in targets if d.get("pid")]
            proc = targets[0].get("name") if targets else None
            if live:
                for d in live:
                    how = ("listening on port %d under the backend" % d["port"]
                           if d.get("port") else
                           "named by AST" if d.get("via") == "name"
                           else "under the backend")
                    say("ok", f"trimmable process '{d['name']}': pid {d['pid']} "
                              f"({how}, {d['reclaimable_gb']:.1f} GB droppable)")
            elif proc and paths.WSL and proc.lower().endswith(".exe"):
                # Expected in a split install, and worth stating plainly: the
                # VRAM reserve still protects the LLM, only DRAM accounting is
                # blind.
                say("warn", f"'{proc}' is a Windows process and /proc cannot see "
                            f"it. VRAM protection still applies; what is lost is "
                            f"DRAM reclaim accounting, which just makes the "
                            f"planner more conservative. See INSTALL.md.")
            elif proc:
                say("warn", f"VIDGEN_LLM_PROCESS names '{proc}', which is not in "
                            f"the process table — DRAM reclaim is off.")
            else:
                # Do not claim to know WHY. Whether the engine runs in AST's
                # own worker or in a process of its own is a fact about AST's
                # configuration, and telling the two apart here would mean
                # re-introducing exactly the built-in engine knowledge this
                # pack no longer keeps. Report what is observable and name
                # both remedies.
                cands = ast.get("process_candidates") or []
                offered = ("  AST offered: " + ", ".join(cands)
                           if cands else "")
                port = astapi.ast_port()
                seen = osmem.listening_pids().get(port) if port else None
                where = (f"  Nothing is listening on port {port} here, so the "
                         f"backend is not a process on this host."
                         if not seen else
                         f"  The backend is pid {seen}, but no process under it "
                         f"could be read.")
                say("warn", "no process under the backend resolves to something "
                            "trimmable, so DRAM reclaim is off. The VRAM guard "
                            "is unaffected. Set VIDGEN_LLM_PROCESS to name the "
                            "process directly if it runs elsewhere." + where
                            + offered)
        else:
            say("warn", f"backend at {ast['url']} is up but no model is loaded. "
                        f"VRAM is still held back for one to return.")

        if shutil.which("ffmpeg") is None:
            say("warn", "ffmpeg not on PATH (only needed for the h3_resume helper)")

        head("ComfyUI server")
        import urllib.request
        reached = None
        for u in paths.comfy_url_candidates():
            try:
                with urllib.request.urlopen(u + "/system_stats", timeout=5) as f:
                    json.load(f)
                reached = u
                break
            except Exception as e:
                say("warn", f"no answer at {u} ({type(e).__name__})")
        if reached:
            say("ok", f"answering at {reached}")
            # --reserve-vram is fixed at launch and is the ONLY thing stopping
            # ComfyUI from allocating into VRAM the LLM already holds. Read it
            # back from the server's own argv instead of assuming.
            import comfyguard
            st = comfyguard.reserve_status(reached)
            # `is not False`: an unreachable backend must not downgrade this
            # to "nothing to protect" -- that is the one wrong answer here.
            llm_up = astapi is None or astapi.llm_loaded() is not False
            if st["reserve_gb"] is not None:
                say("ok", f"launched with --reserve-vram {st['reserve_gb']} GB")
            elif llm_up:
                say("fail", "this ComfyUI was launched WITHOUT --reserve-vram "
                            "while a local LLM is resident. It will treat the "
                            "LLM's VRAM as free and can evict it.\n"
                            f"         Restart it:  python {paths.ENGINE_DIR}"
                            f"/start_comfy.py --force")
            else:
                say("ok", "no --reserve-vram, and no co-resident LLM to protect "
                          "(correct as-is; a reserve is only needed when "
                          "something else holds VRAM)")
        elif paths.COMFY_REMOTE:
            # Remote means nothing here can start it, so this is fatal to any
            # actual render -- unlike the local case where gen.py just launches.
            say("fail", "ComfyUI is unreachable and COMFY_REMOTE is set, so "
                        "nothing here can start it. On the host run:\n"
                        "           python <engine>/start_comfy.py\n"
                        "         It listens on 127.0.0.1 only, which a NAT-mode WSL "
                        "distro cannot reach: turn on WSL mirrored networking "
                        "(.wslconfig: [wsl2] networkingMode=mirrored, then "
                        "`wsl --shutdown`) and plain loopback works.")
        else:
            say("ok", f"not running at {paths.COMFY_URL} — gen.py will start it "
                      f"on demand (this is normal)")

        head("Hermes")
        cfg = Path(paths.HERMES_HOME) / "config.yaml"
        if not cfg.is_file():
            say("warn", f"no Hermes config at {cfg} — is Hermes installed here?")
        else:
            txt = cfg.read_text(encoding="utf-8", errors="replace")
            timeout = None
            in_terminal = False
            for line in txt.splitlines():
                if line.startswith("terminal:"):
                    in_terminal = True
                    continue
                if in_terminal:
                    if line[:1] not in (" ", "\t"):
                        in_terminal = False
                    elif line.strip().startswith("timeout:"):
                        try:
                            timeout = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
            if timeout is None:
                say("warn", f"terminal.timeout not set in {cfg}; set it to 600 so a "
                            f"foreground --wait render is not cut off")
            elif timeout < 600:
                say("warn", f"terminal.timeout={timeout} in {cfg}; renders need 600. "
                            f"(A cut-off is not a failed render — the detached worker "
                            f"finishes anyway — but you lose the foreground result.)")
            else:
                say("ok", f"terminal.timeout={timeout}")

        check_models(paths, wanted_models(Path(paths.WF_DIR)))

    if a.trim_selftest:
        check_trim(paths)

    head("summary")
    print(f"  {len(FATAL)} fatal, {len(WARN)} warning(s)")
    for m in FATAL:
        print("  FATAL: " + m)
    if not FATAL:
        print('  Nothing fatal. Try:  "%s" "%s" zimage "a red fox"'
              % (sys.executable or "python",
                 _script_dir(paths, "imggen") / "gen.py"))
    return 1 if FATAL else 0


if __name__ == "__main__":
    raise SystemExit(main())
