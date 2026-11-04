#!/usr/bin/env python
"""Install the skills and their ComfyUI engine into a Hermes.

Cross-platform on purpose: install.sh and install.ps1 are thin wrappers around
this file, so Windows and Linux cannot drift apart in what they actually do.

    python install.py                              # install with auto-detection
    python install.py --comfy-home /opt/ComfyUI    # say where ComfyUI is
    python install.py --external                   # register instead of copying
    python install.py --link                       # run from this git checkout; `git pull` updates it
    python install.py --dry-run                    # show the plan, change nothing

What it does
------------
1. Copies engine/ to <hermes-home>/comfy-engine   (override with --engine-dir)
2. Copies skills/ to <hermes-home>/skills         (or registers them, --external)
3. Renders each SKILL.md.in, substituting the absolute paths this machine
   actually uses -- the skill documents tell the model to run one exact command,
   so those paths have to be real, not placeholders.
4. Writes hermes-comfy.conf if there isn't one, pre-filled with what was found.
5. Runs tools/doctor.py.

Existing files are backed up to <name>.bak.<timestamp> before being replaced,
so re-running over a working install is safe.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACK = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"

# doctor.py runs as a child and writes straight to the same handle. Without
# line buffering our own prints sit in a block buffer until exit and the whole
# report lands *after* them, which reads as "the verify step printed nothing".
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    except Exception:
        pass

STAMP = time.strftime("%Y%m%d_%H%M%S")
SKILLS = ("imggen", "vidgen", "dramagen", "recast", "motion", "gpushare")
# ComfyUI custom nodes shipped with the pack, copied into <comfy>/custom_nodes.
# hermes_minimax_h3_refkf is what makes `r2v --ref ... --image ...` (reference
# images plus a pinned first frame) possible at all -- see its docstring.
COMFY_PLUGINS = ("hermes_minimax_h3_refkf",)


def hermes_home(explicit: str | None) -> Path:
    """Same rule as hermes_constants.get_hermes_home(), reimplemented so the
    installer does not need Hermes importable on sys.path."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if WINDOWS:
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _comfy_home_from_conf(engine_dir: Path | None) -> Path | None:
    """COMFY_HOME out of an existing hermes-comfy.conf, if there is one.

    A reinstall on a configured box has the answer written down already, and
    step [2/5] deliberately leaves that file alone -- so not reading it meant
    the installer reported "comfy home NOT FOUND" on a machine that was working
    fine, and silently skipped the ComfyUI plugin step. Surfaced on the WSL
    split install, where COMFY_HOME is a /mnt/c path that no probe can guess.
    """
    if engine_dir is None:
        return None
    conf = engine_dir / "hermes-comfy.conf"
    try:
        for line in conf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("COMFY_HOME=") and not line.startswith("#"):
                v = line.split("=", 1)[1].strip()
                if v:
                    return Path(os.path.expandvars(v)).expanduser()
    except OSError:
        pass
    return None


def _engine_paths():
    """The pack's own engine/paths.py, imported from the pack.

    For the layout knowledge only -- where AI Scaler Toolkit puts ComfyUI, and which
    Python a ComfyUI home uses -- so the installer and the engine cannot disagree.
    Importing it folds a hermes-comfy.conf into os.environ, which is what the
    engine will do at run time too.
    """
    sys.path.insert(0, str(PACK / "engine"))
    try:
        import paths
    finally:
        sys.path.pop(0)
    return paths


def find_comfy_home(explicit: str | None, engine_dir: Path | None = None) -> Path | None:
    """Where ComfyUI is. Every shape this pack knows is probed, portable first."""
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    else:
        env = os.environ.get("COMFY_HOME", "").strip()
        if env:
            cands.append(Path(env).expanduser())
        from_conf = _comfy_home_from_conf(engine_dir)
        if from_conf:
            cands.append(from_conf)
        h = Path.home()
        cands += [h / "ComfyUI_windows_portable", h / "ComfyUI", h / "comfy" / "ComfyUI",
                  Path("/opt/ComfyUI"), Path("/srv/ComfyUI"),
                  _engine_paths().ast_comfy_home()]
    for c in cands:
        if (c / "ComfyUI" / "main.py").is_file() or (c / "main.py").is_file():
            return c
    return cands[0] if explicit else None


def find_skill_python(home: Path, explicit: str | None) -> str:
    """The interpreter the SKILL.md commands will name.

    Hermes's own venv is the right default: the skills are launched from
    Hermes's terminal, and a bare `python` there hits the Microsoft Store shim
    on Windows or a system python without deps on Linux. Both were hit in
    practice, which is why the rendered command spells out the interpreter.
    """
    if explicit:
        return str(Path(explicit).expanduser())
    venv = home / "hermes-agent" / "venv"
    cand = venv / ("Scripts/python.exe" if WINDOWS else "bin/python")
    if cand.is_file():
        return str(cand)
    return sys.executable


def backup(p: Path, dry: bool, outside: Path | None = None) -> None:
    """Move `p` aside before it is replaced.

    `outside` matters for skills: Hermes scans every directory under
    <hermes-home>/skills, so a sibling `imggen.bak.<stamp>` is not an inert
    backup -- it is a second skill with the same `name:` in its frontmatter,
    competing for selection. That is the documented failure mode where a small
    model picks the wrong one. Backups therefore land outside the scan path.
    """
    if not p.exists():
        return
    dst = (outside / (p.name + ".bak." + STAMP)) if outside \
        else p.with_name(p.name + ".bak." + STAMP)
    print(f"    backup {p.name} -> {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))


# Runtime state, not payload. The engine writes these beside itself, so running
# the launcher straight out of the pack (which is the normal way to test it)
# leaves logs that would otherwise be copied onto every machine installed from
# that checkout.
RUNTIME_DIRS = {"out", "jobs", "logs", "skill-backups", "__pycache__"}


def copy_tree(src: Path, dst: Path, dry: bool) -> int:
    """Copy src into dst, skipping runtime state. Returns the file count.

    The exclusion is matched against the path RELATIVE to src, never the
    absolute one: an absolute path legitimately contains directories called
    `jobs`, `out` or `logs` (an installer unpacked under ~/.claude/jobs/... hit
    exactly this), and matching on those silently skipped every file in the
    tree while still reporting success.
    """
    n = 0
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)
        if RUNTIME_DIRS & set(rel.parts) or item.suffix in (".pyc", ".log"):
            continue
        target = dst / rel
        if item.is_dir():
            if not dry:
                target.mkdir(parents=True, exist_ok=True)
            continue
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        print(f"    {rel.as_posix()}")
        n += 1
    return n


def render_skill(src: Path, dst: Path, subs: dict, dry: bool) -> None:
    txt = src.read_text(encoding="utf-8")
    for k, v in subs.items():
        txt = txt.replace(k, v)
    left = [ln for ln in txt.splitlines() if "@@" in ln]
    if left:
        print("    WARNING: unsubstituted placeholder(s):")
        for ln in left[:5]:
            print("      " + ln.strip()[:120])
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(txt, encoding="utf-8", newline="\n")


# ---- motion's interpreter ---------------------------------------------------
# motion needs OCR and barcode packages (rapidocr-onnxruntime, zxing-cpp) that
# do not belong in Hermes's own venv: installing them there risks its
# dependencies. It gets a venv of its own in the Hermes home, rebuilt only when
# skills/motion/requirements.txt changes.
def motion_venv(home: Path) -> Path:
    return home / "motion-venv"


def motion_python(home: Path) -> Path:
    return motion_venv(home) / ("Scripts/python.exe" if WINDOWS else "bin/python")


def ensure_motion_venv(home: Path, base_py: str, dry: bool, quiet: bool = False) -> bool:
    say = (lambda *_: None) if quiet else print
    req = PACK / "skills" / "motion" / "requirements.txt"
    if not req.is_file():
        return False
    venv, py = motion_venv(home), motion_python(home)
    stamp = venv / ".requirements.sha1"
    want = _sha(req)
    if py.is_file() and stamp.is_file() and stamp.read_text().strip() == want:
        say(f"    motion venv up to date: {venv}")
        return True
    say(f"    building the motion venv at {venv} (OCR and barcode packages; "
        f"a few minutes the first time)")
    if dry:
        return True
    uv = _engine_paths().find_uv(home)
    if uv:
        # uv builds the venv and fills it: no pip in either interpreter, and the
        # packages come out of the cache the rest of the machine already shares.
        say(f"    using uv: {uv}")
        try:
            if not py.is_file():
                subprocess.run([uv, "venv", "--quiet", "--python", base_py, str(venv)],
                               check=True, capture_output=quiet)
            subprocess.run([uv, "pip", "install", "--quiet", "--python", str(py),
                            "-r", str(req)], check=True, capture_output=quiet)
            stamp.write_text(want)
            say("    motion venv ready")
            return True
        except Exception as e:                  # noqa: BLE001 -- report, never abort the install
            print(f"    WARNING: could not build the motion venv with uv ({e}). The motion "
                  f"skill will not run until it exists:\n      \"{uv}\" venv --python "
                  f"\"{base_py}\" \"{venv}\"\n      \"{uv}\" pip install --python \"{py}\" "
                  f"-r \"{req}\"", file=sys.stderr)
            return False
    # No uv anywhere: the base interpreter's pip, the way it was done before uv.
    try:
        # Without pip of its own, filled by the base interpreter's pip through
        # --python: seeding pip with ensurepip unpacks setuptools' deeply nested
        # test data, which broke Windows' 260-character path limit under a long
        # home directory. A base without pip falls back to the usual way.
        has_pip = subprocess.run([base_py, "-m", "pip", "--version"],
                                 capture_output=True).returncode == 0
        if not py.is_file():
            subprocess.run([base_py, "-m", "venv"] + (["--without-pip"] if has_pip else [])
                           + [str(venv)], check=True, capture_output=quiet)
        pip = ([base_py, "-m", "pip", "--python", str(py)] if has_pip
               else [str(py), "-m", "pip"])
        subprocess.run(pip + ["install", "-q", "-r", str(req)], check=True, capture_output=quiet)
        stamp.write_text(want)
        say("    motion venv ready")
        return True
    except Exception as e:                      # noqa: BLE001 -- report, never abort the install
        print(f"    WARNING: could not build the motion venv ({e}). The motion skill will not "
              f"run until it exists:\n      \"{base_py}\" -m venv \"{venv}\"\n"
              f"      \"{py}\" -m pip install -r \"{req}\"", file=sys.stderr)
        return False


def motion_subs(home: Path, script_root: Path) -> dict:
    return {"@@MOTIONPY@@": motion_python(home).as_posix(),
            "@@MOTION@@": (script_root / "motion" / "motion.py").as_posix(),
            "@@MOTIONLIB@@": (script_root / "motion" / "lib").as_posix()}


# ---- linked install (--link / --refresh) ------------------------------------
# A copy install freezes the pack at install time, so a machine that follows the
# pack with `git pull` still runs the old engine and reads the old SKILL.md until
# someone re-runs this installer by hand. A linked install runs the engine and
# the skill scripts straight from the pack and writes nothing into it; only what
# differs per machine lives in the Hermes home: SKILL.md rendered with this
# machine's paths, hermes-comfy.conf, and the runtime state. A git hook (and a
# check the skills make on start) re-renders the documents after every pull.
LINK_RECORD = "comfyui-skill-link.json"
HOOK_BEGIN = "# >>> comfyui-skill refresh"
HOOK_END = "# <<< comfyui-skill refresh"
GIT_HOOKS = ("post-merge", "post-checkout", "post-rewrite")


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _comfy_bits(comfy: Path | None):
    """(comfy_root, comfy_python) for a ComfyUI home, the way step [3/5] decides."""
    nested = bool(comfy and (comfy / "ComfyUI" / "main.py").is_file())
    comfy_root = (comfy / "ComfyUI") if nested else comfy
    if not comfy:
        return None, Path("<COMFY_HOME>/venv/bin/python")
    eng = _engine_paths()
    comfy_py = (Path(eng.COMFY_PYTHON) if Path(eng.COMFY_HOME) == comfy
                else Path(eng._find_comfy_python(str(comfy), str(comfy_root))))
    return comfy_root, comfy_py


def _set_conf_key(text: str, key: str, value: str) -> str:
    """KEY=value in a conf body: replace the line (commented or not) or append."""
    out, done = [], False
    for line in text.splitlines():
        bare = line.lstrip("#").strip()
        if not done and bare.startswith(key + "="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    return "\n".join(out) + "\n"


def _link_render(home: Path, comfy: Path | None, skill_py: str, dry: bool,
                 quiet: bool, prev: dict | None) -> dict:
    """Render every skill's documents into <home>/skills/<skill>/ with commands that
    point into the pack, copy the rest of each skill's documents beside them, and
    return the record a later --refresh works from."""
    say = (lambda *_: None) if quiet else print
    state = home / "comfy-state"
    skills_root = home / "skills"
    comfy_root, comfy_py = _comfy_bits(comfy)
    subs_common = {
        "@@PY@@": Path(skill_py).as_posix(),
        "@@ENGINE@@": (PACK / "engine").as_posix(),
        "@@OUT@@": (state / "out").as_posix(),
        "@@COMFY_OUTPUT@@": (comfy_root / "output").as_posix() if comfy_root
                            else "<COMFY_HOME>/output",
        "@@COMFY_PY@@": comfy_py.as_posix(),
        "@@IMGGEN@@": (PACK / "skills" / "imggen" / "gen.py").as_posix(),
        "@@VIDGEN@@": (PACK / "skills" / "vidgen" / "gen.py").as_posix(),
        "@@STORY@@": (PACK / "skills" / "dramagen" / "story.py").as_posix(),
        "@@PRODUCE@@": (PACK / "skills" / "dramagen" / "produce.py").as_posix(),
        "@@DRAMA@@": (home / "drama").as_posix(),
        "@@RECAST@@": (PACK / "skills" / "recast" / "recast.py").as_posix(),
        "@@RECASTOUT@@": (home / "recast").as_posix(),
        "@@GPUSHARE@@": (PACK / "skills" / "gpushare" / "gpushare.py").as_posix(),
        **motion_subs(home, PACK / "skills"),
    }
    written, templates = [], {}
    for s in SKILLS:
        src_dir, dst_dir = PACK / "skills" / s, skills_root / s
        say(f"  {s}:  scripts run from {src_dir.as_posix()}")
        # A copy install left the scripts here; a linked skill must not keep a
        # second, frozen gen.py beside its SKILL.md for anything to find.
        if prev is None and dst_dir.is_dir() and any(dst_dir.rglob("*.py")):
            backup(dst_dir, dry, outside=state / "skill-backups")
        rendered = set()
        for tmpl in sorted(src_dir.rglob("*.in")):
            rel = tmpl.relative_to(src_dir)
            render_skill(tmpl, dst_dir / rel.with_suffix(""), subs_common, dry)
            rendered.add(rel.with_suffix(""))
            templates[(Path("skills") / s / rel).as_posix()] = _sha(tmpl)
            written.append((dst_dir / rel.with_suffix("")).as_posix())
            say(f"    {rel.with_suffix('').as_posix()} rendered")
        for item in sorted(src_dir.rglob("*")):
            rel = item.relative_to(src_dir)
            if (not item.is_file() or item.suffix in (".py", ".in", ".pyc", ".log")
                    or rel in rendered or item.name == "engine_dir.txt"
                    or RUNTIME_DIRS & set(rel.parts)):
                continue
            if not dry:
                (dst_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_dir / rel)
            templates[(Path("skills") / s / rel).as_posix()] = _sha(item)
            written.append((dst_dir / rel).as_posix())
            say(f"    {rel.as_posix()}")
    # Documents the pack no longer has must not linger for the model to read.
    for old in (prev or {}).get("written", []):
        if old not in written and Path(old).is_file() and not dry:
            Path(old).unlink()
            say(f"    removed {old} (no longer in the pack)")
    plugins = {}
    for name in COMFY_PLUGINS:
        for f in sorted((PACK / "comfyui-plugins" / name).rglob("*")):
            if f.is_file() and f.suffix != ".pyc":
                plugins[f.relative_to(PACK).as_posix()] = _sha(f)
    return {"pack": PACK.as_posix(), "home": home.as_posix(), "python": skill_py,
            "comfy_home": comfy.as_posix() if comfy else None,
            "written": written, "templates": templates, "plugins": plugins}


def _link_plugins(comfy: Path | None, dry: bool, quiet: bool) -> int:
    comfy_root, _ = _comfy_bits(comfy)
    if comfy_root is None:
        if not quiet:
            print("    SKIPPED: COMFY_HOME is unknown -- copy "
                  f"{(PACK / 'comfyui-plugins').as_posix()}/* into <ComfyUI>/custom_nodes")
        return 0
    n = 0
    for name in COMFY_PLUGINS:
        dst = comfy_root / "custom_nodes" / name
        if quiet:
            for f in (PACK / "comfyui-plugins" / name).rglob("*"):
                if f.is_file() and f.suffix != ".pyc":
                    t = dst / f.relative_to(PACK / "comfyui-plugins" / name)
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, t)
                    n += 1
        else:
            n += copy_tree(PACK / "comfyui-plugins" / name, dst, dry)
    return n


def _install_hooks(home: Path, skill_py: str, dry: bool) -> list[str]:
    """git hooks in the checkout the pack lives in, so `git pull` refreshes.

    Appended as a marked block, never replacing a hook that is already there (a
    product repository has its own). One block per Hermes home, so two homes
    linked to one checkout both stay current.
    """
    try:
        hooks = subprocess.run(["git", "-C", str(PACK), "rev-parse", "--path-format=absolute",
                                "--git-path", "hooks"], capture_output=True, text=True,
                               timeout=20).stdout.strip()
    except Exception:
        hooks = ""
    if not hooks:
        return []
    cmd = (f'"{Path(skill_py).as_posix()}" "{(PACK / "install.py").as_posix()}" --refresh '
           f'--hermes-home "{home.as_posix()}" >/dev/null 2>&1 || true')
    tag = f"{HOOK_BEGIN} ({home.as_posix()})"
    block = f"{tag}\n{cmd}\n{HOOK_END} ({home.as_posix()})\n"
    done = []
    for name in GIT_HOOKS:
        p = Path(hooks) / name
        text = p.read_text(encoding="utf-8") if p.is_file() else "#!/bin/sh\n"
        if tag in text:
            start = text.index(tag)
            end = text.index("\n", text.index(f"{HOOK_END} ({home.as_posix()})", start)) + 1
            text = text[:start] + block + text[end:]
        else:
            text = text.rstrip("\n") + "\n" + block
        if not dry:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
            try:
                p.chmod(0o755)
            except OSError:
                pass
        done.append(p.as_posix())
    return done


def link_main(a, home: Path, comfy: Path | None, skill_py: str, dry: bool) -> int:
    state = home / "comfy-state"
    conf = home / "hermes-comfy.conf"
    engine_dir = PACK / "engine"
    print("hermes-comfy installer, LINKED" + ("  [DRY RUN]" if dry else ""))
    print(f"  pack         {PACK}   (engine and scripts run from here)")
    print(f"  hermes home  {home}")
    print(f"  state        {state}   (jobs, logs, outputs)")
    print(f"  comfy home   {comfy if comfy else 'NOT FOUND -- set it in hermes-comfy.conf'}")
    print(f"  skill python {skill_py}")
    if not home.is_dir():
        print("\nERROR: no Hermes home there. Install Hermes first, or pass --hermes-home.",
              file=sys.stderr)
        return 2
    if (engine_dir / "hermes-comfy.conf").exists():
        print(f"\nWARNING: {engine_dir / 'hermes-comfy.conf'} exists inside the pack and "
              f"would be read before {conf}; delete it.", file=sys.stderr)

    print(f"\n[1/5] config -> {conf}")
    # A machine moving over from a copy install keeps its settings (COMFY_HOME,
    # the backend URL, ...): that install's conf is the starting point.
    old = home / "comfy-engine" / "hermes-comfy.conf"
    if conf.exists():
        body, how = conf.read_text(encoding="utf-8"), "updated"
    elif old.exists():
        body, how = old.read_text(encoding="utf-8"), f"seeded from {old}"
    else:
        body = (PACK / "hermes-comfy.conf.example").read_text(encoding="utf-8")
        if comfy:
            body = body.replace("#COMFY_HOME=", "COMFY_HOME=" + comfy.as_posix())
        how = "written"
    body = _set_conf_key(body, "HERMES_COMFY_ENGINE", engine_dir.as_posix())
    body = _set_conf_key(body, "HERMES_COMFY_STATE", state.as_posix())
    if not dry:
        state.mkdir(parents=True, exist_ok=True)
        conf.write_text(body, encoding="utf-8", newline="\n")
    print(f"    {how}; HERMES_COMFY_ENGINE and HERMES_COMFY_STATE point at the pack and {state}")

    print(f"\n[2/5] skill documents -> {home / 'skills'}")
    rec_path = state / LINK_RECORD
    prev = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else None
    rec = _link_render(home, comfy, skill_py, dry, quiet=False, prev=prev)

    print("\n  motion:")
    ensure_motion_venv(home, skill_py, dry)

    print("\n[3/5] git hooks (refresh after every pull)")
    hooks = _install_hooks(home, skill_py, dry)
    for h in hooks:
        print(f"    {h}")
    if not hooks:
        print("    the pack is not in a git checkout -- run `install.py --refresh` after "
              "updating it (the skills also check on start)")

    print("\n[4/5] ComfyUI plugins")
    n = _link_plugins(comfy, dry, quiet=False)
    if n:
        print("    ComfyUI must be RESTARTED to load them.")
    if not dry:
        rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    print("\n[5/5] verify")
    if dry:
        print("    skipped (dry run)")
        return 0
    env = dict(os.environ, HERMES_COMFY_ENGINE=str(engine_dir), HERMES_HOME=str(home),
               HERMES_COMFY_CONF=str(conf), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(PACK / "tools" / "doctor.py")], env=env)
    print("\nNext: restart ComfyUI (plugins) and Hermes (skills). From now on, updating the "
          "pack with git is all it takes.")
    return r.returncode


def refresh_main(a) -> int:
    """Re-render a linked install from its record. Quiet and fast: run by the git
    hooks after every pull, and by the skills when they find their documents
    older than the pack."""
    home = hermes_home(a.hermes_home)
    rec_path = home / "comfy-state" / LINK_RECORD
    if not rec_path.is_file():
        print(f"no linked install recorded in {rec_path}", file=sys.stderr)
        return 1
    prev = json.loads(rec_path.read_text(encoding="utf-8"))
    comfy = Path(prev["comfy_home"]) if prev.get("comfy_home") else None
    rec = _link_render(home, comfy, prev["python"], dry=False, quiet=True, prev=prev)
    ensure_motion_venv(home, prev["python"], dry=False, quiet=True)
    if rec["plugins"] != prev.get("plugins"):
        _link_plugins(comfy, dry=False, quiet=True)
    rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Install the hermes-comfy skills.")
    ap.add_argument("--hermes-home", help="override the Hermes home directory")
    ap.add_argument("--engine-dir", help="where to put the engine "
                                         "(default <hermes-home>/comfy-engine)")
    ap.add_argument("--comfy-home", help="the directory containing ComfyUI")
    ap.add_argument("--python", dest="skill_python",
                    help="interpreter the SKILL.md commands should name")
    ap.add_argument("--external", action="store_true",
                    help="keep the skills in this pack and register the directory "
                         "via skills.external_dirs instead of copying them into "
                         "the Hermes skills dir (protects them from the curator)")
    ap.add_argument("--link", action="store_true",
                    help="run the engine and skills straight from this pack (a git "
                         "checkout): nothing is copied, and `git pull` updates the "
                         "install -- a git hook re-renders the skill documents")
    ap.add_argument("--refresh", action="store_true",
                    help="re-render a linked install's documents (run by the git hooks)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan only")
    a = ap.parse_args()
    dry = a.dry_run

    if a.refresh:
        return refresh_main(a)
    home = hermes_home(a.hermes_home)
    if a.link:
        return link_main(a, home, find_comfy_home(a.comfy_home, home / "comfy-engine"),
                         find_skill_python(home, a.skill_python), dry)
    engine_dir = Path(a.engine_dir).expanduser() if a.engine_dir else home / "comfy-engine"
    skills_root = PACK / "skills" if a.external else home / "skills"
    comfy = find_comfy_home(a.comfy_home, engine_dir)
    skill_py = find_skill_python(home, a.skill_python)

    print("hermes-comfy installer" + ("  [DRY RUN]" if dry else ""))
    print(f"  platform     {sys.platform}")
    print(f"  hermes home  {home}" + ("" if home.is_dir() else "   <- does not exist yet"))
    print(f"  engine  ->   {engine_dir}")
    print(f"  skills  ->   {skills_root}" + ("   (external, registered not copied)"
                                             if a.external else ""))
    print(f"  comfy home   {comfy if comfy else 'NOT FOUND -- set it in hermes-comfy.conf'}")
    print(f"  skill python {skill_py}")

    if not home.is_dir():
        print("\nERROR: no Hermes home there. Install Hermes first, or pass "
              "--hermes-home.", file=sys.stderr)
        return 2

    # ---- engine ----------------------------------------------------------
    print(f"\n[1/5] engine -> {engine_dir}")
    if not dry:
        engine_dir.mkdir(parents=True, exist_ok=True)
    n_engine = copy_tree(PACK / "engine", engine_dir, dry)
    # An install that copies nothing must not report success. This is the
    # failure mode a filter bug produced once: every file skipped, every step
    # still printing "->", and the breakage only surfacing at first render.
    if n_engine == 0:
        print(f"\nERROR: copied 0 engine files from {PACK / 'engine'}. "
              f"Is the pack complete?", file=sys.stderr)
        return 3
    print(f"    ({n_engine} files)")

    # ---- config ----------------------------------------------------------
    conf = engine_dir / "hermes-comfy.conf"
    print(f"\n[2/5] config -> {conf}")
    if conf.exists():
        print("    already exists, left untouched (delete it to regenerate)")
    else:
        body = (PACK / "hermes-comfy.conf.example").read_text(encoding="utf-8")
        if comfy:
            body = body.replace("#COMFY_HOME=", "COMFY_HOME=" + comfy.as_posix())
        body = body.replace("#HERMES_COMFY_ENGINE=",
                            "HERMES_COMFY_ENGINE=" + engine_dir.as_posix())
        if not dry:
            conf.write_text(body, encoding="utf-8", newline="\n")
        print("    written" + (f" with COMFY_HOME={comfy.as_posix()}" if comfy
                               else " -- YOU MUST SET COMFY_HOME IN IT"))

    # ---- skills ----------------------------------------------------------
    print(f"\n[3/5] skills -> {skills_root}")
    # Nested is not the same as portable. A portable build and AI Scaler Toolkit's
    # install both keep main.py under <home>/ComfyUI, and only the first has a
    # python_embeded beside it -- treating every nested home as portable rendered
    # a python_embeded path that did not exist into the skill documents.
    nested = bool(comfy and (comfy / "ComfyUI" / "main.py").is_file())
    comfy_root = (comfy / "ComfyUI") if nested else comfy
    if comfy:
        eng = _engine_paths()
        # The engine's own answer when it resolved the same home (it has the conf's
        # COMFY_PYTHON and the toolkit's selection); its probe for any other home.
        comfy_py = (Path(eng.COMFY_PYTHON) if Path(eng.COMFY_HOME) == comfy
                    else Path(eng._find_comfy_python(str(comfy), str(comfy_root))))
    else:
        comfy_py = Path("<COMFY_HOME>/venv/bin/python")
    subs_common = {
        "@@PY@@": Path(skill_py).as_posix(),
        "@@ENGINE@@": engine_dir.as_posix(),
        "@@OUT@@": (engine_dir / "out").as_posix(),
        "@@COMFY_OUTPUT@@": (comfy_root / "output").as_posix() if comfy_root
                            else "<COMFY_HOME>/output",
        # ComfyUI's own interpreter, not the one running the skill.
        "@@COMFY_PY@@": comfy_py.as_posix(),
    }
    for s in SKILLS:
        src_dir = PACK / "skills" / s
        dst_dir = skills_root / s
        print(f"  {s}:")
        if not a.external:
            if dst_dir.exists():
                backup(dst_dir, dry, outside=engine_dir / "skill-backups")
            n = copy_tree(src_dir, dst_dir, dry)
            if n == 0:
                print(f"\nERROR: copied 0 files for skill '{s}' from {src_dir}.",
                      file=sys.stderr)
                return 3
        subs = dict(subs_common)
        subs["@@IMGGEN@@"] = (skills_root / "imggen" / "gen.py").as_posix()
        subs["@@VIDGEN@@"] = (skills_root / "vidgen" / "gen.py").as_posix()
        subs["@@STORY@@"] = (skills_root / "dramagen" / "story.py").as_posix()
        subs["@@PRODUCE@@"] = (skills_root / "dramagen" / "produce.py").as_posix()
        subs["@@DRAMA@@"] = (home / "drama").as_posix()
        subs["@@RECAST@@"] = (skills_root / "recast" / "recast.py").as_posix()
        subs["@@RECASTOUT@@"] = (home / "recast").as_posix()
        subs["@@GPUSHARE@@"] = (skills_root / "gpushare" / "gpushare.py").as_posix()
        subs.update(motion_subs(home, skills_root))
        # Every *.in is a template, not just SKILL.md: references/ is linked
        # from SKILL.md and the model reads it, so its commands have to be real
        # paths too. Rendering only SKILL.md left stale ones there once already.
        for tmpl in sorted(src_dir.rglob("*.in")):
            rel = tmpl.relative_to(src_dir)
            dst = dst_dir / rel.with_suffix("")     # foo.md.in -> foo.md
            render_skill(tmpl, dst, subs, dry)
            print(f"    {rel.with_suffix('').as_posix()} rendered")
            stale = dst_dir / rel
            # --external renders in place: there the "copy" IS the pack's own
            # template, and deleting it removed a tracked file from the checkout.
            if stale.exists() and not dry and not a.external:
                stale.unlink()                       # don't ship the template
        # The skill has to find the engine at runtime; env vars are not
        # guaranteed to reach it, so the path is written down next to it.
        if not dry:
            (dst_dir / "engine_dir.txt").write_text(engine_dir.as_posix() + "\n",
                                                    encoding="utf-8")
        print(f"    engine_dir.txt -> {engine_dir}")

    if a.external:
        print(f"\n  Add this to {home / 'config.yaml'} so Hermes loads them:\n"
              f"    skills:\n"
              f"      external_dirs:\n"
              f"        - {skills_root.as_posix()}")

    print("\n  motion:")
    ensure_motion_venv(home, skill_py, dry)

    # ---- ComfyUI plugins --------------------------------------------------
    # comfy_root was resolved for the skill substitutions above.
    print("\n[4/5] ComfyUI plugins")
    if comfy_root is None:
        print("    SKIPPED: COMFY_HOME is unknown, so there is no custom_nodes to "
              "copy into.\n    Set it in hermes-comfy.conf and re-run, or copy "
              f"{(PACK / 'comfyui-plugins').as_posix()}/* into <ComfyUI>/custom_nodes "
              "by hand.\n    Until then `r2v --ref ... --image ...` will fail in the "
              "sampler.")
    else:
        cn = comfy_root / "custom_nodes"
        for name in COMFY_PLUGINS:
            n = copy_tree(PACK / "comfyui-plugins" / name, cn / name, dry)
            print(f"  {name}: {n} files -> {(cn / name).as_posix()}")
            if n == 0:
                print(f"\nERROR: copied 0 files for ComfyUI plugin '{name}'.",
                      file=sys.stderr)
                return 3
        print("    ComfyUI must be RESTARTED to load them.")

    # ---- verify ----------------------------------------------------------
    print("\n[5/5] verify")
    if dry:
        print("    skipped (dry run)")
        return 0
    # HERMES_HOME must be propagated or doctor reports the default profile's
    # skills dir while we just installed into a different one.
    env = dict(os.environ, HERMES_COMFY_ENGINE=str(engine_dir),
               HERMES_HOME=str(home), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(PACK / "tools" / "doctor.py")],
                       env=env)
    print("\nNext:")
    print(f"  1. Check the paths above, then edit {conf} if anything is wrong.")
    print(f"  2. Download any models doctor.py listed as MISSING (see MODELS.md).")
    print(f"  3. Set terminal.timeout: 600 in {home / 'config.yaml'}.")
    print(f"  4. Restart ComfyUI so it loads the custom nodes from [4/5].")
    print(f"  5. Restart Hermes so it picks up the new skills.")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
