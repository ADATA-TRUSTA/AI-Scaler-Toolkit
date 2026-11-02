# comfyui-skill (hermes-comfy-pack)

English | [繁體中文](README.zh-TW.md)

Five skills that let a [Hermes](https://github.com/NousResearch/hermes-agent) agent make images, video and
explainer animation **on your own machine** — images and video on a local ComfyUI, animation by code —
packaged so they can be moved to another machine. Supports **Windows and Linux**. Model weights are not
included (download them separately, see [`MODELS.md`](MODELS.md)).

| Skill | What it does |
|---|---|
| **`imggen`** | Image generation and editing. `qwen21` (Qwen-Image-2.1: best prompt adherence, renders Chinese text, takes reference images), `qwen21_edit` (edit an image: change outfit or background, composite several characters), `qwen21_move` (move a person within the frame), `qwen`, `flux` (Apache license, commercial use OK), `zimage` (fast), `realvis` (photorealistic), `img2img` |
| **`vidgen`** | Video generation. Wan 2.2 I2V (fast, silent) and MiniMax-H3 (**native stereo audio**; can lock a character's look with reference images, pin the first frame, or re-enact a reference video with the people replaced) |
| **`recast`** | A reference video + photos of people → the same video with those people in it. Motion, camera, framing, background and the original audio follow the video. Renders shot by shot, **checks every result automatically and re-rolls the seed when a check fails** |
| **`motion`** | Screenshots and photos → an explainer / promo animation, by code: rows appearing, values outlined, pan and zoom, panels cut from the screen, sound effects. **Every letter on screen stays the original pixels** (nothing re-typed, nothing drawn by a model), so the numbers in the video are the real ones. No GPU |
| **`dramagen`** | One-line premise → a short drama. The LLM writes a shot list, then drives `imggen` and `vidgen` to render stills and clips and joins them into a subtitled film |

`recast` and `dramagen` never touch ComfyUI themselves — they run `vidgen/gen.py` and `imggen/gen.py`,
so the memory protection described below holds for them unchanged.

**[See it working →](docs/showcase/README.md)** every skill driven by Hermes on a **local** LLM, on one laptop:
the requests as typed, the follow-up edits, and the results.

Detailed measurements behind the claims in this README are in **[`TEST_RESULTS.md`](docs/TEST_RESULTS.md)**.

## How this differs from "a quick script that hits the ComfyUI API"

It assumes **an LLM you are actively using lives on the same machine**, and that the agent driving this
whole pipeline *is* that LLM.

- **Memory planner** (`memplan.py`): measures real VRAM/RAM before rendering and decides which tier the
  weights go in (VRAM / DRAM / SSD). If it doesn't fit, it **refuses and explains why**, instead of
  squeezing out the LLM on the same machine.
- **Protected launcher** (`start_comfy.py`): computes `--reserve-vram` from the measured external VRAM
  usage, so ComfyUI cannot eat the LLM's VRAM. ComfyUI is bound to `127.0.0.1` only, and paid cloud API
  nodes are disabled by default (`--disable-api-nodes`; an agent once routed around `gen.py` to call them).
- **Ordered memory hand-off**: the LLM's DRAM is handed over only after ComfyUI has confirmed it is
  rendering. Releasing early means the agent has to re-read its own weights from SSD, which looks like
  Hermes has frozen.
- **GPU lending for engines that fill the card** (`llmsleep.py`): some local engines size themselves to
  the whole GPU on purpose (Strata, via TRUSTA-AST, measured 14.3 GB of a 16 GB card). Holding VRAM back
  for those leaves nothing to render with, so instead each render waits until the agent has finished its
  reply, puts the LLM to sleep — or unloads it when there is not enough RAM for the sleep backup; the
  backend keeps the conversation — renders, and loads it back (~35 s on the reference machine).
- **Measured pitfalls written into SKILL.md**: `t2v` producing pure noise while reporting success,
  reference images shrunk until the face is 78 px, resubmitting after a timeout and paying for the render
  twice, H3 inventing cuts the source video does not have… each is written as a rule the model can read,
  with the measurement that motivated it.

## With AI Scaler Toolkit

AI Scaler Toolkit carries this pack (in its public tree under `hermes-skills/`), so there is nothing to clone.
In its settings, **Components**:

1. **Hermes Agent** — install it first; the skills are installed into it.
2. **Hermes image & video skills** — Install. Runs this pack's `install.py` with Hermes's Python, builds
   `motion`'s environment with uv, and needs no download beyond motion's packages. `motion` works at once.
3. **ComfyUI** — pick the card it runs on. That installs ComfyUI with the `ComfyUI-GGUF` node, then
   installs the skills again so they find it.

Updating AI Scaler Toolkit brings a new version of the skills; their card says so and Reinstall applies it.
The model weights are not included either way: see [`MODELS.md`](MODELS.md).

## Quick start

```
git clone https://github.com/TRUSTA-SW-FAE/comfyui-skill.git
cd comfyui-skill
```

Then install it one of two ways:

| | Command | Updating later |
|---|---|---|
| **Linked** (recommended for a git checkout) | `python install.py --link` | `git pull` — nothing else. Git hooks re-render the skill documents, and the engine and scripts run straight from the checkout |
| **Copied** | Windows `.\install.ps1`, Linux `./install.sh --comfy-home /opt/ComfyUI` | pull or download the new version and run the installer again |

Either way the installer also builds `motion`'s own Python environment (`<hermes home>/motion-venv`) and
copies the pack's ComfyUI node. Then:

```
python tools/doctor.py --models   # check every path, node, and that all 27 models are present
```

Restart ComfyUI (for the node) and Hermes (for the skills).

For full instructions see **[`INSTALL.md`](INSTALL.md)**; for the model list see **[`MODELS.md`](MODELS.md)**.

## Contents

```
comfyui-skill/
├── INSTALL.md (+ .zh-TW.md)    Installation and setup (Windows / Linux) ★ read this first
├── MODELS.md (+ .zh-TW.md)     27 models (+ optional Turbo LoRA): file, size, folder, license, which feature uses it
├── docs/TEST_RESULTS.md (+ .zh-TW.md)  What was measured, on what, and the numbers
├── THIRD_PARTY_NOTICES.md      Licenses of included third-party content; which models are non-commercial
├── install.py                  Cross-platform installer (.sh / .ps1 are thin wrappers)
├── install.sh / install.ps1
├── hermes-comfy.conf.example   ★ Template for the only config file you need to edit
├── engine/                     → installed to <hermes home>/comfy-engine/
│   ├── paths.py                Single source of truth for all paths
│   ├── astapi.py               The only place that reads the TRUSTA-AST backend's status
│   ├── llmsleep.py             Lends a GPU-filling LLM's VRAM to a render and gives it back
│   ├── linkcheck.py            Keeps a linked install's skill documents in step with the checkout
│   ├── osmem.py                OS abstraction layer (processes, memory, per-process VRAM, reclaim, detach)
│   ├── sysmon.py               Real VRAM / RAM
│   ├── memplan.py              Memory-aware planner + ordering guarantee for the hand-off
│   ├── video_gen.py            Video runner (async jobs, detached worker)
│   ├── start_comfy.py          VRAM-protected ComfyUI launcher
│   ├── comfyguard.py           Checks that a running ComfyUI protects the LLM and is on the selected GPU
│   ├── h3_media.py             Audio/video muxing and shot detection for H3 (runs under ComfyUI's python)
│   ├── h3_resume.py / h3_wait_join.py   Resume and join H3 segments
│   ├── prewarm_llm.py          Reads the LLM's GGUF back into the page cache after a render
│   ├── verify_safetensors.py   Checks that a safetensors download is complete
│   └── workflows/*.json        12 verified workflows
├── comfyui-plugins/            → copied into <ComfyUI>/custom_nodes/ (H3 reference images + pinned first frame)
├── skills/                     → installed to <hermes home>/skills/
│   ├── imggen/{SKILL.md.in, gen.py}
│   ├── vidgen/{SKILL.md.in, gen.py, references/traps-and-misuse.md.in}
│   ├── recast/{SKILL.md.in, recast.py}
│   ├── motion/{SKILL.md.in, motion.py, lib/, tests/, requirements.txt}   (own venv: <hermes home>/motion-venv)
│   └── dramagen/{SKILL.md.in, story.py, produce.py, references/screenwriting.md}
├── docs/showcase/              Results made fully locally, with the prompts that made them
├── bench/linux/                A test kit that runs on a fresh Linux box
├── tools/
│   ├── doctor.py               Post-install self-check
│   ├── wsl_bootstrap.sh        WSL split install (Hermes in Ubuntu, ComfyUI stays on Windows)
│   └── linux-reclaim-helper.sh Memory reclaim on Linux (cgroup / process_madvise)
└── tests/test_*.py             Self-tests that run on any platform: backend API parsing, ComfyUI guard,
                                dramagen, multi-vendor VRAM, the Linux branch on a fake /proc, a live
                                process tree, WSL path resolution
```

## GPU vendor neutral

The engine asks only one thing: **how much of this card is already in use, and how much is left**; only
what's left goes to generation. **Vendor tools first, service APIs only if none are available**, tried in
this order:

`VIDGEN_GPU_QUERY_CMD` (custom) → `nvidia-smi` → `intel-ze` (Intel Level Zero)
→ `xpu-smi` (Intel CLI) → `rocm-smi` (AMD)
→ **ComfyUI's `/system_stats`** (reports whatever backend torch is on, no extra tools needed)
→ **TRUSTA-AST's `/system/resources`** (the backend's own measurement, last resort)

The first few read from the driver, so they see usage by **every** process on the card — that is the
number the planner needs. `intel-ze` comes before `xpu-smi` because the Level Zero loader ships with the
Intel graphics driver, while `xpu-smi` is a separately installed CLI. It **runs in a subprocess** because
some machines have the loader but not a matching Level Zero driver, and then the loader crashes outright
instead of returning an error; that must not happen inside the planner. An integrated GPU's memory is
counted once (it is system RAM).

The last two are safety nets so that unfamiliar accelerators also work with zero configuration.
`doctor.py` tells you which source answered.

## Relationship with the local LLM

This pack is **designed to pair with TRUSTA-AST**, and the inference engine underneath AST is swappable.
So the pack never pokes an engine's port: "is a model loaded, how much VRAM does it use, is there a process
whose memory can be reclaimed" are all asked of AST (default `127.0.0.1:8571`; set `VIDGEN_AST_URL` when
it runs elsewhere). Switching engines requires no change here.

What happens to the LLM during a render depends on the engine:

| Engine behaviour | During a render |
|---|---|
| Holds no more VRAM than `VIDGEN_LLM_VRAM_BUDGET` (7 GB by default; e.g. a 35B MoE with experts in RAM) | VRAM is held back for it; its clean DRAM pages are handed to the render once rendering has started |
| Holds more than that — e.g. Strata, which fills the card on purpose | Put to sleep, or unloaded when the sleep backup does not fit in RAM; loaded back afterwards. The agent's conversation is kept |
| Not reachable | The planner **assumes the LLM is there** and reserves VRAM — guessing wrong costs a shorter video, not the agent that drives the pipeline |

`SKILL.md.in` is a template: placeholders such as `@@PY@@` and `@@VIDGEN@@` are replaced at install time
with this machine's absolute paths, because SKILL.md gives the model complete commands to run.

## Requirements

NVIDIA GPU (12 GB+ VRAM recommended), Python 3.9+, ComfyUI (with the `ComfyUI-GGUF` custom node),
40–171 GB of disk for models. Hermes must already be installed. `motion` needs no GPU and no ComfyUI:
only Python and `ffmpeg` (its packages are installed into its own venv). The `qwen21` models need a ComfyUI with
the `TextEncodeQwenImage21` node (verified on 0.37.0). Viggle Turbo for `qwen21` is optional and must be
downloaded separately (research license, not shipped); without it `gen.py` falls back to the slower base
sampler. See `MODELS.md`.

## Verified environments (last updated 2026-10)

| Layout | Status |
|---|---|
| **Windows 11** (ComfyUI portable + Hermes + TRUSTA-AST on one laptop, RTX 5080 Laptop 16 GB, 32 GB RAM) | ✅ **Fully local**: Hermes on a local Strata LLM drove `motion`, `imggen`, `vidgen` and `recast` by natural language, including follow-up edits ([showcase](docs/showcase/README.md)); `recast` also with a remote Qwen3.8-Flash. Linked install updated by `git pull` |
| **WSL2 + Ubuntu 24.04.4** (Hermes in the distro, ComfyUI/models/LLM stay on Windows) | ✅ Image in 16 s, i2v video in 132 s, Hermes agent driven by natural language ×3 |
| **Native Linux** (Ubuntu 24.04.4, RTX 5060 Ti 16 GB) | ✅ End-to-end images; Hermes (Qwen3.8-27B and Qwen3.6-35B-A3B) drove multi-image outfit changes, multi-turn refinement, background + multi-character compositing and moving people |

Not yet run on Linux: the GPU-lending path (`llmsleep.py`), `recast` and `motion`.

The `tests/test_*.py` self-tests and `skills/motion/tests/test_motion.py` run on any platform. On a new
machine, run `tools/doctor.py` first.

## License

This pack is licensed under the [Apache License 2.0](LICENSE), the same as TRUSTA-AST.

Third-party content and model licenses are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
**Some models are non-commercial** (Qwen-Image-2.1 and Viggle Turbo are under the Qwen Research License;
see `MODELS.md`). The weights are never shipped with this pack.
