---
title: The Strata engine
summary: Experts of a large MoE live in DRAM, hot experts are cached on the GPU; deep sleep (NVIDIA) releases the GPU in about a second and keeps the conversation. Optional; the installer carries the package, for NVIDIA and for Intel Arc.
updatedAt: 2026-11-20
---

# The Strata engine

[Strata](https://github.com/Niko1221/Strata) is an inference server built for large MoE models.
AST uses a build with **deep sleep** added.

## The problem it solves

A MoE's experts live in **DRAM**, and **hot experts are cached on the GPU**, with the cache sized to
fill the card.

| | vLLM expert offload | Strata |
|---|---|---|
| Where experts live | Chosen with `vllm_cpu_offload_params=["experts"]` | That is the design |
| What is on the GPU | KV cache + non-expert weights | **A cache of hot experts** |
| Releasing the GPU | sleep: **KV dropped** | deep sleep: **KV and conversation kept** |

## Can it be used

Optional. **The installer carries a Strata package and installs it the first time you load a Strata
model**, so you do not set `STRATA_HOME` yourself. Which package depends on the GPU:

| GPU | Package | Deep sleep |
|---|---|---|
| NVIDIA (checked first: a PC with both gets this one) | CUDA | **Yes** |
| Intel Arc (Windows or Linux) | Intel (SYCL) | **No** (see below) |
| Anything else | None; the `strata` engine is not offered | |

> The package is Strata with **deep sleep** added, compiled, together with Strata's Python side;
> it does not come from Strata's own download. The backend always starts the CUDA engine with
> `--sleep-mode`, so the prebuilt engine Strata's own setup downloads (which lacks that flag)
> cannot be used.

If you run the backend from a source checkout instead, `STRATA_HOME` pointing to an installed
package (holding `serve/server.py`) makes `strata` appear in `GET /health`'s `supported_engines`;
the settings are in the Strata section of [`backend/.env.example`](../../../backend/.env.example).

### Which GGUF files

Strata's setup accepts a fixed list of GGUF files (the first shard's file name): GSQ-RCO
(Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), Swift 1.5 (IQ2_XS, IQ3_XXS, IQ3_S), Coder (IQ1_M), Unsloth
(UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q8_0) and Gyro-S / Gyro-M. Other quantizations,
including Unsloth's UD-IQ3_XXS and UD-Q2_K_XL, are refused. `GET /config/models` marks each GGUF
with `strata_supported`, and the Strata menu only offers those.

### The first load

If the GGUF has no run config yet, the first load installs the package (if needed) and runs
Strata's setup against the GGUF **where it already is**; only the expert pack, the draft layer and
the run config are generated. A file name that is not in the list is refused before any of this.
Later loads start directly.

## Intel Arc (SYCL) package

| | CUDA package | Intel Arc package |
|---|---|---|
| Deep sleep | Yes | **No**: it is built on the CUDA driver's virtual memory, which Level Zero has no equivalent of |
| `POST /inference/sleep`, `/wake` | Allowed | **409** `This Strata engine has no deep sleep…` |
| Image reader | `strata-vision` on the GPU | `strata-vision-sycl` on the GPU, `strata-vision` on the CPU as the fallback |
| `device` in the ready message | `cuda` | `sycl` |

The Intel engine is built to run on **any x86-64 CPU with AVX2**, not only the one it was built on,
and a model that is sharded keeps its own file names (as a Hugging Face cache symlink does).
Both were fixed in the 0.2.11-sleep.2 packages after a build made on a newer CPU died with an
illegal-instruction error on a server CPU without those instructions.

Measured numbers: [Strata on Intel Arc](../benchmarks/strata-intel-arc.md).

> **Newer and less tested than the CUDA path.** The packages were installed and run on a laptop
> with an integrated Arc GPU (Windows) and on a PC with a discrete Arc GPU of about 32 GB (Linux), but through Strata's own launcher; the full path
> through AST's backend has not been run on an Intel machine yet. On Windows the backend sets
> `STRATA_DOORBELL_CHECK=1` for the Intel engine by default (without it the engine stays up but
> its replies can drift).

## How a model is described: the run config

A Strata model is not described by a list of arguments but by the **run config** Strata's setup
writes (`strata-<model>.json`: engine binary, expert pack, GGUF shards, vision encoder).

```
your strata-<model>.json ──copied, this load's overrides applied──► STRATA_RUNTIME_DIR/ ──started from that copy
```

**Your file is never modified.** Without `strata_config_path`, the backend looks in `STRATA_HOME`
for the run config whose first GGUF shard is `model_path`.

## Only four load fields

```
strata_config_path · strata_max_context · strata_vram_limit_gb · strata_auto_wake
```

The other startup flags (KV format, KV placement, speculative window, expert cache) stay in the run
config; edit that file to tune them. This is the opposite of vLLM: **a small API, a rich file**.

### `strata_vram_limit_gb`: less is slower; leave it unset

Strata fills the GPU with its expert cache; this value is room kept for other programs.

| Available VRAM | Decode |
|---|---|
| About 15 GB | 72–84 tok/s |
| About 8 GB | 45–54 tok/s |

**Leave it unset, and `POST /inference/sleep` when another program needs the GPU.**

## Deep sleep: release the GPU in about a second, keep the conversation

Always on; it costs nothing until the first sleep.

| | vLLM sleep | Strata deep sleep |
|---|---|---|
| Weights | Moved to pinned CPU memory | The engine's VRAM is copied to pinned host RAM |
| KV cache | **Dropped** | **Kept** |

Measured (Windows, RTX 5060 Ti 16 GB, GSQ-RCO IQ3_S, two shards of about 78 GB): sleep 1.3 s,
VRAM from 15,866 to 1,498 MiB; wake 1.0 s, and the conversation continued. The endpoints are shared
with vLLM: `POST /inference/sleep`, `POST /inference/wake`; see the sleep/wake section of
[vLLM](vllm.md) for their behaviour.

For a comparison of three ways to give up the GPU at 256K context, see
[three ways to hand off the GPU](../benchmarks/gpu-handoff.md).

### The conversation is saved and restored on the next load

On unload the slot is saved to `STRATA_RUNTIME_DIR/slots/<model>.slot-0.bin` and restored on the
next load. No saved file is not an error: `{"restored": false, "reason": "no saved conversation"}`.

> This is not the server-side chat history of `session_id`: what is saved here is the engine's own
> KV, not a message list.

## Environment defaults at load

```
STRATA_UNBUFFERED_LOAD=1    STRATA_NO_ARENA_THP=1          (all platforms)
STRATA_PLE_AUTO_RAM=0                                       (Windows)
STRATA_DOORBELL_CHECK=1                                     (Windows, Intel engine)
```

Applied only when you have not set them yourself. The first two are explained below; on Windows the
third keeps the engine from locking a large table in RAM, which would leave no room for the sleep
backup (every sleep answered 409), and the fourth is the Intel check described above.

The first two: The reason is a measured incident: loading a
50 GiB expert arena through the page cache kept memory pressure above systemd-oomd's 50% threshold
for long enough (69.8% measured) that **oomd logged out the whole desktop session**. Unbuffered,
with transparent huge pages off, the same load peaked at 0.2% and **read faster**
(9.1–10.0 GiB/s against 6.4).

## How it runs

Like llama-server and vLLM: the worker starts the server (listening only on
`127.0.0.1:STRATA_PORT`) and the main process talks to it over HTTP. The wait for the model to load
is capped by `STRATA_HEALTH_TIMEOUT`, 900 s by default (it reads tens of GB). On failure the last 40
lines of the server log are attached.

## Settings

| Variable | Default | Notes |
|---|---|---|
| `STRATA_HOME` | empty | Strata's install directory (holding `serve/server.py`). Empty: the installer's package is installed on the first load |
| `STRATA_INSTALL_DIR` | empty | Where that package is installed: empty means next to the project; the desktop app uses a `strata` folder in its data directory |
| `STRATA_PYTHON` | — | Interpreter for `serve/server.py`, normally Strata's own `.venv` |
| `STRATA_ENGINE_BINARY` | — | The engine build; written for you when the package is installed |
| `STRATA_PORT` | `5002` | Loopback port |
| `STRATA_HEALTH_TIMEOUT` | `900` | Seconds to wait for the model to load |
| `STRATA_RUNTIME_DIR` | `./.cache/strata` | Run-config copies, server log, saved conversations |

## Related

- [Choosing an inference engine](index.md)
- [vLLM](vllm.md) — the other engine that can sleep
- [Strata on Intel Arc](../benchmarks/strata-intel-arc.md)
- [Prompt reading with AMX](../benchmarks/strata-amx-prefill.md)
- [Strata and vLLM on one 48 GB GPU](../benchmarks/strata-vs-vllm.md)
- [Three ways to hand off the GPU](../benchmarks/gpu-handoff.md)
