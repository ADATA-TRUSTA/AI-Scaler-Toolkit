# GGUF / llama.cpp memory estimation

Estimates the memory usage of GGUF models under llama.cpp. Weights and KV cache are **computed exactly**
from the GGUF tensor table and match llama.cpp's own numbers; the compute buffer uses a calibrated
formula that deliberately overestimates slightly.

All units are **MiB** (matching llama.cpp's output, unlike the older transformers estimator which used GB).

## The five endpoints

| Endpoint | Purpose |
|---|---|
| `POST /inference/estimate_memory/gguf/check` | **The main one for the frontend.** Send the same config as `load_model` and get an answer to "will these settings blow up?" |
| `POST /inference/estimate_memory/gguf` | Single-point estimate + per-layer breakdown |
| `POST /inference/estimate_memory/gguf/sweep` | Feasibility table over `-ngl` × context × KV quantization |
| `POST /inference/estimate_memory/gguf/recommend` | Solve for **one** best setting given the current hardware (or a manual cap) |
| `POST /inference/estimate_memory/gguf/plan` | Solve one per GPU budget × KV quantization and return a **menu** |

---

## 1. Frontend pre-check: `/check`

Send **exactly the same body as `/inference/load_model`** — no need to assemble separate parameters.
llama.cpp flags inside `llama_server_extra_args` are parsed automatically.

```bash
curl -s -X POST 'localhost:8000/inference/estimate_memory/gguf/check' \
  -H 'content-type: application/json' -d '{
    "model_name": "Qwen3-30B-A3B-Q4_K_M",
    "model_path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    "engine": "llama_server",
    "n_gpu_layers": 25,
    "n_ctx": 32768,
    "n_batch": 2048,
    "llama_server_np": 4,
    "llama_server_extra_args": ["-ncmoe","24","-ctk","q8_0","-ctv","q8_0","-ub","1024"]
  }'
```

Key parts of the response:

```jsonc
{
  "fits": true,
  "verdict": "ok",                       // ok | gpu_short | host_short | unknown_budget
  "summary": "Fits: 13574 MiB of 15473 MiB free VRAM, leaving 1899 MiB headroom.",
  "resolved_settings": {                 // the parameters actually in effect after merging config + extra_args
    "n_gpu_layers": 25, "n_ctx": 32768, "n_ubatch": 1024,
    "cache_type_k": "q8_0", "n_cpu_moe": 24, "n_parallel": 4
  },
  "parsed_args": {
    "parsed":  { "n_cpu_moe": 24, "cache_type_k": "q8_0", "n_ubatch": 1024 },
    "ignored": ["--host", "--jinja"],    // memory-irrelevant flags, never silently swallowed
    "warnings": ["--mlock locks the weights in RAM, so the host figure is resident memory that cannot be evicted under pressure."]
  },
  "memory_breakdown_mib": {
    "gpu_total": { "model": 10670, "context": 2560, "compute": 344, "total": 13574 },
    "host":      { "model": 22529, "context": 5632, "compute": 84,  "total": 28245 }
  },
  "budgets_mib": { "gpu_budget_mib": 15473, "host_budget_mib": 120668, "source": "system_monitor" }
}
```

When it does not fit, an extra `suggestion` is included:

```jsonc
{
  "verdict": "gpu_short",
  "summary": "Does not fit: needs 1228 MiB plus a 200 MiB margin, but only 1200 MiB of VRAM is free.",
  "suggestion": {
    "recommended": { "n_gpu_layers": 17, "n_cpu_moe": 0, "n_ctx": 32768, "kv_quant": "f16" },
    "llama_server_args": "-ngl 17 -c 32768 -fa on",
    "memory_breakdown_mib": { "gpu_total": { "total": 979 } }
  }
}
```

If not even a suggestion can be found (e.g. `-ub 4096` alone eats all the VRAM), `dominant_cost` points at the real bottleneck:

> The GPU compute buffer dominates at 2749 MiB, driven by -ub 4096 (and -b 2048).
> Lower the ubatch size first; it scales this buffer almost linearly.

Available query parameters: `margin_mib` (default 1024), `verify`, `include_per_layer`,
`gpu_budget_mib`, `host_budget_mib`.

### Supported `llama_server_extra_args` flags

| Affects memory | Aliases |
|---|---|
| GPU layers | `-ngl` / `--gpu-layers` / `--n-gpu-layers` (incl. `auto` / `all` / `-1`)|
| context | `-c` / `--ctx-size` |
| batch | `-b` / `--batch-size`, `-ub` / `--ubatch-size` |
| parallel slots | `-np` / `--parallel` |
| KV quantization | `-ctk` / `--cache-type-k`, `-ctv` / `--cache-type-v` |
| MoE offload | `-ncmoe` / `--n-cpu-moe`, `-cmoe` / `--cpu-moe` |
| tensor override | `-ot` / `--override-tensor` (the `=CPU` parts are counted)|
| others | `-fa`, `-ts`, `--no-kv-offload` / `-nkvo`, `--swa-full`, `-m` |

Both `--flag=value` and `-flag value` forms work; `--mlock` / `--no-mmap` do not change the total
but do add a note that the host side turns from page cache into real RSS. Remaining flags are listed in `parsed_args.ignored`.

**Precedence**: parameters inside `llama_server_extra_args` override the JSON `n_gpu_layers` / `n_ctx` fields —
consistent with llama-server appending extra args at the end of the command line and llama.cpp letting the later value win.

---

## 2. Single-point estimate + per-layer breakdown: `/gguf`

```bash
curl -s -X POST 'localhost:8000/inference/estimate_memory/gguf' \
  -H 'content-type: application/json' \
  -d '{"model_path":"/models/Qwen3-32B-Q8_0.gguf","n_gpu_layers":20,"n_ctx":32768,"verify":true}'
```

`per_layer_mib` gives the role breakdown of each layer:

```jsonc
{ "index": 0, "device": "host", "attn": 95.65, "ffn": 398.46,
  "moe_exps": 0.0, "shexp": 0.0, "weights": 494.10, "kv": 38.0, "swa": false }
```

`verify: true` additionally runs `llama-fit-params` for exact values (about 0.4 s):

```jsonc
"verification": { "source": "llama_cpp",
  "memory_breakdown_mib": { "gpu_total": { "model": 10176, "context": 2432, "compute": 310 } } }
```

This endpoint also accepts `args` (equivalent to `llama_server_extra_args`):

```json
{ "model_path": "/models/x.gguf", "args": "-ngl 20 -c 32768 -ctk q8_0 -ncmoe 24" }
```

## 3. Feasibility sweep: `/sweep`

```bash
curl -s -X POST 'localhost:8000/inference/estimate_memory/gguf/sweep' \
  -H 'content-type: application/json' -d '{
    "model_path": "/models/Qwen3-32B-Q8_0.gguf",
    "n_gpu_layers_grid": [0,20,40,65],
    "n_ctx_grid": [4096,32768],
    "kv_quant_grid": ["f16","q8_0"]
  }'
```

Purely analytic — no process is started, so the whole grid is nearly free. Each row has `gpu_mib` / `host_mib` /
`fits_gpu` / `fits_host`, ready for the frontend to render as a table or heatmap. Without a budget it reads the currently available VRAM/DRAM.

`-ngl` / `-c` / `-ctk` are the sweep axes; if they appear in `args` they are ignored and explained in `parsed_args.warnings`.

## 4. Automatic suggestion / manual GPU cap: `/recommend`

`gpu_budget_mib` is a **cap**: the result is guaranteed to stay under `gpu_budget_mib - margin_mib`, and it tries to **fill** it,
not merely to "fit".

```bash
# I only want to give this model 12 GiB of VRAM and leave the rest for other things
curl -s -X POST 'localhost:8000/inference/estimate_memory/gguf/recommend' \
  -H 'content-type: application/json' -d '{
    "model_path": "/models/Qwen3-32B-Q8_0.gguf",
    "gpu_budget_mib": 12288,
    "margin_mib": 512,
    "target_utilization": 0.95
  }'
```

Three steps:

1. **Decide the context.** If `n_ctx` is given it is used as is; it is only lowered when it does not fit and
   `allow_ctx_reduction` is set (binary search, not halving). `n_ctx` left at 0 (or omitted) means the search decides,
   starting from `n_ctx_min`.
2. **Maximize offload.** MoE uses `--n-cpu-moe` (attention and KV stay on GPU), otherwise `-ngl` is tuned.
   Layers take priority over context, because offload affects speed far more.
3. **Turn the remaining budget into context.** Only done when `n_ctx` is unspecified; the cap is `n_ctx_max`
   (defaults to the model's trained length).

Measured (Qwen2.5-0.5B, `n_ctx_max=262144`):

| cap | margin | suggested args | llama.cpp actual | utilization |
|---|---|---|---|---|
| 1000 | 50 | `-ngl 25 -c 8960 -fa on` | 924 MiB | 97.3% |
| 1500 | 100 | `-ngl 25 -c 47360 -fa on` | 1353 MiB | 96.6% |
| 3000 | 100 | `-ngl 25 -c 158976 -fa on` | 2835 MiB | 97.8% |

The returned `utilization` tells you directly whether the budget was filled:

```jsonc
"utilization": {
  "gpu_budget_mib": 1500,
  "usable_budget_mib": 1400,     // cap - margin
  "allocated_mib": 1353,
  "allocated_source": "llama_cpp",  // measured value when verify is on, otherwise "analytic"
  "analytic_mib": 1397,
  "headroom_mib": 47,
  "utilization_pct": 96.6,
  "within_budget": true,
  "meets_target": true,
  "correction_mib": 0             // closed-loop correction amount, see below
}
```

**Closed-loop correction**: `verify` is on by default. After the analytic pass, `llama-fit-params` measures once; if the
measured value exceeds the cap, the target is tightened by that difference and the search reruns (up to 3 rounds), with
`correction_mib` recording how much it was tightened. The compute buffer is the only inexact term, and this step keeps
"hugging the cap" safe.

When the budget is not filled, `notes` explains what it is stuck on: `n_ctx` pinned, context already at the `n_ctx_max` ceiling,
or the remaining space cannot hold one more full layer.

---

## 5. Suggestion menu: `/plan`

`/recommend` returns only one setting, and it uses KV quantization only when things "do not fit" — which hides the real trade-off:
**at a fixed budget, quantizing KV buys either context or more offloaded layers**. `/plan` solves once per budget ×
per KV quantization and lays out the menu for the caller to choose from.

```bash
curl -s localhost:8000/inference/estimate_memory/gguf/plan \
  -H 'content-type: application/json' -d '{
    "model_path": "/models/Qwen3-32B-Q8_0.gguf",
    "gpu_budgets_mib": [5120, 10240, 15360],
    "kv_cache_types": ["f16", "q8_0", "q4_0"],
    "margin_mib": 512
  }' | jq '.plans[] | {gpu_budget_mib, recommended}'
```

Returns one `plans[]` entry per budget, each containing a sorted `candidates[]`:

```jsonc
{
  "gpu_budget_mib": 10240,
  "usable_budget_mib": 9728,
  "fits": true,
  "recommended": "-ngl 17 -c 24064 -ctk q4_0 -ctv q4_0 -fa on",  // = candidates[0]
  "candidates": [
    {
      "kv_quant": "q4_0",
      "fits": true,
      "n_gpu_layers": 17, "n_cpu_moe": 0, "n_ctx": 24064,
      "llama_server_args": "-ngl 17 -c 24064 -ctk q4_0 -ctv q4_0 -fa on",
      "gpu_mib": 9724, "gpu_source": "llama_cpp", "host_mib": 25841,
      "utilization_pct": 99.9, "within_budget": true,
      "gpu_weight_mib": 8558, "gpu_weight_pct": 26.0,   // primary sort key
      "blocks_on_gpu": 16, "n_block": 64,
      "constraint": "n_gpu_layers",                      // which knob it is stuck on
      "host_fits": true, "dominated": false, "warnings": []
    }
  ]
}
```

**Sort rule**: compare "what fraction of the model is on the GPU" (`gpu_weight_pct`) first, since it dominates throughput;
gaps within 5 percentage points count as a tie, then context length decides. Without that tolerance band, offloading 1 more
layer out of 30 (3% of the weights) would outrank a 10x longer context, which is not the trade the user wants.

**`dominated: true`** means "precision paid for nothing": a less lossy KV quantization already achieved the same layer count and context.
Budgets that do not fit are not dropped but returned as `fits: false` + `reason`, so a sweep over the whole model library shows why one is missing.

With `verify` on, every candidate is measured once, costing about 1 s each, so the limit is
**32 combinations** (number of budgets × number of KV types).

### Sweeping a whole folder

`scripts/gguf_recommend_report.py` walks every GGUF under a directory (skipping `mmproj-*.gguf` and non-first
shards), calls `plan()` for each model, and emits a markdown table:

```bash
python scripts/gguf_recommend_report.py ~/.cache/huggingface/hub \
    --gpu 5 10 15 --margin 512 -o report.md
python scripts/gguf_recommend_report.py <dir> --no-verify   # purely analytic, fast but less accurate
```

The `mmproj-*.gguf` next to a multimodal model is not counted in the table, but its file size is listed under that model as a reminder —
`--mmproj` is extra VRAM.

---

## Calling from Python

```python
from service.inference.gguf_estimator import gguf_memory_estimator as est
from service.inference.gguf_estimator import parse_llama_server_args, apply_cli_args

est.estimate("/models/x.gguf", n_gpu_layers=20, n_ctx=32768, n_cpu_moe=24)
est.sweep("/models/x.gguf", n_gpu_layers_grid=[0, 20, 40])
est.recommend("/models/x.gguf", n_ctx=32768)
est.recommend("/models/x.gguf", gpu_budget_mib=12288, kv_cache_types=["q8_0"])  # pin KV
est.plan("/models/x.gguf", gpu_budgets_mib=[5120, 10240, 15360])  # menu
est.probe_exact("/models/x.gguf", n_gpu_layers=20)  # exact values; None without the binary

parse_llama_server_args("-ngl 20 -ctk q8_0")  # -> {"settings": {...}, ...}
kwargs, report = apply_cli_args({"n_ctx": 4096}, ["-c", "32768"])
```

`service/inference/gguf_inspect.py` provides the low-level reader:

```python
from service.inference.gguf_inspect import read_gguf_model_info

info = read_gguf_model_info("/models/x.gguf")
info.n_block, info.n_head_kv, info.key_length, info.is_moe
info.file_bytes, info.total_bytes  # file size / actually allocated (differ by the tied output copy)
info.kv_dims_at(il), info.has_kv(il), info.is_recurrent_layer(il)
info.layer_bytes()  # bytes of attn / ffn / moe_exps / shexp per layer
```

---

## Accuracy

Compared against `llama-fit-params` across 6 architectures (qwen2 / qwen3 / qwen35 / qwen35moe / gemma4 /
gpt-oss), dense and MoE, tied and untied, SWA, and hybrid linear attention:

| Item | Error |
|---|---|
| GPU / host weights | at most 1 MiB (rounding only) |
| GPU / host KV cache and recurrent state | at most 1 MiB (rounding only) |
| GPU compute buffer | +0% ~ +26% on real models (always an overestimate) |

Weights and KV **never** underestimated in any cross-check. The compute buffer is the only approximate term: ggml's
allocator reuses the space freed by logits for the attention mask, so the two are combined with max rather than a sum;
add a safety factor and the direction is consistently high. For exact values use `verify: true`, which calls llama.cpp's own
`no_alloc` dry-run. `/recommend` and `/plan` both use it by default.

### A few easily misunderstood behaviors

- **`-ngl 1` is not "the first layer"**: llama.cpp has `n_layer + 1` slots, the GPU takes the last N,
  and the last slot is the output layer. So `-ngl 1` places only the output layer and not a single block.
- **`token_embd.weight` always stays on the CPU**, even with `-ngl 999`.
- **`-ncmoe` does not move the KV cache too**: expert weights go to DRAM, attention and KV stay on the GPU.
- **`n_ctx` is padded up to a multiple of 256**.
- **The GPU still uses memory at `-ngl 0`**: llama.cpp moves the output projection into VRAM for batched compute.
  To estimate a CPU-only build use `n_gpu: 0`.
- **`-fa` defaults to `auto`**. The estimate assumes flash attention is in effect; if it actually falls back to the
  non-flash path, the compute buffer grows a lot (from 344 MiB to 5256 MiB at 32B, 32K ctx). That is why the argument
  strings produced by `/recommend` always spell out `-fa on`.
- **`-ncmoe` does not increase the compute buffer, but lowering `-ngl` does**. Measured: `-ngl 999` and
  `-ngl 999 -ncmoe 2` both give 14 MiB of compute, but `-ngl 2` becomes 18 MiB. The reason is that the buffer-type
  override keeps that layer assigned to the GPU, while layers excluded by `-ngl` run on the CPU backend and need
  scratch space for op-offload. This is another reason `-ncmoe` beats lowering `-ngl`.
- **Tied embeddings cost an extra copy of token_embd**. When a model has no `output.weight`, llama.cpp uses
  `TENSOR_DUPLICATED` to allocate a second copy of token_embd as the output projection, **placed on the output
  layer's device**. So total weights = file size + token_embd. Measured on gemma-4-26B-A4B-Q4_K_M: the file is 16147 MiB,
  but host is 16895 MiB at `-ngl 0`; at `-ngl 1` the GPU holds exactly those 748 MiB. In `model_info`,
  `file_size_mib` is the file size and `weights_mib` is the actually allocated amount.
- **SWA layers may use narrower heads**. Gemma 4's `attention.key_length_swa` is 256 while
  full-attention layers use 512; ignoring it doubles the KV estimate for 25 of the 30 layers.
- **The SWA phase matters**. `set_swa_pattern` defaults to `il % period < period - 1` (the **last** layer of each group
  is dense); only `smallthinker` / `modern-bert` use `il % period != 0`. With the wrong phase the layer count is right,
  but under partial offload you get "which layers' KV is on the GPU" wrong.
- **With hybrid linear attention only some layers have a KV cache**. For Qwen3.5 / Qwen3-Next,
  `recurrent_layer_arr[il] = (il+1) % full_attention_interval != 0`, so 3 of every 4 layers are gated
  delta net and instead hold a fixed-size recurrent state (`(n_embd_r + n_embd_s) × n_seq_max × 4` bytes)
  that **does not grow with context**. Measured on Qwen3.5-9B at c=4096, KV is 178 MiB (8 attention layers 128 MiB
  + 24 linear layers 50 MiB), not the 512 MiB you get by treating all 32 layers as attention.
- **Some layers have no KV cache of their own at all**: Gemma 4's `attention.shared_kv_layers`, the
  `nextn_predict_layers` (MTP layers) at the tail of the GLM/Bailing/MiMo families, and gemma3n's fixed first 20 layers.
- **gpt-oss is MXFP4**, ggml type 39, newer than the pinned `gguf==0.17.1`. `GGUFReader` throws
  `39 is not a valid GGMLQuantizationType` for the whole file, so `gguf_inspect` patches the missing types
  (MXFP4 / NVFP4 / Q1_0) into the enum together with their block geometry. This becomes a no-op once the package is upgraded.

## When llama-fit-params is needed

`verify: true` needs a binary that can run the fit-params probe. Resolution order: the `LLAMA_FIT_PARAMS_BIN`
environment variable, then a standalone `llama-fit-params` from a source build (`$LLAMA_CPP_DIR/build/bin/`, `bin/`,
the root directory), then `llama-fit-params` beside whatever `LLAMA_SERVER_BINARY` resolves to — which is
where `setup_env` puts it (`service/utils/llama-bin`). A unified `llama` also works, through its
`fit-params` subcommand, for anyone who points `LLAMA_SERVER_BINARY` at one. `setup_env` installs it by default, so no
source build and no extra flag is required. If nothing is found the probe is skipped automatically, returning only
the analytic result rather than failing.
