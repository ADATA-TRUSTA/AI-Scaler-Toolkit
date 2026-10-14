# Training Log — Frontend Integration (Handoff)

How the frontend consumes the redesigned fine-tuning log. The backend now emits a
single, structured, per-job event stream over **SSE**, with the durable record on
disk. Progress is no longer carried in the `error` field.

> Scope: this is Phase A. There is **no authentication yet**; see "Auth (later)".

---

## 0. Dynamic base URL

The backend mounts every route at the **root** — there is no `/api` prefix
(`service/app.py` sets no `root_path`). The base URL is still deployment-dependent
because of reverse-proxy mounts, so resolve it once and build every request from it
rather than hardcoding a host.

This is what the shipped SPA does — it fetches `config.json` from its own build root
at boot and uses the `apiBaseUrl` key:

```js
// Matches Trusta-AST-Frontend: GET /config.json -> { apiBaseUrl, timeout, enabledEngines }
async function resolveApiBase() {
  const cfg = await fetch('/config.json').then(r => r.json()).catch(() => ({}));
  return (cfg.apiBaseUrl ?? window.location.origin).replace(/\/+$/, '');
}

const API_BASE = await resolveApiBase();
// All URLs below are built as `${API_BASE}/...`
```

For SSE the URL must be a full string (EventSource takes no options object), so build
it the same way: `new EventSource(`${API_BASE}/training/${jobId}/log/stream`)`.

---

## 1. Flow overview

```
POST {API_BASE}/training/start         → returns job_id
        │
        ▼
EventSource {API_BASE}/training/{job_id}/log/stream   (live, primary)
        │   stage / resource / metric / error / info events
        ▼
close on terminal phase (completed | stopped | error | cancelled)

Auxiliary:
  GET {API_BASE}/training/status                       coarse status + phase
  GET {API_BASE}/training/{job_id}/log?since=<seq>      backfill / history / reconnect
  GET {API_BASE}/training/error_details                 real errors only
```

`job_id` is the universal key. It is returned by `start` and equals the session id.

---

## 2. Endpoints

### 2.1 Start — `POST {API_BASE}/training/start`
Body: the `TrainingConfig` JSON. Response:
```jsonc
{
  "status": "success",
  "message": "Training started",
  "config": { /* echoed config */ },
  "result": { "status": "starting", "session_id": "<job_id>" }
}
```
Take `job_id = result.session_id`.

### 2.2 Live stream — `GET {API_BASE}/training/{job_id}/log/stream`
`Content-Type: text/event-stream`. Each SSE frame has:
- `id: <seq>`     — monotonically increasing sequence number
- `event: <type>` — one of `stage | resource | metric | error | info`
- `data: <json>`  — the full event object (see §3)

The server closes the stream after a terminal event. Resume after a drop with the
`Last-Event-ID` header (EventSource sends it automatically) or `?since=<seq>`.

The stream also emits a bare comment frame (`: keepalive`) about every 15 idle
seconds. `EventSource` swallows these; a hand-rolled parser must skip lines
starting with `:`.

Returns **404** (`"Training job log not found"`) when the job's log directory does
not exist — treat 404 as fatal, not as a retryable drop.

### 2.3 Status — `GET {API_BASE}/training/status`
Returns `TrainingStatus`:
```jsonc
{
  "is_training": true,
  "progress": 0.42,
  "current_step": 42, "total_steps": 100,
  "loss": 1.23, "current_epoch": 0.8, "total_epochs": 3,
  "status": "running",            // coarse: idle|starting|initializing|running|completed|stopped|error
  "phase": "training",            // NEW — fine-grained lifecycle (see §3.1)
  "phase_detail": "Training",     // NEW — human-readable
  "session_id": "<job_id>",
  "job_id": "<job_id>",           // NEW — alias of session_id
  "error": null,                  // CHANGED — real errors ONLY; null during normal runs
  "config": { /* ... */ }
}
```
Use `phase` / `phase_detail` for progress. Poll this if you don't use SSE, or to
recover current state on page reload.

### 2.4 Backfill / history — `GET {API_BASE}/training/{job_id}/log?since=<seq>`
```jsonc
{ "job_id": "<job_id>", "cursor": 87, "events": [ /* events with seq > since */ ] }
```
`cursor` is the highest `seq` returned; pass it as the next `since`. Use for SSE
reconnect backfill or browsing a finished job. Also returns **404** when the job's
log directory does not exist.

### 2.5 Errors — `GET {API_BASE}/training/error_details`
Only returns a real error; during normal runs returns `{ "has_error": false }`.

---

## 3. Event envelope & payloads

Every event:
```jsonc
{ "ts": 1750640000.12, "seq": 42, "job_id": "<id>", "type": "stage", "phase": "...", "msg": "...", "data": { } }
```

### 3.1 `stage` — lifecycle transition (drives the progress UI)
Fields: `phase`, `msg`. Ordered phases:
`pending → queued → resolving_deepspeed → launching_workers → loading_tokenizer →
loading_dataset → loading_model → preparing_trainer → training → saving →
merging_lora → converting_gguf → completed`. Terminal: `completed | stopped | error | cancelled`.
`queued` **is** emitted on every run — it is the first `stage` event of a single job,
not a queue-only phase. Only `pending` (written into `meta.json`, never emitted as an
event) and `cancelled` are unused today.

### 3.2 `resource` — GPU / DRAM / SSD sample (starts BEFORE model loading)
Carries the current `phase` so you can show "this DRAM spike happened during
loading_model". `data` (any of cpu/gpu/disk may be `null`):
```jsonc
{
  "timestamp": 1750640000.1,
  "cpu":  { "cpu_util_percent": 12.3, "dram": { "used_gb": 40.1, "total_gb": 128.0 } },
  "gpu":  { "available": true, "gpus": [
            { "index": 0, "name": "RTX 4090", "gpu_util": 87.0, "used_gb": 22.5, "total_gb": 24.0, "temperature": 71 } ] },
  "disk": { "main": { "total_gb": 1800, "path": "/mnt/nvme", "percent": 63.0,
                       "read_speed_mbps": 410.0, "write_speed_mbps": 380.0, "folder_size_gb": 120.0 } },
  "proc": { "vmrss_mb": 15234, "vmhwm_mb": 16512, "uss_mb": 8432, "cached_mb": 2048, "mem_available_mb": 32768 },
  "cuda": { "alloc_mb": 24576, "reserved_mb": 28672 }
}
```

### 3.3 `metric` — trainer step metrics (only after the `training` phase)
```jsonc
{ "timestamp": 1750640100.0, "step": 10, "total": 100, "progress": 0.10,
  "loss": 1.23, "learning_rate": 0.0001, "epoch": 0.3, "accuracy": null,
  "split": "train" }
```

**`split` matters.** Every metric carries `split`: `"train"` | `"eval"`. When
`eval_split_ratio` is configured, periodic held-out evaluation emits `metric`
events with `split: "eval"` interleaved into the same stream. **Charting every
`metric` event mixes the eval points into the training loss curve** — filter on
`data.split === "train"`.

The final held-out evaluation at the end of a run emits one more `metric` with
`"split": "eval"` and `"final": true`. That event has **no `total` and no
`progress`**, so a handler that does `ui.setProgress(data.progress)`
unconditionally will blank the progress bar at the very end. Guard on
`data.progress !== undefined`.

### 3.4 `error` — a real error
`msg` = short error string; `data`:
```jsonc
{ "error": "CUDA out of memory ...", "traceback": "Traceback ...", "is_oom": true }
```

### 3.5 `info` — misc info
Several distinct `info` messages are emitted, not just one. Switch on `msg`:

| `msg` | When | `data` |
|---|---|---|
| `training environment` | startup | `model_name`, `method`, `num_gpus`, `use_deepspeed`, `deepspeed_profile`, `dataset_path`, `output_dir` |
| `held-out test split created` | after dataset load, when `eval_split_ratio` is set | split sizes |
| `final held-out evaluation` | end of run | `eval_loss`, `eval_accuracy`, `eval_size` |
| `final evaluation skipped` | end of run, when eval could not run | reason |
| *(forwarded)* | multi-GPU runs | messages relayed from the rank-0 DeepSpeed subprocess |

```jsonc
{ "msg": "training environment",
  "data": { "model_name": "...", "method": "lora", "num_gpus": 1, "use_deepspeed": true,
            "deepspeed_profile": "zero3_offload_disk_disk", "dataset_path": "...", "output_dir": "..." } }
```

`final held-out evaluation` is the only place the held-out score reaches SSE clients.

---

## 4. Reference client

```js
async function startTrainingWithLiveLog(trainingConfig) {
  // 1) Start, get job_id
  const res = await fetch(`${API_BASE}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(trainingConfig),
  }).then(r => r.json());
  const jobId = res.result.session_id;

  // 2) Open SSE (full URL string; prefix is dynamic via API_BASE)
  const es = new EventSource(`${API_BASE}/training/${jobId}/log/stream`);
  const TERMINAL = new Set(['completed', 'stopped', 'error', 'cancelled']);

  // NOTE: the server sets `event: <type>`, so `es.onmessage` never fires.
  //       You MUST addEventListener per type.
  es.addEventListener('info', e => {
    const ev = JSON.parse(e.data);
    ui.showEnvironment(ev.data);
  });

  es.addEventListener('stage', e => {
    const ev = JSON.parse(e.data);
    ui.setPhase(ev.phase, ev.msg);
    if (TERMINAL.has(ev.phase)) es.close();   // IMPORTANT: stop auto-reconnect on terminal
  });

  es.addEventListener('resource', e => {
    const ev = JSON.parse(e.data);
    ui.pushResourceSample(ev.ts, ev.data, ev.phase);  // GPU/DRAM/SSD charts
  });

  es.addEventListener('metric', e => {
    const { step, loss, learning_rate, progress, split, final } = JSON.parse(e.data).data;
    if (split === 'eval') {
      ui.pushEvalPoint(step, loss, { final });   // keep eval off the training curve
    } else {
      ui.pushLossPoint(step, loss, learning_rate);
    }
    if (progress !== undefined) ui.setProgress(progress);  // the final eval metric has none
  });

  es.addEventListener('error', e => {            // our application-level error event
    if (e.data) { ui.showError(JSON.parse(e.data).data); es.close(); }
  });

  // Connection-level errors: the browser auto-reconnects and resends Last-Event-ID.
  es.onerror = () => { /* usually nothing to do; backend resumes from last seq */ };

  return { jobId, close: () => es.close() };
}
```

### Reconnect / backfill without EventSource
```js
let cursor = 0;
async function poll(jobId) {
  const { events, cursor: next } = await fetch(
    `${API_BASE}/training/${jobId}/log?since=${cursor}`).then(r => r.json());
  cursor = next;
  for (const ev of events) handleEvent(ev);   // same handlers as above, keyed by ev.type
}
```

---

## 5. Gotchas (read these)

1. **Named events need `addEventListener`.** Because the backend sets `event: <type>`,
   `EventSource.onmessage` receives nothing. Subscribe to each of
   `stage | resource | metric | error | info`.
2. **Close on terminal.** The server ends the stream after a terminal event, but
   `EventSource` treats a closed stream as a drop and **auto-reconnects** (re-reading the
   backlog forever). Call `es.close()` when you receive a terminal `stage` or an `error`.
3. **Progress moved off `error`.** Remove any code that read progress/stage text from
   `status.error` or `error_details`. Read `phase` / `phase_detail` (status) and the
   SSE `stage` events instead. `error` / `error_details` now mean a real failure only.
4. **`resource` starts before loading.** You'll get GPU/DRAM/SSD samples during
   tokenizer/dataset/model loading (not just training) — render them from the first
   event; `metric` events only begin once `phase === "training"`.
5. **Nulls.** In `resource`, `cpu`/`gpu`/`disk` can be `null` (probe failed for that
   tick); guard before charting.

---

## 6. Auth (later — Phase C)

There is no auth in Phase A. When RBAC lands, the log/status/stream endpoints will be
owner-scoped. Browser `EventSource` cannot send a custom `Authorization` header, so SSE
auth will use a query token (`?token=...`) or a session cookie — handled then, no change
needed now.
