"""End-to-end: hermes' reasoning levels really switch MiniCPM5's thinking on and off.

Drives the hermes CLI against the backend, once per `--reasoning` level, on both
engines, and reads what actually crossed the wire through a recording proxy:

    hermes  ->  recording proxy  ->  backend  ->  vLLM / llama-server

What this proves, per level:

1. hermes sends the level the way its OpenAI-compatible ("custom") provider does:
   top-level reasoning_effort, clamped to the OpenAI-compat wire (ultra -> max).
2. The backend turns it into the template switch: "none"/"minimal" give answers
   with no reasoning at all; every other level gives reasoning in its own
   `reasoning_content` field on every reply.
3. Neither ever leaks markup into the answer: no <think>/</think>, and no raw
   <function ...> XML for a declared tool that the parser failed to read. Calls to
   tools the model invented are reported but not failed: that is the model's
   mistake, and vLLM's parser cannot type arguments for a tool it was never given.

Why reasoning presence is a stable thing to assert on a 1B model: MiniCPM5's
template prefills an empty think block when thinking is off and an open <think>
when it is on, so whether a reply reasons is decided by the prompt, not sampled.
Whether the model decides to *call a tool* is sampled, so the tool check is pooled
across all levels: every tool call must parse, and at least one must happen.

Isolation: the backend runs as a uvicorn subprocess on a free port (vLLM and
llama-server get their own), hermes gets a throwaway HERMES_HOME so the user's own
hermes config, memory and sessions are never touched, and
service/configs/current_inference_config.json is snapshotted and restored.

A second test does the same with a model that takes graded levels, gpt-oss-20b:
every hermes level must answer without an error, and "high" must think far more
than "low" -- which proves the level itself reaches the model, not just on/off.

Auto-skips without a hermes binary, without CUDA, or without the model in a local
hub cache (never downloads). The vLLM half also needs Linux + vllm. Overrides:

- HERMES_BIN                 hermes executable (default: `hermes` on PATH). Use one
                             installed on the same OS as the test: a Windows hermes.exe
                             driven from WSL was 30x slower and hung after answering.
- HERMES_E2E_MODEL           HF checkpoint for vLLM (default openbmb/MiniCPM5-1B)
- HERMES_E2E_GGUF_REPO       GGUF repo for llama-server (default openbmb/MiniCPM5-1B-GGUF)
- HERMES_E2E_GRADED_MODEL    gpt-oss-20b checkpoint directory for vLLM
                             (default: openai/gpt-oss-20b from a local hub cache)
- HERMES_E2E_GRADED_GGUF     gpt-oss-20b .gguf file for llama-server
                             (default: ggml-org/gpt-oss-20b-GGUF from a local hub cache)
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx2
import pytest
from model_fixtures import local_snapshot

from service import settings as service_settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = REPO_ROOT / "service" / "configs" / "current_inference_config.json"
LOG_DIR = Path(os.environ.get("TESTS_LOG_DIR", str(REPO_ROOT / "tests" / "results"))) / "e2e_logs"

HERMES_BIN = os.environ.get("HERMES_BIN") or shutil.which("hermes")
HF_MODEL = os.environ.get("HERMES_E2E_MODEL", "openbmb/MiniCPM5-1B")
GGUF_REPO = os.environ.get("HERMES_E2E_GGUF_REPO", "openbmb/MiniCPM5-1B-GGUF")
LLAMA_BINARY = Path(service_settings.LLAMA_SERVER_BINARY)

# hermes' whole ladder. The custom provider clamps to the OpenAI-compat wire, so
# "ultra" goes out as "max"; "none" and "minimal" are the two that mean off.
LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
WIRE_EFFORT = {level: ("max" if level == "ultra" else level) for level in LEVELS}
OFF_LEVELS = frozenset({"none", "minimal"})

PROMPT = "Use the terminal tool to run `echo hermes-e2e-ok`, then tell me what it printed."
HERMES_TIMEOUT_S = 600
LOAD_TIMEOUT_S = 1800
# Bounds a repetition loop; still leaves gpt-oss room to think at "high".
MAX_OUTPUT_TOKENS = 8192

# A graded model: gpt-oss writes "Reasoning: low|medium|high" into its system prompt.
GRADED_HF_REPO = "openai/gpt-oss-20b"
GRADED_GGUF_REPO = "ggml-org/gpt-oss-20b-GGUF"
GRADED_PROMPT = (
    "Without using any tools, work out how many integers from 1 to 600 inclusive are "
    "divisible by 4 or 6 but not by 9. Reply with just the number."
)
# Each end of the scale is sampled this many times, so one short "high" run or one
# long "low" run cannot flip the comparison.
GRADED_REPEATS = 2
# 13.8 GiB of weights that must stay on the GPU, plus KV cache and activations.
GRADED_VLLM_MIN_FREE_GIB = 18.0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_free_gib() -> float:
    """Free memory on GPU 0 via NVML, which -- unlike torch -- opens no CUDA context."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**3
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0.0


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def _gguf_path() -> str | None:
    snapshot = local_snapshot(GGUF_REPO)
    if snapshot is None:
        return None
    files = sorted(Path(snapshot).glob("*.gguf"))
    preferred = [f for f in files if "Q4_K_M" in f.name] or files
    return str(preferred[0]) if preferred else None


pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(HERMES_BIN is None, reason="hermes is not installed (set HERMES_BIN)"),
    pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ------------------------------ recording proxy ------------------------------


class _Recorder:
    """Forward every request to the backend and keep chat-completion exchanges."""

    def __init__(self, upstream: str):
        self.upstream = upstream
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def _forward(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length", "accept-encoding")
                }
                url = recorder.upstream + self.path
                r = httpx2.request(method, url, content=body, headers=headers, timeout=900)
                if self.path.endswith("/chat/completions") and body:
                    with recorder.lock:
                        recorder.calls.append(
                            {"req": json.loads(body), "resp": r.content.decode("utf-8", "replace")}
                        )
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(r.content)))
                self.end_headers()
                self.wfile.write(r.content)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._forward("GET")

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self._forward("POST")

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self) -> list[dict]:
        with self.lock:
            calls, self.calls = self.calls, []
        return calls

    def close(self) -> None:
        self.server.shutdown()


def _parse(resp_text: str) -> dict:
    """Reassemble a (possibly streamed) chat completion into content/reasoning/tool calls."""
    text = resp_text.strip()
    if not text.startswith("data:"):
        msg = (json.loads(text).get("choices") or [{}])[0].get("message") or {}
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": [t["function"] for t in msg.get("tool_calls") or []],
        }
    content, reasoning, calls = "", "", {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[5:]).get("choices") or []:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
    tool_calls = [calls[k] for k in sorted(calls)]
    return {"content": content, "reasoning": reasoning, "tool_calls": tool_calls}


# ------------------------------ fixtures ------------------------------


@pytest.fixture(scope="module")
def backend():
    """An isolated backend on a free port; its vLLM child gets its own free port."""
    backend_port, vllm_port = _free_port(), _free_port()
    base_url = f"http://127.0.0.1:{backend_port}"
    config_backup = CONFIG_FILE.read_bytes() if CONFIG_FILE.exists() else None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"backend_hermes_{backend_port}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "VLLM_PORT": str(vllm_port), "VLLM_SWEEP_PORTS": str(vllm_port)}
    # hermes names no max_tokens, so a 1B model that falls into a repetition loop
    # generates until the 32k window is full: one run took 12 minutes and another
    # timed out. The operator ceiling bounds that without changing what is sent.
    env.setdefault("MAX_OUTPUT_TOKENS_CAP", str(MAX_OUTPUT_TOKENS))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.app:app", "--host", "127.0.0.1",
         "--port", str(backend_port)],
        cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )  # fmt: skip
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = log_path.read_text()[-2000:]
                raise RuntimeError(f"backend exited early; log tail:\n{tail}")
            try:
                if httpx2.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    break
            except httpx2.HTTPError:
                pass
            time.sleep(0.5)
        else:
            tail = log_path.read_text()[-2000:]
            raise RuntimeError(f"backend never became healthy; log tail:\n{tail}")
        yield {"base_url": base_url, "vllm_port": vllm_port}
    finally:
        try:
            httpx2.post(f"{base_url}/inference/unload_model", timeout=60.0)
        except httpx2.HTTPError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_file.close()
        if config_backup is None:
            CONFIG_FILE.unlink(missing_ok=True)
        else:
            CONFIG_FILE.write_bytes(config_backup)
        assert not _port_open(vllm_port), "vLLM server left running after the backend exited"


def _engine_config(engine: str) -> dict:
    if engine == "vllm":
        if sys.platform != "linux" or not _vllm_importable():
            pytest.skip("vLLM needs Linux with vllm installed")
        snapshot = local_snapshot(HF_MODEL)
        if snapshot is None:
            pytest.skip(f"{HF_MODEL} is not in a local hub cache")
        # tool parser "auto" on purpose: it is what a load that names none gets, and
        # MiniCPM5 is exactly the model the probe used to miss.
        return {
            "model_name": snapshot,
            "engine": "vllm",
            "vllm_gpu_memory_utilization": 0.8,
            "vllm_max_model_len": 32768,
            "vllm_tool_call_parser": "auto",
            "vllm_reasoning_parser": "qwen3",
        }
    if not (LLAMA_BINARY.is_file() and os.access(LLAMA_BINARY, os.X_OK)):
        pytest.skip(f"llama-server binary not found/executable: {LLAMA_BINARY}")
    gguf = _gguf_path()
    if gguf is None:
        pytest.skip(f"no GGUF of {GGUF_REPO} in a local hub cache")
    return {
        "model_name": "MiniCPM5-1B-GGUF",
        "model_path": gguf,
        "engine": "llama_server",
        "llama_server_port": _free_port(),
        "n_gpu_layers": -1,
        "n_ctx": 32768,
    }


def _graded_config(engine: str) -> dict:
    """gpt-oss-20b, which takes graded levels: its template writes "Reasoning: <level>"."""
    if engine == "vllm":
        if sys.platform != "linux" or not _vllm_importable():
            pytest.skip("vLLM needs Linux with vllm installed")
        model = os.environ.get("HERMES_E2E_GRADED_MODEL") or local_snapshot(GRADED_HF_REPO)
        if not model or not Path(model).is_dir():
            pytest.skip(f"no {GRADED_HF_REPO} checkpoint (set HERMES_E2E_GRADED_MODEL)")
        # The weights alone take 13.8 GiB, and vLLM's MXFP4 MoE kernel needs them on
        # the GPU: on a 16 GB card there was no room left for KV cache ("No available
        # memory for the cache blocks"), and offloading them is refused outright
        # ("moe_wna16_marlin_gemm ... b_q_weight is not on GPU").
        free = _cuda_free_gib()
        if free < GRADED_VLLM_MIN_FREE_GIB:
            pytest.skip(
                f"gpt-oss-20b on vLLM needs ~{GRADED_VLLM_MIN_FREE_GIB:.0f} GiB of free GPU "
                f"memory, {free:.1f} GiB free"
            )
        return {
            "model_name": model,
            "engine": "vllm",
            "vllm_gpu_memory_utilization": 0.85,
            "vllm_max_model_len": 16384,
            "vllm_tool_call_parser": "auto",
            "vllm_reasoning_parser": "openai_gptoss",
        }
    if not (LLAMA_BINARY.is_file() and os.access(LLAMA_BINARY, os.X_OK)):
        pytest.skip(f"llama-server binary not found/executable: {LLAMA_BINARY}")
    gguf = os.environ.get("HERMES_E2E_GRADED_GGUF")
    if not gguf:
        snapshot = local_snapshot(GRADED_GGUF_REPO)
        files = sorted(Path(snapshot).glob("*.gguf")) if snapshot else []
        gguf = str(files[0]) if files else None
    if not gguf or not Path(gguf).is_file():
        pytest.skip(f"no GGUF of {GRADED_GGUF_REPO} (set HERMES_E2E_GRADED_GGUF)")
    return {
        "model_name": "gpt-oss-20b-GGUF",
        "model_path": gguf,
        "engine": "llama_server",
        "llama_server_port": _free_port(),
        "n_gpu_layers": -1,
        "n_ctx": 32768,
    }


@pytest.fixture(scope="module", params=["vllm", "llama_server"])
def served(request, backend, tmp_path_factory):
    """MiniCPM5 loaded on one engine, a recording proxy, and an isolated hermes home."""
    config = _engine_config(request.param)
    yield from _serve(backend, tmp_path_factory, request.param, config)


@pytest.fixture(scope="module", params=["llama_server", "vllm"])
def served_graded(request, backend, tmp_path_factory):
    """gpt-oss-20b loaded on one engine, set up like `served`."""
    # Unload first: the free-memory check in _graded_config must not count the
    # model the previous fixture left loaded.
    httpx2.post(f"{backend['base_url']}/inference/unload_model", timeout=120.0)
    config = _graded_config(request.param)
    yield from _serve(backend, tmp_path_factory, f"gptoss_{request.param}", config)


def _serve(backend: dict, tmp_path_factory, label: str, config: dict):
    """Load `config`, then yield a recording proxy and an isolated hermes home pointed at it."""
    base_url = backend["base_url"]
    # Module-scoped fixtures of different models are not torn down in between, so
    # whatever the previous one loaded is still there: the backend holds one model
    # and answers 409 to a second load.
    httpx2.post(f"{base_url}/inference/unload_model", timeout=120.0)
    r = httpx2.post(f"{base_url}/inference/load_model", json=config, timeout=60.0)
    assert r.status_code == 200, r.text
    deadline = time.time() + LOAD_TIMEOUT_S
    while time.time() < deadline:
        st = httpx2.get(f"{base_url}/inference/status", timeout=10.0).json()
        if st.get("loaded"):
            break
        if st.get("loading_error"):
            pytest.fail(f"{label} load error: {st['loading_error']}")
        time.sleep(5)
    else:
        pytest.fail(f"{label} load timed out")
    model_id = httpx2.get(f"{base_url}/v1/models", timeout=10.0).json()["data"][0]["id"]

    recorder = _Recorder(base_url)
    home = tmp_path_factory.mktemp(f"hermes_home_{label}")
    workspace = tmp_path_factory.mktemp(f"hermes_ws_{label}")
    # Memory, compression and skills off: each would add model calls of its own that
    # this test is not about. The provider is hermes' generic OpenAI-compatible one.
    (home / "config.yaml").write_text(
        "\n".join([
            "model:",
            f"  default: {json.dumps(model_id)}",
            "  provider: trusta-e2e",
            "agent:",
            "  max_turns: 6",
            "terminal:",
            "  backend: local",
            "  timeout: 60",
            "memory:",
            "  memory_enabled: false",
            "  user_profile_enabled: false",
            "compression:",
            "  enabled: false",
            "telemetry:",
            "  shared_metrics:",
            "    enabled: false",
            "    send: false",
            "providers:",
            "  trusta-e2e:",
            "    name: TRUSTA e2e",
            f"    base_url: http://127.0.0.1:{recorder.port}/v1",
            "    discover_models: false",
            f"    model: {json.dumps(model_id)}",
            "",
        ]),
        encoding="utf-8",
    )  # fmt: skip
    yield {"engine": label, "recorder": recorder, "home": home, "workspace": workspace}

    recorder.close()
    try:
        httpx2.post(f"{base_url}/inference/unload_model", timeout=120.0)
    except httpx2.HTTPError:
        pass


def _run_hermes(served: dict, level: str, prompt: str = PROMPT, tag: str = "") -> list[dict]:
    """One hermes turn at `level`; returns the agent's own chat calls (title calls dropped)."""
    env = {**os.environ, "HERMES_HOME": str(served["home"]), "PYTHONIOENCODING": "utf-8"}
    cmd = [
        HERMES_BIN, "chat", "-Q", "--yolo", "--ignore-rules", "-t", "terminal",
        "--provider", "trusta-e2e", "--reasoning", level,
        "--in", str(served["workspace"]), "-q", prompt,
    ]  # fmt: skip
    log = LOG_DIR / f"hermes_{served['engine']}_{level}{tag}.jsonl"
    try:
        out = subprocess.run(
            cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=HERMES_TIMEOUT_S, check=False,
        )  # fmt: skip
    finally:
        # Written on a timeout too: that is exactly when the traffic is needed.
        calls = served["recorder"].take()
        lines = (json.dumps(c, ensure_ascii=False) for c in calls)
        log.write_text("\n".join(lines), encoding="utf-8")
    assert out.returncode == 0, (
        f"hermes exited {out.returncode}:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}"
    )
    # The agent's own turns carry the tool list; hermes' side calls (session titles) do not.
    agent_calls = [c for c in calls if c["req"].get("tools")]
    assert agent_calls, f"hermes made no agent call at --reasoning {level}; see {log}"
    return agent_calls


# ------------------------------ tests ------------------------------


def _declared_tools(req: dict) -> set[str]:
    return {t["function"]["name"] for t in req.get("tools") or [] if t.get("function")}


def test_every_reasoning_level_switches_thinking(served):
    tool_calls_seen = 0
    failures = []
    invented = []
    for level in LEVELS:
        for call in _run_hermes(served, level):
            sent = call["req"].get("reasoning_effort")
            reply = _parse(call["resp"])
            declared = _declared_tools(call["req"])
            where = f"{served['engine']} --reasoning {level}"
            if sent != WIRE_EFFORT[level]:
                failures.append(f"{where}: hermes sent reasoning_effort={sent!r}")
            if level in OFF_LEVELS and reply["reasoning"].strip():
                failures.append(f"{where}: thinking should be off but the reply reasoned")
            if level not in OFF_LEVELS and not reply["reasoning"].strip():
                failures.append(f"{where}: thinking should be on but the reply has no reasoning")
            if "<think>" in reply["content"] or "</think>" in reply["content"]:
                failures.append(f"{where}: <think> leaked into the answer")
            # Parse health is judged on calls to tools the request declared. A 1B model
            # sometimes invents one (seen: "tool_search" at --reasoning max), and vLLM's
            # minicpm5 parser then leaves its XML in content with empty arguments --
            # that is the model's mistake, not this path's, so it is reported, not failed.
            for name in re.findall(r'<function name="([^"]+)"', reply["content"]):
                if name in declared:
                    failures.append(f"{where}: a call to {name} leaked into the answer unparsed")
                else:
                    invented.append(f"{where}: model invented tool {name!r}")
            for tc in reply["tool_calls"]:
                if tc["name"] not in declared:
                    invented.append(f"{where}: model invented tool {tc['name']!r}")
                    continue
                tool_calls_seen += 1
                try:
                    json.loads(tc["arguments"] or "")
                except ValueError:
                    failures.append(f"{where}: tool call {tc['name']} has invalid JSON arguments")
    if invented:
        print("\n".join(["calls to undeclared tools (model behaviour, not failed):", *invented]))
    assert not failures, "\n".join(failures)
    # A 1B model decides to call a tool only most of the time, so this is pooled
    # over every level rather than demanded of each one.
    assert tool_calls_seen > 0, f"no tool call parsed in any of {len(LEVELS)} hermes runs"


# Harmony's channel markers; seeing one in an answer means the split failed.
_HARMONY_MARKERS = ("<|channel|>", "<|message|>", "<|start|>", "<|end|>")


def test_graded_model_follows_the_level(served_graded):
    """
    gpt-oss reads the level itself, so the backend must hand it on, inside the
    vocabulary every engine accepts: vLLM's Harmony path fails on "none" or "xhigh".
    Every hermes level must therefore answer without an error, and the amount of
    thinking must follow the level -- "high" thinks far more than "low". Measured
    on this prompt through this path: ~500 reasoning characters at low against
    ~7,600 at high, so a factor of two leaves sampling plenty of room. Before the
    level was forwarded both ends ran at the template's medium.
    """
    reasoning: dict[str, list[int]] = {}
    failures = []
    for level in LEVELS:
        repeats = GRADED_REPEATS if level in ("low", "high") else 1
        for n in range(repeats):
            calls = _run_hermes(served_graded, level, GRADED_PROMPT, tag=f"_{n}")
            total = 0
            for call in calls:
                reply = _parse(call["resp"])
                where = f"{served_graded['engine']} --reasoning {level}"
                total += len(reply["reasoning"])
                if any(m in reply["content"] for m in ("<think>", "</think>", *_HARMONY_MARKERS)):
                    failures.append(f"{where}: reasoning markup leaked into the answer")
            reasoning.setdefault(level, []).append(total)
            if total == 0:
                # gpt-oss cannot stop thinking; an off level runs at its default.
                failures.append(f"{served_graded['engine']} --reasoning {level}: no reasoning")
    assert not failures, "\n".join(failures)
    low, high = sum(reasoning["low"]), sum(reasoning["high"])
    print(f"reasoning characters per level: {reasoning}")
    assert high > 2 * low, f"high did not think more than low: {reasoning}"
