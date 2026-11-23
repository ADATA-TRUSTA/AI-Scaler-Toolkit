"""Unit tests for VllmEngine server CLI argument construction."""

import json
import os
import queue
from multiprocessing import Event, Queue
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from service.config_models import InferenceConfig
from service.inference.engines import vllm_engine
from service.inference.engines.vllm_engine import VllmEngine

pytestmark = [pytest.mark.inference, pytest.mark.linux_only]


def _engine():
    return VllmEngine(Queue(), Queue(), Event(), Event())


def test_perf_args_includes_max_num_batched_tokens_when_set():
    config = InferenceConfig(
        engine="vllm",
        model_name="demo",
        vllm_max_num_batched_tokens=4096,
        vllm_max_num_seqs=8,
    )
    args = _engine()._perf_args(config)

    idx = args.index("--max-num-batched-tokens")
    assert args[idx + 1] == "4096"
    seqs_idx = args.index("--max-num-seqs")
    assert args[seqs_idx + 1] == "8"


def test_perf_args_omits_max_num_batched_tokens_by_default():
    config = InferenceConfig(engine="vllm", model_name="demo")
    args = _engine()._perf_args(config)
    assert "--max-num-batched-tokens" not in args


# ==================== tool calling ====================


def _with_template(tmp_path, body="{%- if tools %}call{%- endif %}"):
    (tmp_path / "chat_template.jinja").write_text(body)
    return InferenceConfig(engine="vllm", model_name="demo", model_path=str(tmp_path))


def test_tool_call_parser_off_emits_no_flags():
    """vLLM errors out on a lone --enable-auto-tool-choice, so emit neither."""
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_tool_call_parser="off")
    args = _engine()._template_args(config)
    assert "--enable-auto-tool-choice" not in args
    assert "--tool-call-parser" not in args


def test_explicit_tool_call_parser_is_passed_through():
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_tool_call_parser="qwen3_coder")
    args = _engine()._template_args(config)
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_coder"


def test_auto_uses_the_parser_the_probe_identified(tmp_path, monkeypatch):
    """The default path: the user configures nothing and tool calling still works."""
    engine = _engine()
    monkeypatch.setattr(engine, "_probe_tool_call_parser", lambda _ref: "gemma4")

    args = engine._template_args(_with_template(tmp_path))  # parser defaults to "auto"

    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "gemma4"


def test_auto_stays_off_when_the_probe_identifies_nothing(tmp_path, monkeypatch):
    """No parser could read the syntax: serve chat rather than mangle output."""
    engine = _engine()
    monkeypatch.setattr(engine, "_probe_tool_call_parser", lambda _ref: None)
    assert "--enable-auto-tool-choice" not in engine._template_args(_with_template(tmp_path))


def test_auto_skips_the_probe_when_there_is_no_chat_template():
    """A base checkpoint has nothing to detect, so no subprocess is spawned."""
    engine = _engine()

    def _must_not_probe(_ref):
        raise AssertionError("probe must not run without a chat template")

    engine._probe_tool_call_parser = _must_not_probe  # type: ignore[method-assign]
    config = InferenceConfig(engine="vllm", model_name="demo", model_path="/nonexistent/model")
    assert "--enable-auto-tool-choice" not in engine._template_args(config)


# ==================== reasoning parser ====================


def test_reasoning_parser_off_by_default_emits_no_flag():
    """No config at all: reasoning models still serve, just without the split."""
    config = InferenceConfig(engine="vllm", model_name="demo")
    assert "--reasoning-parser" not in _engine()._template_args(config)


@pytest.mark.parametrize("off_value", ["off", "none", "disabled", "false", "0", " OFF "])
def test_reasoning_parser_off_values_emit_no_flag(off_value):
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_reasoning_parser=off_value)
    assert "--reasoning-parser" not in _engine()._template_args(config)


def test_reasoning_parser_explicit_value_is_passed_through():
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_reasoning_parser="deepseek_r1")
    args = _engine()._template_args(config)
    assert args[args.index("--reasoning-parser") + 1] == "deepseek_r1"


def test_reasoning_parser_value_is_trimmed():
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_reasoning_parser=" qwen3 ")
    args = _engine()._template_args(config)
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"


@pytest.mark.parametrize("as_snapshot_path", [False, True])
def test_reasoning_parser_unset_on_minicpm5_defaults_to_qwen3(tmp_path, as_snapshot_path):
    """MiniCPM5's template prefills <think>; without a parser the chat showed a bare
    </think> in the middle of the answer. Matched by repo id or by a hub snapshot path
    -- one under tmp_path, since the args are built by reading the model directory."""
    model = "openbmb/MiniCPM5-1B"
    if as_snapshot_path:
        snapshot = tmp_path / "hub" / "models--openbmb--MiniCPM5-1B" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        model = str(snapshot)
    config = InferenceConfig(engine="vllm", model_name=model)
    args = _engine()._template_args(config)
    assert args[args.index("--reasoning-parser") + 1] == "qwen3"


def test_reasoning_parser_explicit_off_beats_the_minicpm5_default():
    config = InferenceConfig(
        engine="vllm", model_name="openbmb/MiniCPM5-1B", vllm_reasoning_parser="off"
    )
    assert "--reasoning-parser" not in _engine()._template_args(config)


def test_reasoning_parser_explicit_value_beats_the_minicpm5_default():
    config = InferenceConfig(
        engine="vllm", model_name="openbmb/MiniCPM5-1B", vllm_reasoning_parser="deepseek_r1"
    )
    args = _engine()._template_args(config)
    assert args[args.index("--reasoning-parser") + 1] == "deepseek_r1"


def test_tool_call_and_reasoning_parser_flags_coexist():
    """Both parsers are independent flags; there is no --enable-auto-* for reasoning."""
    config = InferenceConfig(
        engine="vllm",
        model_name="demo",
        vllm_tool_call_parser="hermes",
        vllm_reasoning_parser="deepseek_r1",
    )
    args = _engine()._template_args(config)
    assert args[args.index("--tool-call-parser") + 1] == "hermes"
    assert args[args.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert "--enable-auto-tool-choice" in args


def test_max_model_len_is_omitted_when_not_explicitly_set():
    """Let vLLM choose the runtime default when the app does not specify a limit."""
    config = InferenceConfig(
        engine="vllm",
        model_name="demo",
        model_path="/tmp/demo",
        n_ctx=4096,
        vllm_max_model_len=None,
    )
    args = _engine()._core_args(config)
    assert "--max-model-len" not in args

    config.vllm_max_model_len = 2048
    args = _engine()._core_args(config)
    assert args[args.index("--max-model-len") + 1] == "2048"


# ==================== vllm_server_extra_args ====================


def test_extra_args_defaults_to_empty():
    config = InferenceConfig(engine="vllm", model_name="demo")
    assert _engine()._extra_args(config) == []


def test_extra_args_drops_blank_entries():
    config = InferenceConfig(
        engine="vllm", model_name="demo", vllm_server_extra_args=["--trust-remote-code", "  ", ""]
    )
    assert _engine()._extra_args(config) == ["--trust-remote-code"]


def test_extra_args_are_appended_last_so_a_duplicate_flag_wins(monkeypatch):
    """vLLM's argparse keeps the last occurrence of a repeated flag."""
    config = InferenceConfig(
        engine="vllm",
        model_name="demo",
        vllm_dtype="bfloat16",
        vllm_server_extra_args=["--dtype", "float16"],
    )
    monkeypatch.setattr(vllm_engine, "resolve_local_model_path", lambda _ref: None)
    cmd = _engine()._build_server_cmd(config)
    dtype_indices = [i for i, arg in enumerate(cmd) if arg == "--dtype"]
    assert dtype_indices, "expected --dtype to be emitted"
    assert cmd[dtype_indices[-1] + 1] == "float16"
    assert cmd[-2:] == ["--dtype", "float16"]


# ---- the probe subprocess contract ----


def _fake_run(stdout="", stderr="", returncode=0):
    def _run(*args, **kwargs):
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    return _run


def test_probe_reads_the_marker_out_of_noisy_stdout(monkeypatch):
    """Importing the parsers makes vLLM log to stdout, so the answer needs a marker."""
    noisy = (
        "INFO 08-14 [qwen3xml_tool_parser.py:1157] vLLM Successfully import tool parser !\n"
        "WARNING 08-14 [minimax_tool_parser.py:69] Falling back to string matching.\n"
        "TRUSTA_TOOL_PARSER=gemma4\n"
    )
    monkeypatch.setattr(vllm_engine.subprocess, "run", _fake_run(stdout=noisy))
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") == "gemma4"


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    [
        ("", "TemplateError: Invalid content type", 1),  # template cannot render a call
        ("INFO only, no marker\n", "", 0),  # marker missing
    ],
)
def test_probe_returns_none_on_anything_inconclusive(stdout, stderr, returncode, monkeypatch):
    monkeypatch.setattr(
        vllm_engine.subprocess,
        "run",
        _fake_run(stdout=stdout, stderr=stderr, returncode=returncode),
    )
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") is None


def test_probe_resolves_an_ambiguous_match_via_the_preference_list(monkeypatch):
    """Several parsers round-tripped the probe call; hermes is preferred among them."""
    monkeypatch.setattr(
        vllm_engine.subprocess,
        "run",
        _fake_run(stdout="\nTRUSTA_TOOL_PARSER_AMBIGUOUS=qwen3_coder,hermes\n", returncode=3),
    )
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") == "hermes"


# Every tie-set actually observed live against a real model + real vLLM 0.27.1,
# paired with the parser this project has verified is correct for it (see
# _TOOL_PARSER_AMBIGUOUS_PREFERENCE's comment in vllm_engine.py for why each one
# is right). ADD to this list -- never just reorder or replace an entry -- when a
# new model's tie is found, so a later change made to fix one model's ambiguity
# cannot silently re-break a different model that used to resolve correctly. This
# is the regression net for exactly that "fix A, break B" failure mode.
_KNOWN_AMBIGUOUS_TIES = [
    pytest.param(
        "ernie45,granite4,hermes",
        "hermes",
        id="tool_call-json-body",  # <tool_call>{"name": ..., "arguments": ...}</tool_call>.
        # Seen on Qwen2.5-Coder-14B-Instruct-AWQ (fp8 and AWQ) and separately on
        # Qwen3-VL-4B-Instruct, whose templates independently render this body.
    ),
    pytest.param(
        "mimo,qwen3_coder,qwen3_xml,seed_oss,step3p5",
        "qwen3_xml",
        id="function-parameter-tag-body",  # <function=name><parameter=x>v</parameter></function>.
        # Seen on QuantTrio/Qwen3.5-4B-AWQ. mimo/qwen3_coder/qwen3_xml are
        # literally the same registered Qwen3EngineToolParser class; seed_oss and
        # step3p5 are separate classes from unrelated families that happen to
        # also accept this shape, so they are deliberately not preferred.
    ),
]


@pytest.mark.parametrize(("candidates_csv", "expected"), _KNOWN_AMBIGUOUS_TIES)
def test_probe_resolves_every_known_real_world_ambiguous_tie(candidates_csv, expected, monkeypatch):
    monkeypatch.setattr(
        vllm_engine.subprocess,
        "run",
        _fake_run(stdout=f"\nTRUSTA_TOOL_PARSER_AMBIGUOUS={candidates_csv}\n", returncode=3),
    )
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") == expected


def test_probe_stays_off_when_no_ambiguous_candidate_is_preferred(monkeypatch):
    """None of the tied candidates is trusted enough to guess -- serve chat only."""
    monkeypatch.setattr(
        vllm_engine.subprocess,
        "run",
        _fake_run(stdout="\nTRUSTA_TOOL_PARSER_AMBIGUOUS=mimo,seed_oss\n", returncode=3),
    )
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") is None


def test_pick_ambiguous_tool_call_parser_prefers_hermes():
    assert _engine()._pick_ambiguous_tool_call_parser(["qwen3_xml", "hermes"]) == "hermes"


def test_pick_ambiguous_tool_call_parser_prefers_qwen3_xml_when_hermes_absent():
    assert (
        _engine()._pick_ambiguous_tool_call_parser(["mimo", "qwen3_coder", "qwen3_xml"])
        == "qwen3_xml"
    )


def test_pick_ambiguous_tool_call_parser_returns_none_without_a_preferred_name():
    assert _engine()._pick_ambiguous_tool_call_parser(["qwen3_coder", "seed_oss"]) is None


def test_every_preferred_tool_call_parser_exists_in_the_installed_vllm():
    """
    The preference list is plain strings. If vLLM renames a parser, the name can no
    longer appear among the probe's candidates, every tie then resolves to "off",
    and tool calling is silently lost for those models on the next vLLM bump.
    """
    tool_parsers = pytest.importorskip("vllm.tool_parsers")
    manager = tool_parsers.ToolParserManager
    if hasattr(manager, "list_registered"):
        registered = set(manager.list_registered())
    else:  # pragma: no cover - older registry layout
        registered = set(getattr(manager, "tool_parsers", {})) | set(
            getattr(manager, "lazy_parsers", {})
        )
    missing = set(vllm_engine._TOOL_PARSER_AMBIGUOUS_PREFERENCE) - registered
    assert missing == set(), f"no longer registered in vLLM: {sorted(missing)}"


def test_probe_survives_a_timeout(monkeypatch):
    """A hung probe must not take the model load down with it."""

    def _boom(*args, **kwargs):
        raise vllm_engine.subprocess.TimeoutExpired(cmd="probe", timeout=1)

    monkeypatch.setattr(vllm_engine.subprocess, "run", _boom)
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: "/bin/true")

    assert engine._probe_tool_call_parser("demo") is None


def test_probe_skipped_when_the_vllm_venv_is_missing(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(engine, "_resolve_venv_python", lambda: None)
    assert engine._probe_tool_call_parser("demo") is None


# ==================== admin endpoints vs bind address ====================


@pytest.mark.parametrize(
    "host, is_loopback",
    [
        ("127.0.0.1", True),
        ("127.0.1.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("192.168.1.10", False),
        ("some-host.internal", False),
    ],
)
def test_is_loopback_bind(host, is_loopback):
    assert VllmEngine._is_loopback_bind(host) is is_loopback


_DEMO = InferenceConfig(engine="vllm", model_name="demo")
_SLEEPY = InferenceConfig(
    engine="vllm", model_name="demo", vllm_server_extra_args=["--enable-sleep-mode"]
)


def test_admin_endpoints_are_off_by_default(monkeypatch):
    """/collective_rpc takes a plain POST from any web page on this machine, loopback or not."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", False)
    assert _engine()._admin_endpoint_env(_DEMO) == {"VLLM_SERVER_DEV_MODE": "0"}
    assert _engine()._admin_guard_args(_DEMO) == []


def test_admin_endpoints_when_asked_for_on_loopback(monkeypatch):
    """/reset_prefix_cache only exists under dev mode, so cleanup needs the flag."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", True)
    env = _engine()._admin_endpoint_env(_DEMO)
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    assert _engine()._admin_guard_args(_DEMO) == ["--middleware", "ast_vllm_guard.AdminGuard"]


def test_admin_endpoints_disabled_when_bind_is_routable(monkeypatch):
    """Dev mode also mounts /update_weights and /collective_rpc; never expose those."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", True)
    assert _engine()._admin_endpoint_env(_DEMO) == {"VLLM_SERVER_DEV_MODE": "0"}
    assert _engine()._admin_guard_args(_SLEEPY) == []


def test_build_server_env_overrides_an_inherited_dev_mode(monkeypatch):
    """A VLLM_SERVER_DEV_MODE=1 left in the environment cannot turn them on behind the setting."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", False)
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["VLLM_SERVER_DEV_MODE"] == "0"


def test_build_server_env_turns_pinned_memory_on_under_wsl(monkeypatch):
    """Under WSL2 vLLM disables pinned memory and the engine core dies with
    "UVA is not available" on every load."""
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.setattr(vllm_engine, "_running_under_wsl", lambda: True)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"


def test_build_server_env_leaves_pinned_memory_alone_off_wsl(monkeypatch):
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    monkeypatch.setattr(vllm_engine, "_running_under_wsl", lambda: False)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in env


def test_build_server_env_keeps_an_operator_pinned_memory_choice(monkeypatch):
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    monkeypatch.setattr(vllm_engine, "_running_under_wsl", lambda: True)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "0"


def test_build_server_env_pins_the_prefix_hash_seed(monkeypatch):
    """Offloaded KV files are named by prefix hash; an unset seed renames them per run."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["PYTHONHASHSEED"] == "0"


def test_build_server_env_keeps_an_operator_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "1234")
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["PYTHONHASHSEED"] == "1234"


_GIB = 1024**3


def _ram(monkeypatch, *, gib, cpus):
    monkeypatch.setattr(vllm_engine, "_available_ram_bytes", lambda: int(gib * _GIB))
    monkeypatch.setattr(vllm_engine.os, "cpu_count", lambda: cpus)


def test_build_server_env_caps_max_jobs_by_free_ram(monkeypatch):
    """22 jobs on a 62 GB machine was three OOMs in a row; RAM, not cores, sets the cap."""
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(vllm_engine, "VLLM_JIT_GIB_PER_JOB", 11.0)
    _ram(monkeypatch, gib=60, cpus=20)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    # (60 - 4 reserved) // 11 = 5
    assert env["MAX_JOBS"] == "5"


def test_build_server_env_reserves_the_offload_budget(monkeypatch):
    """Offloaded experts and KV cache take RAM after the environment is built."""
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(vllm_engine, "VLLM_JIT_GIB_PER_JOB", 11.0)
    _ram(monkeypatch, gib=60, cpus=20)
    config = InferenceConfig(
        engine="vllm", model_name="demo", vllm_cpu_offload_gb=16.0, vllm_kv_offloading_size=10.0
    )
    env = _engine()._build_server_env(config)
    # (60 - 4 - 16 - 10) // 11 = 2
    assert env["MAX_JOBS"] == "2"


def test_build_server_env_never_caps_below_one_job_or_above_the_cores(monkeypatch):
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(vllm_engine, "VLLM_JIT_GIB_PER_JOB", 11.0)
    _ram(monkeypatch, gib=2, cpus=20)
    low = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert low["MAX_JOBS"] == "1"
    _ram(monkeypatch, gib=512, cpus=8)
    high = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert high["MAX_JOBS"] == "8"


def test_build_server_env_keeps_an_operator_max_jobs(monkeypatch):
    monkeypatch.setenv("MAX_JOBS", "7")
    _ram(monkeypatch, gib=8, cpus=20)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert env["MAX_JOBS"] == "7"


def test_build_server_env_leaves_max_jobs_unset_when_the_cap_is_off(monkeypatch):
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(vllm_engine, "VLLM_JIT_GIB_PER_JOB", 0.0)
    env = _engine()._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))
    assert "MAX_JOBS" not in env


def test_available_ram_is_lowered_by_a_cgroup_limit(monkeypatch, tmp_path):
    """In a memory-limited container the host's MemAvailable is not the budget."""
    monkeypatch.setattr(
        vllm_engine.psutil, "virtual_memory", lambda: SimpleNamespace(available=100 * _GIB)
    )
    (tmp_path / "memory.max").write_text(str(20 * _GIB))
    (tmp_path / "memory.current").write_text(str(5 * _GIB))
    real_path = vllm_engine.Path
    monkeypatch.setattr(
        vllm_engine,
        "Path",
        lambda p: real_path(str(p).replace("/sys/fs/cgroup", str(tmp_path))),
    )
    assert vllm_engine._available_ram_bytes() == 15 * _GIB
    (tmp_path / "memory.max").write_text("max")
    assert vllm_engine._available_ram_bytes() == 100 * _GIB


def test_available_ram_honours_a_limit_on_the_scope_the_service_runs_in(monkeypatch, tmp_path):
    """systemd's MemoryMax= sits on the unit's cgroup, not on /sys/fs/cgroup itself."""
    monkeypatch.setattr(
        vllm_engine.psutil, "virtual_memory", lambda: SimpleNamespace(available=100 * _GIB)
    )
    scope = tmp_path / "sys" / "user.slice" / "job.scope"
    scope.mkdir(parents=True)
    (scope / "memory.max").write_text(str(42 * _GIB))
    (scope / "memory.current").write_text(str(2 * _GIB))
    (scope.parent / "memory.max").write_text(str(30 * _GIB))  # a tighter limit above it wins
    (scope.parent / "memory.current").write_text(str(5 * _GIB))
    (tmp_path / "sys" / "memory.max").write_text("max")  # the root has no limit
    (tmp_path / "proc_cgroup").write_text("0::/user.slice/job.scope" + chr(10))
    real_path = vllm_engine.Path
    mapping = {"/sys/fs/cgroup": str(tmp_path / "sys"), "/proc/self/cgroup": str(tmp_path / "proc_cgroup")}

    def fake_path(p: str):
        for prefix, target in mapping.items():
            if str(p).startswith(prefix):
                return real_path(str(p).replace(prefix, target, 1))
        return real_path(p)

    monkeypatch.setattr(vllm_engine, "Path", fake_path)
    assert vllm_engine._available_ram_bytes() == 25 * _GIB
    (scope.parent / "memory.max").write_text("max")
    assert vllm_engine._available_ram_bytes() == 40 * _GIB


def test_jit_build_line_is_reported_once_and_only_while_loading():
    engine = _engine()
    line = "INFO Building JIT module fused_moe_120; this can take several minutes on first use"
    engine._note_jit_build(line)
    first = engine.runtime.jit_build_started_at
    engine._note_jit_build(line)
    assert engine.runtime.jit_build_started_at == first
    # A multiprocessing.Queue delivers through a feeder thread, so empty() right after
    # put() is not evidence of anything; read with a timeout instead.
    assert engine.status_queue.get(timeout=5)["stage"] == "vllm_jit_build"
    with pytest.raises(queue.Empty):
        engine.status_queue.get(timeout=0.5)

    serving = _engine()
    serving.runtime.in_startup = False
    serving._note_jit_build(line)
    assert serving.runtime.jit_build_started_at > 0
    with pytest.raises(queue.Empty):
        serving.status_queue.get(timeout=0.5)


class _FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return object()


class _FakeClient:
    def __init__(self, outcome=None):
        self.completions = _FakeCompletions(outcome)
        self.options = None

    def with_options(self, **options):
        self.options = options
        return self


def _warm(engine, client):
    engine.runtime.client = client
    engine.runtime.served_model_name = "demo"
    engine.runtime.in_startup = False
    engine._warm_up_server()


def test_warm_up_sends_one_sampling_request_without_retries(monkeypatch):
    monkeypatch.setattr(vllm_engine, "VLLM_WARMUP_TIMEOUT", 900.0)
    engine, client = _engine(), _FakeClient()
    _warm(engine, client)
    assert client.options == {"timeout": 900.0, "max_retries": 0}
    (call,) = client.completions.calls
    assert call["model"] == "demo" and call["max_tokens"] == 2
    assert call["temperature"] > 0 and call["top_p"] < 1
    # The build lines are INFO; startup forwarding is back off once the warm-up ends.
    assert engine.runtime.in_startup is False


def test_warm_up_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(vllm_engine, "VLLM_WARMUP_TIMEOUT", 0.0)
    engine, client = _engine(), _FakeClient()
    _warm(engine, client)
    assert client.completions.calls == []


def test_a_failed_warm_up_does_not_fail_the_load(monkeypatch):
    """A server that is up and answers a health check still serves; warn and go on."""
    monkeypatch.setattr(vllm_engine, "VLLM_WARMUP_TIMEOUT", 900.0)
    engine = _engine()
    engine.runtime.server_process = SimpleNamespace(poll=lambda: None)
    _warm(engine, _FakeClient(vllm_engine.APIError("bad", request=None, body=None)))
    assert engine.runtime.in_startup is False


def test_a_server_that_died_during_warm_up_is_a_load_failure(monkeypatch):
    monkeypatch.setattr(vllm_engine, "VLLM_WARMUP_TIMEOUT", 900.0)
    engine = _engine()
    engine.runtime.server_process = SimpleNamespace(poll=lambda: -9, returncode=-9)
    monkeypatch.setattr(engine, "_drain_stderr_pump", lambda timeout: None)
    with pytest.raises(RuntimeError, match="exited during startup"):
        _warm(engine, _FakeClient(vllm_engine.APIError("gone", request=None, body=None)))


def test_vllm_native_kv_offloading_emits_the_flag():
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_size=8.0)
    args = _engine()._perf_args(config)

    idx = args.index("--kv-offloading-size")
    assert args[idx + 1] == "8.0"


def test_kv_offloading_leaves_the_hybrid_kv_cache_manager_alone():
    """
    KV offloading must not pin --disable-hybrid-kv-cache-manager.

    The flag is three-state in vLLM: unset means "auto -- disable only when the
    KV connector cannot do hybrid allocation", passing it means "always
    disable". Both connectors --kv-offloading-size selects declare SupportsHMA,
    so pinning the flag turned hybrid allocation off for models that require it
    -- hybrid SSM models (Jamba, Bamba) then fail at startup, and
    sliding-window models lose performance.
    """
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_size=8.0)
    args = _engine()._perf_args(config)

    assert "--disable-hybrid-kv-cache-manager" not in args
    assert "--no-disable-hybrid-kv-cache-manager" not in args, (
        "forcing it on is the opposite mistake: vLLM errors when the runtime must disable it"
    )


# --------------------------------------------------------------------------
# /dev/shm offload regions are reclaimed on unload
#
# vLLM unlinks these itself in SharedOffloadRegion.cleanup(), but only if it
# gets to run -- and _stop_server_process escalates to SIGKILL, which cannot be
# caught. tmpfs is RAM: two observed leaks held 20GB and 48GB, and the second
# made the next load fail with "Insufficient space in /dev/shm".
# --------------------------------------------------------------------------


def _shm_engine(tmp_path, monkeypatch, *, before=(), after=(), mapped=()):
    """An engine whose /dev/shm view is faked, so no real tmpfs is touched."""
    engine = _engine()
    engine.runtime.shm_files_before_launch = frozenset(before)
    monkeypatch.setattr(vllm_engine, "_shm_offload_files", lambda: frozenset(after))
    monkeypatch.setattr(
        vllm_engine, "_paths_mapped_by_live_processes", lambda cands: set(mapped) & set(cands)
    )
    return engine


def _make(tmp_path, name, size=16):
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return str(p)


def test_unload_reclaims_a_region_this_run_leaked(tmp_path, monkeypatch):
    leaked = _make(tmp_path, "vllm_offload_abc.mmap")
    engine = _shm_engine(tmp_path, monkeypatch, before=(), after=(leaked,))

    freed = engine._reclaim_orphaned_shm()

    assert freed == 16
    assert not os.path.exists(leaked)


def test_a_region_that_predates_this_run_is_left_alone(tmp_path, monkeypatch):
    """Not ours to delete: it was already there when this server launched."""
    pre_existing = _make(tmp_path, "vllm_offload_old.mmap")
    engine = _shm_engine(tmp_path, monkeypatch, before=(pre_existing,), after=(pre_existing,))

    assert engine._reclaim_orphaned_shm() == 0
    assert os.path.exists(pre_existing)


def test_a_region_still_mapped_by_a_live_process_is_never_deleted(tmp_path, monkeypatch):
    """
    The guard that makes this safe with two engines running.

    Names carry a uuid generated inside the subprocess, so cleanup cannot
    target a known path -- without this check it would degrade to globbing and
    could delete the region another engine is actively serving from.
    """
    other = _make(tmp_path, "vllm_offload_other.mmap")
    engine = _shm_engine(tmp_path, monkeypatch, before=(), after=(other,), mapped=(other,))

    assert engine._reclaim_orphaned_shm() == 0
    assert os.path.exists(other)


def test_a_failed_unlink_does_not_raise(tmp_path, monkeypatch):
    """Unload must finish; a region we cannot remove is logged, not fatal."""
    leaked = _make(tmp_path, "vllm_offload_stuck.mmap")
    engine = _shm_engine(tmp_path, monkeypatch, before=(), after=(leaked,))
    monkeypatch.setattr(vllm_engine.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("busy")))

    assert engine._reclaim_orphaned_shm() == 0


def test_graceful_stop_timeout_scales_with_the_offload_budget():
    """
    A flat 10s is why the big offloads were the ones that leaked.

    Only the graceful path unmaps and unlinks the region, and a 48GiB teardown
    cannot finish in 10s -- so the larger the budget, the more certain the leak.
    """
    engine = _engine()

    engine.config = None
    assert engine._graceful_stop_timeout() == 10.0

    engine.config = InferenceConfig(engine="vllm", model_name="demo")
    assert engine._graceful_stop_timeout() == 10.0, "no offload -> unchanged"

    engine.config = InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_size=48.0)
    assert engine._graceful_stop_timeout() == 48.0

    engine.config = InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_size=4.0)
    assert engine._graceful_stop_timeout() == 10.0, "floor still applies"

    engine.config = InferenceConfig(engine="vllm", model_name="demo", vllm_kv_offloading_size=500.0)
    assert engine._graceful_stop_timeout() == 120.0, "a wedged server is still killed"


# --------------------------------------------------------------------------
# Startup INFO reaches the service log; serving INFO does not
#
# vLLM's startup facts (KV cache size, chosen kernels) are INFO lines, and at
# the default VLLM_LOGGING_LEVEL=ERROR they were never emitted -- the only way
# to find them was /metrics after the fact. The default stays ERROR on purpose:
# at serving time vLLM logs INFO per request and would bury everything else.
# --------------------------------------------------------------------------


def test_subprocess_is_raised_to_info_so_startup_facts_exist(monkeypatch):
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    monkeypatch.setattr(vllm_engine, "VLLM_LOGGING_LEVEL", "ERROR")
    engine = _engine()

    env = engine._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))

    assert env["VLLM_LOGGING_LEVEL"] == "INFO"
    assert engine.runtime.startup_log_boost is True


def test_an_explicit_logging_level_is_honoured_verbatim(monkeypatch):
    """An operator who set the level gets it, with no filtering layered on top."""
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setattr(vllm_engine, "VLLM_LOGGING_LEVEL", "DEBUG")
    engine = _engine()

    env = engine._build_server_env(InferenceConfig(engine="vllm", model_name="demo"))

    assert env["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert engine.runtime.startup_log_boost is False
    engine.runtime.in_startup = False
    assert engine._should_drop_server_log("INFO 09-17 [core.py:1] Avg prompt throughput") is False


def test_info_is_forwarded_during_startup_and_dropped_after():
    engine = _engine()
    engine.runtime.startup_log_boost = True
    info_line = "INFO 09-17 12:00:00 [kv_cache_utils.py:80] GPU KV cache size: 262,144 tokens"

    engine.runtime.in_startup = True
    assert engine._should_drop_server_log(info_line) is False, "startup facts must get through"

    engine.runtime.in_startup = False
    assert engine._should_drop_server_log(info_line) is True, "per-request INFO must not"


@pytest.mark.parametrize(
    "line",
    [
        "ERROR 09-17 [engine.py:9] died",
        "WARNING 09-17 [engine.py:9] degraded",
        "ValueError: something went wrong",
        '  File "/opt/vllm/config/vllm.py", line 1731, in __post_init__',
        "Traceback (most recent call last):",
    ],
)
def test_nothing_diagnostic_is_ever_dropped(line):
    """A line we cannot classify as INFO is kept -- that is how errors stay visible."""
    engine = _engine()
    engine.runtime.startup_log_boost = True
    engine.runtime.in_startup = False
    assert engine._should_drop_server_log(line) is False


def test_a_message_merely_containing_the_word_info_is_kept():
    engine = _engine()
    engine.runtime.startup_log_boost = True
    engine.runtime.in_startup = False
    assert engine._should_drop_server_log("ERROR: failed to read model info") is False


# ==================== gpu memory utilization on XPU ====================


def _memory(
    engine,
    monkeypatch,
    *,
    vllm_version,
    available_gib,
    total_gib=30.0,
    igpu=False,
    pooled_gib=0.0,
    weights_gib=None,
):
    monkeypatch.setattr(engine, "_resolve_vllm_server_dir", lambda: "/nonexistent")
    monkeypatch.setattr(engine, "_installed_package_version", lambda _venv, _name: vllm_version)
    gib = 1024**3
    monkeypatch.setattr(
        vllm_engine.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=available_gib * gib, total=total_gib * gib),
    )
    # The host running the tests must not decide which path is taken.
    monkeypatch.setattr(vllm_engine, "_intel_igpu_only", lambda: igpu)
    monkeypatch.setattr(vllm_engine, "_gpu_reclaim_bytes", lambda: int(pooled_gib * gib))
    weights = None if weights_gib is None else int(weights_gib * gib)
    monkeypatch.setattr(vllm_engine, "_checkpoint_bytes", lambda _source: weights)


def _flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def test_cuda_build_passes_the_requested_fraction_through(monkeypatch):
    """A discrete card's memory is its own: never second-guessed."""
    engine = _engine()
    _memory(engine, monkeypatch, vllm_version="0.28.0", available_gib=4)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    assert engine._effective_gpu_memory_utilization(config) == 0.9


def test_xpu_build_lowers_a_fraction_that_does_not_fit(monkeypatch):
    """The interface's 0.9 on a 30 GiB laptop with 16 GiB free: (16 - 2) / 30."""
    engine = _engine()
    _memory(engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=16)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    assert engine._effective_gpu_memory_utilization(config) == 0.47
    args = engine._core_args(config)
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.47"


def test_xpu_build_keeps_a_fraction_that_fits(monkeypatch):
    engine = _engine()
    _memory(engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=28)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.3)
    assert engine._effective_gpu_memory_utilization(config) == 0.3


def test_xpu_build_never_goes_below_the_floor(monkeypatch):
    """With almost nothing free, vLLM gets the minimum and says why it cannot start."""
    engine = _engine()
    _memory(engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=1)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    assert engine._effective_gpu_memory_utilization(config) == 0.05


def test_igpu_counts_the_driver_pool_as_free(monkeypatch):
    """
    After a run the driver keeps its pages pooled; the next load reuses them.

    30 GiB laptop, 0.9 requested, 6 GiB free plus 20 GiB pooled, 2 GiB of weights:
    usable = 6 + 20 - 2 = 24, capped at 0.9 * 30 = 27 -> 24; KV = 24 - 2 - 2 (runtime) = 20.
    """
    engine = _engine()
    _memory(
        engine,
        monkeypatch,
        vllm_version="0.32.0+xpu",
        available_gib=6,
        igpu=True,
        pooled_gib=20,
        weights_gib=2,
    )
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    args = engine._core_args(config)
    assert _flag(args, "--kv-cache-memory-bytes") == str(20 * 1024**3)
    # Only vLLM's pre-load check reads this now; the driver's figure leaves out the pool.
    assert _flag(args, "--gpu-memory-utilization") == "0.05"


def test_igpu_kv_cache_stays_within_the_requested_fraction(monkeypatch):
    """0.5 of 30 GiB is 15 GiB however much is free: KV = 15 - 2 (weights) - 2 (runtime)."""
    engine = _engine()
    _memory(
        engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=28, igpu=True, weights_gib=2
    )
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.5)
    assert engine._igpu_kv_cache_bytes(config) == 11 * 1024**3


def test_igpu_kv_cache_leaves_room_for_the_dram_offload_tier(monkeypatch):
    """The DRAM KV tier is system RAM too: 4 GiB of it comes out of the KV cache."""
    engine = _engine()
    _memory(
        engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=28, igpu=True, weights_gib=2
    )
    config = InferenceConfig(
        engine="vllm",
        model_name="demo",
        vllm_gpu_memory_utilization=0.5,
        vllm_kv_offloading_size=4,
    )
    assert engine._igpu_kv_cache_bytes(config) == 7 * 1024**3


def test_igpu_refuses_a_model_that_does_not_fit(monkeypatch):
    """With the KV size fixed vLLM no longer checks the weights fit, so refuse here."""
    engine = _engine()
    _memory(
        engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=16, igpu=True, weights_gib=20
    )
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    with pytest.raises(RuntimeError, match="Not enough memory for demo on this Intel iGPU"):
        engine._core_args(config)


def test_igpu_without_a_checkpoint_size_keeps_the_fraction_path(monkeypatch):
    engine = _engine()
    _memory(engine, monkeypatch, vllm_version="0.32.0+xpu", available_gib=16, igpu=True)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    args = engine._core_args(config)
    assert "--kv-cache-memory-bytes" not in args
    assert _flag(args, "--gpu-memory-utilization") == "0.47"


@pytest.mark.parametrize(
    ("vllm_version", "igpu"),
    [("0.32.0", True), ("0.32.0+xpu", False)],
    ids=["cuda-build", "xpu-not-igpu-only"],
)
def test_igpu_kv_sizing_only_on_an_xpu_build_with_only_the_igpu(monkeypatch, vllm_version, igpu):
    engine = _engine()
    _memory(
        engine,
        monkeypatch,
        vllm_version=vllm_version,
        available_gib=16,
        igpu=igpu,
        weights_gib=2,
    )
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_gpu_memory_utilization=0.9)
    assert "--kv-cache-memory-bytes" not in engine._core_args(config)


def test_checkpoint_bytes_reads_the_index_total(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123456}, "weight_map": {}})
    )
    assert vllm_engine._checkpoint_bytes(str(tmp_path)) == 123456


def test_checkpoint_bytes_falls_back_to_the_single_file(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 2048)
    assert vllm_engine._checkpoint_bytes(str(tmp_path)) == 2048
    assert vllm_engine._checkpoint_bytes(str(tmp_path / "missing")) is None


def test_gpu_reclaim_bytes_reads_meminfo(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  32376252 kB\nGPUActive:  548856 kB\nGPUReclaim: 10247504 kB\n")
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
    )
    assert vllm_engine._gpu_reclaim_bytes() == 10247504 * 1024


def test_sleep_mode_turns_the_admin_endpoints_on_behind_the_guard(monkeypatch):
    """vLLM only mounts /sleep under dev mode; a GPU sharing tool needs it once sleep mode is on."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", False)
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    engine = _engine()
    env = engine._build_server_env(_SLEEPY)
    assert env["VLLM_SERVER_DEV_MODE"] == "1"
    first, *rest = env["PYTHONPATH"].split(os.pathsep)
    assert Path(first, "ast_vllm_guard.py").is_file()
    assert rest == ["/somewhere"]
    cmd = engine._build_server_cmd(_SLEEPY)
    at = cmd.index("--middleware")
    assert cmd[at + 1] == "ast_vllm_guard.AdminGuard"
    assert at < cmd.index("--enable-sleep-mode")


@pytest.mark.parametrize(
    ("args", "on"),
    [
        (["--enable-sleep-mode"], True),
        (["--enable-sleep-mode", "--no-enable-sleep-mode"], False),
        (["--no-enable-sleep-mode", "--enable-sleep-mode"], True),
        (["--enforce-eager"], False),
    ],
)
def test_the_last_sleep_mode_flag_decides(monkeypatch, args, on):
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", False)
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_server_extra_args=args)
    assert (_engine()._admin_guard_args(config) != []) is on


def test_the_sleep_mode_field_mounts_sleep_behind_the_guard_too(monkeypatch):
    """POST /inference/sleep calls vLLM's /sleep: vllm_enable_sleep_mode alone must mount it."""
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_HOST", "127.0.0.1")
    monkeypatch.setattr(vllm_engine, "VLLM_SERVER_DEV_MODE", False)
    engine = _engine()
    config = InferenceConfig(engine="vllm", model_name="demo", vllm_enable_sleep_mode=True)
    assert engine._build_server_env(config)["VLLM_SERVER_DEV_MODE"] == "1"
    cmd = engine._build_server_cmd(config)
    assert cmd[cmd.index("--middleware") + 1] == "ast_vllm_guard.AdminGuard"
    assert "--enable-sleep-mode" in cmd
    # the extra args still have the last word
    off = InferenceConfig(
        engine="vllm",
        model_name="demo",
        vllm_enable_sleep_mode=True,
        vllm_server_extra_args=["--no-enable-sleep-mode"],
    )
    assert engine._admin_guard_args(off) == []


def test_the_guard_cannot_be_turned_off_from_the_extra_args():
    """--middleware is reserved, so the extra args cannot add or replace one."""
    with pytest.raises(ValidationError):
        InferenceConfig(
            engine="vllm",
            model_name="demo",
            vllm_server_extra_args=["--enable-sleep-mode", "--middleware", "x.Y"],
        )


def test_cleanup_returns_its_result_for_the_worker_to_send():
    """The worker tags the reply with the request id; an untagged put was dropped."""
    engine = _engine()
    sent = []
    engine.data_queue = SimpleNamespace(put=sent.append)
    assert engine.cleanup_generation_memory() == "vLLM cleanup skipped: server not loaded"
    assert sent == []


def test_xpu_venv_lib_with_oneccl_joins_ld_library_path(tmp_path):
    """oneCCL 2022.1 dlopens its backend (libccl.so.1) by name, with no RPATH to find it."""
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("libccl.so.2", "libccl.so.1", "libfabric.so.1"):
        (lib / name).write_bytes(b"")
    assert str(lib) in _engine()._discover_runtime_library_dirs(str(tmp_path))


def test_venv_lib_without_runtime_libs_stays_off_ld_library_path(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libpython3.12.so.1.0").write_bytes(b"")
    assert _engine()._discover_runtime_library_dirs(str(tmp_path)) == []
