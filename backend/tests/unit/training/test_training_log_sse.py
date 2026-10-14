"""End-to-end tests for the training SSE log flow.

Covers the whole chain that powers the live training log UI:

    JobLogWriter (producer)  →  events.jsonl  →  read_events (reader)
        →  GET /training/{job_id}/log          (backfill / polling)
        →  GET /training/{job_id}/log/stream   (Server-Sent Events)
        →  client resume via `since` / `Last-Event-ID`
        →  terminal-event close, live tailing, idle keepalive.

Everything is redirected onto a per-test ``tmp_path`` by monkeypatching
``job_logger.LOG_DIR`` so the writer and the FastAPI reader share the same
directory (mirroring the real shared-disk design) without touching real logs.

No GPU / Redis / network needed: the writer's Redis client defaults to ``None``
and every Redis branch is guarded, so these run as fast pure-IO unit tests.
"""

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import service.app as app_module
import service.training.core.job_logger as job_logger
from service.training.core.training_events import EventType, Phase


client = TestClient(app_module.app)


# ── fixtures / helpers ───────────────────────────────────────────────────────
pytestmark = pytest.mark.finetune


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    """Point every job-log read/write at an isolated tmp dir.

    ``job_log_base_dir()`` reads the module-global ``LOG_DIR`` on each call, and
    the API endpoints re-import these helpers per request, so patching the name
    here redirects both the writer (default ``base_dir``) and the readers.
    """
    monkeypatch.setattr(job_logger, "LOG_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def anyio_backend():
    # Run @pytest.mark.anyio tests on asyncio only (no trio dependency).
    return "asyncio"


@pytest.fixture
def fast_poll(monkeypatch):
    """Collapse the stream's 1s poll so tailing/keepalive tests run in <1s.

    Only the generator's long ``await asyncio.sleep(1.0)`` poll is shortened;
    short framework sleeps pass through unchanged.
    """
    real_sleep = asyncio.sleep

    async def _fast(delay):
        await real_sleep(0.01 if (delay or 0) >= 0.5 else (delay or 0))

    monkeypatch.setattr(app_module.asyncio, "sleep", _fast)


def _fake_request(headers=None, disconnected=False):
    """Minimal ASGI ``Request`` for driving the stream generator directly.

    ``is_disconnected()`` polls ``receive`` non-blockingly, so a synchronous
    ``http.disconnect`` reply lets us exercise the client-disconnect branch.
    """
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or [])]

    async def receive():
        return {"type": "http.disconnect" if disconnected else "http.request"}

    scope = {
        "type": "http",
        "method": "GET",
        "headers": raw,
        "query_string": b"",
        "path": "/training/x/log/stream",
    }
    return Request(scope, receive)


def make_writer(job_id, config=None):
    return job_logger.JobLogWriter(job_id, config=config or {"model_name": "demo"})


def parse_sse_records(lines):
    """Turn raw SSE wire lines into ``[{id, event, data(dict)}, ...]``.

    Comment lines (``: keepalive``) are ignored; a blank line flushes a record.
    """
    records, cur = [], {}
    for line in lines:
        if line == "":
            if cur:
                records.append(cur)
                cur = {}
            continue
        if line.startswith(":"):  # comment / keepalive
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "id":
            cur["id"] = value
        elif field == "event":
            cur["event"] = value
        elif field == "data":
            cur["data"] = json.loads(value)
    if cur:
        records.append(cur)
    return records


def read_stream_lines(url, *, headers=None):
    """Drain a self-terminating SSE stream's lines.

    NOTE: Starlette's ``TestClient`` buffers the *entire* streaming response
    before ``stream()`` returns, so it can only be used for streams that close
    on their own (a terminal event already in the backlog). Live tailing and
    idle keepalive are exercised with ``httpx2.AsyncClient`` below, which streams
    incrementally.
    """
    with client.stream("GET", url, headers=headers or {}) as r:
        assert r.status_code == 200, f"unexpected status {r.status_code}"
        return list(r.iter_lines())


async def _drive_stream(job_id, *, headers=None, since=0, disconnected=False, on_chunk=None,
                        stop=None, timeout=15):
    """Drive the endpoint's real SSE generator directly, bypassing transports.

    Returns the list of raw SSE chunks. ``on_chunk(chunk, chunks)`` runs after
    each chunk (e.g. to append a live event); ``stop(chunk)`` ends iteration
    early. An ``asyncio.timeout`` guards against a regression hanging the suite.
    """
    request = _fake_request(headers=headers, disconnected=disconnected)
    resp = await app_module.stream_training_log(job_id, request, since=since)
    chunks = []
    async with asyncio.timeout(timeout):
        try:
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk, chunks)
                if stop is not None and stop(chunk):
                    break
        finally:
            await resp.body_iterator.aclose()
    return chunks


# ── Layer 1: producer → events.jsonl → reader round-trip ─────────────────────
def test_writer_emits_monotonic_seq_and_roundtrips(log_root):
    w = make_writer("rt-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1, "loss": 0.5})
    w.resource({"timestamp": 1.0, "cpu": None, "gpu": None, "disk": None})
    w.info("hello")
    w.close()

    events = job_logger.read_events("rt-job", 0)
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    assert [e["type"] for e in events] == [
        EventType.STAGE,
        EventType.METRIC,
        EventType.RESOURCE,
        EventType.INFO,
    ]
    assert events[0]["phase"] == Phase.LOADING_MODEL
    assert events[0]["msg"]  # default phase label is filled in
    assert events[1]["data"] == {"step": 1, "loss": 0.5}
    # `seq` continuity is what lets the client resume without gaps/dupes.
    assert all(e["job_id"] == "rt-job" for e in events)


def test_read_events_filters_by_since(log_root):
    w = make_writer("since-rt-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1, "loss": 0.5})
    w.stage(Phase.TRAINING)
    w.close()

    assert [e["seq"] for e in job_logger.read_events("since-rt-job", 2)] == [3]
    assert job_logger.read_events("since-rt-job", 99) == []


def test_read_events_missing_file_returns_empty(log_root):
    assert job_logger.read_events("never-existed", 0) == []


def test_read_events_skips_blank_and_partial_lines(log_root):
    """A reader polling mid-flush must skip a half-written line, not crash."""
    job = "partial-job"
    job_logger.job_log_dir(job).mkdir(parents=True, exist_ok=True)
    job_logger.events_file(job).write_text(
        json.dumps({"seq": 1, "type": "stage", "phase": "training"}) + "\n"
        + "\n"  # blank line
        + '{"seq": 2, "type": "metric"'  # truncated mid-flush line (invalid JSON)
        + "\n"
        + json.dumps({"seq": 3, "type": "info", "msg": "ok"}) + "\n",
        encoding="utf-8",
    )
    events = job_logger.read_events(job, 0)
    assert [e["seq"] for e in events] == [1, 3]


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"type": EventType.ERROR}, True),
        ({"type": EventType.STAGE, "phase": Phase.COMPLETED}, True),
        ({"type": EventType.STAGE, "phase": Phase.STOPPED}, True),
        ({"type": EventType.STAGE, "phase": Phase.CANCELLED}, True),
        ({"type": EventType.STAGE, "phase": Phase.ERROR}, True),
        ({"type": EventType.STAGE, "phase": Phase.TRAINING}, False),
        ({"type": EventType.STAGE}, False),  # no phase
        ({"type": EventType.METRIC, "phase": Phase.TRAINING}, False),
        ({"type": EventType.RESOURCE}, False),
        ({"type": EventType.INFO}, False),
    ],
)
def test_is_terminal_event(event, expected):
    assert job_logger.is_terminal_event(event) is expected


def test_error_event_is_terminal_with_payload(log_root):
    w = make_writer("err-job")
    ev = w.error("boom", traceback="trace", is_oom=True)
    w.close()

    assert ev["type"] == EventType.ERROR
    assert ev["phase"] == Phase.ERROR
    assert ev["msg"] == "boom"
    assert ev["data"] == {"error": "boom", "traceback": "trace", "is_oom": True}
    assert job_logger.is_terminal_event(ev) is True
    assert job_logger.read_events("err-job", 0)[-1]["type"] == EventType.ERROR


def test_meta_created_and_finalized(log_root):
    w = make_writer("meta-job", config={"model_name": "demo", "method": "lora"})
    meta = job_logger.read_meta("meta-job")
    assert meta["status"] == Phase.PENDING
    assert meta["job_id"] == "meta-job"
    assert meta["config"]["model_name"] == "demo"

    w.finalize(Phase.COMPLETED, {"output_dir": "out"})
    w.close()
    meta2 = job_logger.read_meta("meta-job")
    assert meta2["status"] == Phase.COMPLETED
    assert meta2["result"] == {"output_dir": "out"}
    assert "finished_at" in meta2


# ── Layer 2: backfill endpoint  GET /training/{job_id}/log ───────────────────
def test_log_endpoint_404_when_job_missing(log_root):
    assert client.get("/training/ghost/log").status_code == 404


def test_log_endpoint_returns_events_and_cursor(log_root):
    w = make_writer("bf-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1, "loss": 0.2})
    w.stage(Phase.TRAINING)
    w.close()

    resp = client.get("/training/bf-job/log")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "bf-job"
    assert body["cursor"] == 3
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    assert body["events"][0]["type"] == EventType.STAGE


def test_log_endpoint_since_filtering(log_root):
    w = make_writer("bf2-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1, "loss": 0.2})
    w.stage(Phase.TRAINING)
    w.close()

    body = client.get("/training/bf2-job/log", params={"since": 2}).json()
    assert [e["seq"] for e in body["events"]] == [3]
    assert body["cursor"] == 3


def test_log_endpoint_empty_but_exists_returns_since_as_cursor(log_root):
    # Writer creates the dir + an empty events.jsonl but emits nothing yet:
    # the endpoint must return 200 (not 404) and echo `since` as the cursor.
    make_writer("empty-job")
    resp = client.get("/training/empty-job/log", params={"since": 7})
    assert resp.status_code == 200
    data = resp.json()
    assert data["events"] == []
    assert data["cursor"] == 7


def test_log_endpoint_rejects_traversal_job_id(log_root):
    assert client.get("/training/evil..id/log").status_code == 400


# ── Layer 3: SSE stream endpoint  GET /training/{job_id}/log/stream ──────────
def test_stream_404_when_job_missing(log_root):
    assert client.get("/training/ghost/log/stream").status_code == 404


def test_stream_backlog_wire_format_and_closes_on_terminal(log_root):
    w = make_writer("stream-job")
    w.stage(Phase.LOADING_MODEL)        # seq 1
    w.metric({"step": 1, "loss": 0.5})  # seq 2
    w.stage(Phase.COMPLETED)            # seq 3 (terminal → stream closes)
    w.close()

    lines = read_stream_lines("/training/stream-job/log/stream")
    records = parse_sse_records(lines)
    # SSE framing: each record carries `id`, `event`, `data` and the stream
    # ended on its own (read_stream_lines would otherwise block forever).
    assert [r["id"] for r in records] == ["1", "2", "3"]
    assert [r["event"] for r in records] == [
        EventType.STAGE,
        EventType.METRIC,
        EventType.STAGE,
    ]
    assert records[0]["data"]["phase"] == Phase.LOADING_MODEL
    assert records[1]["data"]["data"]["loss"] == 0.5
    assert records[2]["data"]["phase"] == Phase.COMPLETED


def test_stream_respects_since_query(log_root):
    w = make_writer("s-since-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1, "loss": 0.1})
    w.stage(Phase.COMPLETED)
    w.close()

    records = parse_sse_records(
        read_stream_lines("/training/s-since-job/log/stream?since=1")
    )
    assert [r["id"] for r in records] == ["2", "3"]


def test_stream_last_event_id_header_overrides_since(log_root):
    w = make_writer("lei-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1})
    w.stage(Phase.COMPLETED)
    w.close()

    records = parse_sse_records(
        read_stream_lines(
            "/training/lei-job/log/stream?since=0",
            headers={"Last-Event-ID": "2"},
        )
    )
    assert [r["id"] for r in records] == ["3"]


def test_stream_invalid_last_event_id_falls_back_to_since(log_root):
    w = make_writer("inv-lei-job")
    w.stage(Phase.LOADING_MODEL)
    w.metric({"step": 1})
    w.stage(Phase.COMPLETED)
    w.close()

    # A non-integer Last-Event-ID must be ignored, leaving `since=2` in effect.
    records = parse_sse_records(
        read_stream_lines(
            "/training/inv-lei-job/log/stream?since=2",
            headers={"Last-Event-ID": "not-an-int"},
        )
    )
    assert [r["id"] for r in records] == ["3"]


@pytest.mark.anyio
async def test_stream_tails_events_appended_after_connect(log_root, anyio_backend, fast_poll):
    """Connect on a live (non-terminal) job, then append a terminal event and
    assert the stream tails it across poll iterations and closes — the core
    live-streaming behaviour.

    Drives the endpoint's real generator directly: both in-process HTTP clients
    (TestClient and httpx2 ASGITransport) fully buffer the response, so they
    deadlock on a stream that only ends after an event written mid-flight.
    """
    w = make_writer("tail-job")
    w.stage(Phase.LOADING_MODEL)  # seq 1, non-terminal: stream stays open

    state = {"appended": False}

    def _append_terminal_after_backlog(chunk, chunks):
        # First chunk == the seq 1 backlog; only now (post-connect) append a
        # terminal event, so the generator must tail it on its next poll.
        if not state["appended"]:
            w.stage(Phase.COMPLETED)  # seq 2, terminal
            state["appended"] = True

    chunks = await _drive_stream(
        "tail-job",
        on_chunk=_append_terminal_after_backlog,
        stop=lambda c: "event: stage" in c and Phase.COMPLETED in c,
    )
    w.close()

    records = parse_sse_records("".join(chunks).split("\n"))
    assert [r["id"] for r in records] == ["1", "2"]
    assert records[0]["data"]["phase"] == Phase.LOADING_MODEL
    assert records[-1]["data"]["phase"] == Phase.COMPLETED


@pytest.mark.anyio
async def test_stream_emits_keepalive_while_idle(log_root, anyio_backend, fast_poll):
    """An open stream with no new events must emit a ``: keepalive`` comment so
    proxies don't drop the idle connection (one every 15 idle poll ticks)."""
    make_writer("idle-job")  # dir + empty events.jsonl, nothing emitted

    chunks = await _drive_stream("idle-job", stop=lambda c: c.startswith(":"))

    assert any(c.startswith(":") for c in chunks), "expected a ': keepalive' comment"


@pytest.mark.anyio
async def test_stream_stops_when_client_disconnects(log_root, anyio_backend, fast_poll):
    """If the client disconnects, the generator must break out of its poll loop
    even though no terminal event was ever written."""
    w = make_writer("disc-job")
    w.stage(Phase.LOADING_MODEL)  # non-terminal backlog; would stream forever
    w.close()

    # `disconnected=True` makes receive() report http.disconnect, which the
    # loop's `is_disconnected()` check honours. If it didn't, the stream would
    # idle forever and the asyncio.timeout guard would fail the test.
    chunks = await _drive_stream("disc-job", disconnected=True)

    assert all(Phase.COMPLETED not in c for c in chunks)  # no terminal close happened


# ── _validate_job_id guard (used by both endpoints) ──────────────────────────
def test_validate_job_id_accepts_clean_id():
    assert app_module._validate_job_id("abc-123") == "abc-123"


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "a..b", ".."])
def test_validate_job_id_rejects_unsafe_ids(bad):
    with pytest.raises(HTTPException) as exc:
        app_module._validate_job_id(bad)
    assert exc.value.status_code == 400
