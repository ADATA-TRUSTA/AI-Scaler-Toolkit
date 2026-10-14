"""
AI API Client for communicating with the AI service backend.

Scope (post-refactor):
- Model lifecycle: load_model / unload_model / get_status
- Fine-tune lifecycle: start_training / stop_training / get_training_status

Inference (chat) is no longer wrapped here. Use the OpenAI Python SDK against
the backend's OpenAI-compatible endpoint (`<backend_url>/v1/chat/completions`)
instead. See `examples/chat/basic.py` and `examples/chat/image.py` for usage.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from types import TracebackType
from typing import Any, TypeVar, overload

import httpx2
from pydantic import BaseModel

from ai_scaler_console.config_models import (
    InferenceConfig,
    ModelListResponse,
    ModelStatus,
    TrainingConfig,
    TrainingErrorDetails,
    TrainingEvent,
    TrainingLogEventsResponse,
    TrainingStatus,
)
from ai_scaler_console.exceptions import AIClientError
from ai_scaler_console.settings import BACKEND_URL, VERIFY_SSL

# Define generic type variable bound to BaseModel
T = TypeVar("T", bound=BaseModel)


class AIClient:
    """
    A client for communicating with the AI service backend.

    Responsibilities:
    - Model load / unload / status (`/inference/load_model`, `/inference/unload_model`,
      `/inference/status`).
    - Fine-tune start / stop / status (`/training/*`).

    Chat / generation is handled outside of this client via the OpenAI-compatible
    API (`<backend_url>/v1/chat/completions`). Use the OpenAI Python SDK directly
    once a model has been loaded.

    Holds an ``httpx2.Client``, so it is usable as a context manager:

        with AIClient() as client:
            client.get_status()
    """

    def __init__(
        self,
        base_url: str = BACKEND_URL,
        timeout: int = 10,
        log_requests: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log_requests = log_requests

        # httpx2 takes timeout and TLS verification on the client, not per-request.
        self.session = httpx2.Client(
            timeout=timeout,
            verify=VERIFY_SSL,
            headers={"Content-Type": "application/json"},
        )

    # --- Resource management ---
    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self.session.close()

    def __enter__(self) -> AIClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Type overload definitions (Overloads) ---
    # kwargs are forwarded verbatim to httpx2.Client.request, whose own signature
    # is the contract; narrowing them here would only duplicate it badly.
    @overload
    def _request(
        self,
        method: str,
        endpoint: str,
        response_model: type[T],
        **kwargs: Any,  # noqa: ANN401
    ) -> T: ...
    @overload
    def _request(
        self,
        method: str,
        endpoint: str,
        response_model: None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> httpx2.Response: ...

    def _request(
        self,
        method: str,
        endpoint: str,
        response_model: type[T] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> T | httpx2.Response:
        """Core request method: URL composition, response parsing, error mapping."""
        url = f"{self.base_url}{endpoint}"
        if self.log_requests:
            print(f"[{method}] {url}")  # Log request path

        try:
            res = self.session.request(method=method, url=url, **kwargs)
            res.raise_for_status()  # Check for 4xx/5xx errors

            # Automatically parse Pydantic model. Both json.JSONDecodeError and
            # pydantic's ValidationError are ValueError subclasses.
            if response_model:
                try:
                    return response_model.model_validate(res.json())
                except ValueError as exc:
                    raise AIClientError(
                        message=f"Failed to parse response model for {endpoint}",
                        status_code=res.status_code,
                        response_content=res.text,
                    ) from exc

            return res

        except httpx2.HTTPStatusError as e:
            try:
                error_content = e.response.json()
            except ValueError:
                error_content = e.response.text

            raise AIClientError(
                message=f"API request failed at {endpoint}",
                status_code=e.response.status_code,
                response_content=error_content,
            ) from e

        except httpx2.RequestError as e:
            raise AIClientError(f"Network error connecting to {url}: {e}") from e

    # --- Type overloads for helper methods ---
    @overload
    def _post(
        self,
        endpoint: str,
        *,  # force keyword-only parameters
        payload: dict | None = None,
        response_model: type[T],
    ) -> T: ...
    @overload
    def _post(
        self,
        endpoint: str,
        *,
        payload: dict | None = None,
        response_model: None = None,
    ) -> httpx2.Response: ...

    def _post(
        self,
        endpoint: str,
        *,
        payload: dict | None = None,
        response_model: type[T] | None = None,
    ) -> T | httpx2.Response:
        return self._request("POST", endpoint, json=payload, response_model=response_model)

    @overload
    def _get(self, endpoint: str, response_model: type[T]) -> T: ...
    @overload
    def _get(self, endpoint: str, response_model: None = None) -> httpx2.Response: ...

    def _get(self, endpoint: str, response_model: type[T] | None = None) -> T | httpx2.Response:
        return self._request("GET", endpoint, response_model=response_model)

    # ==========================================
    # --- Model lifecycle ---
    # ==========================================

    def load_model(self, config: InferenceConfig) -> httpx2.Response:
        """
        Load a model on the backend.

        Send the InferenceConfig to `/inference/load_model`. Backend returns
        success/failure plus the resolved configuration.
        """
        if config is None:
            raise AIClientError("InferenceConfig is required")
        return self._post("/inference/load_model", payload=config.model_dump())

    def unload_model(self) -> httpx2.Response:
        """Unload the currently loaded model and release resources."""
        return self._post("/inference/unload_model")

    def get_status(self) -> ModelStatus:
        """Get the current model status (`/inference/status`)."""
        return self._get("/inference/status", response_model=ModelStatus)

    def get_model_list(self) -> ModelListResponse:
        """Get the unified model list (`/config/models`)."""
        return self._get("/config/models", response_model=ModelListResponse)

    # ==========================================
    # --- Fine-tune lifecycle ---
    # ==========================================

    def start_training(self, config: TrainingConfig) -> httpx2.Response:
        """Start fine-tune training (`/training/start`)."""
        if config is None:
            raise AIClientError("TrainingConfig is required")
        return self._post("/training/start", payload=config.model_dump())

    def stop_training(self) -> httpx2.Response:
        """Stop the in-progress fine-tune training (`/training/stop`)."""
        return self._post("/training/stop")

    def get_training_status(self) -> TrainingStatus:
        """Get the current fine-tune training status (`/training/status`)."""
        return self._get("/training/status", response_model=TrainingStatus)

    # ==========================================
    # --- Structured training log ---
    # ==========================================

    def get_training_log(self, job_id: str, since: int = 0) -> TrainingLogEventsResponse:
        """
        Fetch a job's training-log backlog (`/training/{job_id}/log`).

        Returns events with ``seq > since``. Use it to backfill after an SSE drop
        or to browse a finished job. Raises AIClientError with status 404 when the
        job's log directory does not exist.
        """
        return self._get(
            f"/training/{job_id}/log?since={since}", response_model=TrainingLogEventsResponse
        )

    def get_training_error_details(self) -> TrainingErrorDetails:
        """Fetch the full training error including traceback (`/training/error_details`)."""
        return self._get("/training/error_details", response_model=TrainingErrorDetails)

    def stream_training_log(
        self,
        job_id: str,
        since: int = 0,
        max_reconnects: int = 5,
        startup_timeout: float = 30.0,
    ) -> Iterator[TrainingEvent]:
        """
        Yield a job's training-log events over SSE (`/training/{job_id}/log/stream`).

        The backend sends the backlog first, then tails new events, and closes the
        stream after a terminal event. A dropped connection is resumed from the last
        seen ``seq`` via the ``Last-Event-ID`` header, up to ``max_reconnects`` times.

        A 404 means the job's log directory does not exist. That is fatal *except*
        during startup: `/training/start` returns the job id before the worker has
        created the log directory, so attaching immediately gets a 404 for the first
        second or two. Until the first event arrives, a 404 is retried for up to
        ``startup_timeout`` seconds; pass 0 to make every 404 fatal at once.

        Iteration stops after the terminal event.
        """
        last_seq = since
        attempts = 0
        deadline = time.monotonic() + startup_timeout
        seen_any = False

        while True:
            try:
                for event in self._stream_sse_once(job_id, last_seq):
                    seen_any = True
                    if event.seq:
                        last_seq = event.seq
                    yield event
                return
            except _StreamInterrupted as interrupted:
                # Everything up to interrupted.last_seq was already yielded.
                last_seq = interrupted.last_seq
                attempts += 1
                if attempts > max_reconnects:
                    raise AIClientError(
                        f"Training log stream for {job_id} dropped and exceeded "
                        f"{max_reconnects} reconnect attempts (last seq {last_seq})"
                    ) from interrupted.cause
            except AIClientError as e:
                if e.status_code != 404 or seen_any or time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)

    def _stream_sse_once(self, job_id: str, since: int) -> Iterator[TrainingEvent]:
        """
        Consume one SSE connection, yielding events until it closes.

        Raises _StreamInterrupted (carrying the last seq successfully yielded) when
        the connection drops mid-stream so the caller can resume.
        """
        url = f"{self.base_url}/training/{job_id}/log/stream"
        if self.log_requests:
            print(f"[GET] {url} (SSE, since={since})")

        headers = {"Accept": "text/event-stream"}
        if since:
            headers["Last-Event-ID"] = str(since)

        # The client-wide timeout would kill an idle stream: the backend only sends a
        # keepalive comment every ~15s, well past the default 10s read timeout. Connect
        # still needs a bound, so only the read side is released.
        timeout = httpx2.Timeout(self.timeout, read=None)

        last_seq = since
        try:
            with self.session.stream("GET", url, headers=headers, timeout=timeout) as response:
                # Handled inside the `with`, not via raise_for_status() in an outer
                # except: once the context manager unwinds, the stream is closed and
                # reading the body raises httpx2.StreamClosed, which would mask the
                # real status. A buffered mock transport does not reproduce that.
                if response.is_error:
                    response.read()
                    try:
                        error_content = response.json()
                    except ValueError:
                        error_content = response.text
                    raise AIClientError(
                        message=f"API request failed at /training/{job_id}/log/stream",
                        status_code=response.status_code,
                        response_content=error_content,
                    )

                for event in _iter_sse_events(response.iter_lines()):
                    if event.seq:
                        last_seq = event.seq
                    yield event
                    if event.is_terminal:
                        return
        except httpx2.RequestError as e:
            raise _StreamInterrupted(last_seq, e) from e


class _StreamInterrupted(Exception):
    """Internal: an SSE connection dropped; carries the resume point."""

    def __init__(self, last_seq: int, cause: Exception) -> None:
        super().__init__(f"SSE stream interrupted after seq {last_seq}")
        self.last_seq = last_seq
        self.cause = cause


def _iter_sse_events(lines: Iterator[str]) -> Iterator[TrainingEvent]:
    """
    Parse an SSE line stream into TrainingEvent objects.

    Handles the parts of the wire format the backend actually uses: ``id``/``event``/
    ``data`` fields, a blank line as the event terminator, and bare ``:`` comment
    frames (the backend's keepalive), which carry no event and must be skipped.
    """
    data_lines: list[str] = []
    event_id: str | None = None

    for raw in lines:
        line = raw.rstrip("\r")

        if not line:  # blank line dispatches the buffered event
            event = _build_event(data_lines, event_id)
            if event is not None:
                yield event
            data_lines, event_id = [], None
            continue

        if line.startswith(":"):  # comment frame, e.g. ": keepalive"
            continue

        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value

        if field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value
        # `event:` names the type, but the JSON payload repeats it in `type`, which
        # is what TrainingEvent parses. `retry:` is unused by the backend.

    # A stream that ends without a trailing blank line still has a complete event.
    event = _build_event(data_lines, event_id)
    if event is not None:
        yield event


def _build_event(data_lines: list[str], event_id: str | None) -> TrainingEvent | None:
    """Turn buffered SSE data lines into a TrainingEvent, or None if unparseable."""
    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    # The SSE id is authoritative for resume; fall back to it when the body omits seq.
    if not payload.get("seq") and event_id:
        try:
            payload["seq"] = int(event_id)
        except ValueError:
            pass
    try:
        return TrainingEvent.model_validate(payload)
    except ValueError:
        return None
