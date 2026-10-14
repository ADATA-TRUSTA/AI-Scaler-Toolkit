"""Exceptions raised by the AI Scaler console."""

from typing import Any

# Long enough for the backend's refusal messages, short enough that a 422's
# validation dump does not bury the rest of the line.
_MAX_DETAIL_CHARS = 500


def _explanation(response_content: Any) -> str:  # noqa: ANN401
    """
    The backend's own account of the failure, when it sent one.

    FastAPI puts it under `detail`, which is where the actionable refusals live --
    /training/start returning 501 explains that fine-tuning needs a Linux host, for
    instance. Without this the examples print only "(Status: 501)" and the user has
    nowhere to go.
    """
    if isinstance(response_content, dict):
        detail = response_content.get("detail", response_content)
    else:
        detail = response_content

    if detail is None:
        return ""

    text = detail if isinstance(detail, str) else str(detail)
    text = text.strip()
    if len(text) > _MAX_DETAIL_CHARS:
        text = text[:_MAX_DETAIL_CHARS] + "..."
    return text


class AIClientError(Exception):
    """
    Error that occurs when communicating with AI service.

    Attributes:
        message (str): Error description message.
        status_code (int | None): HTTP status code (e.g. 404, 500). None if connection error.
        response_content (Any | None): Raw content returned from server (usually dict or str).
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        # The backend returns whatever shape the failing endpoint produces, so
        # this genuinely cannot be narrowed.
        response_content: Any = None,  # noqa: ANN401
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.response_content = response_content

        # print(e) is how every example reports a failure, so the status code and
        # the backend's explanation both have to survive into the string form.
        parts = [message]
        if status_code:
            parts.append(f"(Status: {status_code})")
        explanation = _explanation(response_content)
        if explanation:
            parts.append(f"- {explanation}")
        super().__init__(" ".join(parts))


class ModelLoadFailedError(AIClientError):
    """The backend finished loading but reported no loaded model."""


class TrainingFailedError(AIClientError):
    """
    The backend reported that fine-tune training failed.

    Subclasses :class:`AIClientError` so existing broad handlers still catch it,
    while letting the monitoring loop tell a real training failure apart from a
    transient polling error without matching on message text.
    """
