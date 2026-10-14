"""
What `print(e)` shows when a request fails.

Every example reports failures as `print(f"[FAIL] ...: {e}")`, so anything the
exception drops from its string form is lost to the user. The backend writes
actionable refusals into FastAPI's `detail` -- most visibly /training/start
returning 501 to say fine-tuning needs a Linux host -- and those have to survive.
"""

from __future__ import annotations

from ai_scaler_console.exceptions import AIClientError

LINUX_GATE = (
    "Fine-tuning is only supported on Linux; this backend is running on Windows. "
    "Run the backend on Linux (a WSL2 distro counts) to fine-tune models."
)


def test_the_backends_explanation_reaches_the_printed_message() -> None:
    error = AIClientError(
        "API request failed at /training/start",
        status_code=501,
        response_content={"detail": LINUX_GATE},
    )

    assert LINUX_GATE in str(error)
    assert "501" in str(error)


def test_a_plain_string_body_is_shown_too() -> None:
    error = AIClientError("boom", status_code=500, response_content="internal boom")

    assert "internal boom" in str(error)


def test_a_body_that_is_not_a_detail_dict_still_shows() -> None:
    """422s carry a list under `detail`; an unexpected shape must not vanish."""
    error = AIClientError("boom", status_code=422, response_content={"errors": ["bad field"]})

    assert "bad field" in str(error)


def test_no_body_leaves_the_message_unchanged() -> None:
    assert str(AIClientError("boom", status_code=404)) == "boom (Status: 404)"


def test_a_connection_error_has_neither_status_nor_detail() -> None:
    assert str(AIClientError("Network error connecting to host")) == (
        "Network error connecting to host"
    )


def test_a_long_body_is_truncated() -> None:
    error = AIClientError("boom", status_code=422, response_content={"detail": "x" * 5000})

    assert len(str(error)) < 1000
    assert str(error).endswith("...")


def test_the_raw_body_is_still_available_on_the_attribute() -> None:
    """Callers that inspect the response rather than print it keep working."""
    body = {"detail": LINUX_GATE}
    error = AIClientError("boom", status_code=501, response_content=body)

    assert error.response_content == body
    assert error.status_code == 501
    assert error.message == "boom"
