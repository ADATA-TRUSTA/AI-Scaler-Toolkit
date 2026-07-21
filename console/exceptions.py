from typing import Optional, Any


class AIClientError(Exception):
    """
    Error that occurs when communicating with AI service.

    Attributes:
        message (str): Error description message.
        status_code (Optional[int]): HTTP status code (e.g. 404, 500). None if connection error.
        response_content (Optional[Any]): Raw content returned from server (usually dict or str).
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_content: Any = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_content = response_content

        # Combine parent class message for easy viewing of status code when print(e)
        if status_code:
            super().__init__(f"{message} (Status: {status_code})")
        else:
            super().__init__(message)
