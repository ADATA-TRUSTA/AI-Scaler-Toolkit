"""AI API Client for communicating with the AI service backend.

Scope (post-refactor):
- Model lifecycle: load_model / unload_model / get_status
- Fine-tune lifecycle: start_training / stop_training / get_training_status

Inference (chat) is no longer wrapped here. Use the OpenAI Python SDK against
the backend's OpenAI-compatible endpoint (`<backend_url>/v1/chat/completions`)
instead. See `openai-compatible-example.py` and
`openai-compatible-image-example.py` for usage.
"""

from typing import Type, Optional, Union, TypeVar, overload
import requests
import urllib3
from pydantic import BaseModel

from config_models import (
    InferenceConfig,
    ModelListResponse,
    ModelStatus,
    TrainingConfig,
    TrainingStatus,
)
from exceptions import AIClientError

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define generic type variable bound to BaseModel
T = TypeVar("T", bound=BaseModel)


class AIClient:
    """A client for communicating with the AI service backend.

    Responsibilities:
    - Model load / unload / status (`/inference/load_model`, `/inference/unload_model`,
      `/inference/status`).
    - Fine-tune start / stop / status (`/training/*`).

    Chat / generation is handled outside of this client via the OpenAI-compatible
    API (`<backend_url>/v1/chat/completions`). Use the OpenAI Python SDK directly
    once a model has been loaded.
    """

    def __init__(
        self,
        base_url: str = "https://localhost:8000",
        timeout: int = 10,
        log_requests: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log_requests = log_requests

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # --- Type overload definitions (Overloads) ---
    @overload
    def _request(
        self, method: str, endpoint: str, response_model: Type[T], **kwargs
    ) -> T: ...
    @overload
    def _request(
        self, method: str, endpoint: str, response_model: None = None, **kwargs
    ) -> requests.Response: ...

    def _request(
        self,
        method: str,
        endpoint: str,
        response_model: Optional[Type[T]] = None,
        **kwargs,
    ) -> Union[T, requests.Response]:
        """
        Core request method: handles URL composition, timeout, SSL verification,
        and error handling.
        """
        url = f"{self.base_url}{endpoint}"
        if self.log_requests:
            print(f"[{method}] {url}")  # Log request path

        try:
            res = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                verify=False,
                **kwargs,
            )
            res.raise_for_status()  # Check for 4xx/5xx errors

            # Automatically parse Pydantic model
            if response_model:
                try:
                    return response_model.model_validate(res.json())
                except (ValueError, requests.exceptions.JSONDecodeError) as exc:
                    raise AIClientError(
                        message=f"Failed to parse response model for {endpoint}",
                        status_code=res.status_code,
                        response_content=res.text,
                    ) from exc

            return res

        except requests.exceptions.HTTPError as e:
            try:
                error_content = e.response.json()
            except ValueError:
                error_content = e.response.text if e.response else "No response content"

            raise AIClientError(
                message=f"API request failed at {endpoint}",
                status_code=e.response.status_code if e.response else 500,
                response_content=error_content,
            ) from e

        except requests.exceptions.RequestException as e:
            raise AIClientError(f"Network error connecting to {url}: {str(e)}") from e

    # --- Type overloads for helper methods ---
    @overload
    def _post(
        self,
        endpoint: str,
        *,  # force keyword-only parameters
        payload: Optional[dict] = None,
        response_model: Type[T],
    ) -> T: ...
    @overload
    def _post(
        self,
        endpoint: str,
        *,
        payload: Optional[dict] = None,
        response_model: None = None,
    ) -> requests.Response: ...

    def _post(
        self,
        endpoint: str,
        *,
        payload: Optional[dict] = None,
        response_model: Optional[Type[T]] = None,
    ) -> Union[T, requests.Response]:
        return self._request(
            "POST", endpoint, json=payload, response_model=response_model
        )

    @overload
    def _get(self, endpoint: str, response_model: Type[T]) -> T: ...
    @overload
    def _get(self, endpoint: str, response_model: None = None) -> requests.Response: ...

    def _get(
        self, endpoint: str, response_model: Optional[Type[T]] = None
    ) -> Union[T, requests.Response]:
        return self._request("GET", endpoint, response_model=response_model)

    # ==========================================
    # --- Model lifecycle ---
    # ==========================================

    def load_model(self, config: InferenceConfig) -> requests.Response:
        """Load a model on the backend.

        Send the InferenceConfig to `/inference/load_model`. Backend returns
        success/failure plus the resolved configuration.
        """
        if config is None:
            raise AIClientError("InferenceConfig is required")
        return self._post("/inference/load_model", payload=config.model_dump())

    def unload_model(self) -> requests.Response:
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

    def start_training(self, config: TrainingConfig) -> requests.Response:
        """Start fine-tune training (`/training/start`)."""
        if config is None:
            raise AIClientError("TrainingConfig is required")
        return self._post("/training/start", payload=config.model_dump())

    def stop_training(self) -> requests.Response:
        """Stop the in-progress fine-tune training (`/training/stop`)."""
        return self._post("/training/stop")

    def get_training_status(self) -> TrainingStatus:
        """Get the current fine-tune training status (`/training/status`)."""
        return self._get("/training/status", response_model=TrainingStatus)
