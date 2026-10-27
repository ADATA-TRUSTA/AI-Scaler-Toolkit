"""The guard vLLM loads in front of its admin endpoints (service/vllm_middleware)."""

import sys
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "service" / "vllm_middleware"))
import ast_vllm_guard  # noqa: E402

ADMIN = sorted(ast_vllm_guard.FALLBACK_ADMIN)
BLOCKED = sorted(ast_vllm_guard.FALLBACK_ADMIN - ast_vllm_guard.ALLOWED)
ALLOWED = sorted(ast_vllm_guard.ALLOWED)


def _vllm():
    async def ok(request):
        return PlainTextResponse("ok")

    paths = [*ADMIN, "/v1/models", "/health"]
    app = Starlette(routes=[Route(p, ok, methods=["GET", "POST"]) for p in paths])
    app.add_middleware(ast_vllm_guard.AdminGuard)
    return TestClient(app, base_url="http://127.0.0.1:5000")


@pytest.mark.parametrize("path", ALLOWED)
def test_a_program_on_this_machine_can_sleep_and_wake_it(path):
    assert _vllm().post(path).status_code == 200


@pytest.mark.parametrize("path", BLOCKED)
def test_the_rest_of_the_admin_api_is_not_there(path):
    """/collective_rpc and the weight updates answer as they do with dev mode off."""
    response = _vllm().post(path)
    assert response.status_code == 404
    assert _vllm().post(path + "/").status_code == 404


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:8571"},
        {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"},
        {"Sec-Fetch-Site": "same-origin"},
        {"Host": "rebind.evil.example:5000"},
    ],
)
def test_a_web_page_cannot_put_it_to_sleep(headers):
    """Any browser request, or a DNS-rebound host name."""
    assert _vllm().post("/sleep", headers=headers).status_code == 403


@pytest.mark.parametrize("host", ["localhost:5000", "127.0.0.1", "[::1]:5000", "a.localhost"])
def test_loopback_host_names_are_fine(host):
    assert _vllm().get("/is_sleeping", headers={"Host": host}).status_code == 200


def test_the_rest_of_vllm_is_untouched():
    client = _vllm()
    assert client.get("/v1/models", headers={"Origin": "http://evil.example"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_without_vllm_the_known_list_is_used(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(ast_vllm_guard.importlib, "import_module", missing)
    assert ast_vllm_guard.admin_paths() == ast_vllm_guard.FALLBACK_ADMIN


def test_admin_routes_are_read_from_the_installed_vllm(monkeypatch, tmp_path):
    """A route a later vLLM adds under its dev routers is refused, not missed."""
    package = tmp_path / "fakevllm" / "dev" / "newthing"
    package.mkdir(parents=True)
    for d in (tmp_path / "fakevllm", tmp_path / "fakevllm" / "dev", package):
        (d / "__init__.py").write_text("")
    (package / "api_router.py").write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.post('/brand_new_admin_thing')\n"
        "def f(): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(ast_vllm_guard, "_DEV_PACKAGE", "fakevllm.dev")
    assert ast_vllm_guard.admin_paths() == {"/brand_new_admin_thing"}


def test_it_does_not_import_the_service():
    """vLLM's interpreter has only this directory on its path, not the service."""
    source = Path(ast_vllm_guard.__file__).read_text(encoding="utf-8")
    assert "import service" not in source
    assert "from service" not in source
    assert "from ." not in source
