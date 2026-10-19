"""
Who may reach the service, from where (service/remote_access.py, service/server_args.py).

Opened to the network, the service used to answer anyone who could reach its port: chat,
model loads and unloads, downloads, the web interface. And on this machine it answered
every web page open in a browser, which could call it on 127.0.0.1 and read the answers.
These pin what replaced that: another machine gets the inference API with the right key
and nothing else unless SERVICE_REMOTE_ACCESS says "all"; this machine keeps everything
without a key, except a web page from another site; and a load request cannot move the
inference servers off loopback or open them up.
"""

import pytest
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from service import remote_access, server_args
from service.config_models import InferenceConfig
from service.remote_access import (
    RemoteAccessMiddleware,
    is_local,
    is_local_origin,
    local_refusal,
    remote_refusal,
)

pytestmark = pytest.mark.inference

KEY = "k-3f9a1c"
LOCAL = ("127.0.0.1", 50000)
REMOTE = ("192.168.1.20", 50000)
LOCAL_URL = "http://127.0.0.1:8571"

PATHS = [
    "/health",
    "/v1/models",
    "/v1/chat/completions",
    "/inference/status",
    "/config/models",
    "/training/status",
    "/",
    "/frontend/",
    "/assets/index.js",
    "/config.json",
    "/docs",
    "/openapi.json",
]
INFERENCE = ["/v1/models", "/v1/chat/completions"]
MANAGEMENT = ["/inference/status", "/config/models", "/training/status"]
WEB = ["/", "/frontend/", "/assets/index.js", "/config.json", "/docs", "/openapi.json"]


def _app(api_key: str, access: str = "inference") -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route(p, ok, methods=["GET", "POST"]) for p in PATHS])
    app.add_middleware(RemoteAccessMiddleware, api_key=lambda: api_key, access=lambda: access)
    return app


def _get(app, client, path, key=None, base_url=LOCAL_URL, headers=None):
    sent = {"Authorization": f"Bearer {key}"} if key else {}
    sent.update(headers or {})
    return TestClient(app, client=client, base_url=base_url).get(path, headers=sent)


# ------------------------------------------------------------------ another machine


@pytest.mark.parametrize("path", PATHS)
def test_this_machine_reaches_everything_without_a_key(path):
    assert _get(_app(KEY), LOCAL, path).status_code == 200


@pytest.mark.parametrize("path", INFERENCE)
def test_another_machine_reaches_the_inference_api_with_the_key(path):
    app = _app(KEY)
    assert _get(app, REMOTE, path, KEY).status_code == 200
    missing = _get(app, REMOTE, path)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert _get(app, REMOTE, path, "wrong").status_code == 401


@pytest.mark.parametrize("path", MANAGEMENT)
def test_management_api_stays_on_this_machine_by_default(path):
    """Loading models, downloads, training: not for another computer, key or not."""
    response = _get(_app(KEY), REMOTE, path, KEY)
    assert response.status_code == 403
    assert "/v1/" in response.json()["detail"]


@pytest.mark.parametrize("path", MANAGEMENT + INFERENCE)
def test_all_opens_the_management_api_with_the_key(path):
    app = _app(KEY, access="all")
    assert _get(app, REMOTE, path, KEY).status_code == 200
    assert _get(app, REMOTE, path).status_code == 401


@pytest.mark.parametrize("access", ["inference", "all"])
@pytest.mark.parametrize("path", WEB)
def test_another_machine_never_gets_the_web_interface_or_docs(path, access):
    # Not even with the key, not even with "all": they were never meant for another computer.
    assert _get(_app(KEY, access), REMOTE, path, KEY).status_code == 403


def test_health_is_open_to_another_machine():
    assert _get(_app(KEY), REMOTE, "/health").status_code == 200
    assert _get(_app(""), REMOTE, "/health").status_code == 200


def test_without_a_configured_key_another_machine_is_refused():
    response = _get(_app(""), REMOTE, "/v1/models", "anything")
    assert response.status_code == 403
    assert "not enabled" in response.json()["detail"]


def test_an_unknown_access_value_is_the_narrow_one():
    assert remote_refusal("/config/models", f"Bearer {KEY}", KEY, "everything")[0] == 403
    assert remote_refusal("/v1/models", f"Bearer {KEY}", KEY, "everything") is None


def test_the_key_is_read_on_every_request():
    """The launcher changes the key and restarts the service; a test patches settings.
    Either way the middleware must use the current value, not one captured at import."""
    current = {"key": "first"}
    app = Starlette(routes=[Route("/v1/models", lambda r: PlainTextResponse("ok"))])
    app.add_middleware(RemoteAccessMiddleware, api_key=lambda: current["key"])
    assert _get(app, REMOTE, "/v1/models", "first").status_code == 200
    current["key"] = "second"
    assert _get(app, REMOTE, "/v1/models", "first").status_code == 401
    assert _get(app, REMOTE, "/v1/models", "second").status_code == 200


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("127.0.0.1", True),
        ("127.5.6.7", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),  # a dual-stack socket reports IPv4 loopback like this
        (None, True),  # a Unix socket has no address, and is local by definition
        ("192.168.1.20", False),
        ("10.0.0.5", False),
        ("100.117.136.108", False),  # a Tailscale address is another machine too
        ("::ffff:192.168.1.20", False),
        ("fe80::1", False),
        ("testclient", False),  # not an address at all: never trusted
    ],
)
def test_is_local(host, local):
    assert is_local(host) is local


def test_only_the_bearer_scheme_carries_the_key():
    assert remote_refusal("/v1/models", f"Bearer {KEY}", KEY) is None
    assert remote_refusal("/v1/models", f"bearer {KEY}", KEY) is None
    assert remote_refusal("/v1/models", KEY, KEY)[0] == 401
    assert remote_refusal("/v1/models", f"Basic {KEY}", KEY)[0] == 401


def test_a_new_route_is_closed_to_the_network_until_listed():
    # The allowlist, not a denylist: something added later is this machine's only.
    assert remote_refusal("/admin/something", f"Bearer {KEY}", KEY, "all")[0] == 403
    assert not "/admin/something".startswith(remote_access.REMOTE_PREFIXES["all"])


# --------------------------------------------------------------------- this machine


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8571",
        "http://localhost:8571",
        "http://localhost:5173",  # a development server
        "http://[::1]:8571",
    ],
)
def test_pages_from_this_machine_keep_working(origin):
    assert _get(_app(KEY), LOCAL, "/config/models", headers={"Origin": origin}).status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://192.168.1.20:8571",
        "http://127.0.0.1.evil.example",
        "null",  # a sandboxed frame or a file:// page
    ],
)
def test_pages_from_other_sites_are_refused(origin):
    response = _get(_app(KEY), LOCAL, "/config/models", headers={"Origin": origin})
    assert response.status_code == 403


def test_a_cross_site_load_without_an_origin_is_refused():
    """An <img> or <script> from another site sends no Origin, but browsers mark it."""
    headers = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}
    assert _get(_app(KEY), LOCAL, "/config/models", headers=headers).status_code == 403


def test_following_a_link_from_another_site_still_opens_the_interface():
    headers = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    assert _get(_app(KEY), LOCAL, "/frontend/", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:8571", "http://localhost:8571", "http://app.localhost"],
)
def test_local_host_names_are_accepted(base_url):
    assert _get(_app(KEY), LOCAL, "/v1/models", base_url=base_url).status_code == 200


@pytest.mark.parametrize("host", ["[::1]:8571", "[::1]", "192.168.1.5:8571", "LOCALHOST:8571"])
def test_local_host_headers_are_accepted(host):
    # Directly: the test client cannot be pointed at an IPv6 literal.
    assert local_refusal({"host": host}) is None


def test_a_rebound_host_name_is_refused():
    """DNS rebinding: a name the attacker controls, pointed at 127.0.0.1."""
    response = _get(_app(KEY), LOCAL, "/config/models", base_url="http://rebind.evil.example:8571")
    assert response.status_code == 403
    assert "localhost" in response.json()["detail"]


def test_clients_that_are_not_web_pages_send_nothing_to_check():
    assert local_refusal({"host": "127.0.0.1:8571"}) is None
    assert local_refusal({}) is None


@pytest.mark.parametrize(
    ("origin", "local"),
    [
        ("http://127.0.0.1:8571", True),
        ("https://localhost", True),
        ("http://127.0.0.1.nip.io", False),
        ("ftp://127.0.0.1", False),
        ("null", False),
    ],
)
def test_is_local_origin(origin, local):
    assert is_local_origin(origin) is local


# --------------------------------------------------------------------- the real app


def _real_app(monkeypatch, access="inference"):
    import service.app as app_module
    from service import settings

    monkeypatch.setattr(settings, "SERVICE_API_KEY", KEY)
    monkeypatch.setattr(settings, "SERVICE_REMOTE_ACCESS", access)
    return app_module.app


def test_the_real_app_applies_it(monkeypatch):
    """The middleware is installed on the service itself, reading its settings."""
    app = _real_app(monkeypatch)
    remote = TestClient(app, client=REMOTE)
    auth = {"Authorization": f"Bearer {KEY}"}
    assert remote.get("/health").status_code == 200
    assert remote.get("/v1/models").status_code == 401
    assert remote.get("/v1/models", headers=auth).status_code == 200
    assert remote.get("/inference/status", headers=auth).status_code == 403
    assert remote.get("/docs").status_code == 403


def test_the_real_app_opens_management_with_all(monkeypatch):
    app = _real_app(monkeypatch, access="all")
    remote = TestClient(app, client=REMOTE)
    assert remote.get("/inference/status", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_the_real_app_answers_cors_for_this_machine_only(monkeypatch):
    app = _real_app(monkeypatch)
    local = TestClient(app, client=LOCAL, base_url=LOCAL_URL)
    preflight = {"Access-Control-Request-Method": "POST"}

    ours = local.options("/config/models", headers={"Origin": "http://localhost:5173", **preflight})
    assert ours.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert ours.headers.get("access-control-allow-credentials") is None

    theirs = local.options("/config/models", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in theirs.headers


# --------------------------------------------------- what a load request cannot change


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "example.com"])
def test_llama_server_cannot_be_moved_off_loopback(host):
    with pytest.raises(ValidationError, match="loopback"):
        InferenceConfig(model_name="m", engine="llama_server", llama_server_host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_llama_server_on_loopback_is_fine(host):
    InferenceConfig(model_name="m", engine="llama_server", llama_server_host=host)


@pytest.mark.parametrize(
    "args",
    [
        ["--host", "0.0.0.0"],
        ["--host=0.0.0.0"],
        ["--port", "80"],
        ["--api-key", "mine"],
        ["--cors-origins", "*"],
        ["--path", "/"],
        ["--slot-save-path", "/tmp"],
        ["--mlock", "--log-file", "/tmp/x"],
    ],
)
def test_llama_extra_args_cannot_take_what_the_service_decides(args):
    with pytest.raises(ValidationError, match="cannot set"):
        InferenceConfig(model_name="m", engine="llama_server", llama_server_extra_args=args)


@pytest.mark.parametrize(
    "args",
    [
        ["--host", "0.0.0.0"],
        ["--allowed-origins", '["*"]'],
        ["--middleware", "evil.module.Middleware"],
        ["--api-key=mine"],
        ["--uds", "/tmp/s"],
    ],
)
def test_vllm_extra_args_cannot_take_what_the_service_decides(args):
    with pytest.raises(ValidationError, match="cannot set"):
        InferenceConfig(model_name="m", engine="vllm", vllm_server_extra_args=args)


def test_tuning_flags_still_pass():
    InferenceConfig(
        model_name="m",
        engine="llama_server",
        llama_server_extra_args=["--mlock", "--no-mmap", "-ot", "exps=CPU", "--mmproj", "/m/p.gguf"],
    )
    InferenceConfig(
        model_name="m",
        engine="vllm",
        vllm_server_extra_args=["--trust-remote-code", "--tool-call-parser", "hermes"],
    )


def test_local_service_origins_follow_the_port(monkeypatch):
    from service import settings

    monkeypatch.setattr(settings, "SERVICE_PORT", 9123)
    assert server_args.local_service_origins() == [
        "http://127.0.0.1:9123",
        "http://localhost:9123",
    ]


# ------------------------------------------------------------------- wrong-key lockout


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _locking_app(clock):
    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route(p, ok) for p in ("/health", "/v1/models")])
    lockout = remote_access.KeyLockout(limit=3, window=60, duration=300, clock=clock)
    app.add_middleware(RemoteAccessMiddleware, api_key=lambda: KEY, lockout=lockout)
    return app


def test_repeated_wrong_keys_lock_the_address_out():
    clock = _Clock()
    app = _locking_app(clock)
    for _ in range(3):
        assert _get(app, REMOTE, "/v1/models", "wrong").status_code == 401
    locked = _get(app, REMOTE, "/v1/models", KEY)  # even the right key, for now
    assert locked.status_code == 429
    assert 0 < int(locked.headers["retry-after"]) <= 301
    assert _get(app, REMOTE, "/health").status_code == 200  # still says it is up

    clock.now += 301
    assert _get(app, REMOTE, "/v1/models", KEY).status_code == 200


def test_wrong_keys_spread_out_do_not_lock():
    clock = _Clock()
    app = _locking_app(clock)
    for _ in range(5):
        assert _get(app, REMOTE, "/v1/models", "wrong").status_code == 401
        clock.now += 61
    assert _get(app, REMOTE, "/v1/models", KEY).status_code == 200


def test_the_lockout_is_per_address():
    clock = _Clock()
    app = _locking_app(clock)
    for _ in range(3):
        _get(app, REMOTE, "/v1/models", "wrong")
    assert _get(app, ("192.168.1.21", 50000), "/v1/models", KEY).status_code == 200
    assert _get(app, LOCAL, "/v1/models").status_code == 200


def test_a_refused_path_is_not_a_wrong_key():
    """A 403 for the management API is not a guess at the key."""
    clock = _Clock()
    app = _locking_app(clock)
    for _ in range(5):
        assert _get(app, REMOTE, "/inference/status", KEY).status_code in (403, 404)
    assert _get(app, REMOTE, "/v1/models", KEY).status_code == 200


# ------------------------------------------------------------ a server of the request's


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:5001", "http://localhost:8080/", "http://[::1]:5001", "", None]
)
def test_llama_server_url_on_this_machine_is_fine(url):
    InferenceConfig(model_name="m", engine="llama_server", llama_server_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.9:9999",
        "http://169.254.169.254/latest",  # a cloud metadata service
        "http://example.com:5001",
        "file:///etc/passwd",
        "http://127.0.0.1.evil.example",
    ],
)
def test_llama_server_url_elsewhere_is_refused(url):
    with pytest.raises(ValidationError, match="this machine"):
        InferenceConfig(model_name="m", engine="llama_server", llama_server_url=url)
