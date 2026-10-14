"""Whether first-time setup can reach what it downloads, asked before it downloads.

Setup fetches a Python, the packages, the GPU build of PyTorch and the llama runtime,
all from the internet, and without a connection it failed eight seconds in -- measured
on a Windows VM with its network adapter unplugged -- showing the tail of uv's log: a
GitHub URL for a Python build, "Request failed after 3 retries". True, and no help to
anyone who did not already know the machine was offline.

So the hosts setup actually uses are asked first, in parallel, briefly. None of them
answering means no connection; some of them not answering is usually a firewall or a
proxy, and naming which is the useful part. Only asked when there is setup to do: an
installed machine runs without a network, and must not be told it needs one.

Through urllib, not a raw socket, so a proxy is honoured the way a download would
honour it -- HTTPS_PROXY, and on Windows the system proxy setting. A direct TCP probe
would call a machine behind a proxy offline while every download on it worked.
"""

from __future__ import annotations

import http.client
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import applog

log = applog.get("connectivity")

# What setup downloads from, and what for. Hugging Face is not here: only the starter
# model comes from it, after setup, and that step already carries on without it.
HOSTS: tuple[tuple[str, str, str], ...] = (
    ("GitHub", "https://github.com", "Python and the llama runtime"),
    ("PyPI", "https://pypi.org/simple/", "Python packages"),
    ("PyTorch", "https://download.pytorch.org/whl/", "the GPU build of PyTorch"),
)

# A host that misses the first is asked once more, with longer: a slow answer is not a
# missing one. With a single 6 s try, a Panther Lake laptop was told PyPI was blocked --
# setup stopped there -- while PyPI answered a HEAD in 0.25 s before and after; the
# moment it was asked, on Wi-Fi straight after an apt install, it took 13 s. An
# unplugged machine still finds out at once: name resolution fails on both tries.
_TIMEOUTS_SECONDS = (8, 15)


@dataclass(frozen=True)
class Result:
    unreachable: tuple[tuple[str, str, str], ...]

    @property
    def offline(self) -> bool:
        return len(self.unreachable) == len(HOSTS)

    @property
    def ok(self) -> bool:
        return not self.unreachable


def _connect_ipv4(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):  # noqa: ANN001, ANN202 - http.client's own signature
    host, port = address
    error: OSError | None = None
    for family, kind, proto, _, sockaddr in socket.getaddrinfo(host, port, socket.AF_INET,
                                                               socket.SOCK_STREAM):
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            error = exc
            sock.close()
    raise error or OSError(f"no IPv4 address for {host}")


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_ipv4


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):  # noqa: ANN001, ANN201
        return self.do_open(_IPv4HTTPSConnection, req, context=self._context)


# A fallback over IPv4 only. urllib tries a host's addresses in order and moves to the
# next only when a connection cannot be made; one that is made and then reset is a
# failure. A Panther Lake laptop on two different networks had exactly that over IPv6 --
# the handshake with download.pytorch.org reset six times in eight while IPv4 answered
# every time -- and setup was stopped as "PyTorch unreachable" although uv, which falls
# back by itself, would have downloaded everything.
_ipv4_opener = urllib.request.build_opener(_IPv4HTTPSHandler())


def _reachable(url: str) -> bool:
    for timeout in _TIMEOUTS_SECONDS:
        for opener in (urllib.request.urlopen, _ipv4_opener.open):
            request = urllib.request.Request(url, method="HEAD",
                                             headers={"User-Agent": "AI-Scaler-Toolkit"})
            try:
                with opener(request, timeout=timeout):
                    return True
            except urllib.error.HTTPError:
                # An answer is an answer: a 403 or 405 to a HEAD still proves the host is there.
                return True
            except ValueError:
                return False
            except OSError:
                continue
    return False


def check() -> Result:
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        answers = list(pool.map(lambda host: _reachable(host[1]), HOSTS))
    unreachable = tuple(host for host, ok in zip(HOSTS, answers, strict=True) if not ok)
    log.info("setup hosts: %s", ", ".join(
        f"{host[0]} {'ok' if ok else 'UNREACHABLE'}" for host, ok in zip(HOSTS, answers, strict=True)))
    return Result(unreachable=unreachable)


def explain(result: Result) -> tuple[str, str]:
    """(title, message) for a result that is not ok."""
    if result.offline:
        return ("No internet connection",
                "First-time setup downloads the GPU runtime, several gigabytes of it, "
                "from the internet. Connect to the internet, then open AI Scaler "
                "Toolkit again.")
    names = "\n".join(f"  {name} ({url.split('/')[2]}), for {purpose}"
                      for name, url, purpose in result.unreachable)
    return ("Setup cannot reach everything it needs",
            f"These could not be reached:\n\n{names}\n\n"
            "The rest of the internet answered, so this is usually a firewall or a "
            "proxy blocking them. Once they can be reached, open AI Scaler Toolkit "
            "again.")
