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

_TIMEOUT_SECONDS = 6


@dataclass(frozen=True)
class Result:
    unreachable: tuple[tuple[str, str, str], ...]

    @property
    def offline(self) -> bool:
        return len(self.unreachable) == len(HOSTS)

    @property
    def ok(self) -> bool:
        return not self.unreachable


def _reachable(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "AI-Scaler-Toolkit"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS):
            return True
    except urllib.error.HTTPError:
        # An answer is an answer: a 403 or 405 to a HEAD still proves the host is there.
        return True
    except (OSError, ValueError):
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
