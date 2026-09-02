"""Read the public releases feed. No token, no writes."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

API = "https://api.github.com/repos/{repo}/releases/latest"
DEFAULT_REPO = "ADATA-TRUSTA/AI-Scaler-Toolkit"

VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class Release:
    tag: str
    name: str
    url: str
    asset_url: str | None
    asset_name: str | None
    prerelease: bool


def parse_version(tag: str) -> tuple[int, int, int] | None:
    match = VERSION_RE.match(tag or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_newer(candidate: str, current: str | None) -> bool:
    """Compare numerically, not as strings -- "v2.10.0" sorts before "v2.9.0" as text."""
    new = parse_version(candidate)
    if new is None:
        return False
    old = parse_version(current or "")
    return old is None or new > old


class NoReleases(Exception):
    """The repository is reachable and has published nothing yet.

    Worth its own type: it is indistinguishable from a network failure at the HTTP
    level -- GitHub answers /releases/latest with 404 in both the "no releases" and
    "no such repo" cases -- but telling a user "could not reach the releases page"
    when the page is fine and simply empty sends them looking for a problem that is
    not there.
    """


def latest(repo: str = DEFAULT_REPO, asset_suffix: str = ".exe", timeout: float = 10.0) -> Release | None:
    request = urllib.request.Request(
        API.format(repo=repo),
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ai-scaler-toolkit-launcher"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NoReleases(repo) from exc
        return None
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        # No network, or rate limited. An update check that cannot reach the internet
        # is a check that found nothing, not a failure worth interrupting anyone over.
        return None

    asset = next(
        (a for a in body.get("assets", []) if str(a.get("name", "")).endswith(asset_suffix)),
        None,
    )
    return Release(
        tag=body.get("tag_name", ""),
        name=body.get("name") or body.get("tag_name", ""),
        url=body.get("html_url", ""),
        asset_url=asset.get("browser_download_url") if asset else None,
        asset_name=asset.get("name") if asset else None,
        prerelease=bool(body.get("prerelease")),
    )
