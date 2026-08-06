"""
Session management with Redis (fallback to in-memory store).
- Primary: Redis using REDIS_URL env (default: redis://localhost:6379/0)
- Optional TTL via SESSION_TTL_SECONDS env (default: 86400 seconds).

Stored format per session_id: JSON list of messages
  [{"role": "system|user|assistant", "content": "..."}, ...]
"""

from __future__ import annotations

import json
import logging
import os
from typing import cast

logger = logging.getLogger(__name__)


class BaseStore:
    """Abstract session store interface."""

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the stored message history for a session."""
        raise NotImplementedError

    def set_history(
        self, session_id: str, history: list[dict[str, str]], ttl: int | None = None
    ) -> None:
        """Replace the stored message history for a session."""
        raise NotImplementedError

    def append_message(
        self, session_id: str, message: dict[str, str], ttl: int | None = None
    ) -> None:
        """Append a single message to a session's history."""
        raise NotImplementedError

    def reset(self, session_id: str) -> None:
        """Delete all stored history for a session."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying store resources."""


class InMemoryStore(BaseStore):
    """In-process session store used when Redis is unavailable."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of the session's message history."""
        return list(self._data.get(session_id, []))

    def set_history(
        self, session_id: str, history: list[dict[str, str]], ttl: int | None = None
    ) -> None:
        """Replace the session's message history (TTL ignored in memory)."""
        # TTL is ignored in memory store
        self._data[session_id] = list(history)

    def append_message(
        self, session_id: str, message: dict[str, str], ttl: int | None = None
    ) -> None:
        """Append a single message to the session's history."""
        self._data.setdefault(session_id, []).append(dict(message))

    def reset(self, session_id: str) -> None:
        """Delete the session's stored history."""
        self._data.pop(session_id, None)


class RedisStore(BaseStore):
    """Redis-backed session store with JSON-encoded history."""

    def __init__(self, url: str) -> None:
        import redis  # type: ignore

        self._redis = redis.Redis.from_url(url, decode_responses=True)
        # simple ping to validate connectivity
        self._redis.ping()

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the session's message history, filtering malformed entries."""
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            # decode_responses=True makes get() return str; redis stubs widen it to ResponseT.
            val = json.loads(cast(str, raw))
            if isinstance(val, list):
                return [m for m in val if isinstance(m, dict) and "role" in m and "content" in m]
        except Exception:
            logger.warning("Invalid history JSON for session %s", session_id)
        return []

    def set_history(
        self, session_id: str, history: list[dict[str, str]], ttl: int | None = None
    ) -> None:
        """Replace the session's message history, applying TTL when positive."""
        data = json.dumps(history, ensure_ascii=False)
        key = self._key(session_id)
        if ttl and ttl > 0:
            self._redis.setex(key, ttl, data)
        else:
            self._redis.set(key, data)

    def append_message(
        self, session_id: str, message: dict[str, str], ttl: int | None = None
    ) -> None:
        """Append a single message to the session's history."""
        history = self.get_history(session_id)
        history.append(message)
        self.set_history(session_id, history, ttl=ttl)

    def reset(self, session_id: str) -> None:
        """Delete the session's stored history from Redis."""
        self._redis.delete(self._key(session_id))

    def close(self) -> None:
        """Close the underlying Redis connection."""
        try:
            self._redis.close()
        except Exception:
            pass


class SessionManager:
    """Facade over a Redis or in-memory session store."""

    def __init__(self) -> None:
        self.ttl = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._store: BaseStore
        try:
            self._store = RedisStore(url)
            logger.info("Session store: Redis connected at %s", url)
        except Exception as e:
            logger.warning(
                "Redis unavailable (%s). Falling back to in-memory session store.",
                str(e).rstrip("."),
            )
            self._store = InMemoryStore()

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """Return the session's history, or empty when no session id is given."""
        if not session_id:
            return []
        return self._store.get_history(session_id)

    def set_history(self, session_id: str | None, history: list[dict[str, str]]) -> None:
        """Replace the session's history using the configured TTL."""
        if not session_id:
            return
        self._store.set_history(session_id, history, ttl=self.ttl)

    def append_message(self, session_id: str | None, message: dict[str, str]) -> None:
        """Append a single message using the configured TTL."""
        if not session_id:
            return
        self._store.append_message(session_id, message, ttl=self.ttl)

    def reset(self, session_id: str | None) -> None:
        """Delete the session's stored history."""
        if not session_id:
            return
        self._store.reset(session_id)

    def close(self) -> None:
        """Close the underlying store."""
        try:
            self._store.close()
        except Exception:
            pass


# Singleton instance
session_manager = SessionManager()
