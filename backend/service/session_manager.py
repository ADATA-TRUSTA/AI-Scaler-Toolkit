"""
Session management with Redis (fallback to in-memory store).
- Primary: Redis using REDIS_URL env (default: redis://localhost:6379/0)
- Optional TTL via SESSION_TTL_SECONDS env (default: 86400 seconds)

Stored format per session_id: JSON list of messages
  [{"role": "system|user|assistant", "content": "..."}, ...]
"""
from __future__ import annotations
import json
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BaseStore:
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        raise NotImplementedError

    def set_history(self, session_id: str, history: List[Dict[str, str]], ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def append_message(self, session_id: str, message: Dict[str, str], ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def reset(self, session_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class InMemoryStore(BaseStore):
    def __init__(self) -> None:
        self._data: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._data.get(session_id, []))

    def set_history(self, session_id: str, history: List[Dict[str, str]], ttl: Optional[int] = None) -> None:
        # TTL is ignored in memory store
        self._data[session_id] = list(history)

    def append_message(self, session_id: str, message: Dict[str, str], ttl: Optional[int] = None) -> None:
        self._data.setdefault(session_id, []).append(dict(message))

    def reset(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisStore(BaseStore):
    def __init__(self, url: str) -> None:
        import redis  # type: ignore
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        # simple ping to validate connectivity
        self._redis.ping()

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        raw = self._redis.get(self._key(session_id))
        if not raw:
            return []
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [m for m in val if isinstance(m, dict) and "role" in m and "content" in m]
        except Exception:
            logger.warning("Invalid history JSON for session %s", session_id)
        return []

    def set_history(self, session_id: str, history: List[Dict[str, str]], ttl: Optional[int] = None) -> None:
        data = json.dumps(history, ensure_ascii=False)
        key = self._key(session_id)
        if ttl and ttl > 0:
            self._redis.setex(key, ttl, data)
        else:
            self._redis.set(key, data)

    def append_message(self, session_id: str, message: Dict[str, str], ttl: Optional[int] = None) -> None:
        history = self.get_history(session_id)
        history.append(message)
        self.set_history(session_id, history, ttl=ttl)

    def reset(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass


class SessionManager:
    def __init__(self) -> None:
        self.ttl = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._store: BaseStore
        try:
            self._store = RedisStore(url)
            logger.info("Session store: Redis connected at %s", url)
        except Exception as e:
            logger.warning("Redis unavailable (%s). Falling back to in-memory session store.", str(e))
            self._store = InMemoryStore()

    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id:
            return []
        return self._store.get_history(session_id)

    def set_history(self, session_id: Optional[str], history: List[Dict[str, str]]) -> None:
        if not session_id:
            return
        self._store.set_history(session_id, history, ttl=self.ttl)

    def append_message(self, session_id: Optional[str], message: Dict[str, str]) -> None:
        if not session_id:
            return
        self._store.append_message(session_id, message, ttl=self.ttl)

    def reset(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        self._store.reset(session_id)

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:
            pass


# Singleton instance
session_manager = SessionManager()
