"""
backend/services/cache_service.py

Simple in-memory TTL cache keyed on location string (city / coordinates).
Avoids redundant model inference for repeated requests within the TTL window.
"""
import time
import threading
import logging

logger = logging.getLogger("cache_service")

DEFAULT_TTL_SECONDS = 300   # 5 minutes


class CacheService:
    """Thread-safe in-memory prediction cache."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._store: dict = {}         # key → {"payload": …, "expires_at": float}
                obj._ttl = DEFAULT_TTL_SECONDS
                cls._instance = obj
        return cls._instance

    @classmethod
    def instance(cls) -> "CacheService":
        return cls()

    # ---------- public API ----------

    def get(self, key: str):
        """Return cached payload or None if missing/expired."""
        entry = self._store.get(self._normalise(key))
        if entry is None:
            return None
        if time.monotonic() > entry["expires_at"]:
            self._store.pop(self._normalise(key), None)
            logger.debug(f"Cache EXPIRED for key: {key}")
            return None
        logger.debug(f"Cache HIT for key: {key}")
        return entry["payload"]

    def set(self, key: str, payload: dict):
        """Store payload with TTL."""
        self._store[self._normalise(key)] = {
            "payload": payload,
            "expires_at": time.monotonic() + self._ttl
        }
        logger.debug(f"Cache SET for key: {key} (TTL={self._ttl}s)")

    def invalidate(self, key: str):
        self._store.pop(self._normalise(key), None)

    def clear(self):
        self._store.clear()

    # ---------- private helpers ----------

    @staticmethod
    def _normalise(key: str) -> str:
        return key.lower().strip().replace(" ", "_")
