"""Bounded, expiring in-memory artifacts. Resume content never enters cookies."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class ResumeArtifact:
    latex: str
    pdf: bytes | None = None
    page_size: str = "letter"
    expires_at: float = 0

    @property
    def size(self) -> int:
        return len(self.latex.encode("utf-8")) + len(self.pdf or b"")


class ArtifactStore:
    """Per-process storage with absolute TTL, byte and entry limits, and LRU eviction."""

    def __init__(self, *, ttl: int = 3600, max_entries: int = 64,
                 max_bytes: int = 32_000_000):
        self.ttl = max(1, ttl)
        self.max_entries = max(1, max_entries)
        self.max_bytes = max(1, max_bytes)
        self._items: OrderedDict[str, ResumeArtifact] = OrderedDict()
        self._lock = RLock()

    def _purge(self) -> None:
        now = time.monotonic()
        for key in [key for key, item in self._items.items() if item.expires_at <= now]:
            self._items.pop(key, None)

    def get(self, key: str) -> ResumeArtifact | None:
        with self._lock:
            self._purge()
            item = self._items.get(key)
            if item:
                self._items.move_to_end(key)
            return item

    def put(self, key: str, *, latex: str, pdf: bytes | None = None,
            page_size: str = "letter") -> bool:
        with self._lock:
            self._purge()
            self._items.pop(key, None)
            item = ResumeArtifact(latex, pdf, page_size, time.monotonic() + self.ttl)
            if item.size > self.max_bytes:
                return False
            self._items[key] = item
            while (len(self._items) > self.max_entries
                   or sum(value.size for value in self._items.values()) > self.max_bytes):
                self._items.popitem(last=False)
            return True

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)
