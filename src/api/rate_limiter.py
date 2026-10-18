import math
import threading
import time

from collections import deque

class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, key: str, now: float | None = None) -> tuple[bool,int]:
        now = time.monotonic() if now is None else now
        cutoff = now - self.window
        with self._lock:
            self._calls += 1
            if self._calls % 1000 == 0:
                self._purge(cutoff)

            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                retry_after = math.ceil(hits[0] + self.window - now)
                return False, max(retry_after, 1)

            hits.append(now)
            return True, 0

    def _purge(self, cutoff: float) -> None:
        stale = [k for k,d in self._hits.items() if not d or d[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
