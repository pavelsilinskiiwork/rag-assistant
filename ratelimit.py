"""Abuse protection for the public demo: a per-IP hourly limit plus a global daily cap.

In-memory and single-process, which is enough for a one-instance demo. It is only
active when DEMO_MODE=1, so local development and the test suite are unaffected.
"""

import os
import threading
import time
from collections import defaultdict, deque

HOUR = 3600
DAY = 86400


class RateLimiter:
    def __init__(self, per_ip_per_hour: int = 10, daily_total: int = 150,
                 enabled: bool = True, clock=time.time):
        self.per_ip = per_ip_per_hour
        self.daily_total = daily_total
        self.enabled = enabled
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._day = int(clock() // DAY)
        self._daily = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RateLimiter":
        return cls(
            per_ip_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "10")),
            daily_total=int(os.getenv("DAILY_LIMIT", "150")),
            enabled=os.getenv("DEMO_MODE") == "1",
        )

    def check(self, ip: str) -> tuple[str, int] | None:
        """Record one request. Returns None if allowed, else (reason, retry_after_seconds)."""
        if not self.enabled:
            return None
        with self._lock:
            now = self.clock()
            day = int(now // DAY)
            if day != self._day:  # new UTC day: reset the global counter
                self._day, self._daily = day, 0
            if self._daily >= self.daily_total:
                return "daily", int((day + 1) * DAY - now)
            hits = self._hits[ip]
            while hits and hits[0] <= now - HOUR:
                hits.popleft()
            if len(hits) >= self.per_ip:
                return "ip", int(hits[0] + HOUR - now) + 1
            hits.append(now)
            self._daily += 1
            if len(self._hits) > 10_000:  # keep memory bounded
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return None
