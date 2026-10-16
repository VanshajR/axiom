"""The site's published data (portfolio data/*.yaml -> profile.json), cached.

Axiom only knows what that file says, so the site and the assistant can't drift.
"""
import json
import os
import time

import httpx

TTL_S = 600


class ProfileStore:
    def __init__(self, url: str, http: httpx.AsyncClient | None = None, path: str | None = None):
        self.url = url
        self.path = path or os.getenv("PROFILE_PATH")  # a local file, for tests and offline dev
        # shipped with the image: used if the site is unreachable before the first good fetch
        self.snapshot = os.getenv("PROFILE_SNAPSHOT", "profile.snapshot.json")
        self.http = http
        self._data: dict | None = None
        self._loaded_at = 0.0

    async def get(self) -> dict:
        if self._data is not None and time.time() - self._loaded_at < TTL_S:
            return self._data
        try:
            if self.path:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                r = await self.http.get(self.url, timeout=15)
                r.raise_for_status()
                data = r.json()
            self._data, self._loaded_at = data, time.time()
        except Exception:
            # keep serving the last good copy if the site is briefly unreachable,
            # and fall back to the bundled snapshot if there's never been one
            if self._data is None:
                if not os.path.exists(self.snapshot):
                    raise
                with open(self.snapshot, encoding="utf-8") as f:
                    self._data = json.load(f)
                # retry the site sooner than the normal TTL
                self._loaded_at = time.time() - TTL_S + 60
        return self._data

    def set(self, data: dict) -> None:
        self._data, self._loaded_at = data, time.time()
