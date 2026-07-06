"""Real candidate generation (DATA_SPEC Part A) — replaces gold-entity seeding so
comparisons to GRAMS+ are on a legitimate (entity-linking-included) protocol.

For each cell mention we retrieve candidate Wikidata entities via wbsearchentities
(label + alias search), cached per normalized mention (reproducible, polite). This
is intentionally the boring GRAMS+-style retrieval; CEA is not our contribution.
Cache: data/cache/candidates/<hash>.json -> list[{id,label}].
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from flint.utils.paths import cache_dir

_API = "https://www.wikidata.org/w/api.php"


class _FetchError(Exception):
    pass


class CachedCandidateGenerator:
    """mention -> [entity_id, ...] via wbsearchentities, cached on disk.

    offline=True serves only the cache (returns [] for misses) for reproducible
    runs after warming. k is the retrieval cap (sweepable; DATA_SPEC uses 100).
    """

    def __init__(self, cache_path: Path | None = None, *, offline: bool = False,
                 k: int = 50, timeout: float = 30.0) -> None:
        self.cache_path = cache_path or (cache_dir() / "candidates")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.k = k
        self.timeout = timeout
        self._mem: dict[str, list[dict]] = {}

    @staticmethod
    def _norm(mention: str) -> str:
        return " ".join((mention or "").strip().lower().split())

    def _file(self, mention: str) -> Path:
        h = hashlib.blake2b(self._norm(mention).encode(), digest_size=12).hexdigest()
        return self.cache_path / f"{h}.json"

    def _raw(self, mention: str) -> list[dict]:
        key = self._norm(mention)
        if not key:
            return []
        if key in self._mem:
            return self._mem[key]
        fp = self._file(mention)
        if fp.exists():
            data = json.loads(fp.read_text())
            self._mem[key] = data
            return data
        if self.offline:
            return []
        try:
            data = self._fetch(key)
        except _FetchError:
            return []  # transient: do not cache (retry later)
        fp.write_text(json.dumps(data))
        self._mem[key] = data
        return data

    def _fetch(self, mention: str, *, retries: int = 4) -> list[dict]:
        params = {
            "action": "wbsearchentities", "search": mention, "language": "en",
            "uselang": "en", "type": "item", "limit": str(min(self.k, 50)), "format": "json",
        }
        url = f"{_API}?{urllib.parse.urlencode(params)}"
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(url, headers={"User-Agent": "ocgnn-research/0.1 (research)"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return [{"id": s["id"], "label": s.get("label", "")}
                        for s in payload.get("search", []) if s.get("id", "").startswith("Q")]
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    self._backoff(attempt)
                    continue
                raise _FetchError(str(e)) from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                self._backoff(attempt)
        raise _FetchError(f"{mention}: {last}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        import time
        time.sleep(min(2.0 ** attempt, 8.0))

    def candidates(self, mention: str, k: int | None = None) -> list[str]:
        """Top-k candidate entity ids for a mention (cached)."""
        k = k or self.k
        return [c["id"] for c in self._raw(mention)[:k]]

    def warm_concurrent(self, mentions: list[str], *, workers: int = 12) -> int:
        from concurrent.futures import ThreadPoolExecutor
        todo = [m for m in dict.fromkeys(self._norm(x) for x in mentions if self._norm(x))
                if not self._file(m).exists()]
        if not todo:
            return 0

        def _one(m: str) -> bool:
            try:
                data = self._fetch(m)
            except _FetchError:
                return False
            self._file(m).write_text(json.dumps(data))
            return True

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return sum(ex.map(_one, todo))
