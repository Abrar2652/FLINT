"""Wikidata ontology access (cached, reproducible).

This is the shared substrate for three things:
  - the CTA cscore metric (needs the subclass-of hierarchy)  -> eval/metrics.py
  - the ontology subgraph attached to each table graph        -> graph/builder.py
  - the joint-loss consistency term (property domain/range)   -> models/flint.py

REPRODUCIBILITY (DATA_SPEC.md Part C): we bootstrap from live Wikidata but cache
every entity's trimmed JSON to disk ONCE; all queries are then served from the
frozen cache, so a run is reproducible given the cache. Record the fetch date as
the snapshot. GRAMS+ used the 2023-06-19 dump; the subclass-of hierarchy we rely
on is stable, and our contribution does not hinge on entity-level reproduction of
their pipeline (we use the fallback baseline, CLAUDE.md s4). The cache dir hash
goes into each run dir.

The `KG` Protocol lets tests inject a deterministic in-memory `FakeKG` with no
network (see tests/conftest fixtures), while production uses `CachedWikidataKG`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Protocol, runtime_checkable

from flint.utils.paths import cache_dir

_ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
_P_INSTANCE_OF = "P31"
_P_SUBCLASS_OF = "P279"


class _FetchError(Exception):
    """Transient fetch failure (network / rate-limit). Must NOT be cached."""


@runtime_checkable
class KG(Protocol):
    """Minimal Wikidata ontology interface consumed across the project."""

    def instance_of(self, entity_id: str) -> list[str]:
        """Class ids (P31 targets) of an entity."""

    def subclass_of(self, class_id: str) -> list[str]:
        """Direct superclass ids (P279 targets) of a class."""

    def statements(self, entity_id: str) -> list[tuple[str, str]]:
        """(property_id, object_entity_id) statements with entity values."""

    def literal_statements(self, entity_id: str) -> list[tuple[str, str, str]]:
        """(property_id, value_str, kind) statements with literal values."""

    def label(self, qid: str) -> str:
        """English label, or the qid itself if unavailable."""


# --------------------------------------------------------------------------- #
# Hierarchy helpers (work against any KG implementation)
# --------------------------------------------------------------------------- #
def ancestors_with_hops(kg: KG, class_id: str, *, max_hops: int | None = None) -> dict[str, int]:
    """BFS up subclass-of from `class_id`. Returns {ancestor_id: min_hop}.

    hop 0 is `class_id` itself. `max_hops=None` walks the full closure. Cycles in
    Wikidata's P279 graph (they exist) are handled by the visited set.
    """
    dist: dict[str, int] = {class_id: 0}
    q: deque[str] = deque([class_id])
    while q:
        cur = q.popleft()
        h = dist[cur]
        if max_hops is not None and h >= max_hops:
            continue
        for parent in kg.subclass_of(cur):
            if parent not in dist:
                dist[parent] = h + 1
                q.append(parent)
    return dist


def hierarchy_distance(kg: KG, a: str, b: str, *, max_hops: int = 10) -> int | None:
    """Shortest |hops| between a and b along subclass-of, if one is the other's
    ancestor (the only relation cscore credits). None if unrelated within max_hops.
    """
    if a == b:
        return 0
    up_a = ancestors_with_hops(kg, a, max_hops=max_hops)
    if b in up_a:
        return up_a[b]  # a is a descendant of b (b is ancestor of a)
    up_b = ancestors_with_hops(kg, b, max_hops=max_hops)
    if a in up_b:
        return up_b[a]
    return None


def is_ancestor(kg: KG, a: str, b: str, *, max_hops: int = 10) -> bool:
    """True iff `a` is an ancestor of `b` (b subclass-of* a)."""
    if a == b:
        return False
    return a in ancestors_with_hops(kg, b, max_hops=max_hops)


# --------------------------------------------------------------------------- #
# Cached live-Wikidata implementation
# --------------------------------------------------------------------------- #
class CachedWikidataKG:
    """Fetches each entity's JSON once from Special:EntityData, caches the trimmed
    claims to disk, and serves all queries from cache.

    Args:
        cache_path: directory for per-entity json (default: data/cache/wikidata).
        offline: if True, never hit the network; missing entities return empty
            (use after a cache has been warmed, for fully reproducible runs).
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        offline: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.cache_path = cache_path or (cache_dir() / "wikidata")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout
        self._mem: dict[str, dict] = {}

    TRIM_VERSION = 2  # bump when _trim's schema changes (triggers lazy re-fetch)

    def _trim(self, entity: dict) -> dict:
        """Keep P31, P279, entity-valued statements, LITERAL-valued statements, label.

        Literal statements (quantity/time/string/monolingualtext) feed the
        entity->literal CPA edge: an entity's statement VALUE matched against a
        literal cell's value (e.g. P1351 number-of-points = the Points cell)."""
        claims = entity.get("claims", {})

        def _targets(pid: str) -> list[str]:
            out = []
            for st in claims.get(pid, []):
                dv = st.get("mainsnak", {}).get("datavalue", {})
                if dv.get("type") == "wikibase-entityid":
                    out.append(dv["value"]["id"])
            return out

        stmts: list[list[str]] = []
        literals: list[list[str]] = []  # [pid, normalized_value, kind]
        for pid, sts in claims.items():
            if pid in (_P_INSTANCE_OF, _P_SUBCLASS_OF):
                continue
            for st in sts:
                dv = st.get("mainsnak", {}).get("datavalue", {})
                typ = dv.get("type")
                if typ == "wikibase-entityid":
                    stmts.append([pid, dv["value"]["id"]])
                elif typ == "quantity":
                    amt = str(dv["value"].get("amount", "")).lstrip("+")
                    if amt:
                        literals.append([pid, amt, "quantity"])
                elif typ == "time":
                    t = str(dv["value"].get("time", ""))  # e.g. +1990-00-00T00:00:00Z
                    if t:
                        literals.append([pid, t, "time"])
                elif typ in ("string", "monolingualtext"):
                    v = dv["value"] if typ == "string" else dv["value"].get("text", "")
                    if v:
                        literals.append([pid, str(v), "string"])
        labels = entity.get("labels", {})
        return {
            "v": self.TRIM_VERSION,
            "P31": _targets(_P_INSTANCE_OF),
            "P279": _targets(_P_SUBCLASS_OF),
            "statements": stmts,
            "literals": literals,
            "label": labels.get("en", {}).get("value", entity.get("id", "")),
        }

    _EMPTY = {"P31": [], "P279": [], "statements": [], "label": None}

    def _get(self, qid: str) -> dict:
        if qid in self._mem:
            return self._mem[qid]
        fp = self.cache_path / f"{qid}.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            # lazily upgrade stale-schema cache (e.g. missing literals from v1)
            if data.get("v", 1) >= self.TRIM_VERSION or self.offline:
                self._mem[qid] = data
                return data
        if self.offline:
            return {"P31": [], "P279": [], "statements": [], "literals": [], "label": qid}
        try:
            data = self._fetch(qid)
        except _FetchError:
            # transient failure: DO NOT cache (so it is retried later); return empty.
            return {"P31": [], "P279": [], "statements": [], "label": qid}
        fp.write_text(json.dumps(data))
        self._mem[qid] = data
        return data

    def _fetch(self, qid: str, *, retries: int = 4) -> dict:
        """Fetch + trim one entity. Raises _FetchError on transient failure (so the
        caller does NOT poison the cache). A 200 response with a present-but-empty
        entity is a legitimate empty result and is returned (and cached)."""
        url = _ENTITY_DATA_URL.format(qid=qid)
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(url, headers={"User-Agent": "ocgnn-research/0.1 (research)"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                entity = payload.get("entities", {}).get(qid)
                if entity is None:  # redirect/missing-id: treat as legitimately empty
                    return {"P31": [], "P279": [], "statements": [], "label": qid}
                return self._trim(entity)
            except urllib.error.HTTPError as e:  # noqa: PERF203
                last = e
                if e.code in (429, 500, 502, 503, 504):  # rate-limit / transient
                    self._backoff(attempt)
                    continue
                raise _FetchError(str(e)) from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                self._backoff(attempt)
                continue
        raise _FetchError(f"{qid}: {last}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        import time

        time.sleep(min(2.0 ** attempt, 8.0))

    # --- KG protocol -------------------------------------------------------- #
    def instance_of(self, entity_id: str) -> list[str]:
        return list(self._get(entity_id)["P31"])

    def subclass_of(self, class_id: str) -> list[str]:
        return list(self._get(class_id)["P279"])

    def statements(self, entity_id: str) -> list[tuple[str, str]]:
        return [(p, o) for p, o in self._get(entity_id)["statements"]]

    def literal_statements(self, entity_id: str) -> list[tuple[str, str, str]]:
        """(property_id, value_str, kind) for literal-valued statements (quantity/
        time/string). Empty for v1 (pre-literals) cache entries."""
        return [(p, v, k) for p, v, k in self._get(entity_id).get("literals", [])]

    def label(self, qid: str) -> str:
        return self._get(qid)["label"]

    def _stale(self, qid: str) -> bool:
        fp = self.cache_path / f"{qid}.json"
        if not fp.exists():
            return True
        try:
            return json.loads(fp.read_text()).get("v", 1) < self.TRIM_VERSION
        except Exception:  # noqa: BLE001
            return True

    def warm(self, qids: list[str]) -> int:
        """Pre-fetch a batch of entities into the cache; return #newly fetched."""
        n = 0
        for qid in qids:
            if self._stale(qid):
                self._mem.pop(qid, None)
                self._get(qid)
                n += 1
        return n

    def warm_concurrent(self, qids: list[str], *, workers: int = 16) -> int:
        """Concurrently (re-)fetch MISSING or STALE-schema entities. Returns #fetched.

        Polite to Wikidata (bounded workers); cache writes are per-file so safe.
        """
        from concurrent.futures import ThreadPoolExecutor

        todo = [q for q in dict.fromkeys(qids) if self._stale(q)]
        if not todo:
            return 0

        def _one(qid: str) -> bool:
            try:
                data = self._fetch(qid)
            except _FetchError:
                return False  # leave uncached so a later pass retries it
            (self.cache_path / f"{qid}.json").write_text(json.dumps(data))
            return True

        with ThreadPoolExecutor(max_workers=workers) as ex:
            ok = sum(ex.map(_one, todo))
        return ok


# --------------------------------------------------------------------------- #
# Deterministic in-memory KG for tests (no network)
# --------------------------------------------------------------------------- #
class FakeKG:
    """In-memory KG built from plain dicts; used by unit tests and fixtures."""

    def __init__(
        self,
        *,
        instance_of: dict[str, list[str]] | None = None,
        subclass_of: dict[str, list[str]] | None = None,
        statements: dict[str, list[tuple[str, str]]] | None = None,
        literals: dict[str, list[tuple[str, str, str]]] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._iof = instance_of or {}
        self._sub = subclass_of or {}
        self._st = statements or {}
        self._lit = literals or {}
        self._lab = labels or {}

    def instance_of(self, entity_id: str) -> list[str]:
        return list(self._iof.get(entity_id, []))

    def subclass_of(self, class_id: str) -> list[str]:
        return list(self._sub.get(class_id, []))

    def statements(self, entity_id: str) -> list[tuple[str, str]]:
        return list(self._st.get(entity_id, []))

    def literal_statements(self, entity_id: str) -> list[tuple[str, str, str]]:
        return list(self._lit.get(entity_id, []))

    def label(self, qid: str) -> str:
        return self._lab.get(qid, qid)


# --------------------------------------------------------------------------- #
# Literal value matching (entity statement value <-> literal cell value)
# --------------------------------------------------------------------------- #
def _norm_num(s: str) -> float | None:
    s = s.strip().replace(",", "").lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None


def literal_matches(cell: str, value: str, kind: str) -> bool:
    """True if a literal cell value matches an entity statement value.

    quantity: numeric equality (tolerant); time: shared 4-digit year (cells are
    often just years); string: case-insensitive containment either way. Kept
    deliberately simple + deterministic (GRAMS+ does analogous value matching)."""
    cell = (cell or "").strip()
    if not cell or not value:
        return False
    if kind == "quantity":
        a, b = _norm_num(cell), _norm_num(value)
        return a is not None and b is not None and abs(a - b) < 1e-6
    if kind == "time":
        import re
        cy = re.findall(r"\d{4}", cell)
        vy = re.findall(r"\d{4}", value)
        return bool(cy and vy and cy[0] == vy[0])
    # string / monolingualtext
    c, v = cell.lower(), value.lower()
    return c == v or (len(c) >= 3 and (c in v or v in c))
