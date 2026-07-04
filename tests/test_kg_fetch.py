"""Regression test: transient fetch failures must NEVER be cached (the cache-
poisoning bug of 2026-06-05). A failed fetch returns empty but leaves no file,
so a later pass retries it; a successful (even empty) fetch is cached.
"""
from __future__ import annotations

import json

from flint.data.kg import CachedWikidataKG, _FetchError


def test_transient_failure_is_not_cached(tmp_path):
    kg = CachedWikidataKG(cache_path=tmp_path)

    calls = {"n": 0}

    def boom(qid, **kw):  # noqa: ANN001
        calls["n"] += 1
        raise _FetchError("simulated rate limit")

    kg._fetch = boom  # type: ignore[method-assign]
    assert kg.instance_of("Q999999") == []          # returns empty on failure
    assert not (tmp_path / "Q999999.json").exists()  # but NOTHING is cached
    kg._mem.clear()
    assert kg.instance_of("Q999999") == []          # so a retry actually re-fetches
    assert calls["n"] == 2                            # not served from a poisoned cache


def test_successful_fetch_is_cached(tmp_path):
    kg = CachedWikidataKG(cache_path=tmp_path)
    payload = {"P31": ["Q5"], "P279": [], "statements": [("P54", "Qx")], "label": "thing"}
    kg._fetch = lambda qid, **kw: dict(payload)  # type: ignore[method-assign]
    assert kg.instance_of("Q42") == ["Q5"]
    assert (tmp_path / "Q42.json").exists()
    # served from disk on a fresh instance (no network)
    kg2 = CachedWikidataKG(cache_path=tmp_path, offline=True)
    assert kg2.instance_of("Q42") == ["Q5"]
    assert kg2.statements("Q42") == [("P54", "Qx")]


def test_legitimately_empty_entity_is_cached(tmp_path):
    """A real entity with no P31 (200 response, empty claims) is a valid empty, cached."""
    kg = CachedWikidataKG(cache_path=tmp_path)
    kg._fetch = lambda qid, **kw: {"P31": [], "P279": [], "statements": [], "label": "x"}  # type: ignore[method-assign]
    assert kg.instance_of("Q1") == []
    assert (tmp_path / "Q1.json").exists()  # cached (label != qid distinguishes from failure)
    blob = json.loads((tmp_path / "Q1.json").read_text())
    assert blob["label"] == "x"
