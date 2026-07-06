"""Centralised path resolution (CLAUDE.md s5: no hard-coded absolute paths).

Everything is anchored at the repo root, discovered by walking up from this file
until we find the project marker (pyproject.toml). Override with FLINT_DATA_DIR /
FLINT_RAW_DIR env vars for non-standard layouts (e.g. data on a scratch disk).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # fallback: src/flint/utils/paths.py -> repo is 3 levels up from src
    return p.parents[3]


def data_dir() -> Path:
    return Path(os.environ.get("FLINT_DATA_DIR", repo_root() / "data"))


def raw_dir() -> Path:
    return Path(os.environ.get("FLINT_RAW_DIR", data_dir() / "raw"))


def processed_dir() -> Path:
    return data_dir() / "processed"


def cache_dir() -> Path:
    return data_dir() / "cache"


# Known dataset locations under data/raw, resolved by logical name.
_DATASET_SUBPATHS = {
    "250wt": "sm-datasets/sm_datasets/datasets/250wt",
    "hardtables": "zenodo/HardTablesR1",
    "wikidatatables": "zenodo/WikidataTables2023R1",
    "toughtables": "zenodo/2T_WD",
    "tfood": "zenodo/tfood",
}


def dataset_dir(name: str) -> Path:
    """Absolute path to a raw dataset by logical name (see _DATASET_SUBPATHS)."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _DATASET_SUBPATHS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(_DATASET_SUBPATHS)}")
    return raw_dir() / _DATASET_SUBPATHS[key]
