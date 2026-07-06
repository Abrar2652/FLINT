"""Feature encoders for graph node initial features (SCHEMA.md "initial features").

Two implementations behind one tiny interface:
  - HashingEncoder: deterministic, dependency-free, fixed-dim. Used by unit tests
    and any time we only care about graph STRUCTURE (Gate 2), not learned features.
  - SentenceTransformerEncoder: the real text encoder (lazy import), used for runs.

Both expose `.dim` and `.encode(list[str]) -> Tensor[len, dim]`.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import torch


class TextEncoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> torch.Tensor: ...


class HashingEncoder:
    """Deterministic hashing-based text embedding (no learning, no network).

    Maps each token to buckets via blake2b and L2-normalises. Stable across runs
    and processes, so permutation/structure tests are exactly reproducible.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim)
        for tok in (text or "").lower().split():
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        n = v.norm()
        return v / n if n > 0 else v

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dim))
        return torch.stack([self._vec(t) for t in texts])


class SentenceTransformerEncoder:
    """Real Sentence-Transformer encoder (matches GRAMS+'s use of SBERT).

    Lazy: the model loads on first encode so importing this module is cheap.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 384  # all-MiniLM-L6-v2

    def _ensure(self) -> None:
        if self._model is None:
            self._fix_hf_home()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()

    @staticmethod
    def _fix_hf_home() -> None:
        """Default HF_HOME to a writable repo cache if unset or the user default
        (~/.cache/huggingface) is missing/broken. Avoids the broken-symlink crash
        and keeps the model cache reproducible under data/cache/."""
        import os

        if os.environ.get("HF_HOME"):
            return
        default = os.path.expanduser("~/.cache/huggingface")
        if os.path.isdir(default) and os.path.exists(default):
            return
        from flint.utils.paths import cache_dir

        hf = cache_dir() / "hf"
        hf.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dim))
        self._ensure()
        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return torch.from_numpy(emb).float()
