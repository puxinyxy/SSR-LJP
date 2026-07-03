"""
Lightweight in-memory vector index for text retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from ljp_hybrid_retrieval import HybridSearchConfig, HybridStringIndex


class SimpleVectorIndex:
    def __init__(self, emb_model, texts: List[str]):
        self.emb_model = emb_model
        self.texts = list(texts)
        self._vectors = self._embed_texts(self.texts)

    def _embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 10,
        max_chars: int = 8000,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            clipped = [
                (t[:max_chars] if isinstance(t, str) and len(t) > max_chars else t)
                for t in batch
            ]
            embeddings = self.emb_model.embed_list(list(clipped))
            for emb in embeddings:
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
        return np.vstack(vectors)

    def search(self, query: str, top_k: int) -> List[str]:
        if not self.texts or top_k <= 0:
            return []
        query_text = query if isinstance(query, str) else str(query)
        q_vecs = self._embed_texts([query_text])
        if q_vecs.size == 0:
            return []
        q_vec = q_vecs[0]
        scores = self._vectors @ q_vec
        top_k = min(top_k, len(self.texts))
        if top_k == 1:
            best_idx = int(np.argmax(scores))
            return [self.texts[best_idx]]
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.texts[i] for i in idx]


class SimpleHybridIndex:
    def __init__(
        self,
        emb_model,
        texts: List[str],
        dense_top_k: int = 50,
        bm25_top_k: int = 50,
        keyword_top_k: int = 30,
        join_top_k: int = 50,
        rrf_k: float = 60.0,
        dense_weight: float = 2.0,
        bm25_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fusion_mode: str = "rrf",
        dense_score_weight: float = 0.65,
        bm25_score_weight: float = 0.25,
        keyword_score_weight: float = 0.10,
        rerank_score_weight: float = 0.20,
        dense_anchor: bool = False,
        dense_margin_threshold: float = 0.02,
        dense_override_threshold: float = 0.08,
        legal_aware_rerank: bool = False,
        lexical_include_numeric: bool = False,
        query_max_chars: int = 2000,
        use_rerank: bool = False,
        rerank_top_k: int = 30,
        rerank_model: str = "qwen3-rerank",
        rerank_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        rerank_api_key: str | None = None,
        rerank_timeout: int = 60,
        batch_size: int = 10,
        cache_dir: Path | None = None,
        cache_prefix: str = "simple_hybrid",
        cache_meta: dict[str, Any] | None = None,
    ):
        self.emb_model = emb_model
        self.texts = list(texts)
        self._impl = HybridStringIndex(
            emb_model,
            self.texts,
            config=HybridSearchConfig(
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                keyword_top_k=keyword_top_k,
                join_top_k=join_top_k,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
                keyword_weight=keyword_weight,
                fusion_mode=fusion_mode,
                dense_score_weight=dense_score_weight,
                bm25_score_weight=bm25_score_weight,
                keyword_score_weight=keyword_score_weight,
                rerank_score_weight=rerank_score_weight,
                dense_anchor=dense_anchor,
                dense_margin_threshold=dense_margin_threshold,
                dense_override_threshold=dense_override_threshold,
                legal_aware_rerank=legal_aware_rerank,
                lexical_include_numeric=lexical_include_numeric,
                query_max_chars=query_max_chars,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k,
                rerank_model=rerank_model,
                rerank_url=rerank_url,
                rerank_api_key=rerank_api_key,
                rerank_timeout=rerank_timeout,
            ),
            batch_size=batch_size,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_meta=cache_meta,
        )

    def search(self, query: str, top_k: int) -> List[str]:
        return self._impl.search(query, top_k)

    @property
    def last_search_debug(self) -> dict[str, Any]:
        return self._impl.last_search_debug
