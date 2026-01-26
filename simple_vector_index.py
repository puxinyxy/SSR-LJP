"""
Lightweight in-memory vector index for text retrieval.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


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
