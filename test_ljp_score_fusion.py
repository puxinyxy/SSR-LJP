from __future__ import annotations

import numpy as np

from ljp_hybrid_retrieval import HybridIndex, HybridSearchConfig
from ljp_law_query import LawRetrievalQueries
from ljp_tools import TextItem


class FakeEmbedder:
    vocab = ("盗窃", "抢劫", "手机", "财物", "诈骗", "贷款")

    def embed_list(self, texts):
        return [
            [float(str(text).count(token)) for token in self.vocab]
            for text in texts
        ]


def _vectors(embedder, texts):
    return [
        np.array(vector, dtype=np.float32)
        for vector in embedder.embed_list(texts)
    ]


def _build_index(config):
    embedder = FakeEmbedder()
    items = [
        TextItem("盗窃 手机", {"article_id": 264}),
        TextItem("盗窃 财物", {"article_id": 265}),
        TextItem("抢劫 手机", {"article_id": 263}),
        TextItem("诈骗 贷款", {"article_id": 193}),
        TextItem("故意 伤害", {"article_id": 234}),
    ]
    return HybridIndex(
        items,
        _vectors(embedder, [item.text for item in items]),
        embedder,
        config=config,
    )


def test_dense_bm25_keyword_raw_scores_are_descending():
    index = _build_index(
        HybridSearchConfig(
            dense_top_k=5,
            bm25_top_k=5,
            keyword_top_k=5,
            join_top_k=5,
        )
    )

    dense = index._core._dense_hits("盗窃 手机")
    bm25 = index._core._bm25_hits("盗窃 手机")
    keyword = index._core._keyword_hits("盗窃 手机")

    assert all(dense[i][1] >= dense[i + 1][1] for i in range(len(dense) - 1))
    assert all(bm25[i][1] >= bm25[i + 1][1] for i in range(len(bm25) - 1))
    assert all(
        keyword[i][1] >= keyword[i + 1][1]
        for i in range(len(keyword) - 1)
    )
    assert dense[0][0] == 0
    assert bm25[0][0] == 0
    assert keyword[0][0] == 0


def test_dense_anchor_allows_override_when_margin_is_low():
    index = _build_index(
        HybridSearchConfig(
            dense_top_k=2,
            bm25_top_k=2,
            keyword_top_k=0,
            join_top_k=2,
            fusion_mode="score",
            dense_score_weight=0.40,
            bm25_score_weight=0.60,
            keyword_score_weight=0.0,
            dense_anchor=True,
            dense_margin_threshold=0.02,
            dense_override_threshold=0.25,
        )
    )
    index._core._dense_hits = lambda query: [(0, 0.90), (1, 0.89)]
    index._core._bm25_hits = lambda query: [(1, 10.0), (0, 9.0)]

    hits = index.search("盗窃", 2)

    assert hits[0].meta["article_id"] == 265
    assert index.last_search_debug["dense_anchor"]["applied"] is False


def test_reranker_receives_legal_rerank_query_and_final_top_k():
    import ljp_hybrid_retrieval as module

    index = _build_index(
        HybridSearchConfig(
            dense_top_k=5,
            bm25_top_k=5,
            keyword_top_k=5,
            join_top_k=5,
            use_rerank=True,
            rerank_top_k=5,
            legal_aware_rerank=True,
        )
    )
    captured = {}

    def fake_rerank(query, candidate_ids, texts, config, top_k, metas=None):
        captured["query"] = query
        captured["top_k"] = top_k
        return list(candidate_ids[:top_k]), {
            "enabled": True,
            "candidate_pool_size": len(candidate_ids),
            "returned_count": 0,
            "ranking_complete": False,
            "ranked_results": [],
            "success": False,
            "fallback_reason": "offline_test",
        }

    original = module._remote_rerank_item_ids
    module._remote_rerank_item_ids = fake_rerank
    try:
        queries = LawRetrievalQueries(
            dense_query="完整事实",
            lexical_query="盗窃 手机",
            rerank_query="法律任务：选择主要定罪法条。盗窃 手机",
            circumstance_query="退赃",
        )
        hits = index.search(queries, 3)
    finally:
        module._remote_rerank_item_ids = original

    assert captured["query"] == queries.rerank_query
    assert captured["top_k"] == 3
    assert len(hits) == 3


def run_all():
    test_dense_bm25_keyword_raw_scores_are_descending()
    test_dense_anchor_allows_override_when_margin_is_low()
    test_reranker_receives_legal_rerank_query_and_final_top_k()


if __name__ == "__main__":
    run_all()
    print("ok")
