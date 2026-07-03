from __future__ import annotations

import json
import numpy as np

from calc_metrics import _calculate_retrieval_at_k
from ljp_hybrid_retrieval import (
    HybridIndex,
    HybridSearchConfig,
    HybridStringIndex,
    KeywordRecord,
    _remote_rerank_item_ids,
    keyword_scores,
    lexical_meta_to_text,
    normalize_scored_hits,
    prepare_retrieval_query,
    rrf_fuse,
    score_fuse,
)
from ljp_law_query import LawRetrievalQueries
from ljp_tools import (
    TextItem,
    _group_law_document,
    extract_article_id,
    extract_article_key,
    search_index,
)


class FakeEmbedder:
    vocab = ["盗窃", "抢劫", "诈骗", "退赃", "刑法", "自首"]

    def embed_list(self, texts):
        vectors = []
        for text in texts:
            text = str(text)
            vectors.append([float(text.count(token)) for token in self.vocab])
        return vectors


def _vectors(embedder, texts):
    return [np.array(vec, dtype=np.float32) for vec in embedder.embed_list(texts)]


def test_rrf_fuse_scores_and_dedup():
    fused = rrf_fuse([[0, 1], [1, 2], [1]], top_k=3, rrf_k=60)
    assert fused[0][0] == 1
    expected = 1 / 62 + 1 / 61 + 1 / 61
    assert abs(fused[0][1] - expected) < 1e-9
    assert len([item_id for item_id, _ in fused if item_id == 1]) == 1


def test_rrf_weights_keep_dense_first():
    fused = rrf_fuse(
        [[0], [1], [1]],
        top_k=2,
        rrf_k=60,
        weights=[2.0, 0.7, 0.3],
    )
    assert fused[0][0] == 0


def test_score_fusion_keeps_high_confidence_dense_first():
    fused, branches = score_fuse(
        dense_hits=[(0, 0.90), (1, 0.80)],
        bm25_hits=[(1, 10.0), (0, 1.0)],
        keyword_hits=[],
        top_k=2,
        dense_weight=0.65,
        bm25_weight=0.25,
        keyword_weight=0.10,
    )
    assert fused[0][0] == 0
    assert branches["dense"][0]["raw_score"] == 0.90
    assert branches["bm25"][0]["item_id"] == 1


def test_score_normalization_handles_equal_values():
    normalized = normalize_scored_hits([(0, 1.0), (1, 1.0)])
    assert [record["normalized_score"] for record in normalized] == [0.0, 0.0]


def test_technical_metadata_is_not_lexical_text():
    meta = {
        "chunk_id": 2022,
        "article_id": 17,
        "source": "law_article",
        "accusation": ["盗窃"],
    }
    text = lexical_meta_to_text(meta)
    assert "2022" not in text
    assert "17" not in text
    assert "law_article" not in text
    assert "盗窃" in text


def test_group_law_document_by_article():
    text = (
        "目录\n"
        "第一条 总则内容\n"
        "这是第一条的续行\n"
        "第二条 第一款\n"
        "这是第二条的续行\n"
        "第二条之一 特别规定\n"
    )
    articles = _group_law_document(text)
    assert len(articles) == 3
    assert "这是第一条的续行" in articles[0]
    assert extract_article_id(articles[0]) == 1
    assert extract_article_key(articles[2]) == "2-1"


def test_prepare_retrieval_query_keeps_head_and_tail():
    query = "A" * 20 + " middle " + "Z" * 20
    prepared = prepare_retrieval_query(query, max_chars=20)
    assert prepared.startswith("A" * 10)
    assert prepared.endswith("Z" * 10)


def test_rerank_uses_candidate_pool_larger_than_final_top_k():
    import ljp_hybrid_retrieval as module

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {"index": 4, "relevance_score": 1.0},
                        {"index": 3, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.8},
                        {"index": 1, "relevance_score": 0.7},
                        {"index": 0, "relevance_score": 0.6},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    original = module.urllib.request.urlopen
    module.urllib.request.urlopen = fake_urlopen
    try:
        hits, debug = _remote_rerank_item_ids(
            "query",
            list(range(10)),
            [f"doc-{i}" for i in range(10)],
            HybridSearchConfig(
                use_rerank=True,
                rerank_top_k=5,
                rerank_api_key="test-key",
            ),
            top_k=3,
        )
    finally:
        module.urllib.request.urlopen = original

    assert len(captured["payload"]["documents"]) == 5
    assert captured["payload"]["top_n"] == 5
    assert hits == [4, 3, 2]
    assert debug["success"] is True
    assert debug["ranking_complete"] is True
    assert debug["returned_count"] == 5
    assert debug["ranked_results"][0] == {
        "rank": 1,
        "item_id": 4,
        "score": 1.0,
    }


def test_legal_aware_rerank_compresses_law_candidates():
    import ljp_hybrid_retrieval as module

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {"results": [{"index": 0, "relevance_score": 0.9}]}
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    original = module.urllib.request.urlopen
    module.urllib.request.urlopen = fake_urlopen
    try:
        hits, debug = _remote_rerank_item_ids(
            "寻找主要罪名条文",
            [0],
            ["【盗窃罪】刑法第二百六十四条，盗窃公私财物，数额较大的。"],
            HybridSearchConfig(
                use_rerank=True,
                rerank_top_k=1,
                rerank_api_key="test-key",
                legal_aware_rerank=True,
            ),
            top_k=1,
            metas=[{"article_id": 264}],
        )
    finally:
        module.urllib.request.urlopen = original

    document = captured["payload"]["documents"][0]
    assert hits == [0]
    assert debug["legal_aware"] is True
    assert "法条编号：第264条" in document
    assert "标题：盗窃罪" in document
    assert "核心规定：" in document


def test_retrieval_metrics_use_full_rerank_ranking():
    records = [
        {
            "gold": {"law_articles": ["264"]},
            "retrieval": {
                "candidates": [
                    {"meta": {"article_id": 67}},
                    {"meta": {"article_id": 264}},
                    {"meta": {"article_id": 265}},
                ],
                "debug": {
                    "rerank": {
                        "candidate_pool_size": 5,
                        "ranking_complete": True,
                        "ranked_results": [
                            {"rank": 1, "item_id": 0, "score": 0.9, "meta": {"article_id": 67}},
                            {"rank": 2, "item_id": 1, "score": 0.8, "meta": {"article_id": 264}},
                            {"rank": 3, "item_id": 2, "score": 0.7, "meta": {"article_id": 265}},
                            {"rank": 4, "item_id": 3, "score": 0.6, "meta": {"article_id": 266}},
                            {"rank": 5, "item_id": 4, "score": 0.5, "meta": {"article_id": 267}},
                        ],
                    }
                },
            },
        }
    ]
    at_1 = _calculate_retrieval_at_k(records, 1)
    at_3 = _calculate_retrieval_at_k(records, 3)
    at_5 = _calculate_retrieval_at_k(records, 5)
    assert at_1["recall"] == 0.0
    assert at_3["recall"] == 1.0
    assert at_3["mrr"] == 0.5
    assert at_5["recall"] == 1.0
    assert at_5["insufficient"] == 0


def test_retrieval_metrics_prefer_final_fusion_ranking():
    records = [
        {
            "gold": {"law_articles": ["264"]},
            "retrieval": {
                "candidates": [
                    {"meta": {"article_id": 67}},
                    {"meta": {"article_id": 264}},
                ],
                "debug": {
                    "final_scores": [
                        {
                            "item_id": 1,
                            "final_score": 0.9,
                            "meta": {"article_id": 264},
                        },
                        {
                            "item_id": 0,
                            "final_score": 0.8,
                            "meta": {"article_id": 67},
                        },
                    ],
                    "rerank": {
                        "candidate_pool_size": 2,
                        "ranking_complete": True,
                        "ranked_results": [
                            {
                                "rank": 1,
                                "item_id": 0,
                                "score": 0.9,
                                "meta": {"article_id": 67},
                            },
                            {
                                "rank": 2,
                                "item_id": 1,
                                "score": 0.8,
                                "meta": {"article_id": 264},
                            },
                        ],
                    },
                },
            },
        }
    ]
    metrics = _calculate_retrieval_at_k(records, 1)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["sources"] == ["final_fusion"]


def test_retrieval_metrics_use_saved_embedding_ranking_candidates():
    records = [
        {
            "gold": {"law_articles": ["264"]},
            "retrieval": {
                "candidates": [{"meta": {"article_id": 67}}],
                "ranking_candidates": [
                    {"meta": {"article_id": 67}},
                    {"meta": {"article_id": 263}},
                    {"meta": {"article_id": 264}},
                    {"meta": {"article_id": 265}},
                    {"meta": {"article_id": 266}},
                ],
            },
        }
    ]
    metrics = _calculate_retrieval_at_k(records, 5)
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1 / 3
    assert metrics["sources"] == ["ranking_candidates"]


def test_old_top3_result_is_insufficient_for_recall_at_5():
    records = [
        {
            "gold": {"law_articles": ["264"]},
            "retrieval": {
                "candidates": [
                    {"meta": {"article_id": 264}},
                    {"meta": {"article_id": 67}},
                    {"meta": {"article_id": 265}},
                ]
            },
        }
    ]
    metrics = _calculate_retrieval_at_k(records, 5)
    assert metrics["evaluated"] == 0
    assert metrics["insufficient"] == 1


def test_keyword_scores_components():
    records = [
        KeywordRecord(
            item_id=0,
            text="被告人盗窃手机并退赃",
            meta_text="accusation 盗窃 article 264",
            searchable_text="accusation 盗窃 article 264 被告人盗窃手机并退赃",
            tokens={"盗窃", "手机", "退赃"},
        ),
        KeywordRecord(
            item_id=1,
            text="被告人抢劫他人财物",
            meta_text="accusation 抢劫 article 263",
            searchable_text="accusation 抢劫 article 263 被告人抢劫他人财物",
            tokens={"抢劫", "财物"},
        ),
    ]
    scored = keyword_scores("盗窃 退赃 抢劫", records)
    assert scored[0][0] == 0
    assert scored[0][1] > scored[1][1]


def test_hybrid_index_returns_text_items():
    embedder = FakeEmbedder()
    items = [
        TextItem("被告人抢劫他人财物", {"accusation": ["抢劫"]}),
        TextItem("被告人盗窃手机后退赃", {"accusation": ["盗窃"]}),
        TextItem("被告人诈骗钱款", {"accusation": ["诈骗"]}),
    ]
    vectors = _vectors(embedder, [item.text for item in items])
    index = HybridIndex(
        items,
        vectors,
        embedder,
        config=HybridSearchConfig(dense_top_k=3, bm25_top_k=3, keyword_top_k=3, join_top_k=3),
    )
    hits = index.search("盗窃 退赃", 2)
    assert all(isinstance(hit, TextItem) for hit in hits)
    assert hits[0].meta["accusation"] == ["盗窃"]
    assert search_index(index, "盗窃 退赃", 1)[0].text == hits[0].text


def test_hybrid_index_adds_metadata_to_saved_rerank_results():
    import ljp_hybrid_retrieval as module

    embedder = FakeEmbedder()
    items = [
        TextItem("刑法第263条 抢劫罪", {"article_id": 263}),
        TextItem("刑法第264条 盗窃罪", {"article_id": 264}),
    ]
    vectors = _vectors(embedder, [item.text for item in items])
    index = HybridIndex(
        items,
        vectors,
        embedder,
        config=HybridSearchConfig(
            dense_top_k=2,
            bm25_top_k=2,
            keyword_top_k=2,
            join_top_k=2,
            use_rerank=True,
        ),
    )

    def fake_rerank(query, candidate_ids, texts, config, top_k, metas=None):
        item_id = candidate_ids[0]
        return [item_id], {
            "enabled": True,
            "candidate_pool_size": len(candidate_ids),
            "returned_count": 1,
            "ranking_complete": len(candidate_ids) == 1,
            "ranked_results": [
                {"rank": 1, "item_id": item_id, "score": 0.9}
            ],
            "success": True,
            "fallback_reason": None,
        }

    original = module._remote_rerank_item_ids
    module._remote_rerank_item_ids = fake_rerank
    try:
        index.search("盗窃", 1)
    finally:
        module._remote_rerank_item_ids = original

    result = index.last_search_debug["rerank"]["ranked_results"][0]
    assert result["meta"]["article_id"] in {263, 264}


def test_score_fusion_dense_anchor_restores_dense_top1():
    embedder = FakeEmbedder()
    items = [
        TextItem("刑法第264条 盗窃罪", {"article_id": 264}),
        TextItem("刑法第263条 抢劫罪", {"article_id": 263}),
    ]
    vectors = _vectors(embedder, [item.text for item in items])
    index = HybridIndex(
        items,
        vectors,
        embedder,
        config=HybridSearchConfig(
            dense_top_k=2,
            bm25_top_k=2,
            keyword_top_k=0,
            join_top_k=2,
            fusion_mode="score",
            dense_score_weight=0.40,
            bm25_score_weight=0.60,
            keyword_score_weight=0.0,
            dense_anchor=True,
            dense_margin_threshold=0.10,
            dense_override_threshold=0.25,
        ),
    )
    index._core._dense_hits = lambda query: [(0, 0.90), (1, 0.70)]
    index._core._bm25_hits = lambda query: [(1, 10.0), (0, 9.0)]

    hits = index.search("盗窃手机", 2)

    assert hits[0].meta["article_id"] == 264
    assert index.last_search_debug["dense_anchor"]["applied"] is True


def test_hybrid_branches_use_branch_specific_queries():
    embedder = FakeEmbedder()
    items = [
        TextItem("刑法第264条 盗窃罪", {"article_id": 264}),
        TextItem("刑法第263条 抢劫罪", {"article_id": 263}),
    ]
    vectors = _vectors(embedder, [item.text for item in items])
    index = HybridIndex(
        items,
        vectors,
        embedder,
        config=HybridSearchConfig(
            dense_top_k=2,
            bm25_top_k=2,
            keyword_top_k=2,
            join_top_k=2,
        ),
    )
    captured = {}

    def dense_hits(query):
        captured["dense"] = query
        return [(0, 0.9)]

    def bm25_hits(query):
        captured["bm25"] = query
        return [(0, 1.0)]

    def keyword_hits(query):
        captured["keyword"] = query
        return [(0, 1.0)]

    index._core._dense_hits = dense_hits
    index._core._bm25_hits = bm25_hits
    index._core._keyword_hits = keyword_hits
    queries = LawRetrievalQueries(
        dense_query="完整案情",
        lexical_query="盗窃 手机",
        rerank_query="寻找主要罪名条文：盗窃手机",
        circumstance_query="退赃",
    )

    index.search(queries, 1)

    assert captured == {
        "dense": "完整案情",
        "bm25": "盗窃 手机",
        "keyword": "盗窃 手机",
    }
    assert index.last_search_debug["queries"]["rerank"] == queries.rerank_query


def test_legal_rerank_expands_join_pool_to_rerank_top_k():
    import ljp_hybrid_retrieval as module

    embedder = FakeEmbedder()
    items = [
        TextItem(f"刑法第{i + 1}条 盗窃", {"article_id": i + 1})
        for i in range(12)
    ]
    vectors = _vectors(embedder, [item.text for item in items])
    index = HybridIndex(
        items,
        vectors,
        embedder,
        config=HybridSearchConfig(
            dense_top_k=12,
            bm25_top_k=0,
            keyword_top_k=0,
            join_top_k=3,
            use_rerank=True,
            rerank_top_k=8,
            legal_aware_rerank=True,
        ),
    )
    captured = {}

    def fake_rerank(query, candidate_ids, texts, config, top_k, metas=None):
        captured["candidate_count"] = len(candidate_ids)
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
        index.search("盗窃", 3)
    finally:
        module._remote_rerank_item_ids = original

    assert captured["candidate_count"] == 8


def test_hybrid_string_index_returns_strings():
    embedder = FakeEmbedder()
    index = HybridStringIndex(
        embedder,
        ["抢劫财物", "盗窃手机并退赃", "诈骗钱款"],
        config=HybridSearchConfig(dense_top_k=3, bm25_top_k=3, keyword_top_k=3, join_top_k=3),
    )
    hits = index.search("盗窃 退赃", 2)
    assert isinstance(hits[0], str)
    assert hits[0] == "盗窃手机并退赃"


def run_all():
    test_rrf_fuse_scores_and_dedup()
    test_rrf_weights_keep_dense_first()
    test_score_fusion_keeps_high_confidence_dense_first()
    test_score_normalization_handles_equal_values()
    test_technical_metadata_is_not_lexical_text()
    test_group_law_document_by_article()
    test_prepare_retrieval_query_keeps_head_and_tail()
    test_rerank_uses_candidate_pool_larger_than_final_top_k()
    test_legal_aware_rerank_compresses_law_candidates()
    test_retrieval_metrics_use_full_rerank_ranking()
    test_retrieval_metrics_prefer_final_fusion_ranking()
    test_retrieval_metrics_use_saved_embedding_ranking_candidates()
    test_old_top3_result_is_insufficient_for_recall_at_5()
    test_keyword_scores_components()
    test_hybrid_index_returns_text_items()
    test_hybrid_index_adds_metadata_to_saved_rerank_results()
    test_score_fusion_dense_anchor_restores_dense_top1()
    test_hybrid_branches_use_branch_specific_queries()
    test_legal_rerank_expands_join_pool_to_rerank_top_k()
    test_hybrid_string_index_returns_strings()


if __name__ == "__main__":
    run_all()
    print("ok")
