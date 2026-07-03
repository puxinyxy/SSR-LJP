from __future__ import annotations

from eval_llm_rag_law import _parse_prediction, _prepare_retrieval_input
from ljp_law_query import LawRetrievalQueries, prepare_law_retrieval_queries


def test_law_query_separates_evidence_and_sentencing_circumstances():
    fact = (
        "经审理查明，被告人秘密窃取一部手机，价值5000元。"
        "上述事实有证人证言、鉴定意见等证据证实。"
        "案发后被告人退赃并取得被害人谅解。"
    )

    queries = prepare_law_retrieval_queries(fact, max_chars=2000)

    assert "秘密窃取" in queries.lexical_query
    assert "手机" in queries.lexical_query
    assert "5000元" in queries.lexical_query
    assert "证人证言" not in queries.lexical_query
    assert "退赃" not in queries.lexical_query
    assert "退赃" in queries.circumstance_query
    assert "证人证言" in queries.dense_query
    assert "寻找最直接规定" in queries.rerank_query


def test_law_query_matches_markers_with_cjo22_style_spaces():
    fact = (
        "经 审理 查明 ： 被告人 盗窃 他人 手机 ， 价值 5000 元 。"
        "案发 后 如实 供述 并 退赃 。"
        "上述 事实 有 证人 证言 和 鉴定 意见 证实 。"
    )

    queries = prepare_law_retrieval_queries(fact)

    assert "经 审理 查明" not in queries.lexical_query
    assert "盗窃 他人 手机" in queries.lexical_query
    assert "如实 供述" not in queries.lexical_query
    assert "如实 供述" in queries.circumstance_query
    assert "证人 证言" not in queries.lexical_query


def test_law_query_falls_back_to_dense_when_no_offense_clause_survives():
    fact = "上述事实有证人证言证实。"
    queries = prepare_law_retrieval_queries(fact)
    assert queries.lexical_query == queries.dense_query


def test_law_query_preserves_dense_head_and_tail():
    fact = "A" * 20 + "中间内容" + "Z" * 20
    queries = prepare_law_retrieval_queries(fact, max_chars=20)
    assert queries.dense_query.startswith("A" * 10)
    assert queries.dense_query.endswith("Z" * 10)


def test_prediction_parser_defaults_to_one_law_article():
    prediction = _parse_prediction(
        '{"law_articles": ["刑法第264条", "刑法第67条"]}'
    )
    assert prediction["law_articles"] == ["刑法第264条"]


def test_prediction_parser_can_keep_multiple_for_future_experiments():
    prediction = _parse_prediction(
        '{"law_articles": ["刑法第264条", "刑法第67条"]}',
        max_output_articles=0,
    )
    assert prediction["law_articles"] == ["刑法第264条", "刑法第67条"]


def test_retrieval_input_can_disable_law_query_rewrite():
    fact = "经审理查明，被告人盗窃手机。案发后退赃。"
    queries, retrieval_input = _prepare_retrieval_input(
        fact,
        max_chars=2000,
        retrieval_mode="hybrid",
        law_query_rewrite=False,
    )
    assert isinstance(retrieval_input, str)
    assert retrieval_input == queries.dense_query


def test_retrieval_input_uses_structured_queries_when_enabled():
    fact = "经审理查明，被告人盗窃手机。案发后退赃。"
    queries, retrieval_input = _prepare_retrieval_input(
        fact,
        max_chars=2000,
        retrieval_mode="hybrid",
        law_query_rewrite=True,
    )
    assert isinstance(retrieval_input, LawRetrievalQueries)
    assert retrieval_input.lexical_query == queries.lexical_query
    assert "退赃" not in retrieval_input.lexical_query


def run_all():
    test_law_query_separates_evidence_and_sentencing_circumstances()
    test_law_query_matches_markers_with_cjo22_style_spaces()
    test_law_query_falls_back_to_dense_when_no_offense_clause_survives()
    test_law_query_preserves_dense_head_and_tail()
    test_prediction_parser_defaults_to_one_law_article()
    test_prediction_parser_can_keep_multiple_for_future_experiments()
    test_retrieval_input_can_disable_law_query_rewrite()
    test_retrieval_input_uses_structured_queries_when_enabled()


if __name__ == "__main__":
    run_all()
    print("ok")
