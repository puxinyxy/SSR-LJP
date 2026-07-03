from __future__ import annotations

from calc_metrics import (
    _calculate_penalty_retrieval_at_k,
    compare_law_results,
    extract_pred_pt_cls,
)


def _record(case_id, prediction, gold):
    return {
        "case_id": case_id,
        "prediction": {"law_articles": prediction},
        "gold": {"law_articles": gold},
    }


def test_compare_law_results_is_paired_by_case_id():
    baseline = [
        _record(1, ["刑法第264条"], ["264"]),
        _record(2, ["刑法第263条"], ["264"]),
        _record(3, ["刑法第234条"], ["234"]),
        _record(4, ["刑法第264条"], ["263"]),
    ]
    candidate = [
        _record(4, ["刑法第263条"], ["263"]),
        _record(3, ["刑法第264条"], ["234"]),
        _record(2, ["刑法第264条"], ["264"]),
        _record(1, ["刑法第264条"], ["264"]),
    ]

    result = compare_law_results(
        baseline,
        candidate,
        bootstrap_samples=200,
        seed=42,
    )

    assert result["paired_samples"] == 4
    assert result["baseline_accuracy"] == 0.5
    assert result["candidate_accuracy"] == 0.75
    assert result["accuracy_difference"] == 0.25
    assert result["mcnemar"]["candidate_only_correct"] == 2
    assert result["mcnemar"]["baseline_only_correct"] == 1
    assert result["bootstrap_ci_95"] is not None


def test_compare_law_results_reports_unmatched_records():
    baseline = [
        _record(1, ["264"], ["264"]),
        _record(2, ["263"], ["263"]),
    ]
    candidate = [
        _record(2, ["263"], ["263"]),
        _record(3, ["234"], ["234"]),
    ]

    result = compare_law_results(
        baseline,
        candidate,
        bootstrap_samples=0,
    )

    assert result["paired_samples"] == 1
    assert result["baseline_unmatched"] == 1
    assert result["candidate_unmatched"] == 1
    assert result["bootstrap_ci_95"] is None


def test_penalty_retrieval_metrics_use_structured_precedents():
    records = [
        {
            "gold": {"imprisonment_months": 8, "law_articles": ["234"]},
            "retrieval": {
                "precedent_candidates": [
                    {
                        "text": "case-a",
                        "meta": {
                            "term_of_imprisonment": {"imprisonment": 7},
                            "relevant_articles": ["234"],
                        },
                    },
                    {
                        "text": "case-b",
                        "meta": {
                            "term_of_imprisonment": {"imprisonment": 36},
                            "relevant_articles": ["266"],
                        },
                    },
                ]
            },
        }
    ]

    result = _calculate_penalty_retrieval_at_k(records, 2)

    assert result["cls_hit"] == 1.0
    assert result["cls_mrr"] == 1.0
    assert result["month_mae"] == 1.0
    assert result["article_hit"] == 1.0


def test_penalty_prediction_class_prefers_explicit_bucket():
    record = {
        "prediction": {
            "penalty_class": "5",
            "imprisonment_months": 6,
        }
    }

    assert extract_pred_pt_cls(record) == 5


def run_all():
    test_compare_law_results_is_paired_by_case_id()
    test_compare_law_results_reports_unmatched_records()
    test_penalty_retrieval_metrics_use_structured_precedents()
    test_penalty_prediction_class_prefers_explicit_bucket()


if __name__ == "__main__":
    run_all()
    print("ok")
