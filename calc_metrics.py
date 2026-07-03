# calc_metrics.py
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "baseline_output"
    / "plain_prompt_cjo22_qwen3-max_LOT_prompt"
    / "LOT_prompt_results_qwen3-max.jsonl"
)


def iter_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict) and obj.get("record_type") != "summary":
                        yield obj
                return
        except json.JSONDecodeError:
            pass
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("record_type") != "summary":
                yield obj


def maybe_parse_json(val: Any) -> Any:
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return val
    return val


def get_nested(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def extract_container(record: dict, keys: Sequence[str]) -> Optional[dict]:
    for k in keys:
        val = maybe_parse_json(record.get(k))
        if isinstance(val, dict):
            return val
    return None


def extract_law_articles(record: dict, role: str) -> Any:
    if role == "pred":
        container = extract_container(record, ("prediction", "pred", "output", "result"))
    else:
        container = extract_container(record, ("gold", "label", "truth", "target"))
    if isinstance(container, dict):
        for k in ("law_articles", "articles"):
            if k in container:
                return container[k]
    for k in ("law_articles", "articles"):
        if k in record:
            return record[k]
    if role == "gold":
        meta = record.get("meta")
        if isinstance(meta, dict) and "relevant_articles" in meta:
            return meta["relevant_articles"]
    return []


def safe_json_load(text: str) -> Optional[dict]:
    if not text:
        return None
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        return obj
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None
    return None


def extract_pred_judgment(record: dict) -> dict:
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict):
        judgment = pred.get("judgment")
        if isinstance(judgment, dict):
            return judgment
        if isinstance(judgment, str):
            parsed = safe_json_load(judgment)
            if isinstance(parsed, dict):
                return parsed
    return {}


def extract_truth_obj(record: dict) -> Optional[dict]:
    truth = maybe_parse_json(record.get("truth"))
    return truth if isinstance(truth, dict) else None


def _fullwidth_to_halfwidth(s: str) -> str:
    fw_digits = "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"
    hw_digits = "0123456789"
    return s.translate(str.maketrans(fw_digits, hw_digits))


CN_NUM = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u96f6\u4e24"


def _chinese_num_to_int(s: str) -> int:
    digit_map = {
        "\u96f6": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e24": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
    }
    unit_map = {"\u5341": 10, "\u767e": 100, "\u5343": 1000}
    total = 0
    num = 0
    for ch in s:
        if ch in digit_map:
            num = digit_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
    total += num
    return total


def _extract_article_numbers(text: str) -> set[int]:
    text_hw = _fullwidth_to_halfwidth(text)
    nums: set[int] = set()
    pattern = re.compile(rf"\u7b2c\s*(\d{{1,4}}|[{CN_NUM}]+)\s*\u6761")
    for match in pattern.finditer(text_hw):
        raw = match.group(1)
        try:
            val = int(raw) if raw.isdigit() else _chinese_num_to_int(raw)
            if val:
                nums.add(val)
        except Exception:
            continue
    if not nums and re.fullmatch(r"\s*\d{1,4}\s*", text_hw):
        nums.add(int(text_hw.strip()))
    return nums


def to_article_set(values: Any) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    nums: set[int] = set()
    for val in values:
        if val is None or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            nums.add(int(val))
            continue
        text = str(val).strip()
        if not text:
            continue
        nums.update(_extract_article_numbers(text))
    return nums


def normalize_acc_name(name: str) -> str:
    s = str(name).strip()
    if s.endswith("\u7f6a"):
        s = s[:-1]
    return s


def to_acc_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    items: set[str] = set()
    for val in values:
        if val is None:
            continue
        name = normalize_acc_name(str(val))
        if name:
            items.add(name)
    return items


def extract_accusations_from_text(text: str, vocab: Sequence[str]) -> set[str]:
    if not text:
        return set()
    found = {name for name in vocab if name and name in text}
    return {normalize_acc_name(name) for name in found}


def extract_pred_articles(record: dict) -> set[int]:
    judgment = extract_pred_judgment(record)
    if isinstance(judgment, dict):
        val = judgment.get("articles", judgment.get("law_articles"))
        if val is not None:
            return to_article_set(val)
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict):
        law_resp = pred.get("law_resp")
        if isinstance(law_resp, str):
            return to_article_set(law_resp)
    return to_article_set(extract_law_articles(record, "pred"))


def extract_truth_articles(record: dict) -> set[int]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        val = truth.get("relevant_articles", truth.get("law_articles"))
        if val is not None:
            return to_article_set(val)
    return to_article_set(extract_law_articles(record, "gold"))


def extract_pred_accusations(record: dict, vocab: Sequence[str]) -> set[str]:
    judgment = extract_pred_judgment(record)
    if isinstance(judgment, dict):
        val = judgment.get("accusations", judgment.get("accusation"))
        if val is not None:
            return to_acc_set(val)
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict):
        acc_resp = pred.get("acc_resp")
        if isinstance(acc_resp, str):
            return extract_accusations_from_text(acc_resp, vocab)
    return set()


def extract_truth_accusations(record: dict) -> set[str]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        val = truth.get("accusation", truth.get("accusations"))
        if val is not None:
            return to_acc_set(val)
    return set()


def extract_pred_months(record: dict) -> Optional[int]:
    judgment = extract_pred_judgment(record)
    if isinstance(judgment, dict) and "imprisonment_months" in judgment:
        return normalize_months(judgment.get("imprisonment_months"))
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict) and "imprisonment_months" in pred:
        return normalize_months(pred.get("imprisonment_months"))
    return extract_months(record, role="pred")


def extract_pred_pt_cls(record: dict) -> Optional[int]:
    containers: List[dict] = []
    judgment = extract_pred_judgment(record)
    if isinstance(judgment, dict):
        containers.append(judgment)
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict):
        containers.append(pred)
    container = extract_container(record, ("prediction", "output", "result"))
    if isinstance(container, dict):
        containers.append(container)
    containers.append(record)

    for container in containers:
        for key in ("pt_cls", "penalty_class", "sentence_class"):
            cls = normalize_pt_cls(container.get(key))
            if cls is not None:
                return cls

    pred_months = extract_pred_months(record)
    return get_pt_cls(pred_months) if pred_months is not None else None


def extract_truth_pt_cls(record: dict) -> Optional[int]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        for key in ("pt_cls", "penalty_class", "sentence_class"):
            cls = normalize_pt_cls(truth.get(key))
            if cls is not None:
                return cls
        term = truth.get("term_of_imprisonment", {})
        if isinstance(term, dict):
            months = normalize_months(term.get("imprisonment"))
        else:
            months = None
        if months is None:
            months = normalize_months(truth.get("imprisonment_months"))
        if months is None:
            months = normalize_months(truth.get("penalty"))
        return get_pt_cls(months) if months is not None else None
    return gold_pt_cls(record)


def update_label_stats(stats: dict, pred_set: set, gold_set: set) -> None:
    for label in pred_set:
        entry = stats.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
        if label in gold_set:
            entry["tp"] += 1
        else:
            entry["fp"] += 1
    for label in gold_set:
        if label not in pred_set:
            entry = stats.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            entry["fn"] += 1


def macro_from_label_stats(stats: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for entry in stats.values():
        tp = entry.get("tp", 0)
        fp = entry.get("fp", 0)
        fn = entry.get("fn", 0)
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return safe_mean(precisions), safe_mean(recalls), safe_mean(f1s)


def precision_recall_f1(pred_set: set[int], gold_set: set[int]) -> tuple[float, float, float]:
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    inter = len(pred_set & gold_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def normalize_months(val: Any) -> Optional[int]:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(round(float(val)))
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return int(round(float(s)))
        except (TypeError, ValueError):
            return None
    return None


def normalize_pt_cls(val: Any) -> Optional[int]:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        cls = int(round(float(val)))
        return cls if 0 <= cls <= 9 else None
    if isinstance(val, str):
        match = re.search(r"\d+", val)
        if not match:
            return None
        cls = int(match.group(0))
        return cls if 0 <= cls <= 9 else None
    return None


def get_pt_cls(months: Optional[float]) -> Optional[int]:
    if months is None:
        return None
    if months > 10 * 12:
        return 9
    if months > 7 * 12:
        return 8
    if months > 5 * 12:
        return 7
    if months > 3 * 12:
        return 6
    if months > 2 * 12:
        return 5
    if months > 1 * 12:
        return 4
    if months > 9:
        return 3
    if months > 6:
        return 2
    if months > 0:
        return 1
    return 0


def extract_months(record: dict, role: str) -> Optional[int]:
    if role == "pred":
        container = extract_container(record, ("prediction", "pred", "output", "result"))
    else:
        container = extract_container(record, ("gold", "label", "truth", "target"))
    if isinstance(container, dict) and "imprisonment_months" in container:
        return normalize_months(container.get("imprisonment_months"))
    if "imprisonment_months" in record:
        return normalize_months(record.get("imprisonment_months"))
    if role == "gold":
        meta = record.get("meta")
        if isinstance(meta, dict):
            term = meta.get("term_of_imprisonment", {})
            if isinstance(term, dict):
                return normalize_months(term.get("imprisonment"))
    return None


def gold_pt_cls(record: dict) -> Optional[int]:
    meta = record.get("meta")
    if isinstance(meta, dict):
        pt_cls = meta.get("pt_cls")
        if isinstance(pt_cls, (int, float)):
            return int(pt_cls)
    months = extract_months(record, role="gold")
    return get_pt_cls(months) if months is not None else None


def _retrieval_candidate_articles(candidate: Any) -> set[int]:
    if isinstance(candidate, dict):
        meta = candidate.get("meta")
        if isinstance(meta, dict):
            article_id = meta.get("article_id")
            if isinstance(article_id, (int, float)):
                return {int(article_id)}
            if article_id is not None:
                parsed = to_article_set(article_id)
                if parsed:
                    return parsed
        return to_article_set(candidate.get("text"))
    return to_article_set(candidate)


def _extract_retrieval_ranking(record: dict) -> tuple[List[set[int]], bool, str]:
    retrieval = record.get("retrieval")
    if not isinstance(retrieval, dict):
        return [], False, "missing"

    debug = retrieval.get("debug")
    if isinstance(debug, dict):
        final_scores = debug.get("final_scores")
        if isinstance(final_scores, list) and final_scores:
            ranked_articles = [
                _retrieval_candidate_articles(candidate)
                for candidate in final_scores
            ]
            return ranked_articles, False, "final_fusion"

    rerank = debug.get("rerank") if isinstance(debug, dict) else None
    if isinstance(rerank, dict):
        ranked_results = rerank.get("ranked_results")
        if isinstance(ranked_results, list) and ranked_results:
            ranked_articles = [
                _retrieval_candidate_articles(candidate)
                for candidate in ranked_results
            ]
            pool_size = rerank.get("candidate_pool_size")
            ranking_complete = bool(rerank.get("ranking_complete"))
            if isinstance(pool_size, int) and len(ranked_articles) >= pool_size:
                ranking_complete = True
            return ranked_articles, ranking_complete, "rerank"

    candidates = retrieval.get("ranking_candidates")
    source = "ranking_candidates"
    if not isinstance(candidates, list):
        candidates = retrieval.get("candidates")
        source = "final_candidates"
    if not isinstance(candidates, list):
        return [], False, "missing"
    return [
        _retrieval_candidate_articles(candidate)
        for candidate in candidates
    ], False, source


def _calculate_retrieval_at_k(records: Sequence[dict], k: int) -> dict[str, Any]:
    evaluated = 0
    insufficient = 0
    any_hit = 0
    all_hit = 0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    sources: set[str] = set()

    for record in records:
        gold = extract_truth_articles(record)
        if not gold:
            continue
        ranked_articles, ranking_complete, source = _extract_retrieval_ranking(record)
        if not ranked_articles:
            continue
        if len(ranked_articles) < k and not ranking_complete:
            insufficient += 1
            continue

        sources.add(source)
        top_k_articles = ranked_articles[:k]
        retrieved = set().union(*top_k_articles) if top_k_articles else set()
        overlap = retrieved & gold
        evaluated += 1
        any_hit += int(bool(overlap))
        all_hit += int(gold <= retrieved)
        recall_sum += len(overlap) / len(gold)
        first_rank = next(
            (
                rank
                for rank, article_set in enumerate(top_k_articles, 1)
                if article_set & gold
            ),
            None,
        )
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

    return {
        "k": k,
        "evaluated": evaluated,
        "insufficient": insufficient,
        "sources": sorted(sources),
        "any_hit": any_hit / evaluated if evaluated else 0.0,
        "all_hit": all_hit / evaluated if evaluated else 0.0,
        "recall": recall_sum / evaluated if evaluated else 0.0,
        "mrr": reciprocal_rank_sum / evaluated if evaluated else 0.0,
    }


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("text", ""))
    return str(candidate)


def _candidate_meta(candidate: Any) -> dict:
    if not isinstance(candidate, dict):
        return {}
    meta = candidate.get("meta")
    return meta if isinstance(meta, dict) else {}


def _candidate_months(candidate: Any) -> Optional[int]:
    meta = _candidate_meta(candidate)
    term = meta.get("term_of_imprisonment")
    if isinstance(term, dict):
        months = normalize_months(term.get("imprisonment"))
        if months is not None:
            return months
    for key in ("imprisonment_months", "sentence_months", "penalty"):
        months = normalize_months(meta.get(key))
        if months is not None:
            return months
    text = _candidate_text(candidate)
    numbers = [int(match.group(1)) for match in re.finditer(r"(\d{1,4})", text)]
    return numbers[-1] if numbers else None


def _candidate_pt_cls(candidate: Any) -> Optional[int]:
    months = _candidate_months(candidate)
    return get_pt_cls(months) if months is not None else None


def _candidate_articles(candidate: Any) -> set[int]:
    meta = _candidate_meta(candidate)
    for key in ("relevant_articles", "law_articles", "articles", "article_id"):
        if key in meta:
            parsed = to_article_set(meta.get(key))
            if parsed:
                return parsed
    return to_article_set(_candidate_text(candidate))


def _extract_precedent_ranking(record: dict) -> tuple[List[Any], bool, str]:
    retrieval = record.get("retrieval")
    if not isinstance(retrieval, dict):
        return [], False, "missing"

    candidates = retrieval.get("ranking_precedent_candidates")
    source = "ranking_precedent_candidates"
    if not isinstance(candidates, list):
        candidates = retrieval.get("precedent_candidates")
        source = "precedent_candidates"
    if isinstance(candidates, list) and candidates:
        return candidates, False, source

    debug = retrieval.get("precedent_debug")
    if isinstance(debug, dict):
        final_scores = debug.get("final_scores")
        if isinstance(final_scores, list) and final_scores:
            return final_scores, False, "precedent_final_fusion"
        rerank = debug.get("rerank")
        if isinstance(rerank, dict):
            ranked_results = rerank.get("ranked_results")
            if isinstance(ranked_results, list) and ranked_results:
                pool_size = rerank.get("candidate_pool_size")
                ranking_complete = bool(rerank.get("ranking_complete"))
                if isinstance(pool_size, int) and len(ranked_results) >= pool_size:
                    ranking_complete = True
                return ranked_results, ranking_complete, "precedent_rerank"

    return [], False, "missing"


def _calculate_penalty_retrieval_at_k(records: Sequence[dict], k: int) -> dict[str, Any]:
    evaluated = 0
    insufficient = 0
    cls_hit = 0
    reciprocal_rank_sum = 0.0
    month_errors: list[float] = []
    article_evaluated = 0
    article_hit = 0
    sources: set[str] = set()

    for record in records:
        gold_months = extract_months(record, role="gold")
        gold_cls = get_pt_cls(gold_months) if gold_months is not None else gold_pt_cls(record)
        if gold_cls is None:
            continue
        ranked_candidates, ranking_complete, source = _extract_precedent_ranking(record)
        if not ranked_candidates:
            continue
        if len(ranked_candidates) < k and not ranking_complete:
            insufficient += 1
            continue

        top_k_candidates = ranked_candidates[:k]
        sources.add(source)
        evaluated += 1

        first_cls_rank = None
        candidate_months: list[int] = []
        for rank, candidate in enumerate(top_k_candidates, 1):
            months = _candidate_months(candidate)
            if months is not None:
                candidate_months.append(months)
            if first_cls_rank is None and _candidate_pt_cls(candidate) == gold_cls:
                first_cls_rank = rank
        if first_cls_rank is not None:
            cls_hit += 1
            reciprocal_rank_sum += 1.0 / first_cls_rank
        if gold_months is not None and candidate_months:
            month_errors.append(
                float(min(abs(months - gold_months) for months in candidate_months))
            )

        gold_articles = extract_truth_articles(record)
        if gold_articles:
            candidate_articles = set().union(
                *(_candidate_articles(candidate) for candidate in top_k_candidates)
            )
            if candidate_articles:
                article_evaluated += 1
                article_hit += int(bool(candidate_articles & gold_articles))

    return {
        "k": k,
        "evaluated": evaluated,
        "insufficient": insufficient,
        "sources": sorted(sources),
        "cls_hit": cls_hit / evaluated if evaluated else 0.0,
        "cls_mrr": reciprocal_rank_sum / evaluated if evaluated else 0.0,
        "month_mae": safe_mean(month_errors),
        "month_samples": len(month_errors),
        "article_hit": article_hit / article_evaluated if article_evaluated else None,
        "article_samples": article_evaluated,
    }


def _law_prediction_correct(record: dict) -> bool:
    pred_set = to_article_set(extract_law_articles(record, "pred"))
    gold_set = to_article_set(extract_law_articles(record, "gold"))
    return pred_set == gold_set


def _record_case_id(record: dict) -> Any:
    return record.get("case_id", record.get("caseID", record.get("id")))


def _pair_records(
    baseline_records: Sequence[dict],
    candidate_records: Sequence[dict],
) -> tuple[list[tuple[dict, dict]], int, int]:
    baseline_ids = [_record_case_id(record) for record in baseline_records]
    candidate_ids = [_record_case_id(record) for record in candidate_records]
    if all(case_id is not None for case_id in baseline_ids + candidate_ids):
        baseline_map = {
            case_id: record
            for case_id, record in zip(baseline_ids, baseline_records)
        }
        candidate_map = {
            case_id: record
            for case_id, record in zip(candidate_ids, candidate_records)
        }
        if len(baseline_map) != len(baseline_records):
            raise ValueError("Duplicate case IDs found in baseline results")
        if len(candidate_map) != len(candidate_records):
            raise ValueError("Duplicate case IDs found in candidate results")
        shared_ids = [
            case_id for case_id in baseline_ids if case_id in candidate_map
        ]
        pairs = [
            (baseline_map[case_id], candidate_map[case_id])
            for case_id in shared_ids
        ]
        return (
            pairs,
            len(baseline_records) - len(pairs),
            len(candidate_records) - len(pairs),
        )
    if len(baseline_records) != len(candidate_records):
        raise ValueError(
            "Results without case IDs must contain the same number of records"
        )
    return list(zip(baseline_records, candidate_records)), 0, 0


def _mcnemar_exact_p_value(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, candidate_only)
    lower_tail = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def compare_law_results(
    baseline_records: Sequence[dict],
    candidate_records: Sequence[dict],
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    pairs, baseline_unmatched, candidate_unmatched = _pair_records(
        baseline_records,
        candidate_records,
    )
    if not pairs:
        raise ValueError("No paired records found")
    baseline_correct = [
        _law_prediction_correct(baseline)
        for baseline, _ in pairs
    ]
    candidate_correct = [
        _law_prediction_correct(candidate)
        for _, candidate in pairs
    ]
    differences = [
        float(candidate) - float(baseline)
        for baseline, candidate in zip(
            baseline_correct,
            candidate_correct,
        )
    ]
    candidate_only = sum(
        1
        for baseline, candidate in zip(
            baseline_correct,
            candidate_correct,
        )
        if candidate and not baseline
    )
    baseline_only = sum(
        1
        for baseline, candidate in zip(
            baseline_correct,
            candidate_correct,
        )
        if baseline and not candidate
    )
    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    if bootstrap_samples > 0:
        pair_count = len(pairs)
        for _ in range(bootstrap_samples):
            sample_sum = sum(
                differences[rng.randrange(pair_count)]
                for _ in range(pair_count)
            )
            bootstrap_differences.append(sample_sum / pair_count)
        bootstrap_differences.sort()
    point_difference = sum(differences) / len(differences)
    return {
        "paired_samples": len(pairs),
        "baseline_unmatched": baseline_unmatched,
        "candidate_unmatched": candidate_unmatched,
        "baseline_accuracy": sum(baseline_correct) / len(pairs),
        "candidate_accuracy": sum(candidate_correct) / len(pairs),
        "accuracy_difference": point_difference,
        "bootstrap_ci_95": (
            [
                _percentile(bootstrap_differences, 0.025),
                _percentile(bootstrap_differences, 0.975),
            ]
            if bootstrap_differences
            else None
        ),
        "mcnemar": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "exact_p_value": _mcnemar_exact_p_value(
                baseline_only,
                candidate_only,
            ),
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to results JSONL (default points to baseline_output).",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "law",
            "law-compare",
            "penalty",
            "penalty-retrieval",
            "retrieval",
            "metrics",
            "ljp",
        ),
        default="law",
    )
    parser.add_argument(
        "--baseline-input",
        default=None,
        help="Baseline JSONL used by mode=law-compare.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Paired bootstrap samples used by mode=law-compare.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics_key",
        default="metrics",
        help="When mode=metrics, average numeric values under this key.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        nargs="+",
        default=[3],
        metavar="K",
        help="One or more cutoffs for mode=retrieval, e.g. --retrieval-k 1 3 5 10.",
    )
    args = parser.parse_args()

    records = list(iter_records(Path(args.input)))
    if not records:
        raise SystemExit("No records found")

    if args.mode == "law-compare":
        if not args.baseline_input:
            raise SystemExit("--baseline-input is required for mode=law-compare")
        if args.bootstrap_samples < 0:
            raise SystemExit("--bootstrap-samples must be >= 0")
        baseline_records = list(iter_records(Path(args.baseline_input)))
        if not baseline_records:
            raise SystemExit("No baseline records found")
        try:
            comparison = compare_law_results(
                baseline_records,
                records,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return

    if args.mode == "retrieval":
        retrieval_ks = list(dict.fromkeys(args.retrieval_k))
        if any(k <= 0 for k in retrieval_ks):
            raise SystemExit("--retrieval-k values must be positive integers")
        for k in retrieval_ks:
            metrics = _calculate_retrieval_at_k(records, k)
            if metrics["insufficient"]:
                raise SystemExit(
                    f"Recall@{k} cannot be computed for {metrics['insufficient']} records: "
                    "the result file does not contain a complete ranking to that depth. "
                    "Use a result generated after full rerank ranking was enabled."
                )
            if metrics["evaluated"] == 0:
                raise SystemExit("No retrieval candidates found in records")
            print(f"retrieval_any_hit@{k}:", metrics["any_hit"])
            print(f"retrieval_all_hit@{k}:", metrics["all_hit"])
            print(f"retrieval_recall@{k}:", metrics["recall"])
            print(f"retrieval_mrr@{k}:", metrics["mrr"])
            print(f"ranking_source@{k}:", ",".join(metrics["sources"]))
            print(f"samples@{k}:", metrics["evaluated"])
        return

    if args.mode == "penalty-retrieval":
        retrieval_ks = list(dict.fromkeys(args.retrieval_k))
        if any(k <= 0 for k in retrieval_ks):
            raise SystemExit("--retrieval-k values must be positive integers")
        for k in retrieval_ks:
            metrics = _calculate_penalty_retrieval_at_k(records, k)
            if metrics["insufficient"]:
                raise SystemExit(
                    f"Penalty retrieval@{k} cannot be computed for "
                    f"{metrics['insufficient']} records: the result file does not "
                    "contain a complete precedent ranking to that depth."
                )
            if metrics["evaluated"] == 0:
                raise SystemExit("No precedent retrieval candidates found in records")
            print(f"penalty_retrieval_cls_hit@{k}:", metrics["cls_hit"])
            print(f"penalty_retrieval_cls_mrr@{k}:", metrics["cls_mrr"])
            print(f"penalty_retrieval_month_mae@{k}:", metrics["month_mae"])
            print(f"penalty_retrieval_month_samples@{k}:", metrics["month_samples"])
            print(f"penalty_retrieval_article_hit@{k}:", metrics["article_hit"])
            print(f"penalty_retrieval_article_samples@{k}:", metrics["article_samples"])
            print(f"penalty_ranking_source@{k}:", ",".join(metrics["sources"]))
            print(f"penalty_samples@{k}:", metrics["evaluated"])
        return

    if args.mode == "metrics":
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in records:
            metrics = maybe_parse_json(rec.get(args.metrics_key))
            if not isinstance(metrics, dict):
                continue
            for key, val in metrics.items():
                if isinstance(val, bool) or val is None:
                    continue
                if isinstance(val, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(val)
                    counts[key] = counts.get(key, 0) + 1
        if not sums:
            raise SystemExit(f"No numeric metrics found under key '{args.metrics_key}'")
        for key in sorted(sums):
            mean = sums[key] / counts[key]
            print(f"{key}: {mean}")
        print("samples:", len(records))
        return

    if args.mode == "ljp":
        law_stats: dict = {}
        acc_stats: dict = {}
        law_correct = 0
        acc_correct = 0

        acc_vocab = set()
        for rec in records:
            acc_vocab |= extract_truth_accusations(rec)
        acc_vocab = {normalize_acc_name(a) for a in acc_vocab if a}

        penalty_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(10)}
        penalty_correct = 0
        penalty_total = 0
        penalty_skipped = 0

        for rec in records:
            pred_law = extract_pred_articles(rec)
            gold_law = extract_truth_articles(rec)
            if pred_law == gold_law:
                law_correct += 1
            update_label_stats(law_stats, pred_law, gold_law)

            pred_acc = extract_pred_accusations(rec, acc_vocab)
            gold_acc = extract_truth_accusations(rec)
            if pred_acc == gold_acc:
                acc_correct += 1
            update_label_stats(acc_stats, pred_acc, gold_acc)

            gold_cls = extract_truth_pt_cls(rec)
            pred_cls = extract_pred_pt_cls(rec)
            if gold_cls is None:
                penalty_skipped += 1
            else:
                penalty_total += 1
                if pred_cls == gold_cls:
                    penalty_correct += 1
                    penalty_stats[gold_cls]["tp"] += 1
                else:
                    penalty_stats[gold_cls]["fn"] += 1
                    if pred_cls is not None and 0 <= pred_cls < 10:
                        penalty_stats[pred_cls]["fp"] += 1

        law_ma_p, law_ma_r, law_ma_f = macro_from_label_stats(law_stats)
        acc_ma_p, acc_ma_r, acc_ma_f = macro_from_label_stats(acc_stats)
        pen_ma_p, pen_ma_r, pen_ma_f = macro_from_label_stats(penalty_stats)
        law_acc = (law_correct / len(records)) if records else None
        acc_acc = (acc_correct / len(records)) if records else None
        pen_acc = (penalty_correct / penalty_total) if penalty_total else None

        print("law_acc:", law_acc)
        print("law_Ma-P:", law_ma_p)
        print("law_Ma-R:", law_ma_r)
        print("law_Ma-F:", law_ma_f)
        print("acc_acc:", acc_acc)
        print("acc_Ma-P:", acc_ma_p)
        print("acc_Ma-R:", acc_ma_r)
        print("acc_Ma-F:", acc_ma_f)
        print("penalty_acc:", pen_acc)
        print("penalty_Ma-P:", pen_ma_p)
        print("penalty_Ma-R:", pen_ma_r)
        print("penalty_Ma-F:", pen_ma_f)
        print("samples:", len(records))
        print("penalty_eval_samples:", penalty_total)
        print("penalty_skipped:", penalty_skipped)
        return

    if args.mode == "law":
        acc_scores: List[float] = []
        p_scores: List[float] = []
        r_scores: List[float] = []
        f_scores: List[float] = []
        primary_hit_scores: List[float] = []
        single_gold_acc_scores: List[float] = []
        multi_gold_recall_scores: List[float] = []
        for rec in records:
            pred_set = to_article_set(extract_law_articles(rec, "pred"))
            gold_set = to_article_set(extract_law_articles(rec, "gold"))
            p, r, f1 = precision_recall_f1(pred_set, gold_set)
            acc = 1.0 if pred_set == gold_set else 0.0
            acc_scores.append(acc)
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(f1)
            primary_hit_scores.append(1.0 if pred_set & gold_set else 0.0)
            if len(gold_set) == 1:
                single_gold_acc_scores.append(acc)
            elif len(gold_set) > 1:
                multi_gold_recall_scores.append(r)
        print("acc:", safe_mean(acc_scores))
        print("Ma-P:", safe_mean(p_scores))
        print("Ma-R:", safe_mean(r_scores))
        print("Ma-F:", safe_mean(f_scores))
        print("primary_hit_acc:", safe_mean(primary_hit_scores))
        print("single_gold_acc:", safe_mean(single_gold_acc_scores))
        print(
            "multi_gold_primary_recall:",
            safe_mean(multi_gold_recall_scores),
        )
        print("samples:", len(acc_scores))
        print("single_gold_samples:", len(single_gold_acc_scores))
        print("multi_gold_samples:", len(multi_gold_recall_scores))
        return

    num_classes = 10
    cls_stats = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(num_classes)]
    correct = 0
    total_eval = 0
    skipped = 0
    for rec in records:
        gold_cls = gold_pt_cls(rec)
        pred_cls = extract_pred_pt_cls(rec)
        if gold_cls is None:
            skipped += 1
            continue
        total_eval += 1
        if pred_cls == gold_cls:
            correct += 1
            cls_stats[gold_cls]["tp"] += 1
        else:
            cls_stats[gold_cls]["fn"] += 1
            if pred_cls is not None and 0 <= pred_cls < num_classes:
                cls_stats[pred_cls]["fp"] += 1

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for s in cls_stats:
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    acc = correct / total_eval if total_eval else None
    print("acc:", acc)
    print("Ma-P:", safe_mean(precisions))
    print("Ma-R:", safe_mean(recalls))
    print("Ma-F:", safe_mean(f1s))
    print("eval_samples:", total_eval)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
