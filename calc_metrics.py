# calc_metrics.py
from __future__ import annotations

import argparse
import json
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
    fw_digits = "??????????"
    hw_digits = "0123456789"
    return s.translate(str.maketrans(fw_digits, hw_digits))


CN_NUM = "??????????????"


def _chinese_num_to_int(s: str) -> int:
    digit_map = {
        "?": 0,
        "?": 1,
        "?": 2,
        "?": 2,
        "?": 3,
        "?": 4,
        "?": 5,
        "?": 6,
        "?": 7,
        "?": 8,
        "?": 9,
    }
    unit_map = {"?": 10, "?": 100, "?": 1000}
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
    nums = {int(n) for n in re.findall(r"\d{1,4}", text_hw)}
    pattern = rf"[?]?([{CN_NUM}]+)[??]?"
    for m in re.findall(pattern, text):
        try:
            val = _chinese_num_to_int(m)
            if val:
                nums.add(val)
        except Exception:
            continue
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


def extract_truth_pt_cls(record: dict) -> Optional[int]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        pt_cls = truth.get("pt_cls")
        if isinstance(pt_cls, (int, float)):
            return int(pt_cls)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to results JSONL (default points to baseline_output).",
    )
    parser.add_argument(
        "--mode",
        choices=("law", "penalty", "metrics", "ljp"),
        default="law",
    )
    parser.add_argument(
        "--metrics_key",
        default="metrics",
        help="When mode=metrics, average numeric values under this key.",
    )
    args = parser.parse_args()

    records = list(iter_records(Path(args.input)))
    if not records:
        raise SystemExit("No records found")

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
            pred_months = extract_pred_months(rec)
            pred_cls = get_pt_cls(pred_months) if pred_months is not None else None
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
        for rec in records:
            pred_set = to_article_set(extract_law_articles(rec, "pred"))
            gold_set = to_article_set(extract_law_articles(rec, "gold"))
            p, r, f1 = precision_recall_f1(pred_set, gold_set)
            acc_scores.append(1.0 if pred_set == gold_set else 0.0)
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(f1)
        print("acc:", safe_mean(acc_scores))
        print("Ma-P:", safe_mean(p_scores))
        print("Ma-R:", safe_mean(r_scores))
        print("Ma-F:", safe_mean(f_scores))
        print("samples:", len(acc_scores))
        return

    num_classes = 10
    cls_stats = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(num_classes)]
    correct = 0
    total_eval = 0
    skipped = 0
    for rec in records:
        gold_cls = gold_pt_cls(rec)
        pred_months = extract_months(rec, role="pred")
        pred_cls = get_pt_cls(pred_months) if pred_months is not None else None
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
