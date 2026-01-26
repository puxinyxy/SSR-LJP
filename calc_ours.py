from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "ljp_eval_20251225_094605.jsonl"
)

TRAILING_CRIME_CHAR = "\u7f6a"
FULLWIDTH_DIGITS = "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"
HALFWIDTH_DIGITS = "0123456789"
CN_NUM = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u96f6\u4e24"


def iter_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
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
            if isinstance(obj, dict):
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


def fullwidth_to_halfwidth(text: str) -> str:
    return text.translate(str.maketrans(FULLWIDTH_DIGITS, HALFWIDTH_DIGITS))


def chinese_num_to_int(text: str) -> int:
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
    for ch in text:
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


def extract_article_numbers(text: str) -> set[int]:
    if not text:
        return set()
    norm = fullwidth_to_halfwidth(str(text))
    nums = {int(n) for n in re.findall(r"\d{1,4}", norm)}
    pattern = rf"[\u7b2c]?([{CN_NUM}]+)[\u6761\u7bc7]?"
    for m in re.findall(pattern, str(text)):
        try:
            val = chinese_num_to_int(m)
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
        nums.update(extract_article_numbers(str(val)))
    return nums


def normalize_acc_name(name: str) -> str:
    s = str(name).strip()
    if s.endswith(TRAILING_CRIME_CHAR):
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


def extract_accusations_from_text(text: str, vocab: set[str]) -> set[str]:
    if not text:
        return set()
    found = {name for name in vocab if name and name in text}
    return {normalize_acc_name(name) for name in found}


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
    return set()


def extract_truth_articles(record: dict) -> set[int]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        val = truth.get("relevant_articles", truth.get("law_articles"))
        if val is not None:
            return to_article_set(val)
    return set()


def extract_pred_accusations(record: dict, vocab: set[str]) -> set[str]:
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


def extract_pred_months(record: dict) -> Optional[int]:
    judgment = extract_pred_judgment(record)
    if isinstance(judgment, dict) and "imprisonment_months" in judgment:
        return normalize_months(judgment.get("imprisonment_months"))
    pred = maybe_parse_json(record.get("pred"))
    if isinstance(pred, dict) and "imprisonment_months" in pred:
        return normalize_months(pred.get("imprisonment_months"))
    return None


def extract_truth_pt_cls(record: dict) -> Optional[int]:
    truth = extract_truth_obj(record)
    if isinstance(truth, dict):
        pt_cls = truth.get("pt_cls")
        if isinstance(pt_cls, (int, float)):
            return int(pt_cls)
        term = truth.get("term_of_imprisonment", {})
        months = None
        if isinstance(term, dict):
            months = normalize_months(term.get("imprisonment"))
        if months is None:
            months = normalize_months(truth.get("imprisonment_months"))
        if months is None:
            months = normalize_months(truth.get("penalty"))
        return get_pt_cls(months) if months is not None else None
    return None


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
    if not precisions:
        return None, None, None
    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to ljp_eval JSONL (default points to output).",
    )
    args = parser.parse_args()

    records = list(iter_records(Path(args.input)))
    if not records:
        raise SystemExit("No records found")

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


if __name__ == "__main__":
    main()
