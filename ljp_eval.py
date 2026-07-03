"""
Evaluate the LJP multi-agent pipeline on the testset.

Metrics:
- law_recall: overlap between predicted law articles and truth set
- acc_recall: overlap between predicted accusations and truth set
- imprison_abs_err: absolute error of imprisonment months; if truth is life/death, matching category counts as 0 else None

Usage: --limit controls evaluation size; --offset controls starting caseID.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Optional, Set, Tuple

import json as _json_for_excel
from logger_utils import log_metrics, setup_run_logger
from ljp_config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    JOIN_TOP_K,
    KEYWORD_TOP_K,
    KEYWORD_WEIGHT,
    RERANK_TOP_K,
    RETRIEVAL_MODE,
    RETRIEVAL_QUERY_MAX_CHARS,
    RRF_K,
    USE_RERANK,
)

# Compatibility shim: openai<1.44 lacks ParsedChatCompletion.
try:  # pragma: no cover - best-effort guard for old SDK
    from openai.types.chat import ParsedChatCompletion as _ParsedChatCompletion  # type: ignore
except Exception:  # noqa: BLE001
    import sys
    import types

    try:
        import openai  # type: ignore
    except Exception:  # noqa: BLE001
        openai = None  # type: ignore

    chat_mod = sys.modules.get("openai.types.chat")
    if chat_mod is None:
        chat_mod = types.ModuleType("openai.types.chat")
        sys.modules["openai.types.chat"] = chat_mod
    if not hasattr(chat_mod, "ParsedChatCompletion"):
        class ParsedChatCompletion:  # Minimal placeholder for older SDKs
            ...

        chat_mod.ParsedChatCompletion = ParsedChatCompletion  # type: ignore


def _cell_safe(val):
    """Convert lists/dicts to JSON strings for Excel cells."""
    if isinstance(val, (list, dict)):
        return _json_for_excel.dumps(val, ensure_ascii=False)
    return val


# ------------------------ Parsing helpers ------------------------ #
def _fullwidth_to_halfwidth(s: str) -> str:
    """Convert fullwidth digits to halfwidth."""
    fw_digits = "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19"
    hw_digits = "0123456789"
    return s.translate(str.maketrans(fw_digits, hw_digits))


def _chinese_num_to_int(s: str) -> int:
    """Lightweight parser for Chinese numerals up to thousands."""
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


CN_NUM = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u96f6\u4e24"


def extract_articles(text: str) -> Set[int]:
    text_hw = _fullwidth_to_halfwidth(text)
    nums = {int(n) for n in re.findall(r"\d{2,4}", text_hw)}

    pattern = rf"[\u7b2c]?([{CN_NUM}]+)[\u6761\u7bc7]?"
    for m in re.findall(pattern, text):
        try:
            val = _chinese_num_to_int(m)
            if val:
                nums.add(val)
        except Exception:
            continue
    return nums


def extract_accusations(text: str, truth_vocab: Iterable[str]) -> Set[str]:
    found = {a for a in truth_vocab if a and a in text}
    if found:
        return found
    spans = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    return set(spans)


# ------------------------ Penalty labels & parsing ---------------- #
def get_pt_cls(months: Optional[float]) -> Optional[int]:
    """Map imprisonment months to penalty class buckets."""
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


def _parse_judgment_json(judgment_text: str | dict) -> Optional[dict]:
    """Parse judgment JSON (string or dict); return dict or None."""
    if not judgment_text:
        return None
    if isinstance(judgment_text, dict):
        return judgment_text
    try:
        obj = json.loads(judgment_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_imprisonment_safe(text: str) -> Tuple[Optional[float], bool, bool]:
    """
    Parse imprisonment info from model output.
    Returns (months, life, death); months=None when not numeric.
    Supported: X年Y个月 / X年 / X月 / 标签文本 / -1 (life) / -2 (death).
    """
    if not text:
        return None, False, False

    norm = _fullwidth_to_halfwidth(text)
    if "\u6b7b\u5211" in norm or "-2" in norm:
        return None, False, True
    if "\u65e0\u671f" in norm or "-1" in norm:
        return None, True, False

    m = re.search(r"(-?\d+)\s*\u5e74\s*(\d+)\s*\u6708?", norm)
    if m:
        years = float(m.group(1))
        months = float(m.group(2))
        return years * 12.0 + months, False, False

    m = re.search(r"(-?\d+)\s*\u5e74", norm)
    if m:
        years = float(m.group(1))
        if years >= 0:
            return years * 12.0, False, False

    m = re.search(r"(-?\d+)\s*(\u4e2a?\u6708|\u6708)", norm)
    if m:
        months = float(m.group(1))
        if months >= 0:
            return months, False, False

    label_mid = {
        "\u516d\u4e2a\u6708\u4ee5\u5185": 3,
        "\u516d\u5230\u4e5d\u4e2a\u6708": 7.5,
        "\u4e5d\u4e2a\u6708\u5230\u4e00\u5e74": 10.5,
        "\u4e00\u5230\u4e24\u5e74": 18,
        "\u4e8c\u5230\u4e09\u5e74": 30,
        "\u4e09\u5230\u4e94\u5e74": 48,
        "\u4e94\u5230\u4e03\u5e74": 72,
        "\u4e03\u5230\u5341\u5e74": 102,
        "\u5341\u5e74\u4ee5\u4e0a": 120,
    }
    for label, mid in label_mid.items():
        if label in norm:
            return float(mid), False, False

    return None, False, False


# ------------------------ Data loading --------------------------- #
def iter_testset(path: Path, offset: int = 0, limit: Optional[int] = None):
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    count = 0
    yielded = 0
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                if count >= offset:
                    yield obj
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                count += 1
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
            if count >= offset:
                yield obj
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            count += 1


def count_testset(path: Path) -> Optional[int]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return len(data)
    except json.JSONDecodeError:
        pass

    total = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                total += 1
    return total if total > 0 else None


# ------------------------ Evaluation ----------------------------- #
def _normalize_acc(name: str) -> str:
    """Normalize accusation for loose matching (e.g., strip trailing '罪')."""
    if not isinstance(name, str):
        return ""
    n = name.strip()
    if n.endswith("\u7f6a"):
        n = n[:-1]
    return n


def evaluate_case(pred: Dict[str, str], truth: Dict) -> Dict[str, Optional[float]]:
    truth_meta = truth.get("meta", {})
    truth_laws = {int(x) for x in truth_meta.get("relevant_articles", []) if str(x).isdigit()}
    truth_accs = {_normalize_acc(a) for a in truth_meta.get("accusation", []) if a}
    truth_pt_cls_raw = truth_meta.get("pt_cls")
    term = truth_meta.get("term_of_imprisonment", {})
    truth_imp = term.get("imprisonment")
    truth_life = term.get("life_imprisonment", False)
    truth_death = term.get("death_penalty", False)

    judgment_text = pred.get("judgment", "")
    judgment_obj = _parse_judgment_json(judgment_text)

    if judgment_obj:
        pred_laws = {
            int(x)
            for x in judgment_obj.get("articles", [])
            if (isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()))
        }
        pred_accs = {_normalize_acc(str(x)) for x in judgment_obj.get("accusations", []) if x}
    else:
        pred_laws = extract_articles(pred.get("law_resp", ""))
        pred_accs = {_normalize_acc(a) for a in extract_accusations(pred.get("acc_resp", ""), truth_accs)}

    # Predicted imprisonment (months only; life/death handled via special flags)
    pred_imp: Optional[float] = None
    pred_life = False
    pred_death = False
    if judgment_obj and "imprisonment_months" in judgment_obj:
        imp_val = judgment_obj.get("imprisonment_months")
        if isinstance(imp_val, (int, float)):
            pred_imp = float(imp_val)
        else:
            try:
                pred_imp = float(str(imp_val).strip())
            except (TypeError, ValueError):
                pred_imp = None
        if pred_imp is None:
            pred_imp, pred_life, pred_death = parse_imprisonment_safe(judgment_text)
    else:
        pred_imp, pred_life, pred_death = parse_imprisonment_safe(judgment_text)

    metrics: Dict[str, Optional[float]] = {}
    metrics["law_recall"] = (len(truth_laws & pred_laws) / len(truth_laws)) if truth_laws else None
    metrics["acc_recall"] = (len(truth_accs & pred_accs) / len(truth_accs)) if truth_accs else None

    metrics["imprison_abs_err"] = None
    if truth_life or truth_death:
        if (truth_life and pred_life) or (truth_death and pred_death):
            metrics["imprison_abs_err"] = 0.0
    elif isinstance(truth_imp, (int, float)) and isinstance(pred_imp, (int, float)):
        metrics["imprison_abs_err"] = abs(float(truth_imp) - float(pred_imp))

    # Penalty class accuracy: compare mapped predicted class with truth pt_cls
    truth_cls = None
    if isinstance(truth_pt_cls_raw, (int, float)):
        truth_cls = int(truth_pt_cls_raw)
    elif isinstance(truth_imp, (int, float)):
        truth_cls = get_pt_cls(float(truth_imp))

    pred_cls = get_pt_cls(pred_imp) if isinstance(pred_imp, (int, float)) else None
    if truth_cls is None and pred_cls is None:
        metrics["penalty_cls_acc"] = None
    elif truth_cls is not None and pred_cls is not None:
        metrics["penalty_cls_acc"] = 1.0 if truth_cls == pred_cls else 0.0
    else:
        metrics["penalty_cls_acc"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LJP multi-agent on testset")
    parser.add_argument("--limit", type=int, default=5, help="Number of cases to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Start from this index")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k for retrieval")
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "embedding"),
        default=RETRIEVAL_MODE,
        help="Retrieval mode. hybrid uses dense+BM25+keyword RRF.",
    )
    parser.add_argument("--dense-top-k", type=int, default=DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=BM25_TOP_K)
    parser.add_argument("--keyword-top-k", type=int, default=KEYWORD_TOP_K)
    parser.add_argument("--join-top-k", type=int, default=JOIN_TOP_K)
    parser.add_argument("--rrf-k", type=float, default=RRF_K)
    parser.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    parser.add_argument("--keyword-weight", type=float, default=KEYWORD_WEIGHT)
    parser.add_argument(
        "--retrieval-query-max-chars",
        type=int,
        default=RETRIEVAL_QUERY_MAX_CHARS,
    )
    parser.add_argument("--use-rerank", dest="use_rerank", action="store_true")
    parser.add_argument("--no-rerank", dest="use_rerank", action="store_false")
    parser.set_defaults(use_rerank=USE_RERANK)
    parser.add_argument("--rerank-top-k", type=int, default=RERANK_TOP_K)
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results")
    args = parser.parse_args()

    logger, run_dir, run_id = setup_run_logger(
        run_name="ljp_eval",
        args=vars(args),
        extra={"cwd": str(Path.cwd())},
    )

    from openai import BadRequestError
    from camel.models.model_manager import ModelProcessingError
    from openpyxl import Workbook
    from ljp_workflow import build_resources, predict_case
    from ljp_multi_agent import parse_args as pipeline_args

    import sys

    _argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    pipe_args = pipeline_args()
    sys.argv = _argv_backup
    if args.top_k is not None:
        pipe_args.top_k = args.top_k
    pipe_args.retrieval_mode = args.retrieval_mode
    pipe_args.dense_top_k = args.dense_top_k
    pipe_args.bm25_top_k = args.bm25_top_k
    pipe_args.keyword_top_k = args.keyword_top_k
    pipe_args.join_top_k = args.join_top_k
    pipe_args.rrf_k = args.rrf_k
    pipe_args.dense_weight = args.dense_weight
    pipe_args.bm25_weight = args.bm25_weight
    pipe_args.keyword_weight = args.keyword_weight
    pipe_args.retrieval_query_max_chars = args.retrieval_query_max_chars
    pipe_args.use_rerank = args.use_rerank
    pipe_args.rerank_top_k = args.rerank_top_k

    test_path = Path("data/testset/testset.json")
    total_raw = count_testset(test_path) or 0
    if total_raw and args.limit is not None:
        total_eval = max(0, min(args.limit, total_raw - args.offset))
    elif total_raw:
        total_eval = max(0, total_raw - args.offset)
    else:
        total_eval = args.limit or 0

    logger.info(
        "config test_path=%s total_raw=%s total_eval=%s",
        str(test_path),
        total_raw,
        total_eval,
    )

    resources = build_resources(pipe_args)
    logger.info("resources_built top_k=%s", pipe_args.top_k)

    all_metrics = {
        "law_recall": [],
        "acc_recall": [],
        "imprison_abs_err": [],
        "penalty_cls_acc": [],
    }
    per_case_records = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ljp_eval_{ts}.jsonl"
    jsonl_file = out_path.open("w", encoding="utf-8")
    xlsx_path = out_dir / f"ljp_eval_{ts}.xlsx"

    wb = Workbook()
    ws_records = wb.active
    ws_records.title = "records"
    ws_records.append(
        [
            "caseID",
            "law_recall",
            "acc_recall",
            "imprison_abs_err",
            "penalty_cls_acc",
            "pred_law",
            "pred_acc",
            "pred_prec",
            "pred_judgment",
            "penalty_summary",
            "usage_total_tokens",
            "truth_laws",
            "truth_accusations",
            "truth_imprisonment",
        ]
    )
    evaluated = 0
    skipped = []
    for case_obj in iter_testset(test_path, offset=args.offset, limit=args.limit):
        fact = case_obj.get("fact", "")
        try:
            pred = predict_case(fact, resources, top_k=pipe_args.top_k)
        except (BadRequestError, ModelProcessingError, ValueError) as e:
            parts = [
                str(e),
                repr(e),
                str(getattr(e, "__cause__", "")),
                repr(getattr(e, "__cause__", "")),
                str(getattr(e, "__context__", "")),
                repr(getattr(e, "__context__", "")),
            ]
            full_msg = " ".join(p for p in parts if p)
            if "data_inspection_failed" in full_msg or "inappropriate content" in full_msg:
                skipped.append(case_obj.get("caseID"))
                print(f"Skip CaseID={case_obj.get('caseID')} due to content inspection", flush=True)
                logger.warning(
                    "case_skipped caseID=%s reason=content_inspection err_type=%s err=%s",
                    case_obj.get("caseID"),
                    type(e).__name__,
                    full_msg,
                )
                logger.info("case_skipped caseID=%s reason=content_inspection", case_obj.get("caseID"))
                continue
            if isinstance(e, (ModelProcessingError, ValueError)):
                skipped.append(case_obj.get("caseID"))
                print(f"Skip CaseID={case_obj.get('caseID')} due to model processing error", flush=True)
                logger.exception(
                    "case_error caseID=%s reason=model_processing err_type=%s err=%s",
                    case_obj.get("caseID"),
                    type(e).__name__,
                    full_msg,
                )
                logger.info("case_skipped caseID=%s reason=model_processing", case_obj.get("caseID"))
                continue
            raise
        m = evaluate_case(pred, case_obj)
        logger.info("case_done caseID=%s metrics=%s", case_obj.get("caseID"), m)
        for k, v in m.items():
            if v is not None:
                all_metrics[k].append(v)
        evaluated += 1
        progress = f"[{evaluated}/{total_eval}]" if total_eval else f"[{evaluated}]"
        record = {
            "caseID": case_obj.get("caseID"),
            "metrics": m,
            "pred": {
                "law_resp": pred.get("law_resp"),
                "acc_resp": pred.get("acc_resp"),
                "prec_resp": pred.get("prec_resp"),
                "judgment": pred.get("judgment"),
                "penalty_summary": pred.get("penalty_summary"),
                "law_hits": [{"text": h.text, "meta": h.meta} for h in pred.get("law_hits", [])],
                "acc_hits": [{"text": h.text, "meta": h.meta} for h in pred.get("acc_hits", [])],
                "cand_hits": [{"text": h.text, "meta": h.meta} for h in pred.get("cand_hits", [])],
                "usage": pred.get("usage", {}),
            },
            "truth": case_obj.get("meta", {}),
        }
        per_case_records.append(record)
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{progress} CaseID={case_obj.get('caseID')}, metrics={m}", flush=True)
        usage = record["pred"].get("usage", {})
        total_tokens = sum(u.get("total_tokens", 0) for u in usage.values() if isinstance(u, dict))
        ws_records.append(
            [
                record.get("caseID"),
                m.get("law_recall"),
                m.get("acc_recall"),
                m.get("imprison_abs_err"),
                m.get("penalty_cls_acc"),
                record["pred"].get("law_resp"),
                record["pred"].get("acc_resp"),
                record["pred"].get("prec_resp"),
                record["pred"].get("judgment"),
                record["pred"].get("penalty_summary"),
                total_tokens,
                _cell_safe(record["truth"].get("relevant_articles")),
                _cell_safe(record["truth"].get("accusation")),
                _cell_safe(record["truth"].get("term_of_imprisonment")),
            ]
        )
        wb.save(xlsx_path)
    jsonl_file.close()

    summary = {k: (mean(v) if v else None) for k, v in all_metrics.items()}
    print("\n===== Evaluation Summary =====")
    print(f"Samples evaluated: {evaluated}")
    if skipped:
        print(f"Skipped (content inspection/model errors): {len(skipped)} -> {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    for k, v in summary.items():
        print(f"{k}: {v}")
    log_metrics(logger, summary, prefix="summary")

    ws_summary = wb.create_sheet("summary")
    ws_summary.append(["metric", "value"])
    for k, v in summary.items():
        ws_summary.append([k, v])
    wb.save(xlsx_path)
    print(f"Saved Excel results to: {xlsx_path}")


if __name__ == "__main__":
    main()
