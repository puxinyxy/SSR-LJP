"""
Evaluate the LJP multi-agent pipeline on the CAIL2018 first-stage test set.

Differences vs ljp_eval.py:
- Reads cases from data/testset/test.json (CAIL2018 format).
- Writes outputs under output_cail2018/.
- Imprisonment is evaluated by mapping months to the penalty class bucket (get_pt_cls)
  and comparing predicted vs truth classes (no absolute-error metric here).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, Optional, Set

from openpyxl import Workbook

from ljp_agents import make_llm, make_agent  # noqa: F401 (kept for parity with ljp_eval imports)
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
from ljp_eval import (
    _normalize_acc,
    _parse_judgment_json,
    extract_accusations,
    extract_articles,
    get_pt_cls,
    parse_imprisonment_safe,
)
from ljp_workflow import build_resources, predict_case


# ------------------------ Data loading --------------------------- #
def iter_testset(path: Path, offset: int = 0, limit: Optional[int] = None):
    """
    Iterate over CAIL2018 testset. Supports JSONL and JSON list.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    # Try full JSON (list) first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for idx, obj in enumerate(data):
                if idx < offset:
                    continue
                if limit is not None and idx >= offset + limit:
                    return
                if isinstance(obj, dict):
                    yield obj
            return
    except json.JSONDecodeError:
        pass

    # Fallback to JSONL
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if limit is not None and idx >= offset + limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield obj


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
def evaluate_case(pred: Dict[str, str], truth: Dict) -> Dict[str, Optional[float]]:
    truth_meta = truth.get("meta", {})
    truth_laws = {int(x) for x in truth_meta.get("relevant_articles", []) if str(x).isdigit()}
    truth_accs = {_normalize_acc(a) for a in truth_meta.get("accusation", []) if a}
    term = truth_meta.get("term_of_imprisonment", {})
    truth_imp = term.get("imprisonment")
    truth_cls = get_pt_cls(float(truth_imp)) if isinstance(truth_imp, (int, float)) else None

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

    pred_imp: Optional[float] = None
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
            pred_imp, _, _ = parse_imprisonment_safe(judgment_text)
    else:
        pred_imp, _, _ = parse_imprisonment_safe(judgment_text)

    pred_cls = get_pt_cls(pred_imp) if isinstance(pred_imp, (int, float)) else None

    metrics: Dict[str, Optional[float]] = {}
    metrics["law_acc"] = None if not truth_laws else (1.0 if pred_laws == truth_laws else 0.0)
    metrics["acc_acc"] = None if not truth_accs else (1.0 if pred_accs == truth_accs else 0.0)
    if truth_cls is None and pred_cls is None:
        metrics["penalty_cls_acc"] = None
    elif truth_cls is not None and pred_cls is not None:
        metrics["penalty_cls_acc"] = 1.0 if truth_cls == pred_cls else 0.0
    else:
        metrics["penalty_cls_acc"] = 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LJP multi-agent on CAIL2018 testset")
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
    parser.add_argument(
        "--dataset-path",
        type=str,
        # default="data/testset/cail_sampled.json",
        default="data/testset/test_sampled_single_seed_42.json",
        help="Path to CAIL2018 test JSONL (default: data/testset/cail_sampled.json)",
    )
    parser.add_argument("--output-dir", type=str, default="output_cail2018", help="Directory to save results")
    parser.add_argument(
        "--candidates-path",
        type=str,
        default=None,
        help="Precedent candidates path (default: data/candidates/precedents_cail.json)",
    )
    args = parser.parse_args()

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
    # Use CAIL-specific candidates file; allow override via CLI
    pipe_args.candidates_path = args.candidates_path or "data/candidates/precedents_cail.json"

    test_path = Path(args.dataset_path)
    total_raw = count_testset(test_path) or 0
    if total_raw and args.limit is not None:
        total_eval = max(0, min(args.limit, total_raw - args.offset))
    elif total_raw:
        total_eval = max(0, total_raw - args.offset)
    else:
        total_eval = args.limit or 0

    resources = build_resources(pipe_args)

    all_metrics = {
        "law_acc": [],
        "acc_acc": [],
        "penalty_cls_acc": [],
    }
    per_case_records = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ljp_eval_cail2018_{ts}.jsonl"
    jsonl_file = out_path.open("w", encoding="utf-8")
    xlsx_path = out_dir / f"ljp_eval_cail2018_{ts}.xlsx"

    wb = Workbook()
    ws_records = wb.active
    ws_records.title = "records"
    ws_records.append(
        [
            "caseID",
            "law_acc",
            "acc_acc",
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
        except Exception as e:
            skipped.append(case_obj.get("caseID"))
            print(f"Skip CaseID={case_obj.get('caseID')} due to model processing error: {e}", flush=True)
            continue
        m = evaluate_case(pred, case_obj)
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
                m.get("law_acc"),
                m.get("acc_acc"),
                m.get("penalty_cls_acc"),
                record["pred"].get("law_resp"),
                record["pred"].get("acc_resp"),
                record["pred"].get("prec_resp"),
                record["pred"].get("judgment"),
                record["pred"].get("penalty_summary"),
                total_tokens,
                json.dumps(record["truth"].get("relevant_articles"), ensure_ascii=False),
                json.dumps(record["truth"].get("accusation"), ensure_ascii=False),
                json.dumps(record["truth"].get("term_of_imprisonment"), ensure_ascii=False),
            ]
        )
        wb.save(xlsx_path)
    jsonl_file.close()

    summary = {k: (mean(v) if v else None) for k, v in all_metrics.items()}
    print("\n===== Evaluation Summary =====")
    print(f"Samples evaluated: {evaluated}")
    if skipped:
        print(f"Skipped (model errors): {len(skipped)} -> {skipped[:10]}{'...' if len(skipped)>10 else ''}")
    for k, v in summary.items():
        print(f"{k}: {v}")

    ws_summary = wb.create_sheet("summary")
    ws_summary.append(["metric", "value"])
    for k, v in summary.items():
        ws_summary.append([k, v])
    wb.save(xlsx_path)
    print(f"Saved results to: {xlsx_path}")


if __name__ == "__main__":
    main()
