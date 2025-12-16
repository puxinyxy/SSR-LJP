"""
Plain/COT/LOT prompt baselines for LJP.

- Sends a prompt to the LLM to predict articles/accusations/imprisonment.
- Model must return JSON: { "articles": [int], "accusations": [str], "imprisonment_months": int }
- Evaluates strict accuracy metrics (law_acc / acc_acc / penalty_cls_acc).

Usage:
    python baseline/plain_prompt.py --task cjo22 --prompt-type plain_prompt --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from openai import OpenAI

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from ljp_eval import (
    _normalize_acc,
    _parse_judgment_json,
    extract_accusations,
    extract_articles,
    get_pt_cls,
    parse_imprisonment_safe,
)
from ljp_eval_cail2018 import iter_testset as iter_testset_cail
from ljp_eval_cail2018 import count_testset as count_testset_cail
from ljp_eval import iter_testset as iter_testset_cjo
from ljp_eval import count_testset as count_testset_cjo


# ------------------------ Config (editable) ------------------------ #
# DEFAULT_MODEL = "qwen3-235b-a22b"
# DEFAULT_MODEL = "qwen3-max"
DEFAULT_MODEL = "qwen3-8b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY = "sk-d103be2645ca438d91892867a65cfd2c"
DEFAULT_DATASET_CAIL = "data/testset/cail_sampled.json"  # CAIL2018-format testset
DEFAULT_DATASET_CJO22 = "data/testset/testset.json"  # cjo22-format testset
DEFAULT_OUTPUT_DIR_CAIL = "baseline_output/plain_prompt_cail2018"
DEFAULT_OUTPUT_DIR_CJO22 = "baseline_output/plain_prompt_cjo22"

# Prompt variants
PROMPTS = {
    "plain_prompt": (
        "请根据案件事实，直接输出一个合法的 JSON 对象，不能有多余文字。\n"
        "JSON 字段必须且仅有 3 个：\n"
        '  1) \"articles\": 法条编号整数列表，例如 [263, 264]\n'
        '  2) \"accusations\": 罪名字符串列表，例如 [\"抢劫罪\"]\n'
        '  3) \"imprisonment_months\": 总刑期（整数，月份）；无期=-1，死刑=-2，无拘役=0\n'
        "务必返回合法 JSON。"
    ),
    "COT_prompt": (
        "让我们一步步思考，输出被告人构成的法条、罪名、刑期。最终只输出一个合法的 JSON 对象，不能有多余文字。\n"
        "JSON 字段必须且仅有 3 个：\n"
        '  1) \"articles\": 法条编号整数列表，例如 [263, 264]\n'
        '  2( \"accusations\": 罪名字符串列表，例如 [\"抢劫罪\"]\n'
        '  3) \"imprisonment_months\": 总刑期（整数，月份）；无期=-1，死刑=-2，无拘役=0\n'
        "务必返回合法 JSON。"
    ),
    "LOT_prompt": (
        "你在司法三段论中，大前提是具体的法律规范，小前提是案件事实，结论是判决结果。 \n"
        "让我们用司法三段论思考，输出被告人构成的法条、罪名、刑期，最后输出被告人构成的罪名最终只输出一个合法的 JSON 对象，不能有多余文字。\n"
        "JSON 字段必须且仅有 3 个：\n"
        '  1) \"articles\": 法条编号整数列表，例如 [263, 264]\n'
        '  2) \"accusations\": 罪名字符串列表，例如 [\"抢劫罪\"]\n'
        '  3) \"imprisonment_months\": 总刑期（整数，月份）；无期=-1，死刑=-2，无拘役=0\n'
        "务必返回合法 JSON。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-based LJP baseline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="plain_prompt",
        choices=list(PROMPTS.keys()),
        help="Choose prompt template: plain_prompt / COT_prompt / LOT_prompt",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cail2018",
        choices=["cail2018", "cjo22"],
        help="Which eval logic/dataset to use.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=1698)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8000)
    return parser.parse_args()


def call_llm(
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
    max_tokens: int = 16000,
) -> str:
    """Minimal OpenAI-compatible chat completion call using the official client."""
    client = OpenAI(api_key=api_key or None, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    if not resp.choices:
        return ""
    msg = resp.choices[0].message
    return msg.content or ""


def _strip_code_fence(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    if not text:
        return text
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop leading ```
        stripped = stripped.lstrip("`")
        # remove optional language tag
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        # drop trailing ```
        stripped = stripped.rstrip("`").strip()
    return stripped


def _extract_imprisonment_months_from_text(text: str) -> Optional[float]:
    """Heuristic: grab `imprisonment_months: <num>` from raw text (even if fenced)."""
    if not text:
        return None
    m = re.search(r"imprisonment_months\"?\s*:\s*(-?\d+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def evaluate_case_strict_cjo(pred: Dict[str, str], truth: Dict) -> Dict[str, Optional[float]]:
    truth_meta = truth.get("meta", truth)
    truth_laws = {int(x) for x in truth_meta.get("relevant_articles", []) if str(x).isdigit()}
    truth_accs = {_normalize_acc(a) for a in truth_meta.get("accusation", []) if a}
    term = truth_meta.get("term_of_imprisonment", {})
    truth_imp = term.get("imprisonment")
    truth_life = term.get("life_imprisonment", False)
    truth_death = term.get("death_penalty", False)

    judgment_text = pred.get("judgment", "")
    judgment_obj = _parse_judgment_json(_strip_code_fence(judgment_text))

    if judgment_obj:
        pred_laws = {
            int(x)
            for x in judgment_obj.get("articles", [])
            if (isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()))
        }
        pred_accs = {_normalize_acc(str(x)) for x in judgment_obj.get("accusations", []) if x}
        imp_val = judgment_obj.get("imprisonment_months")
        if isinstance(imp_val, (int, float)):
            pred_imp, pred_life, pred_death = float(imp_val), False, False
        else:
            pred_imp, pred_life, pred_death = parse_imprisonment_safe(judgment_text)
    else:
        # Fallback: extract directly from judgment_text
        pred_laws = extract_articles(judgment_text)
        pred_accs = {_normalize_acc(a) for a in extract_accusations(judgment_text, truth_accs)}
        # Try to parse imprisonment from raw text first, then fallback to regex parser
        imp_guess = _extract_imprisonment_months_from_text(judgment_text)
        if imp_guess is not None:
            pred_imp, pred_life, pred_death = imp_guess, False, False
        else:
            pred_imp, pred_life, pred_death = parse_imprisonment_safe(judgment_text)

    metrics: Dict[str, Optional[float]] = {}
    metrics["law_acc"] = 1.0 if truth_laws and pred_laws == truth_laws else (None if not truth_laws else 0.0)
    metrics["acc_acc"] = 1.0 if truth_accs and pred_accs == truth_accs else (None if not truth_accs else 0.0)

    metrics["imprison_abs_err"] = None
    if truth_life or truth_death:
        if truth_life and pred_life:
            metrics["imprison_abs_err"] = 0.0
        elif truth_death and pred_death:
            metrics["imprison_abs_err"] = 0.0
    else:
        if isinstance(truth_imp, (int, float)) and isinstance(pred_imp, (int, float)):
            metrics["imprison_abs_err"] = abs(float(truth_imp) - float(pred_imp))

    truth_cls = None
    if isinstance(term.get("pt_cls"), (int, float)):
        truth_cls = int(term.get("pt_cls"))
    else:
        truth_cls = get_pt_cls(float(truth_imp) if isinstance(truth_imp, (int, float)) else None)
    pred_cls = get_pt_cls(pred_imp) if isinstance(pred_imp, (int, float)) else None

    if truth_cls is not None and pred_cls is not None:
        metrics["penalty_cls_acc"] = 1.0 if truth_cls == pred_cls else 0.0
    else:
        metrics["penalty_cls_acc"] = None

    return metrics


def evaluate_case_strict_cail(pred: Dict[str, str], truth: Dict) -> Dict[str, Optional[float]]:
    truth_meta = truth.get("meta", truth)
    truth_laws = {int(x) for x in truth_meta.get("relevant_articles", []) if str(x).isdigit()}
    truth_accs = {_normalize_acc(a) for a in truth_meta.get("accusation", []) if a}
    term = truth_meta.get("term_of_imprisonment", {})
    truth_imp = term.get("imprisonment")
    truth_cls = get_pt_cls(float(truth_imp)) if isinstance(truth_imp, (int, float)) else None

    judgment_text = pred.get("judgment", "")
    judgment_obj = _parse_judgment_json(_strip_code_fence(judgment_text))

    if judgment_obj:
        pred_laws = {
            int(x)
            for x in judgment_obj.get("articles", [])
            if (isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()))
        }
        pred_accs = {_normalize_acc(str(x)) for x in judgment_obj.get("accusations", []) if x}
        imp_val = judgment_obj.get("imprisonment_months")
        if isinstance(imp_val, (int, float)):
            pred_imp = float(imp_val)
        else:
            pred_imp, _, _ = parse_imprisonment_safe(judgment_text)
    else:
        pred_laws = extract_articles(judgment_text)
        pred_accs = {_normalize_acc(a) for a in extract_accusations(judgment_text, truth_accs)}
        imp_guess = _extract_imprisonment_months_from_text(judgment_text)
        if imp_guess is not None:
            pred_imp = imp_guess
        else:
            pred_imp, _, _ = parse_imprisonment_safe(judgment_text)

    pred_cls = get_pt_cls(pred_imp) if isinstance(pred_imp, (int, float)) else None

    metrics: Dict[str, Optional[float]] = {}
    metrics["law_acc"] = 1.0 if truth_laws and pred_laws == truth_laws else (None if not truth_laws else 0.0)
    metrics["acc_acc"] = 1.0 if truth_accs and pred_accs == truth_accs else (None if not truth_accs else 0.0)
    if truth_cls is None and pred_cls is None:
        metrics["penalty_cls_acc"] = None
    elif truth_cls is not None and pred_cls is not None:
        metrics["penalty_cls_acc"] = 1.0 if truth_cls == pred_cls else 0.0
    else:
        metrics["penalty_cls_acc"] = 0.0
    return metrics


def main() -> None:
    args = parse_args()
    system_prompt = PROMPTS[args.prompt_type]

    # Choose eval utilities based on task
    if args.task == "cail2018":
        dataset_path = Path(args.dataset or DEFAULT_DATASET_CAIL)
        out_dir = Path(args.output_dir or f"{DEFAULT_OUTPUT_DIR_CAIL}_{args.model}_{args.prompt_type}")
        metric_keys = ["law_acc", "acc_acc", "penalty_cls_acc"]
        iter_func = iter_testset_cail
        count_func = count_testset_cail
        eval_func = evaluate_case_strict_cail
    else:  # cjo22
        dataset_path = Path(args.dataset or DEFAULT_DATASET_CJO22)
        out_dir = Path(args.output_dir or f"{DEFAULT_OUTPUT_DIR_CJO22}_{args.model}_{args.prompt_type}")
        metric_keys = ["law_acc", "acc_acc", "imprison_abs_err", "penalty_cls_acc"]
        iter_func = iter_testset_cjo
        count_func = count_testset_cjo
        eval_func = evaluate_case_strict_cjo

    out_dir.mkdir(parents=True, exist_ok=True)

    total_raw = count_func(dataset_path) or 0
    if total_raw and args.limit is not None:
        total_eval = max(0, min(args.limit, total_raw - args.offset))
    elif total_raw:
        total_eval = max(0, total_raw - args.offset)
    else:
        total_eval = args.limit or 0

    records: List[Dict] = []
    metrics_accum = {k: [] for k in metric_keys}

    jsonl_path = out_dir / f"{args.prompt_type}_results_{args.model}.jsonl"
    jsonl_file = jsonl_path.open("w", encoding="utf-8")

    evaluated = 0
    for case_obj in iter_func(dataset_path, offset=args.offset, limit=args.limit):
        fact = case_obj.get("fact", "")
        user_prompt = f"案件事实：{fact}\n请按要求返回 JSON。"
        try:
            content = call_llm(
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"[error] caseID={case_obj.get('caseID')} -> {e}")
            content = ""

        pred = {
            "judgment": content,
            "law_resp": "",
            "acc_resp": "",
        }
        m = eval_func(pred, case_obj)
        for k in metrics_accum:
            if m.get(k) is not None:
                metrics_accum[k].append(m[k])

        record = {
            "caseID": case_obj.get("caseID"),
            "metrics": m,
            "pred": pred,
            "truth": case_obj.get("meta", {}),
        }
        records.append(record)
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        evaluated += 1
        progress = f"[{evaluated}/{total_eval}]" if total_eval else f"[{evaluated}]"
        print(f"{progress} caseID={case_obj.get('caseID')}, metrics={m}")

    jsonl_file.close()

    def avg(lst: List[float]) -> Optional[float]:
        return mean(lst) if lst else None

    summary = {k: avg(v) for k, v in metrics_accum.items()}
    print("\n===== Summary =====")
    print(f"Samples evaluated: {evaluated}")
    for k, v in summary.items():
        print(f"{k}: {v}")
    summary_path = out_dir / f"{args.prompt_type}_summary_{args.model}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved jsonl to {jsonl_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
