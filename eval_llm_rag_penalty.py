"""
LLM+RAG baseline for penalty prediction (single LLM call).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI
from logger_utils import log_metrics, setup_run_logger

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camel.embeddings import OpenAICompatibleEmbedding

from ljp_config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    TOP_K,
)
from simple_vector_index import SimpleVectorIndex


SYSTEM_PROMPT = (
    "你是一名经验丰富的中国刑事审判法官。你会看到一个案件事实、若干候选法律条文以及一些历史相似案例。"
    "请综合这些信息给出合理的量刑建议（有期徒刑月数）。"
)
SYSTEM_PROMPT += (
    "\nReturn ONLY a valid JSON object with keys "
    "\"law_articles\" (array of strings) and \"imprisonment_months\" (number). "
    "Select law_articles from the candidate law articles. "
    "Example: {\"law_articles\": [\"???264?\"], \"imprisonment_months\": 12}."
)


def load_dataset(path: str | Path) -> List[dict]:
    """Load JSONL (or JSON array) into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [obj for obj in data if isinstance(obj, dict)]
    except json.JSONDecodeError:
        pass

    items: List[dict] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def load_law_articles(law_dir: str | Path) -> List[str]:
    """Load law articles from *.txt files (one article per file)."""
    law_path = Path(law_dir)
    if not law_path.exists():
        raise FileNotFoundError(f"Law dir not found: {law_path}")
    files = sorted(law_path.glob("*.txt"))
    articles: List[str] = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            articles.append(text)
    return articles


def build_embedder(embedding_model: str) -> OpenAICompatibleEmbedding:
    return OpenAICompatibleEmbedding(
        model_type=embedding_model,
        api_key=EMBEDDING_API_KEY,
        url=EMBEDDING_BASE_URL,
    )


def _format_join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return "、".join(parts)
    return str(value).strip()


def load_precedents(path: str | Path) -> List[str]:
    """
    Load precedent cases and format as concatenated strings for retrieval.

    Format:
    【事实】{fact} 【裁判结论】罪名：{accusations}；适用法条：{law_articles}；有期徒刑：{imprisonment_months}个月
    """
    data = load_dataset(path)
    results: List[str] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        fact = obj.get("fact", "")
        accusations = obj.get("accusations")
        law_articles = obj.get("law_articles")
        imprisonment = obj.get("imprisonment_months")

        if accusations is None:
            meta = obj.get("meta", {})
            accusations = meta.get("accusation")
        if law_articles is None:
            meta = obj.get("meta", {})
            law_articles = meta.get("relevant_articles")
        if imprisonment is None:
            meta = obj.get("meta", {})
            term = meta.get("term_of_imprisonment", {})
            imprisonment = term.get("imprisonment")

        acc_text = _format_join(accusations) or "未知"
        law_text = _format_join(law_articles) or "未知"
        if imprisonment is None or (isinstance(imprisonment, str) and not imprisonment.strip()):
            imp_text = "未知"
        else:
            imp_text = str(imprisonment).strip()

        fact_text = fact if isinstance(fact, str) else str(fact)
        if not fact_text.strip():
            continue
        results.append(
            f"【事实】{fact_text} 【裁判结论】罪名：{acc_text}；适用法条：{law_text}；有期徒刑：{imp_text}个月"
        )
    return results


def _clean_json_text(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
        stripped = stripped.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
        return stripped
    if stripped.lower().startswith("json"):
        return stripped[4:].lstrip()
    return stripped


def _safe_json_load(text: str) -> Optional[dict]:
    if not text:
        return None
    stripped = _clean_json_text(text)
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


def _normalize_months(val: Any) -> Optional[int]:
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


def _get_case_pt_cls(obj: dict) -> Optional[int]:
    meta = obj.get("meta", {})
    pt_cls = meta.get("pt_cls")
    if isinstance(pt_cls, (int, float)):
        return int(pt_cls)
    months = None
    if "imprisonment_months" in obj:
        months = _normalize_months(obj.get("imprisonment_months"))
    if months is None:
        term = meta.get("term_of_imprisonment", {})
        months = _normalize_months(term.get("imprisonment"))
    return get_pt_cls(months) if months is not None else None


def _normalize_law_articles(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(v).strip() for v in val if v is not None and str(v).strip()]
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    return []


def _fullwidth_to_halfwidth(s: str) -> str:
    """Convert fullwidth digits to halfwidth digits."""
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


def _extract_article_numbers(text: str) -> set[int]:
    text_hw = _fullwidth_to_halfwidth(text)
    nums = {int(n) for n in re.findall(r"\d{1,4}", text_hw)}

    pattern = rf"[\u7b2c]?([{CN_NUM}]+)[\u6761\u7bc7]?"
    for m in re.findall(pattern, text):
        try:
            val = _chinese_num_to_int(m)
            if val:
                nums.add(val)
        except Exception:
            continue
    return nums


def _to_article_set(values: Sequence[Any]) -> set[int]:
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


def _precision_recall_f1(pred_set: set[int], gold_set: set[int]) -> tuple[float, float, float]:
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    inter = len(pred_set & gold_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _macro_from_stats(
    stats: Sequence[dict],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for s in stats:
        tp = int(s.get("tp", 0))
        fp = int(s.get("fp", 0))
        fn = int(s.get("fn", 0))
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return _safe_mean(precisions), _safe_mean(recalls), _safe_mean(f1s)


def _get_case_id(obj: dict) -> Any:
    return obj.get("case_id", obj.get("caseID", obj.get("id")))


def _get_case_law_articles(obj: dict) -> List[str]:
    if "law_articles" in obj:
        return _normalize_law_articles(obj.get("law_articles"))
    meta = obj.get("meta", {})
    return _normalize_law_articles(meta.get("relevant_articles", []))


def _get_case_imprisonment(obj: dict) -> int:
    if "imprisonment_months" in obj:
        val = _normalize_months(obj.get("imprisonment_months"))
        return val if val is not None else 0
    meta = obj.get("meta", {})
    term = meta.get("term_of_imprisonment", {})
    val = _normalize_months(term.get("imprisonment"))
    return val if val is not None else 0


def _parse_prediction(text: str) -> Dict[str, Any]:
    obj = _safe_json_load(text)
    if not isinstance(obj, dict):
        return {
            "law_articles": [],
            "imprisonment_months": 0,
            "raw_output": text,
        }
    pred = dict(obj)
    law_val = pred.get("law_articles", pred.get("articles"))
    pred["law_articles"] = _normalize_law_articles(law_val)
    months = _normalize_months(pred.get("imprisonment_months"))
    pred["imprisonment_months"] = months if months is not None else 0
    return pred


def _format_law_candidates(hits: Sequence[str], max_chars: int) -> str:
    if not hits:
        return "无"
    lines = []
    for i, hit in enumerate(hits, 1):
        snippet = hit.strip().replace("\n", " ")
        if max_chars > 0:
            snippet = snippet[:max_chars]
        lines.append(f"候选{i}：{snippet}")
    return "\n".join(lines)


def _format_precedent_candidates(hits: Sequence[str], max_chars: int) -> str:
    if not hits:
        return "无"
    lines = []
    for i, hit in enumerate(hits, 1):
        fact = hit.strip().replace("\n", " ")
        if max_chars > 0:
            fact = fact[:max_chars]
        lines.append(f"案例{i}：{fact}")
    return "\n".join(lines)


def call_llm(
    model_name: str,
    api_key: str,
    api_base: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = OpenAI(api_key=api_key or None, base_url=api_base or None)
    resp = client.chat.completions.create(
        model=model_name,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM+RAG baseline for penalty prediction")
    parser.add_argument(
        "--dataset_path",
        "--dataset",
        dest="dataset_path",
        type=str,
        default="data/testset/testset.json",
    )
    parser.add_argument(
        "--precedent_file",
        "--precedent-path",
        type=str,
        default="data/candidates/precedent_case.json",
        help="JSONL precedents file for retrieval",
    )
    parser.add_argument(
        "--law_dir",
        "--law-dir",
        dest="law_dir",
        type=str,
        default="data/law_articles",
    )
    parser.add_argument("--output_path", type=str, default="embedding_output/llm_rag_penalty.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--topk_law", type=int, default=TOP_K)
    parser.add_argument("--topk_case", type=int, default=TOP_K)
    parser.add_argument("--max-law-chars", type=int, default=400)
    parser.add_argument("--max-prec-chars", type=int, default=400)
    parser.add_argument("--max-fact-chars", type=int, default=2000)
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--model_name", "--model", dest="model_name", type=str, default=LLM_MODEL)
    parser.add_argument("--api_base", "--base-url", dest="api_base", type=str, default=LLM_BASE_URL)
    parser.add_argument("--api_key", type=str, default=LLM_API_KEY)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_tokens", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, run_dir, run_id = setup_run_logger(
        run_name="eval_llm_rag_penalty",
        args=vars(args),
        extra={"cwd": str(Path.cwd())},
    )
    data = load_dataset(args.dataset_path)
    if args.offset > 0:
        data = data[args.offset :]
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    law_texts = load_law_articles(args.law_dir)
    if not law_texts:
        raise ValueError(f"No law articles found in {args.law_dir}")
    precedent_texts = load_precedents(args.precedent_file)
    if not precedent_texts:
        raise ValueError(f"No precedents found in {args.precedent_file}")

    embedder = build_embedder(args.embedding_model)
    law_index = SimpleVectorIndex(embedder, law_texts)
    case_index = SimpleVectorIndex(embedder, precedent_texts)
    logger.info(
        "config dataset_path=%s precedent_file=%s law_dir=%s topk_case=%s topk_law=%s",
        args.dataset_path,
        args.precedent_file,
        args.law_dir,
        args.topk_case,
        args.topk_law,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = 10
    cls_stats = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(num_classes)]
    total_eval = 0
    correct = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as jf:
        for idx, case in enumerate(data, 1):
            case_id = _get_case_id(case)
            fact = case.get("fact", "")
            prompt_fact = fact
            if args.max_fact_chars > 0:
                prompt_fact = prompt_fact[: args.max_fact_chars]
            law_hits = law_index.search(fact, args.topk_law)
            prec_hits = case_index.search(fact, args.topk_case)
            law_block = _format_law_candidates(law_hits, args.max_law_chars)
            prec_block = _format_precedent_candidates(prec_hits, args.max_prec_chars)

            user_prompt = (
                f"[案件事实]：{prompt_fact}\n"
                f"[检索到的法条候选]：\n{law_block}\n"
                f"[检索到的相似案例]：\n{prec_block}\n"
                "[任务说明]：请参考本案事实、上述候选法条和相似案例，在法律允许范围内给出本案被告人应判处的有期徒刑月数。"
            )
            user_prompt += (
                "\n[Output] Return ONLY a JSON object with keys "
                "\"law_articles\" (array of strings) and \"imprisonment_months\" (number). "
                "Select law_articles from the candidate law articles above."
            )

            try:
                content = call_llm(
                    model_name=args.model_name,
                    api_key=args.api_key,
                    api_base=args.api_base,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                print(f"[error] case_id={case_id} -> {e}")
                content = ""

            prediction = _parse_prediction(content)
            gold = {
                "law_articles": _get_case_law_articles(case),
                "imprisonment_months": _get_case_imprisonment(case),
            }
            gold_cls = _get_case_pt_cls(case)
            pred_months = _normalize_months(prediction.get("imprisonment_months"))
            pred_cls = get_pt_cls(pred_months) if pred_months is not None else None

            if gold_cls is None:
                skipped += 1
            else:
                total_eval += 1
                if pred_cls == gold_cls:
                    correct += 1
                    cls_stats[gold_cls]["tp"] += 1
                else:
                    cls_stats[gold_cls]["fn"] += 1
                    if pred_cls is not None and 0 <= pred_cls < num_classes:
                        cls_stats[pred_cls]["fp"] += 1
            record = {
                "case_id": case_id,
                "fact": fact,
                "prediction": prediction,
                "gold": gold,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(data)}] case_id={case_id}")
            logger.info("case_done case_id=%s", case_id)

        ma_p, ma_r, ma_f = _macro_from_stats(cls_stats)
        acc = (correct / total_eval) if total_eval else None
        summary = {
            "record_type": "summary",
            "num_samples": len(data),
            "num_eval_samples": total_eval,
            "num_skipped": skipped,
            "metrics": {
                "acc": acc,
                "Ma-P": ma_p,
                "Ma-R": ma_r,
                "Ma-F": ma_f,
            },
        }
        jf.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log_metrics(logger, summary.get("metrics", {}), prefix="summary")

    if data:
        print("===== Metrics =====")
        print(f"acc: {summary['metrics']['acc']}")
        print(f"Ma-P: {summary['metrics']['Ma-P']}")
        print(f"Ma-R: {summary['metrics']['Ma-R']}")
        print(f"Ma-F: {summary['metrics']['Ma-F']}")
        print(f"eval_samples: {summary.get('num_eval_samples')}")
        print(f"skipped: {summary.get('num_skipped')}")
    print(f"Saved jsonl to {output_path}")


if __name__ == "__main__":
    main()
