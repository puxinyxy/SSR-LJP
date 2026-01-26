"""
LLM+RAG baseline for law article prediction (single LLM call).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
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
    "请从法律适用角度选择本案应当适用的主要刑法条文。"
    "最终只输出一个合法 JSON 对象，不要输出任何解释或 Markdown。"
    "JSON 仅包含字段 \"law_articles\"，值为字符串数组，"
    "例如：{ \"law_articles\": [\"刑法第264条\", \"刑法第340条\"] }。"
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
        return {"law_articles": [], "raw_output": text}
    if "law_articles" in obj:
        law_val = obj.get("law_articles")
    elif "articles" in obj:
        law_val = obj.get("articles")
    else:
        return {"law_articles": [], "raw_output": text}

    law_articles = _normalize_law_articles(law_val)
    if not law_articles and not (isinstance(law_val, list) and len(law_val) == 0):
        return {"law_articles": [], "raw_output": text}
    return {"law_articles": law_articles}


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
    parser = argparse.ArgumentParser(description="LLM+RAG baseline for law article prediction")
    parser.add_argument(
        "--dataset_path",
        "--dataset",
        dest="dataset_path",
        type=str,
        default="data/testset/testset.json",
    )
    parser.add_argument(
        "--law_dir",
        "--law-dir",
        dest="law_dir",
        type=str,
        default="data/law_articles",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--topk_law", type=int, default=TOP_K)
    parser.add_argument("--topk_case", type=int, default=TOP_K, help="Unused in law-only eval")
    parser.add_argument("--max-law-chars", type=int, default=400)
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
        run_name="eval_llm_rag_law",
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

    embedder = build_embedder(args.embedding_model)
    law_index = SimpleVectorIndex(embedder, law_texts)
    logger.info(
        "config dataset_path=%s law_dir=%s topk_law=%s",
        args.dataset_path,
        args.law_dir,
        args.topk_law,
    )

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("embedding_output") / f"llm_rag_law_{ts}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc_scores: List[float] = []
    ma_p_scores: List[float] = []
    ma_r_scores: List[float] = []
    ma_f_scores: List[float] = []

    with output_path.open("w", encoding="utf-8") as jf:
        for idx, case in enumerate(data, 1):
            case_id = _get_case_id(case)
            fact = case.get("fact", "")
            prompt_fact = fact
            if args.max_fact_chars > 0:
                prompt_fact = prompt_fact[: args.max_fact_chars]
            law_hits = law_index.search(fact, args.topk_law)
            law_block = _format_law_candidates(law_hits, args.max_law_chars)

            user_prompt = (
                f"[案件事实]：{prompt_fact}\n"
                f"[检索到的法条候选]：\n{law_block}\n"
                "[任务说明]：请只在这些候选条文中进行选择，不要创造列表以外的新条文。"
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

            pred_set = _to_article_set(prediction.get("law_articles", []))
            gold_set = _to_article_set(gold.get("law_articles", []))
            precision, recall, f1 = _precision_recall_f1(pred_set, gold_set)
            acc = 1.0 if pred_set == gold_set else 0.0
            acc_scores.append(acc)
            ma_p_scores.append(precision)
            ma_r_scores.append(recall)
            ma_f_scores.append(f1)

            record = {
                "case_id": case_id,
                "fact": fact,
                "prediction": prediction,
                "gold": gold,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(data)}] case_id={case_id}")
            logger.info("case_done case_id=%s", case_id)

        summary = {
            "record_type": "summary",
            "num_samples": len(data),
            "metrics": {
                "acc": _safe_mean(acc_scores),
                "Ma-P": _safe_mean(ma_p_scores),
                "Ma-R": _safe_mean(ma_r_scores),
                "Ma-F": _safe_mean(ma_f_scores),
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
    print(f"Saved jsonl to {output_path}")


if __name__ == "__main__":
    main()
