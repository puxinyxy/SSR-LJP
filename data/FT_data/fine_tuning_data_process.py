"""
Convert the CAIL2018 training set into OpenAI chat fine-tuning format and
split it into multiple <=200MB jsonl files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _project_root() -> Path:
    """Return repository root so we can import ljp_agents for the prompt."""
    return Path(__file__).resolve().parents[2]


def load_system_prompt() -> str:
    """
    Try to reuse the judgment system prompt from ljp_agents for consistency
    with the LJP workflow; fall back to a minimal version if import fails.
    """
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        from ljp_agents import JUDGMENT_SYSTEM as prompt  # type: ignore
        return prompt
    except Exception:
        return (
            "你是判决智能体。根据案件事实直接输出 JSON，对象仅包含 articles "
            "（整数列表）、accusations（字符串列表）、imprisonment_months（整数）"
            "无期=-1，死刑=-2，缓刑/不判实刑=0）、reason（1-2 句话理由）。"
        )


def normalize_articles(meta: Dict) -> List[int]:
    arts = meta.get("relevant_articles", [])
    normalized: List[int] = []
    for a in arts or []:
        try:
            val = int(str(a).strip())
        except (TypeError, ValueError):
            continue
        normalized.append(val)
    return normalized


def normalize_accusations(meta: Dict) -> List[str]:
    accs = meta.get("accusation", [])
    return [str(a) for a in accs if a]


def normalize_imprisonment(term: Dict) -> int:
    """
    Map imprisonment info to months integer expected by the judge prompt.
    -2: death; -1: life; 0: none/unclear; >0: months
    """
    if not isinstance(term, dict):
        return 0
    if term.get("death_penalty"):
        return -2
    if term.get("life_imprisonment"):
        return -1
    imp = term.get("imprisonment")
    if isinstance(imp, (int, float)):
        return int(round(imp))
    try:
        return int(str(imp).strip())
    except Exception:
        return 0


def penalty_phrase(months: int) -> str:
    if months == -2:
        return "判处死刑"
    if months == -1:
        return "判处无期徒刑"
    if months == 0:
        return "未判处实刑或无法确定刑期"
    years, remain = divmod(max(months, 0), 12)
    parts = []
    if years:
        parts.append(f"{years}年")
    if remain:
        parts.append(f"{remain}个月")
    span = "".join(parts) if parts else f"{months}个月"
    return f"判处有期徒刑{span}"


def build_reason(accusations: List[str], articles: List[int], imprisonment_months: int) -> str:
    acc_text = "、".join(accusations) if accusations else "相关罪名"
    law_text = "、".join(f"第{a}条" for a in articles) if articles else "相关法条"
    penalty_text = penalty_phrase(imprisonment_months)
    return f"根据案件事实，认定构成{acc_text}，适用刑法{law_text}，{penalty_text}。"


def case_to_conversation(case_obj: Dict, system_prompt: str) -> Dict:
    fact = str(case_obj.get("fact", "")).strip()
    meta = case_obj.get("meta", {}) or {}
    articles = normalize_articles(meta)
    accusations = normalize_accusations(meta)
    imprisonment_months = normalize_imprisonment(meta.get("term_of_imprisonment", {}))
    assistant_payload = {
        "articles": articles,
        "accusations": accusations,
        "imprisonment_months": imprisonment_months,
        "reason": build_reason(accusations, articles, imprisonment_months),
    }
    user_content = (
        f"案件事实：{fact}\n"
        "请依据上述事实，直接给出只包含 articles、accusations、imprisonment_months、reason "
        "四个字段的 JSON。"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ]
    }


def iter_cases(path: Path, limit: Optional[int] = None) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def write_splits(
    cases: Iterable[Dict],
    output_dir: Path,
    system_prompt: str,
    max_bytes: int,
) -> tuple[int, int]:
    """
    Write conversations to jsonl splits capped by max_bytes.
    Returns (num_files, num_samples).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_idx = 1
    num_samples = 0
    current_path = output_dir / f"Trainingdata_part_{file_idx:03d}.jsonl"
    current_file = current_path.open("w", encoding="utf-8")
    current_bytes = 0

    def _new_file(idx: int):
        return output_dir / f"Trainingdata_part_{idx:03d}.jsonl"

    for case in cases:
        conv = case_to_conversation(case, system_prompt)
        line = json.dumps(conv, ensure_ascii=False)
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
        if current_bytes + line_bytes > max_bytes and current_bytes > 0:
            current_file.close()
            file_idx += 1
            current_path = _new_file(file_idx)
            current_file = current_path.open("w", encoding="utf-8")
            current_bytes = 0
        current_file.write(line + "\n")
        current_bytes += line_bytes
        num_samples += 1

    current_file.close()
    # Remove empty trailing file if no samples were written.
    if num_samples == 0 and current_path.exists():
        current_path.unlink()
        file_idx = 0
    return file_idx, num_samples


def guess_input_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path)
    candidates = [
        Path(__file__).resolve().parent / "cail_2018" / "train.data",
        Path(__file__).resolve().parent / "cail_2018" / "train.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError("No train.data/train.json found under data/FT_data/cail_2018/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CAIL2018 train set to chat fine-tuning jsonl and split under 200MB."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CAIL2018 train json/jsonl (default: data/FT_data/cail_2018/train.data or train.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "cail_2018" / "ft_splits"),
        help="Directory to write split jsonl files.",
    )
    parser.add_argument(
        "--max-mb",
        type=int,
        default=190,
        help="Max file size per jsonl split in MB (use <200 to stay under fine-tune limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples for a dry run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = guess_input_path(args.input)
    output_dir = Path(args.output_dir)
    max_bytes = int(args.max_mb * 1024 * 1024)
    system_prompt = load_system_prompt()

    print(f"Reading cases from: {input_path}")
    print(f"Writing splits to : {output_dir} (max {args.max_mb} MB each)")
    file_count, sample_count = write_splits(
        cases=iter_cases(input_path, limit=args.limit),
        output_dir=output_dir,
        system_prompt=system_prompt,
        max_bytes=max_bytes,
    )
    print(f"Done. Wrote {sample_count} samples into {file_count} file(s).")


if __name__ == "__main__":
    main()
