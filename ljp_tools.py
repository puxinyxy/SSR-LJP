"""
Utility functions: data loading, text chunking, embedding, and vector search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from camel.embeddings import OpenAICompatibleEmbedding


@dataclass
class TextItem:
    text: str
    meta: dict


@dataclass
class VectorIndex:
    texts: List[TextItem]
    vectors: List[np.ndarray]
    embedder: OpenAICompatibleEmbedding


def chunk_text(text: str, max_chars: int = 480) -> List[str]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        for i in range(0, len(para), max_chars):
            chunks.append(para[i : i + max_chars])
    return chunks


def load_law_articles(path: Path, max_chunks: int) -> List[TextItem]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw)
    limited = chunks[:max_chunks]
    return [
        TextItem(text=c, meta={"source": "law_article", "chunk_id": i})
        for i, c in enumerate(limited)
    ]


def load_candidates(path: Path, max_items: int) -> List[TextItem]:
    items: List[TextItem] = []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    # Detect JSON array vs JSONL
    if text.startswith("["):
        try:
            data = json.loads(text)
            iterable = data if isinstance(data, list) else []
        except json.JSONDecodeError:
            iterable = []
        for i, obj in enumerate(iterable):
            if i >= max_items:
                break
            if not isinstance(obj, dict):
                continue
            fact = obj.get("fact", "")
            meta = obj.get("meta", {})
            items.append(
                TextItem(
                    text=fact,
                    meta={
                        "case_id": obj.get("caseID", i),
                        "accusation": meta.get("accusation", []),
                        "relevant_articles": meta.get("relevant_articles", []),
                        "term_of_imprisonment": meta.get("term_of_imprisonment", {}),
                        "punish_of_money": meta.get("punish_of_money", 0),
                    },
                )
            )
        return items

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fact = obj.get("fact", "")
            meta = obj.get("meta", {})
            items.append(
                TextItem(
                    text=fact,
                    meta={
                        "case_id": obj.get("caseID", i),
                        "accusation": meta.get("accusation", []),
                        "relevant_articles": meta.get("relevant_articles", []),
                        "term_of_imprisonment": meta.get("term_of_imprisonment", {}),
                        "punish_of_money": meta.get("punish_of_money", 0),
                    },
                )
            )
    return items


def load_accusations(candidates: Sequence[TextItem]) -> List[TextItem]:
    seen = {}
    for item in candidates:
        for acc in item.meta.get("accusation", []):
            if acc not in seen:
                seen[acc] = item.text[:200]
    return [
        TextItem(text=f"{name} 示例: {snippet}", meta={"accusation": name})
        for name, snippet in seen.items()
    ]


def load_test_case(path: Path, case_id: int) -> dict:
    """
    Robust loader: supports JSON array or JSON Lines. Skips malformed lines.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict) and obj.get("caseID") == case_id:
                    return obj
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("caseID") == case_id:
                return obj
    raise ValueError(f"caseID {case_id} not found in {path}")


def batch_embed(
    embedder: OpenAICompatibleEmbedding,
    texts: Sequence[str],
    batch_size: int = 32,
    max_chars: int = 8000,
) -> List[np.ndarray]:
    results: List[np.ndarray] = []
    # DashScope compatible models often limit batch <= 10
    effective_batch = max(1, min(batch_size, 10))
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        clipped = [
            t[:max_chars] if isinstance(t, str) and len(t) > max_chars else t
            for t in batch
        ]
        vecs = embedder.embed_list(list(clipped))
        results.extend(np.array(v, dtype=np.float32) for v in vecs)
    return results


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def build_index(
    embedder: OpenAICompatibleEmbedding,
    items: Sequence[TextItem],
    batch_size: int = 32,
) -> VectorIndex:
    vectors = batch_embed(embedder, [item.text for item in items], batch_size)
    return VectorIndex(texts=list(items), vectors=vectors, embedder=embedder)


def search_index(
    index: VectorIndex,
    query: str,
    top_k: int = 3,
) -> List[TextItem]:
    q_vec = batch_embed(index.embedder, [query])[0]
    scored = [
        (cosine_sim(q_vec, vec), item) for vec, item in zip(index.vectors, index.texts)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def penalty_stats(cases: Sequence[TextItem]) -> str:
    """Summarize imprisonment (months) from retrieved cases."""
    imprison_months = []
    life_count = 0
    death_count = 0
    for c in cases:
        meta = c.meta or {}
        term = meta.get("term_of_imprisonment", {})
        if term.get("life_imprisonment"):
            life_count += 1
        if term.get("death_penalty"):
            death_count += 1
        imp = term.get("imprisonment")
        if isinstance(imp, (int, float)):
            imprison_months.append(float(imp))

    def summarize(nums: list[float]) -> str:
        if not nums:
            return "无数据"
        avg = sum(nums) / len(nums)
        return f"均值≈{avg:.1f}，最小={min(nums):.0f}，最大={max(nums):.0f}"

    imp_summary = summarize(imprison_months)
    flags = []
    if life_count:
        flags.append(f"无期:{life_count}")
    if death_count:
        flags.append(f"死刑:{death_count}")
    flag_txt = f"（{', '.join(flags)}）" if flags else ""
    return f"刑期统计: {imp_summary}{flag_txt}"


def penalty_stats_structured(records: Sequence[dict]) -> str:
    """
    Summarize imprisonment months from structured precedent outputs.
    Accepts a list of dicts containing sentence_months and penalty_factors.
    """
    imprison_months: list[float] = []
    life_count = 0
    death_count = 0
    suspended_count = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        months = rec.get("sentence_months")
        if isinstance(months, str):
            try:
                months = float(months.strip())
            except (TypeError, ValueError):
                months = None
        factors = rec.get("penalty_factors") if isinstance(rec.get("penalty_factors"), dict) else {}
        if months == -1:
            life_count += 1
            continue
        if months == -2:
            death_count += 1
            continue
        if isinstance(months, (int, float)):
            imprison_months.append(float(months))
        if isinstance(factors, dict) and factors.get("suspended") is True:
            suspended_count += 1

    def summarize(nums: list[float]) -> str:
        if not nums:
            return "无数据"
        avg = sum(nums) / len(nums)
        return f"均值≈{avg:.1f}，最小={min(nums):.0f}，最大={max(nums):.0f}"

    imp_summary = summarize(imprison_months)
    flags = []
    if life_count:
        flags.append(f"无期:{life_count}")
    if death_count:
        flags.append(f"死刑:{death_count}")
    if suspended_count:
        flags.append(f"缓刑:{suspended_count}")
    flag_txt = f"（{', '.join(flags)}）" if flags else ""
    return f"刑期统计: {imp_summary}{flag_txt}"


# ------------------------ Optional compression (LLMLingua-2) ------------------------ #
_LLM2_COMPRESSOR = None


def lingua_compress(
    text: str,
    rate: float = 0.5,
    min_chars: int = 300,
    token_limit: int = 16000,
    approx_chars_per_token: float = 2.0,
    force_tokens: list[str] | None = None,
) -> str:
    """
    Use LLMLingua-2 PromptCompressor to compress text to a given rate.
    - Only compress when len(text) >= min_chars AND estimated tokens exceed token_limit
    - On any exception or empty result, fall back to original text
    """
    if not isinstance(text, str) or not text:
        return text
    if len(text) < min_chars:
        return text
    try:
        est_tokens = len(text) / max(approx_chars_per_token, 0.5)
        if est_tokens <= token_limit:
            return text
    except Exception:
        pass

    try:
        from llmlingua import PromptCompressor  # type: ignore
        import torch
    except Exception:
        return text

    global _LLM2_COMPRESSOR
    if _LLM2_COMPRESSOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _LLM2_COMPRESSOR = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map=device,
            )
        except Exception:
            return text

    separators = force_tokens or ["\n", "。", "？", "！", "："]
    try:
        result = _LLM2_COMPRESSOR.compress_prompt(
            text,
            rate=rate,
            force_tokens=separators,
        )
        compressed = result.get("compressed_prompt") if isinstance(result, dict) else None
        if compressed:
            return compressed
    except Exception:
        return text
    return text
