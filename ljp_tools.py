"""
Utility functions: data loading, text chunking, embedding, and vector search.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import re

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


_CN_NUM = "一二三四五六七八九十百千零两"
_FW_DIGITS = "０１２３４５６７８９"
_HW_DIGITS = "0123456789"


def _fullwidth_to_halfwidth(s: str) -> str:
    return s.translate(str.maketrans(_FW_DIGITS, _HW_DIGITS))


def _chinese_num_to_int(s: str) -> int:
    digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {"十": 10, "百": 100, "千": 1000}
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


def _extract_article_id(text: str) -> int | None:
    if not text:
        return None
    norm = _fullwidth_to_halfwidth(text)
    m = re.search(r"第\s*(\d{1,4})\s*条", norm)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m = re.search(rf"第\s*([{_CN_NUM}]+)\s*条", text)
    if m:
        try:
            val = _chinese_num_to_int(m.group(1))
            return val if val else None
        except Exception:
            return None
    return None


def extract_article_id(text: str) -> int | None:
    return _extract_article_id(text)


def extract_article_key(text: str) -> str | None:
    if not text:
        return None
    norm = _fullwidth_to_halfwidth(text)
    pattern = (
        rf"^\s*第\s*(\d{{1,4}}|[{_CN_NUM}]+)\s*条"
        rf"(?:\s*之\s*(\d{{1,3}}|[{_CN_NUM}]+))?"
    )
    match = re.search(pattern, norm)
    if not match:
        return None

    def _part_to_int(value: str | None) -> int | None:
        if not value:
            return None
        if value.isdigit():
            return int(value)
        parsed = _chinese_num_to_int(value)
        return parsed if parsed > 0 else None

    article_id = _part_to_int(match.group(1))
    if article_id is None:
        return None
    suffix = _part_to_int(match.group(2))
    return str(article_id) if suffix is None else f"{article_id}-{suffix}"


def _group_law_document(text: str) -> List[str]:
    articles: List[str] = []
    current_lines: List[str] = []
    found_header = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if extract_article_key(line) is not None:
            found_header = True
            if current_lines:
                articles.append("\n".join(current_lines))
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)

    if current_lines:
        articles.append("\n".join(current_lines))
    if found_header:
        return articles

    stripped = text.strip()
    return [stripped] if stripped else []


def load_grouped_law_texts(path: str | Path) -> List[str]:
    law_path = Path(path)
    if not law_path.exists():
        raise FileNotFoundError(f"Law path not found: {law_path}")
    files = [law_path] if law_path.is_file() else sorted(law_path.glob("*.txt"))
    articles: List[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        articles.extend(_group_law_document(text))
    return articles


def load_law_articles(path: Path, max_chunks: int) -> List[TextItem]:
    limited = load_grouped_law_texts(path)[:max_chunks]
    return [
        TextItem(
            text=article,
            meta={
                "source": "law_article",
                "item_id": i,
                "article_id": _extract_article_id(article),
                "article_key": extract_article_key(article),
            },
        )
        for i, article in enumerate(limited)
    ]


def load_candidates(path: Path, max_items: int | None) -> List[TextItem]:
    items: List[TextItem] = []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    max_items_limit = max_items if isinstance(max_items, int) and max_items > 0 else None
    # Detect JSON array vs JSONL
    if text.startswith("["):
        try:
            data = json.loads(text)
            iterable = data if isinstance(data, list) else []
        except json.JSONDecodeError:
            iterable = []
        for i, obj in enumerate(iterable):
            if max_items_limit is not None and i >= max_items_limit:
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
            if max_items_limit is not None and i >= max_items_limit:
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


def _meta_signature(meta: dict) -> str:
    payload = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def build_index_cached(
    embedder: OpenAICompatibleEmbedding,
    items: Sequence[TextItem],
    batch_size: int,
    cache_dir: Path,
    cache_prefix: str,
    meta: dict,
) -> VectorIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _meta_signature(meta)
    cache_name = f"{cache_prefix}_{signature[:12]}"
    vec_path = cache_dir / f"{cache_name}.npy"
    meta_path = cache_dir / f"{cache_name}.json"

    if vec_path.exists() and meta_path.exists():
        try:
            cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached_meta.get("signature") == signature and cached_meta.get("num_items") == len(items):
                vecs = np.load(vec_path)
                if vecs.ndim == 2 and vecs.shape[0] == len(items):
                    vectors = [vecs[i] for i in range(vecs.shape[0])]
                    return VectorIndex(texts=list(items), vectors=vectors, embedder=embedder)
        except Exception:
            pass

    vectors = batch_embed(embedder, [item.text for item in items], batch_size)
    if vectors:
        vec_arr = np.stack(vectors).astype(np.float32, copy=False)
    else:
        vec_arr = np.zeros((0, 0), dtype=np.float32)
    np.save(vec_path, vec_arr)
    meta_out: dict[str, Any] = dict(meta)
    meta_out.update(
        {
            "signature": signature,
            "num_items": len(items),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return VectorIndex(texts=list(items), vectors=vectors, embedder=embedder)


def build_retrieval_index_cached(
    embedder: OpenAICompatibleEmbedding,
    items: Sequence[TextItem],
    batch_size: int,
    cache_dir: Path,
    cache_prefix: str,
    meta: dict,
    retrieval_mode: str = "hybrid",
    dense_top_k: int = 50,
    bm25_top_k: int = 50,
    keyword_top_k: int = 30,
    join_top_k: int = 50,
    rrf_k: float = 60.0,
    dense_weight: float = 2.0,
    bm25_weight: float = 0.7,
    keyword_weight: float = 0.3,
    fusion_mode: str = "rrf",
    dense_score_weight: float = 0.65,
    bm25_score_weight: float = 0.25,
    keyword_score_weight: float = 0.10,
    rerank_score_weight: float = 0.20,
    dense_anchor: bool = False,
    dense_margin_threshold: float = 0.02,
    dense_override_threshold: float = 0.08,
    legal_aware_rerank: bool = False,
    lexical_include_numeric: bool = False,
    query_max_chars: int = 2000,
    use_rerank: bool = False,
    rerank_top_k: int = 30,
    rerank_model: str = "qwen3-rerank",
    rerank_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
    rerank_api_key: str | None = None,
    rerank_timeout: int = 60,
    retrieval_cache_dir: Path | None = None,
) -> Any:
    dense_index = build_index_cached(
        embedder,
        items,
        batch_size=batch_size,
        cache_dir=cache_dir,
        cache_prefix=cache_prefix,
        meta=meta,
    )
    if retrieval_mode == "embedding":
        return dense_index
    if retrieval_mode != "hybrid":
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    from ljp_hybrid_retrieval import HybridIndex, HybridSearchConfig

    lexical_cache_dir = retrieval_cache_dir or (cache_dir.parent / "retrieval_cache")
    hybrid_meta = dict(meta)
    hybrid_meta.update(
        {
            "retrieval_mode": retrieval_mode,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "keyword_top_k": keyword_top_k,
            "join_top_k": join_top_k,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "keyword_weight": keyword_weight,
            "fusion_mode": fusion_mode,
            "dense_score_weight": dense_score_weight,
            "bm25_score_weight": bm25_score_weight,
            "keyword_score_weight": keyword_score_weight,
            "rerank_score_weight": rerank_score_weight,
            "dense_anchor": dense_anchor,
            "dense_margin_threshold": dense_margin_threshold,
            "dense_override_threshold": dense_override_threshold,
            "legal_aware_rerank": legal_aware_rerank,
            "lexical_include_numeric": lexical_include_numeric,
            "query_max_chars": query_max_chars,
            "use_rerank": use_rerank,
            "rerank_top_k": rerank_top_k,
            "rerank_model": rerank_model,
            "rerank_url": rerank_url,
        }
    )
    return HybridIndex(
        dense_index.texts,
        dense_index.vectors,
        dense_index.embedder,
        config=HybridSearchConfig(
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            keyword_top_k=keyword_top_k,
            join_top_k=join_top_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            keyword_weight=keyword_weight,
            fusion_mode=fusion_mode,
            dense_score_weight=dense_score_weight,
            bm25_score_weight=bm25_score_weight,
            keyword_score_weight=keyword_score_weight,
            rerank_score_weight=rerank_score_weight,
            dense_anchor=dense_anchor,
            dense_margin_threshold=dense_margin_threshold,
            dense_override_threshold=dense_override_threshold,
            legal_aware_rerank=legal_aware_rerank,
            lexical_include_numeric=lexical_include_numeric,
            query_max_chars=query_max_chars,
            use_rerank=use_rerank,
            rerank_top_k=rerank_top_k,
            rerank_model=rerank_model,
            rerank_url=rerank_url,
            rerank_api_key=rerank_api_key,
            rerank_timeout=rerank_timeout,
        ),
        cache_dir=lexical_cache_dir,
        cache_prefix=f"{cache_prefix}_hybrid",
        cache_meta=hybrid_meta,
    )


def search_index(
    index: Any,
    query: Any,
    top_k: int = 3,
) -> List[TextItem]:
    search_method = getattr(index, "search", None)
    if callable(search_method):
        return list(search_method(query, top_k))

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
