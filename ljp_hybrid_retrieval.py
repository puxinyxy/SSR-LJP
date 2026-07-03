"""
Hybrid retrieval utilities for the LJP experiments.

The dense, BM25, and keyword branches all operate on the same item list.
The stable item identity is the original list index, which is also the key
used by RRF fusion.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ljp_law_query import (
    LawRetrievalQueries,
    coerce_law_retrieval_queries,
)


_QUERY_SPAN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_()（）-]{2,}")
_LAW_TITLE_RE = re.compile(r"【([^】]+)】")


@dataclass(frozen=True)
class HybridSearchConfig:
    dense_top_k: int = 50
    bm25_top_k: int = 50
    keyword_top_k: int = 30
    join_top_k: int = 50
    rrf_k: float = 60.0
    dense_weight: float = 2.0
    bm25_weight: float = 0.7
    keyword_weight: float = 0.3
    fusion_mode: str = "rrf"
    dense_score_weight: float = 0.65
    bm25_score_weight: float = 0.25
    keyword_score_weight: float = 0.10
    rerank_score_weight: float = 0.20
    dense_anchor: bool = False
    dense_margin_threshold: float = 0.02
    dense_override_threshold: float = 0.08
    legal_aware_rerank: bool = False
    lexical_include_numeric: bool = False
    query_max_chars: int = 2000
    use_rerank: bool = False
    rerank_top_k: int = 30
    rerank_model: str = "qwen3-rerank"
    rerank_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    rerank_api_key: str | None = None
    rerank_timeout: int = 60


@dataclass
class KeywordRecord:
    item_id: int
    text: str
    meta_text: str
    searchable_text: str
    tokens: set[str]


def _require_hybrid_deps() -> tuple[Any, Any, Any]:
    try:
        import jieba  # type: ignore
        from rank_bm25 import BM25Okapi  # type: ignore
        from rapidfuzz import fuzz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Hybrid LJP retrieval requires jieba, rank-bm25, and rapidfuzz. "
            "Install them with: pip install -r requirements-ljp.txt"
        ) from exc
    return jieba, BM25Okapi, fuzz


def prepare_retrieval_query(text: str, max_chars: int = 2000) -> str:
    normalized = " ".join(str(text).split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{normalized[:head_chars]} {normalized[-tail_chars:]}"


def tokenize_zh(text: str, include_numeric: bool = False) -> list[str]:
    jieba, _, _ = _require_hybrid_deps()
    tokens: list[str] = []
    for raw_token in jieba.cut(str(text)):
        token = raw_token.strip().lower()
        if not token:
            continue
        if not include_numeric and re.fullmatch(r"[\d.,:%年月日时分秒元万亿吨克千]+", token):
            continue
        if not re.search(
            r"[\u4e00-\u9fffA-Za-z0-9]" if include_numeric else r"[\u4e00-\u9fffA-Za-z]",
            token,
        ):
            continue
        tokens.append(token)
    return tokens


def extract_query_terms(query: str) -> list[str]:
    normalized = " ".join(str(query).split())
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize_zh(normalized):
        if len(token) > 1 and token not in seen:
            terms.append(token)
            seen.add(token)
    for span in _QUERY_SPAN_RE.findall(normalized):
        key = span.lower()
        if key not in seen:
            terms.append(key)
            seen.add(key)
    return terms


def meta_to_text(meta: Any) -> str:
    if meta is None:
        return ""
    if isinstance(meta, dict):
        parts: list[str] = []
        for key, value in sorted(meta.items(), key=lambda item: str(item[0])):
            parts.append(str(key))
            parts.append(meta_to_text(value))
        return " ".join(part for part in parts if part)
    if isinstance(meta, (list, tuple, set)):
        return " ".join(meta_to_text(value) for value in meta)
    return str(meta)


_SEMANTIC_META_KEYS = {
    "accusation",
    "accusations",
    "charge",
    "charges",
    "keywords",
    "title",
}


def lexical_meta_to_text(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    parts: list[str] = []
    for key, value in meta.items():
        if str(key).lower() not in _SEMANTIC_META_KEYS:
            continue
        parts.append(meta_to_text(value))
    return " ".join(part for part in parts if part)


def build_keyword_records(
    texts: Sequence[str],
    metas: Sequence[Any],
    include_numeric: bool = False,
) -> list[KeywordRecord]:
    records: list[KeywordRecord] = []
    for item_id, text in enumerate(texts):
        meta_text = lexical_meta_to_text(metas[item_id] if item_id < len(metas) else {})
        searchable_text = f"{meta_text}\n{text}".strip()
        records.append(
            KeywordRecord(
                item_id=item_id,
                text=text,
                meta_text=meta_text,
                searchable_text=searchable_text,
                tokens=set(tokenize_zh(searchable_text, include_numeric=include_numeric)),
            )
        )
    return records


def keyword_scores(
    query: str,
    records: Sequence[KeywordRecord],
    include_numeric: bool = False,
) -> list[tuple[int, float]]:
    _, _, fuzz = _require_hybrid_deps()
    query_text = " ".join(str(query).split())
    query_l = query_text.lower()
    terms = extract_query_terms(query_text)
    query_tokens = set(tokenize_zh(query_text, include_numeric=include_numeric))
    scored: list[tuple[int, float]] = []

    for record in records:
        text_l = record.text.lower()
        meta_l = record.meta_text.lower()
        exact_meta_hit = 0.0
        exact_text_hit = 0.0
        for term in terms:
            if term and term in meta_l:
                exact_meta_hit += 1.0
            if term and term in text_l:
                exact_text_hit += 1.0

        token_overlap = 0.0
        if query_tokens:
            token_overlap = len(query_tokens & record.tokens) / max(len(query_tokens), 1)

        fuzzy_raw = fuzz.partial_ratio(query_l, record.searchable_text[:1000].lower()) / 100.0
        fuzzy_bonus = fuzzy_raw if fuzzy_raw >= 0.5 else 0.0
        score = exact_meta_hit * 2.0 + exact_text_hit * 2.0 + token_overlap * 2.0 + fuzzy_bonus * 0.5
        if score > 0:
            scored.append((record.item_id, float(score)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def rrf_fuse(
    hit_lists: Sequence[Sequence[int]],
    top_k: int,
    rrf_k: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []
    scores: dict[int, float] = {}
    weights = weights or [1.0] * len(hit_lists)
    for list_idx, item_ids in enumerate(hit_lists):
        weight = float(weights[list_idx]) if list_idx < len(weights) else 1.0
        if weight <= 0:
            continue
        for rank, item_id in enumerate(item_ids, start=1):
            scores[int(item_id)] = scores.get(int(item_id), 0.0) + weight / (float(rrf_k) + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


def normalize_scored_hits(
    scored_hits: Sequence[tuple[int, float]],
    log_scale: bool = False,
) -> list[dict[str, float | int]]:
    if not scored_hits:
        return []
    transformed = [
        float(np.log1p(max(score, 0.0))) if log_scale else float(score)
        for _, score in scored_hits
    ]
    min_score = min(transformed)
    max_score = max(transformed)
    span = max_score - min_score
    normalized: list[dict[str, float | int]] = []
    for (item_id, raw_score), value in zip(scored_hits, transformed):
        norm = 0.0 if span <= 1e-12 else (value - min_score) / span
        normalized.append(
            {
                "item_id": int(item_id),
                "raw_score": float(raw_score),
                "normalized_score": float(norm),
            }
        )
    return normalized


def score_fuse(
    dense_hits: Sequence[tuple[int, float]],
    bm25_hits: Sequence[tuple[int, float]],
    keyword_hits: Sequence[tuple[int, float]],
    top_k: int,
    dense_weight: float = 0.65,
    bm25_weight: float = 0.25,
    keyword_weight: float = 0.10,
) -> tuple[list[tuple[int, float]], dict[str, list[dict[str, float | int]]]]:
    branches = {
        "dense": normalize_scored_hits(dense_hits),
        "bm25": normalize_scored_hits(bm25_hits, log_scale=True),
        "keyword": normalize_scored_hits(keyword_hits, log_scale=True),
    }
    weights = {
        "dense": float(dense_weight),
        "bm25": float(bm25_weight),
        "keyword": float(keyword_weight),
    }
    scores: dict[int, float] = {}
    branch_ranks: dict[str, dict[int, int]] = {}
    for branch_name, records in branches.items():
        weight = weights[branch_name]
        branch_ranks[branch_name] = {
            int(record["item_id"]): rank
            for rank, record in enumerate(records, 1)
        }
        if weight <= 0:
            continue
        for record in records:
            item_id = int(record["item_id"])
            scores[item_id] = scores.get(item_id, 0.0) + weight * float(
                record["normalized_score"]
            )
    missing_rank = 10**9
    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            branch_ranks["dense"].get(item[0], missing_rank),
            branch_ranks["bm25"].get(item[0], missing_rank),
            branch_ranks["keyword"].get(item[0], missing_rank),
            item[0],
        ),
    )
    return ranked[:top_k], branches


def _parse_rerank_response(result: dict[str, Any]) -> list[tuple[int, float]]:
    raw = result.get("results")
    if raw is None and isinstance(result.get("output"), dict):
        raw = result["output"].get("results")
    if raw is None and isinstance(result.get("data"), list):
        raw = result["data"]
    if not isinstance(raw, list):
        return []
    parsed: list[tuple[int, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index", item.get("document_index"))
        score = item.get("relevance_score", item.get("score"))
        if index is not None and score is not None:
            parsed.append((int(index), float(score)))
    return parsed


def _compress_law_candidate(text: str, meta: Any, max_chars: int = 600) -> str:
    normalized = " ".join(str(text).split())
    article_id = meta.get("article_id") if isinstance(meta, dict) else None
    title_match = _LAW_TITLE_RE.search(normalized)
    title = title_match.group(1).strip() if title_match else ""
    core = _LAW_TITLE_RE.sub("", normalized, count=1).strip()
    if max_chars > 0:
        core = core[:max_chars]
    parts: list[str] = []
    if article_id is not None:
        parts.append(f"法条编号：第{article_id}条")
    if title:
        parts.append(f"标题：{title}")
    parts.append(f"核心规定：{core}")
    return "\n".join(parts)


def _remote_rerank_item_ids(
    query: str,
    candidate_ids: Sequence[int],
    texts: Sequence[str],
    config: HybridSearchConfig,
    top_k: int,
    metas: Sequence[Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": bool(config.use_rerank),
        "candidate_pool_size": 0,
        "returned_count": 0,
        "ranking_complete": False,
        "ranked_results": [],
        "success": False,
        "fallback_reason": None,
    }
    if not config.use_rerank or not candidate_ids:
        debug["fallback_reason"] = "disabled_or_empty"
        return list(candidate_ids[:top_k]), debug
    if not config.rerank_api_key:
        debug["fallback_reason"] = "missing_api_key"
        return list(candidate_ids[:top_k]), debug

    candidate_ids = list(candidate_ids[: max(top_k, config.rerank_top_k)])
    debug["candidate_pool_size"] = len(candidate_ids)
    documents = []
    for item_id in candidate_ids:
        meta = metas[item_id] if metas is not None and item_id < len(metas) else {}
        if config.legal_aware_rerank:
            documents.append(_compress_law_candidate(texts[item_id], meta))
        else:
            documents.append(texts[item_id])
    payload = {
        "model": config.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": len(candidate_ids),
        "return_documents": False,
    }
    debug["query"] = query
    debug["legal_aware"] = bool(config.legal_aware_rerank)
    req = urllib.request.Request(
        config.rerank_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.rerank_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.rerank_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        debug["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return list(candidate_ids[:top_k]), debug

    parsed = _parse_rerank_response(result)
    reranked: list[int] = []
    ranked_results: list[dict[str, Any]] = []
    seen_item_ids: set[int] = set()
    for index, score in parsed:
        if 0 <= index < len(candidate_ids):
            item_id = candidate_ids[index]
            if item_id in seen_item_ids:
                continue
            seen_item_ids.add(item_id)
            reranked.append(item_id)
            ranked_results.append(
                {
                    "rank": len(ranked_results) + 1,
                    "item_id": item_id,
                    "score": score,
                }
            )
    if not reranked:
        debug["fallback_reason"] = "empty_rerank_result"
        return list(candidate_ids[:top_k]), debug
    debug["returned_count"] = len(ranked_results)
    debug["ranking_complete"] = len(ranked_results) >= len(candidate_ids)
    debug["ranked_results"] = ranked_results
    debug["success"] = True
    return reranked[:top_k], debug


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _embed_one(embedder: Any, text: str, max_chars: int = 8000) -> np.ndarray:
    clipped = text[:max_chars] if isinstance(text, str) and len(text) > max_chars else text
    emb = embedder.embed_list([clipped])[0]
    return np.array(emb, dtype=np.float32)


def _embed_texts(
    embedder: Any,
    texts: Sequence[str],
    batch_size: int = 10,
    max_chars: int = 8000,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    effective_batch = max(1, min(batch_size, 10))
    for i in range(0, len(texts), effective_batch):
        batch = texts[i : i + effective_batch]
        clipped = [t[:max_chars] if isinstance(t, str) and len(t) > max_chars else t for t in batch]
        vectors.extend(np.array(emb, dtype=np.float32) for emb in embedder.embed_list(list(clipped)))
    return vectors


def _top_item_ids_from_scores(scores: Sequence[float], top_k: int, positive_only: bool = False) -> list[int]:
    if top_k <= 0 or len(scores) == 0:
        return []
    scored = [
        (idx, float(score))
        for idx, score in enumerate(scores)
        if not positive_only or float(score) > 0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [idx for idx, _ in scored[: min(top_k, len(scored))]]


def _top_scored_hits_from_scores(
    scores: Sequence[float],
    top_k: int,
    positive_only: bool = False,
) -> list[tuple[int, float]]:
    if top_k <= 0 or len(scores) == 0:
        return []
    scored = [
        (idx, float(score))
        for idx, score in enumerate(scores)
        if not positive_only or float(score) > 0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[: min(top_k, len(scored))]


def _items_digest(texts: Sequence[str], metas: Sequence[Any]) -> str:
    digest = hashlib.md5()
    digest.update(str(len(texts)).encode("utf-8"))
    for idx, text in enumerate(texts):
        digest.update(str(idx).encode("utf-8"))
        digest.update(str(text).encode("utf-8", errors="ignore"))
        meta = metas[idx] if idx < len(metas) else {}
        digest.update(json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _cache_signature(meta: dict[str, Any], config: HybridSearchConfig, texts: Sequence[str], metas: Sequence[Any]) -> str:
    payload = {
        "meta": meta,
        "config": {
            "dense_top_k": config.dense_top_k,
            "bm25_top_k": config.bm25_top_k,
            "keyword_top_k": config.keyword_top_k,
            "join_top_k": config.join_top_k,
            "rrf_k": config.rrf_k,
            "dense_weight": config.dense_weight,
            "bm25_weight": config.bm25_weight,
            "keyword_weight": config.keyword_weight,
            "fusion_mode": config.fusion_mode,
            "dense_score_weight": config.dense_score_weight,
            "bm25_score_weight": config.bm25_score_weight,
            "keyword_score_weight": config.keyword_score_weight,
            "rerank_score_weight": config.rerank_score_weight,
            "dense_anchor": config.dense_anchor,
            "dense_margin_threshold": config.dense_margin_threshold,
            "dense_override_threshold": config.dense_override_threshold,
            "legal_aware_rerank": config.legal_aware_rerank,
            "lexical_include_numeric": config.lexical_include_numeric,
            "query_max_chars": config.query_max_chars,
        },
        "tokenizer": "jieba",
        "items_digest": _items_digest(texts, metas),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


class _HybridCore:
    def __init__(
        self,
        texts: Sequence[str],
        metas: Sequence[Any],
        vectors: Sequence[np.ndarray],
        embedder: Any,
        config: HybridSearchConfig,
        cache_dir: Path | None = None,
        cache_prefix: str = "hybrid",
        cache_meta: dict[str, Any] | None = None,
    ) -> None:
        self.texts = list(texts)
        self.metas = list(metas)
        self.vectors = [np.array(vec, dtype=np.float32) for vec in vectors]
        self.embedder = embedder
        self.config = config
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix
        self.cache_meta = cache_meta or {}
        self.bm25: Any = None
        self.keyword_records: list[KeywordRecord] = []
        self.last_search_debug: dict[str, Any] = {}
        self._build_or_load_lexical_indexes()

    def _build_or_load_lexical_indexes(self) -> None:
        _, BM25Okapi, _ = _require_hybrid_deps()
        signature = _cache_signature(self.cache_meta, self.config, self.texts, self.metas)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = f"{self.cache_prefix}_{signature[:12]}"
            bm25_path = self.cache_dir / f"{cache_name}_bm25.pkl"
            keyword_path = self.cache_dir / f"{cache_name}_keyword.pkl"
            meta_path = self.cache_dir / f"{cache_name}_hybrid_meta.json"
            if bm25_path.exists() and keyword_path.exists() and meta_path.exists():
                try:
                    cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if cached_meta.get("signature") == signature and cached_meta.get("num_items") == len(self.texts):
                        with bm25_path.open("rb") as f:
                            self.bm25 = pickle.load(f)
                        with keyword_path.open("rb") as f:
                            self.keyword_records = pickle.load(f)
                        return
                except Exception:
                    pass

        tokenized = [
            tokenize_zh(
                f"{lexical_meta_to_text(meta)}\n{text}",
                include_numeric=self.config.lexical_include_numeric,
            )
            for text, meta in zip(self.texts, self.metas)
        ]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self.keyword_records = build_keyword_records(
            self.texts,
            self.metas,
            include_numeric=self.config.lexical_include_numeric,
        )

        if self.cache_dir is not None:
            with bm25_path.open("wb") as f:
                pickle.dump(self.bm25, f)
            with keyword_path.open("wb") as f:
                pickle.dump(self.keyword_records, f)
            meta_out = {
                **self.cache_meta,
                "signature": signature,
                "num_items": len(self.texts),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _dense_hits(self, query: str) -> list[tuple[int, float]]:
        if not self.texts or not self.vectors:
            return []
        q_vec = _embed_one(self.embedder, query)
        scores = [_cosine_sim(q_vec, vec) for vec in self.vectors]
        return _top_scored_hits_from_scores(scores, self.config.dense_top_k)

    def _bm25_hits(self, query: str) -> list[tuple[int, float]]:
        if self.bm25 is None or not self.texts:
            return []
        scores = self.bm25.get_scores(
            tokenize_zh(
                query,
                include_numeric=self.config.lexical_include_numeric,
            )
        )
        return _top_scored_hits_from_scores(
            scores,
            self.config.bm25_top_k,
            positive_only=True,
        )

    def _keyword_hits(self, query: str) -> list[tuple[int, float]]:
        scored = keyword_scores(
            query,
            self.keyword_records,
            include_numeric=self.config.lexical_include_numeric,
        )
        return scored[: self.config.keyword_top_k]

    def search_item_ids(
        self,
        query: str | LawRetrievalQueries,
        top_k: int,
    ) -> list[int]:
        if top_k <= 0:
            return []
        queries = coerce_law_retrieval_queries(query, self.config.query_max_chars)
        score_mode = self.config.fusion_mode == "score"
        if self.config.fusion_mode not in {"rrf", "score"}:
            raise ValueError(f"Unknown fusion_mode: {self.config.fusion_mode}")

        dense_enabled = (
            self.config.dense_score_weight > 0
            if score_mode
            else self.config.dense_weight > 0
        )
        bm25_enabled = (
            self.config.bm25_score_weight > 0
            if score_mode
            else self.config.bm25_weight > 0
        )
        keyword_enabled = (
            self.config.keyword_score_weight > 0
            if score_mode
            else self.config.keyword_weight > 0
        )
        dense_hits = self._dense_hits(queries.dense_query) if dense_enabled else []
        bm25_hits = self._bm25_hits(queries.lexical_query) if bm25_enabled else []
        keyword_hits = (
            self._keyword_hits(queries.lexical_query)
            if keyword_enabled
            else []
        )
        legal_rerank_pool_k = (
            self.config.rerank_top_k
            if self.config.use_rerank and self.config.legal_aware_rerank
            else 0
        )
        join_top_k = max(
            top_k,
            self.config.join_top_k,
            legal_rerank_pool_k,
        )

        if score_mode:
            fused, branch_scores = score_fuse(
                dense_hits,
                bm25_hits,
                keyword_hits,
                top_k=join_top_k,
                dense_weight=self.config.dense_score_weight,
                bm25_weight=self.config.bm25_score_weight,
                keyword_weight=self.config.keyword_score_weight,
            )
        else:
            fused = rrf_fuse(
                [
                    [item_id for item_id, _ in dense_hits],
                    [item_id for item_id, _ in bm25_hits],
                    [item_id for item_id, _ in keyword_hits],
                ],
                top_k=join_top_k,
                rrf_k=self.config.rrf_k,
                weights=[
                    self.config.dense_weight,
                    self.config.bm25_weight,
                    self.config.keyword_weight,
                ],
            )
            branch_scores = {
                "dense": normalize_scored_hits(dense_hits),
                "bm25": normalize_scored_hits(bm25_hits, log_scale=True),
                "keyword": normalize_scored_hits(keyword_hits, log_scale=True),
            }

        fused_ids = [item_id for item_id, _ in fused]
        reranked_ids, rerank_debug = _remote_rerank_item_ids(
            queries.rerank_query,
            fused_ids,
            self.texts,
            self.config,
            top_k,
            metas=self.metas,
        )
        ranked_results = rerank_debug.get("ranked_results")
        if isinstance(ranked_results, list):
            for result in ranked_results:
                if not isinstance(result, dict):
                    continue
                item_id = result.get("item_id")
                if isinstance(item_id, int) and 0 <= item_id < len(self.metas):
                    result["meta"] = self.metas[item_id]
        rerank_pairs = []
        if isinstance(ranked_results, list):
            rerank_pairs = [
                (int(result["item_id"]), float(result["score"]))
                for result in ranked_results
                if isinstance(result, dict)
                and isinstance(result.get("item_id"), int)
                and isinstance(result.get("score"), (int, float))
            ]
        normalized_rerank = normalize_scored_hits(rerank_pairs)
        branch_scores["rerank"] = normalized_rerank

        dense_margin = None
        if len(dense_hits) >= 2:
            dense_margin = float(dense_hits[0][1] - dense_hits[1][1])

        anchor_debug = {
            "enabled": bool(self.config.dense_anchor and score_mode),
            "applied": False,
            "dense_top1": dense_hits[0][0] if dense_hits else None,
            "dense_margin": dense_margin,
            "fused_advantage": None,
        }
        final_scores: list[dict[str, float | int]] = []
        if score_mode:
            rerank_norm_map = {
                int(record["item_id"]): float(record["normalized_score"])
                for record in normalized_rerank
            }
            rerank_raw_map = {
                int(record["item_id"]): float(record["raw_score"])
                for record in normalized_rerank
            }
            rerank_weight = (
                min(max(float(self.config.rerank_score_weight), 0.0), 1.0)
                if rerank_debug.get("success")
                else 0.0
            )
            for item_id, pre_score in fused:
                rerank_score = rerank_norm_map.get(item_id, 0.0)
                final_score = (
                    (1.0 - rerank_weight) * float(pre_score)
                    + rerank_weight * rerank_score
                )
                final_scores.append(
                    {
                        "item_id": item_id,
                        "pre_score": float(pre_score),
                        "rerank_raw_score": rerank_raw_map.get(item_id, 0.0),
                        "rerank_normalized_score": rerank_score,
                        "final_score": final_score,
                    }
                )
            pre_rank = {item_id: rank for rank, (item_id, _) in enumerate(fused)}
            final_scores.sort(
                key=lambda record: (
                    -float(record["final_score"]),
                    pre_rank.get(int(record["item_id"]), 10**9),
                    int(record["item_id"]),
                )
            )

            if (
                self.config.dense_anchor
                and dense_hits
                and dense_margin is not None
                and dense_margin >= self.config.dense_margin_threshold
                and final_scores
            ):
                dense_top1 = dense_hits[0][0]
                final_top1 = int(final_scores[0]["item_id"])
                score_map = {
                    int(record["item_id"]): float(record["final_score"])
                    for record in final_scores
                }
                if dense_top1 in score_map and final_top1 != dense_top1:
                    advantage = score_map[final_top1] - score_map[dense_top1]
                    anchor_debug["fused_advantage"] = advantage
                    if advantage < self.config.dense_override_threshold:
                        dense_record = next(
                            record
                            for record in final_scores
                            if int(record["item_id"]) == dense_top1
                        )
                        final_scores = [
                            dense_record,
                            *[
                                record
                                for record in final_scores
                                if int(record["item_id"]) != dense_top1
                            ],
                        ]
                        anchor_debug["applied"] = True
            final_ids = [
                int(record["item_id"])
                for record in final_scores[:top_k]
            ]
        else:
            final_ids = reranked_ids

        self.last_search_debug = {
            "query_chars": len(queries.dense_query),
            "queries": {
                "dense": queries.dense_query,
                "lexical": queries.lexical_query,
                "rerank": queries.rerank_query,
                "circumstance": queries.circumstance_query,
            },
            "fusion_mode": self.config.fusion_mode,
            "dense_hits": [item_id for item_id, _ in dense_hits],
            "bm25_hits": [item_id for item_id, _ in bm25_hits],
            "keyword_hits": [item_id for item_id, _ in keyword_hits],
            "branch_scores": branch_scores,
            "fused": [
                {"item_id": item_id, "score": score}
                for item_id, score in fused
            ],
            "pre_fusion_scores": [
                {"item_id": item_id, "score": score}
                for item_id, score in fused
            ],
            "final_scores": final_scores,
            "final_ids": final_ids,
            "dense_margin": dense_margin,
            "dense_anchor_applied": anchor_debug["applied"],
            "weights": {
                "rrf": {
                    "dense": self.config.dense_weight,
                    "bm25": self.config.bm25_weight,
                    "keyword": self.config.keyword_weight,
                },
                "score": {
                    "dense": self.config.dense_score_weight,
                    "bm25": self.config.bm25_score_weight,
                    "keyword": self.config.keyword_score_weight,
                    "rerank": self.config.rerank_score_weight,
                },
            },
            "dense_anchor": anchor_debug,
            "rerank": rerank_debug,
        }
        for record in self.last_search_debug["final_scores"]:
            item_id = record.get("item_id")
            if isinstance(item_id, int) and 0 <= item_id < len(self.metas):
                record["meta"] = self.metas[item_id]
        return final_ids


class HybridIndex:
    def __init__(
        self,
        items: Sequence[Any],
        vectors: Sequence[np.ndarray],
        embedder: Any,
        config: HybridSearchConfig | None = None,
        cache_dir: Path | None = None,
        cache_prefix: str = "hybrid",
        cache_meta: dict[str, Any] | None = None,
    ) -> None:
        self.texts = list(items)
        self.vectors = list(vectors)
        self.embedder = embedder
        self.config = config or HybridSearchConfig()
        item_texts = [str(getattr(item, "text", "")) for item in self.texts]
        item_metas = [getattr(item, "meta", {}) for item in self.texts]
        self._core = _HybridCore(
            item_texts,
            item_metas,
            self.vectors,
            embedder,
            self.config,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_meta=cache_meta,
        )

    def search(
        self,
        query: str | LawRetrievalQueries,
        top_k: int,
    ) -> list[Any]:
        return [self.texts[item_id] for item_id in self._core.search_item_ids(query, top_k)]

    @property
    def last_search_debug(self) -> dict[str, Any]:
        return self._core.last_search_debug


class HybridStringIndex:
    def __init__(
        self,
        embedder: Any,
        texts: Sequence[str],
        config: HybridSearchConfig | None = None,
        batch_size: int = 10,
        cache_dir: Path | None = None,
        cache_prefix: str = "simple_hybrid",
        cache_meta: dict[str, Any] | None = None,
        meta_builder: Callable[[int, str], Any] | None = None,
    ) -> None:
        self.emb_model = embedder
        self.texts = list(texts)
        self.config = config or HybridSearchConfig()
        metas = [
            meta_builder(idx, text) if meta_builder is not None else {}
            for idx, text in enumerate(self.texts)
        ]
        self._vectors = _embed_texts(embedder, self.texts, batch_size=batch_size)
        self._core = _HybridCore(
            self.texts,
            metas,
            self._vectors,
            embedder,
            self.config,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_meta=cache_meta,
        )

    def search(
        self,
        query: str | LawRetrievalQueries,
        top_k: int,
    ) -> list[str]:
        return [self.texts[item_id] for item_id in self._core.search_item_ids(query, top_k)]

    @property
    def last_search_debug(self) -> dict[str, Any]:
        return self._core.last_search_debug
