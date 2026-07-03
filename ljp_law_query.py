"""Deterministic query rewriting for law-article retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\r?\n+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,。！？!?；;])")
_SPACE_RE = re.compile(r"\s+")

_EVIDENCE_MARKERS = (
    "上述事实",
    "下列证据",
    "证据证实",
    "证人证言",
    "证人的证言",
    "辨认笔录",
    "勘验笔录",
    "检查笔录",
    "搜查笔录",
    "鉴定意见",
    "检验报告",
    "庭审质证",
    "当庭质证",
    "户籍资料",
    "户籍证明",
    "抓获经过",
    "到案经过",
    "发破案经过",
    "案件来源",
    "公诉机关出示",
    "公诉人出示",
    "本院予以确认",
    "足以认定",
)

_PROCEDURE_MARKERS = (
    "经审理查明",
    "公诉机关指控",
    "人民检察院指控",
    "本院认为",
    "开庭审理",
    "审理过程中",
    "提起公诉",
    "建议判处",
    "量刑建议",
    "依法判处",
)

_CIRCUMSTANCE_MARKERS = (
    "自首",
    "坦白",
    "如实供述",
    "退赃",
    "退赔",
    "赔偿",
    "谅解",
    "认罪认罚",
    "自愿认罪",
    "累犯",
    "前科",
    "立功",
    "缓刑",
    "从轻处罚",
    "从重处罚",
    "减轻处罚",
    "免除处罚",
)

_RERANK_INSTRUCTION = (
    "任务：寻找最直接规定本案犯罪构成和罪名的主要刑法条文。\n"
    "优先依据：犯罪行为、犯罪对象、数额或数量、行为方式、主观目的、危害结果。\n"
    "不要因为案件出现自首、坦白、退赃、谅解、累犯、立功、缓刑、没收、"
    "从轻或从重情节，就优先选择一般量刑条文。\n"
    "案件犯罪要件：\n"
)


@dataclass(frozen=True)
class LawRetrievalQueries:
    dense_query: str
    lexical_query: str
    rerank_query: str
    circumstance_query: str


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text)).strip()


def _compact_for_matching(text: str) -> str:
    return _SPACE_RE.sub("", str(text))


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    compact = _compact_for_matching(text)
    return any(_compact_for_matching(marker) in compact for marker in markers)


def _strip_leading_procedure_text(clause: str) -> str:
    cleaned = clause.strip()
    for marker in _PROCEDURE_MARKERS:
        marker_pattern = r"\s*".join(re.escape(ch) for ch in marker)
        cleaned, count = re.subn(
            rf"^{marker_pattern}",
            "",
            cleaned,
            count=1,
        )
        if count:
            cleaned = cleaned.lstrip("：:，,。；; ")
    return cleaned


def prepare_law_retrieval_queries(
    fact: str,
    max_chars: int = 2000,
) -> LawRetrievalQueries:
    """Split one case fact into branch-specific deterministic queries."""
    dense_query = _normalize(fact)
    if max_chars > 0 and len(dense_query) > max_chars:
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        dense_query = f"{dense_query[:head_chars]} {dense_query[-tail_chars:]}"

    offense_clauses: list[str] = []
    circumstance_clauses: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(str(fact)):
        sentence = _normalize(sentence)
        if not sentence or _contains_any(sentence, _EVIDENCE_MARKERS):
            continue
        for raw_clause in _CLAUSE_SPLIT_RE.split(sentence):
            clause = _strip_leading_procedure_text(_normalize(raw_clause))
            if not clause:
                continue
            if _contains_any(clause, _EVIDENCE_MARKERS):
                continue
            if _contains_any(clause, _CIRCUMSTANCE_MARKERS):
                circumstance_clauses.append(clause)
                continue
            if _compact_for_matching(clause) in {
                _compact_for_matching(marker)
                for marker in _PROCEDURE_MARKERS
            }:
                continue
            offense_clauses.append(clause)

    lexical_query = _normalize(" ".join(offense_clauses))
    circumstance_query = _normalize(" ".join(circumstance_clauses))
    if not lexical_query:
        lexical_query = dense_query
    if max_chars > 0:
        lexical_query = lexical_query[:max_chars]
        circumstance_query = circumstance_query[:max_chars]

    rerank_query = f"{_RERANK_INSTRUCTION}{lexical_query}"
    return LawRetrievalQueries(
        dense_query=dense_query,
        lexical_query=lexical_query,
        rerank_query=rerank_query,
        circumstance_query=circumstance_query,
    )


def coerce_law_retrieval_queries(
    query: str | LawRetrievalQueries,
    max_chars: int = 2000,
) -> LawRetrievalQueries:
    if isinstance(query, LawRetrievalQueries):
        return query
    normalized = _normalize(query)
    if max_chars > 0 and len(normalized) > max_chars:
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        normalized = f"{normalized[:head_chars]} {normalized[-tail_chars:]}"
    return LawRetrievalQueries(
        dense_query=normalized,
        lexical_query=normalized,
        rerank_query=normalized,
        circumstance_query="",
    )
