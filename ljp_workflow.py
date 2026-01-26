"""
Orchestrates the LJP multi-agent workflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from camel.embeddings import OpenAICompatibleEmbedding

from ljp_agents import (
    ACC_SYSTEM,
    JUDGMENT_SYSTEM,
    LAW_SYSTEM,
    PRECEDENT_SYSTEM,
    make_agent,
    make_llm,
)
from ljp_config import (
    EMBED_BATCH,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from ljp_tools import (
    build_index,
    load_accusations,
    load_candidates,
    load_law_articles,
    load_test_case,
    penalty_stats,
    penalty_stats_structured,
    search_index,
    TextItem,
)


@dataclass
class PipelineResources:
    embedder: OpenAICompatibleEmbedding
    law_index: Any
    acc_index: Any
    cand_index: Any
    law_items: List[TextItem]
    accusation_items: List[TextItem]
    candidates: List[TextItem]
    agents: Dict[str, Any]


def build_resources(args, candidates_path: Path | None = None) -> PipelineResources:
    data_dir = Path("data")
    law_path = Path(r"G:\\graduate_1\\Code\\Camel\\data\\meta\\laws.txt")
    if not law_path.exists():
        raise FileNotFoundError(f"Law articles file not found: {law_path}")
    if candidates_path is None:
        if getattr(args, "candidates_path", None):
            candidates_path = Path(args.candidates_path)
        else:
            candidates_path = data_dir / "candidates" / "precedent_case.json"

    # Load data
    law_items = load_law_articles(law_path, max_chunks=args.max_law_chunks)
    candidates = load_candidates(candidates_path, max_items=args.max_candidates)
    accusation_items = load_accusations(candidates)

    # Embedding model is locked to config
    embedder = OpenAICompatibleEmbedding(
        model_type=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        url=EMBEDDING_BASE_URL,
    )
    law_index = build_index(embedder, law_items, batch_size=args.embed_batch)
    acc_index = build_index(embedder, accusation_items, batch_size=args.embed_batch)
    cand_index = build_index(embedder, candidates, batch_size=args.embed_batch)

    # LLM for agents
    llm = make_llm(
        model_name=args.model or LLM_MODEL,
        api_key=args.api_key or LLM_API_KEY,
        base_url=args.base_url or LLM_BASE_URL,
    )
    agents = {
        "law": make_agent(llm, LAW_SYSTEM),
        "acc": make_agent(llm, ACC_SYSTEM),
        "prec": make_agent(llm, PRECEDENT_SYSTEM),
        "judge": make_agent(llm, JUDGMENT_SYSTEM),
    }

    return PipelineResources(
        embedder=embedder,
        law_index=law_index,
        acc_index=acc_index,
        cand_index=cand_index,
        law_items=law_items,
        accusation_items=accusation_items,
        candidates=candidates,
        agents=agents,
    )


def predict_case(case_fact: str, resources: PipelineResources, top_k: int):
    """Predict law, accusation, similar cases, and judgment."""
    # Reset agent memories to avoid cross-case accumulation
    for ag_key in ("law", "acc", "prec", "judge"):
        if hasattr(resources.agents.get(ag_key), "reset"):
            try:
                resources.agents[ag_key].reset()
            except Exception:
                pass

    # Retrieve candidates
    law_hits = search_index(resources.law_index, case_fact, top_k)
    acc_hits = search_index(resources.acc_index, case_fact, top_k)
    prelim_laws = [f"[{h.meta['chunk_id']}] {h.text[:150]}" for h in law_hits]
    prelim_acc = [h.meta["accusation"] for h in acc_hits]

    cand_query = (
        case_fact[:800]
        + "\n候选法条: "
        + ", ".join(prelim_laws[:3])
        + "\n候选罪名: "
        + ", ".join(prelim_acc)
    )
    cand_hits = search_index(resources.cand_index, cand_query, top_k)

    # Agents
    agents = resources.agents
    law_agent = agents["law"]
    acc_agent = agents["acc"]
    prec_agent = agents["prec"]
    judge_agent = agents["judge"]

    law_prompt = (
        f"案件事实：{case_fact}\n"
        f"候选法条片段（按相似度排序）：\n" + "\n".join(prelim_laws)
    )
    acc_prompt = (
        f"案件事实：{case_fact}\n"
        f"候选罪名示例：\n" + "\n".join(f"- {h.text}" for h in acc_hits)
    )

    def _format_term(meta: dict) -> str:
        term = meta.get("term_of_imprisonment", {}) if isinstance(meta, dict) else {}
        if term.get("death_penalty"):
            return "死刑"
        if term.get("life_imprisonment"):
            return "无期"
        imp = term.get("imprisonment")
        if isinstance(imp, (int, float)):
            return f"{imp}个月"
        return "未知"

    cand_blocks = []
    for h in cand_hits:
        meta = h.meta or {}
        raw_text = h.text.strip()
        compressed_text = raw_text
        block = (
            f"- case_id={meta.get('case_id')} | 罪名={meta.get('accusation')} | "
            f"法条={meta.get('relevant_articles')} | 量刑信息={_format_term(meta)}\n"
            f"  案例原文/摘要：{compressed_text}"
        )
        cand_blocks.append(block)
    prec_prompt = (
        f"案件事实：{case_fact}\n"
        f"预测法条：{prelim_laws}\n"
        f"预测罪名：{prelim_acc}\n"
        f"候选案例（逐条抽取量刑因子并按 JSON 数组输出，仅输出 JSON）：\n"
        + "\n\n".join(cand_blocks)
    )

    law_step = law_agent.step(law_prompt)
    acc_step = acc_agent.step(acc_prompt)
    prec_step = prec_agent.step(prec_prompt)

    if not law_step.msgs or not acc_step.msgs or not prec_step.msgs:
        raise ValueError("Empty response from one of the agents (law/acc/prec)")

    law_resp = law_step.msgs[0].content
    acc_resp = acc_step.msgs[0].content
    prec_resp = prec_step.msgs[0].content

    prec_structured: List[Dict[str, Any]] = []
    if prec_resp:
        try:
            parsed = json.loads(prec_resp)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            prec_structured = [p for p in parsed if isinstance(p, dict)]
        if not prec_structured:
            try:
                start = prec_resp.find("[")
                end = prec_resp.rfind("]")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(prec_resp[start : end + 1])
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    if isinstance(parsed, list):
                        prec_structured = [p for p in parsed if isinstance(p, dict)]
            except Exception:
                prec_structured = []

    penalty_summary = (
        penalty_stats_structured(prec_structured) if prec_structured else penalty_stats(cand_hits)
    )
    prec_structured_text = json.dumps(prec_structured, ensure_ascii=False) if prec_structured else "[]"

    judge_prompt = (
        f"案件事实：{case_fact}\n"
        f"法条预测：{law_resp}\n"
        f"罪名预测：{acc_resp}\n"
        f"相似案例结构化量刑因子（JSON 数组）：{prec_structured_text}\n"
        f"相似案例原始摘要/回退：{prec_resp}\n"
        f"相似案例量化统计（供量刑参考）：{penalty_summary}"
    )
    judge_step = judge_agent.step(judge_prompt)
    if not judge_step.msgs:
        raise ValueError("Empty response from judge agent")
    judgment = judge_step.msgs[0].content

    return {
        "law_resp": law_resp,
        "acc_resp": acc_resp,
        "prec_resp": prec_resp,
        "prec_structured": prec_structured,
        "judgment": judgment,
        "law_hits": law_hits,
        "acc_hits": acc_hits,
        "cand_hits": cand_hits,
        "penalty_summary": penalty_summary,
        "usage": {
            "law": law_step.info.get("usage", {}),
            "acc": acc_step.info.get("usage", {}),
            "prec": prec_step.info.get("usage", {}),
            "judge": judge_step.info.get("usage", {}),
        },
    }


def run_pipeline(args) -> None:
    data_dir = Path("data")
    test_path = data_dir / "testset" / "testset.json"

    test_case = load_test_case(test_path, case_id=args.case_id)
    resources = build_resources(args)
    outputs = predict_case(test_case["fact"], resources, top_k=args.top_k)

    print("\n=== 法条预测 ===\n", outputs["law_resp"])
    print("\n=== 罪名预测 ===\n", outputs["acc_resp"])
    print("\n=== 相似案例检索 ===\n", outputs["prec_resp"])
    print("\n=== 判决预测 ===\n", outputs["judgment"])
