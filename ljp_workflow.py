"""
Orchestrates the LJP multi-agent workflow.
"""

from __future__ import annotations

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


def build_resources(args) -> PipelineResources:
    data_dir = Path("data")
    law_dir = data_dir / "law_articles"
    law_path_candidates = list(law_dir.glob("*.txt"))
    if not law_path_candidates:
        raise FileNotFoundError(f"No law article .txt found in {law_dir}")
    law_path = law_path_candidates[0]
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
    penalty_summary = penalty_stats(cand_hits)

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
    prec_prompt = (
        f"案件事实：{case_fact}\n"
        f"预测法条：{prelim_laws}\n"
        f"预测罪名：{prelim_acc}\n"
        f"候选案例（只显示前 {top_k} 个，含 case_id）：\n"
        + "\n".join(
            f"[case_id={h.meta.get('case_id')}] {h.text[:200]}..." for h in cand_hits
        )
    )

    law_step = law_agent.step(law_prompt)
    acc_step = acc_agent.step(acc_prompt)
    prec_step = prec_agent.step(prec_prompt)

    if not law_step.msgs or not acc_step.msgs or not prec_step.msgs:
        raise ValueError("Empty response from one of the agents (law/acc/prec)")

    law_resp = law_step.msgs[0].content
    acc_resp = acc_step.msgs[0].content
    prec_resp = prec_step.msgs[0].content

    judge_prompt = (
        f"案件事实：{case_fact}\n"
        f"法条预测：{law_resp}\n"
        f"罪名预测：{acc_resp}\n"
        f"相似案例摘要：{prec_resp}\n"
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
