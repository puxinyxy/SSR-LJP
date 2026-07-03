from __future__ import annotations

import sys
from pathlib import Path

import run_cail_ablation_v3_sequence as sequence


sequence.OUTPUT_DIR = sequence.ROOT / "embedding_output" / "cjo22_ablation_v3"
sequence.REPORT_PATH = sequence.OUTPUT_DIR / "CJO22_RAG_ablation_results.md"
sequence.STATE_PATH = sequence.OUTPUT_DIR / "sequence_state.json"
sequence.LOG_PATH = sequence.OUTPUT_DIR / "sequence_monitor.log"
sequence.EXPECTED_SAMPLES = 1698
sequence.DATASET_NAME = "CJO22"


def law_args(output_name: str) -> list[str]:
    return [
        "eval_llm_rag_law.py",
        "--dataset_path",
        "data/testset/testset.json",
        "--law_dir",
        "data/meta/laws.txt",
        "--limit",
        "1698",
        "--offset",
        "0",
        "--topk_law",
        "3",
        "--retrieval-query-max-chars",
        "2000",
        "--output_path",
        f"embedding_output\\cjo22_ablation_v3\\{output_name}",
    ]


def penalty_args(output_name: str) -> list[str]:
    return [
        "eval_llm_rag_penalty.py",
        "--dataset_path",
        "data/testset/testset.json",
        "--precedent_file",
        "data/candidates/precedent_case.json",
        "--law_dir",
        "data/law_articles",
        "--limit",
        "1698",
        "--offset",
        "0",
        "--topk_law",
        "3",
        "--topk_case",
        "3",
        "--retrieval-query-max-chars",
        "2000",
        "--retrieval-record-top-k",
        "30",
        "--output_path",
        f"embedding_output\\cjo22_ablation_v3\\{output_name}",
    ]


sequence.EXPERIMENTS = [
    {
        "id": "law_embedding",
        "name": "法条预测 - Embedding",
        "task": "law",
        "output": "cjo22_law_embedding.jsonl",
        "args": law_args("cjo22_law_embedding.jsonl")
        + ["--retrieval-mode", "embedding", "--no-rerank"],
    },
    {
        "id": "law_hybrid_rrf",
        "name": "法条预测 - Hybrid RRF",
        "task": "law",
        "output": "cjo22_law_hybrid_rrf.jsonl",
        "args": law_args("cjo22_law_hybrid_rrf.jsonl")
        + sequence.HYBRID_ARGS
        + ["--no-rerank"],
    },
    {
        "id": "law_hybrid_rerank",
        "name": "法条预测 - Hybrid RRF + Rerank",
        "task": "law",
        "output": "cjo22_law_hybrid_rerank.jsonl",
        "args": law_args("cjo22_law_hybrid_rerank.jsonl")
        + sequence.HYBRID_ARGS
        + ["--use-rerank", "--rerank-top-k", "30"],
    },
    {
        "id": "penalty_embedding",
        "name": "刑期预测 - Embedding",
        "task": "penalty",
        "output": "cjo22_penalty_embedding.jsonl",
        "args": penalty_args("cjo22_penalty_embedding.jsonl")
        + ["--retrieval-mode", "embedding", "--no-rerank"],
    },
    {
        "id": "penalty_hybrid_rrf",
        "name": "刑期预测 - Hybrid RRF",
        "task": "penalty",
        "output": "cjo22_penalty_hybrid_rrf.jsonl",
        "args": penalty_args("cjo22_penalty_hybrid_rrf.jsonl")
        + sequence.HYBRID_ARGS
        + ["--no-rerank"],
    },
    {
        "id": "penalty_hybrid_rerank",
        "name": "刑期预测 - Hybrid RRF + Rerank",
        "task": "penalty",
        "output": "cjo22_penalty_hybrid_rerank.jsonl",
        "args": penalty_args("cjo22_penalty_hybrid_rerank.jsonl")
        + sequence.HYBRID_ARGS
        + ["--use-rerank", "--rerank-top-k", "30"],
    },
]


if __name__ == "__main__":
    sys.exit(sequence.main())
