"""
Entry point for the LJP multi-agent demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from logger_utils import setup_run_logger
from ljp_config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    EMBED_BATCH,
    JOIN_TOP_K,
    KEYWORD_TOP_K,
    KEYWORD_WEIGHT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_CANDIDATES,
    MAX_LAW_CHUNKS,
    RERANK_TOP_K,
    RETRIEVAL_MODE,
    RETRIEVAL_QUERY_MAX_CHARS,
    RRF_K,
    TOP_K,
    USE_RERANK,
)
from ljp_workflow import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LJP multi-agent demo")
    parser.add_argument("--case-id", type=int, default=0, help="caseID in testset")
    parser.add_argument("--max-law-chunks", type=int, default=MAX_LAW_CHUNKS)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES, help="0 means no limit")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k retrieval")
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH)
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "embedding"),
        default=RETRIEVAL_MODE,
        help="Retrieval mode. hybrid uses dense+BM25+keyword RRF.",
    )
    parser.add_argument("--dense-top-k", type=int, default=DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=BM25_TOP_K)
    parser.add_argument("--keyword-top-k", type=int, default=KEYWORD_TOP_K)
    parser.add_argument("--join-top-k", type=int, default=JOIN_TOP_K)
    parser.add_argument("--rrf-k", type=float, default=RRF_K)
    parser.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    parser.add_argument("--keyword-weight", type=float, default=KEYWORD_WEIGHT)
    parser.add_argument(
        "--retrieval-query-max-chars",
        type=int,
        default=RETRIEVAL_QUERY_MAX_CHARS,
    )
    parser.add_argument("--use-rerank", dest="use_rerank", action="store_true")
    parser.add_argument("--no-rerank", dest="use_rerank", action="store_false")
    parser.set_defaults(use_rerank=USE_RERANK)
    parser.add_argument("--rerank-top-k", type=int, default=RERANK_TOP_K)

    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help="LLM model name (OpenAI compatible)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=LLM_API_KEY,
        help="LLM API key (hardcoded defaults apply).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=LLM_BASE_URL,
        help="LLM base URL (OpenAI compatible).",
    )
    parser.add_argument(
        "--candidates-path",
        type=str,
        default=None,
        help="Path to precedent candidates JSONL (defaults to data/candidates/precedent_case.json for cjo, precedents_cail.json can be used for cail).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger, run_dir, run_id = setup_run_logger(
        run_name="ljp_multi_agent",
        args=vars(args),
        extra={"cwd": str(Path.cwd())},
    )
    logger.info(
        "config case_id=%s top_k=%s max_law_chunks=%s max_candidates=%s",
        args.case_id,
        args.top_k,
        args.max_law_chunks,
        args.max_candidates,
    )
    run_pipeline(args)
