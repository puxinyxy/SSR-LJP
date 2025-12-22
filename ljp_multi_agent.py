"""
Entry point for the LJP multi-agent demo.
"""

from __future__ import annotations

import argparse
from ljp_config import (
    EMBED_BATCH,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_CANDIDATES,
    MAX_LAW_CHUNKS,
    TOP_K,
)
from ljp_workflow import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LJP multi-agent demo")
    parser.add_argument("--case-id", type=int, default=0, help="caseID in testset")
    parser.add_argument("--max-law-chunks", type=int, default=MAX_LAW_CHUNKS)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--top-k", type=int, default=TOP_K, help="top-k retrieval")
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH)

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
    run_pipeline(args)
