"""
Configuration constants for the LJP multi-agent demo.

Note: This file intentionally hardcodes the embedding model/base URL/API key
per user request. Be careful not to commit secrets if this repo is shared.
"""

from __future__ import annotations

# Embedding service (OpenAI-compatible)
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_API_KEY = "sk-d103be2645ca438d91892867a65cfd2c"

# LLM for the agents (can still be overridden via CLI)
# LLM候选
LLM_MODEL = "qwen3-max"
# LLM_MODEL = "qwen3-235b-a22b"
# LLM_MODEL = "qwen3-14b"
# LLM_MODEL = "qwen3-8b"
# LLM_MODEL = "siliconflow/deepseek-v3.2"
# 微调模型
# LLM_MODEL = "qwen3-32b-ft-202512041704-057e"
# LLM_MODEL = "qwen3-8b-ft-202512041940-a343"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-d103be2645ca438d91892867a65cfd2c"

# Defaults for pipeline limits
MAX_LAW_CHUNKS = 1200
MAX_CANDIDATES = 0  # 0 means no limit
TOP_K = 3
EMBED_BATCH = 10  # DashScope embedding batch limit (<=10)

# Hybrid retrieval defaults
RETRIEVAL_MODE = "hybrid"
DENSE_TOP_K = 10
BM25_TOP_K = 10
KEYWORD_TOP_K = 10
JOIN_TOP_K = 30
RRF_K = 60.0
DENSE_WEIGHT = 2.0
BM25_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
RETRIEVAL_QUERY_MAX_CHARS = 2000
USE_RERANK = True
LAW_MAX_OUTPUT_ARTICLES = 1

# Law-only Hybrid RAG V2 defaults. Generic and penalty indexes keep their
# previous RRF behavior unless these options are passed explicitly.
FUSION_MODE = "score"
DENSE_SCORE_WEIGHT = 0.65
BM25_SCORE_WEIGHT = 0.25
KEYWORD_SCORE_WEIGHT = 0.10
RERANK_SCORE_WEIGHT = 0.20
DENSE_ANCHOR = True
DENSE_MARGIN_THRESHOLD = 0.02
DENSE_OVERRIDE_THRESHOLD = 0.08
LEGAL_AWARE_RERANK = True
LEXICAL_INCLUDE_NUMERIC = True
# Number of fused candidates sent to rerank. The final number returned to the
# caller is still controlled by TOP_K / --topk-law / --top-k.
RERANK_TOP_K = 30
RERANK_MODEL = "qwen3-rerank"
RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
RERANK_TIMEOUT = 60
