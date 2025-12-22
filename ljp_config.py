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
# LLM_MODEL = "qwen3-max"
LLM_MODEL = "qwen3-235b-a22b"
# LLM_MODEL = "qwen3-14b"
# LLM_MODEL = "qwen3-8b"
# 微调模型
# LLM_MODEL = "qwen3-32b-ft-202512041704-057e"
# LLM_MODEL = "qwen3-8b-ft-202512041940-a343"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-d103be2645ca438d91892867a65cfd2c"

# Defaults for pipeline limits
MAX_LAW_CHUNKS = 600
MAX_CANDIDATES = 2000
TOP_K = 3
EMBED_BATCH = 10  # DashScope embedding batch limit (<=10)
