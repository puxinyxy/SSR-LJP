# LJP Hybrid RAG Upgrade Plan

## Summary

将当前 LJP 流程中的纯 embedding 检索升级为参考 KGQA 文本 RAG 的三路混合检索：`dense + BM25 + keyword`，使用 RRF 融合；远程 rerank 默认开启，并提供 `--no-rerank` 用于消融或避免额外远程调用。

运行环境为 conda 环境 `Camel`，所有依赖安装和验证命令默认在该环境中执行。

## Key Design Decisions

- 三路检索共享同一份候选列表，不允许各检索器各自重新切块。每个候选使用稳定 `item_id`，即候选在原始 `List[TextItem]` 或 `List[str]` 中的下标。
- dense、BM25、keyword 都基于同一个 `item_id -> text/meta` 语料视图构建索引。RRF 融合和去重只按 `item_id` 执行，避免同一法条或案例因分块不一致被重复计入。
- 保留现有自由函数 `search_index(index, query, top_k)` 作为兼容入口；新 `HybridIndex` 自身提供 `.search(query, top_k)` 方法，`search_index()` 只做薄适配：如果对象有 `.search()` 就调用对象方法，否则走旧的 dense 逻辑。
- 不在本次改动中处理 graph 检索，也不重构现有跨文件重复工具函数，避免扩大变更范围。

## Key Changes

- 新增 LJP 专用混合检索模块 `ljp_hybrid_retrieval.py`，避免直接引入 KGQA 的 Haystack/Qdrant/GraphRAG。
- 新增 `HybridIndex`：
  - 输入为现有 `items: Sequence[TextItem]`、已缓存 dense vectors 和 embedder。
  - 内部保存 `item_id`、原文、meta、jieba tokens、BM25 模型、keyword token 视图。
  - `.search(query, top_k)` 返回 `List[TextItem]`，保持下游契约。
- 新增 `SimpleHybridIndex`：
  - 输入为 `texts: List[str]` 和 embedder。
  - 用于 `eval_llm_rag_penalty.py` 的字符串列表检索场景。
  - `.search(query, top_k)` 返回 `List[str]`，兼容 `SimpleVectorIndex.search()`。
- 检索流程：
  - dense：复用现有 embedding 向量缓存。
  - BM25：使用 `jieba` 分词和 `rank-bm25`，基于同一候选列表构建。
  - keyword：使用确定性关键词打分公式，见下一节。
  - fusion：使用 RRF 融合三路召回结果，默认 `rrf_k=60`，按 `item_id` 去重。
- 新增 `requirements-ljp.txt`：
  - `jieba`
  - `rank-bm25`
  - `rapidfuzz`
- 更新入口：
  - `ljp_tools.py`：新增 hybrid index 构建函数，并让 `search_index()` 兼容 `.search()`。
  - `simple_vector_index.py`：新增 `SimpleHybridIndex`。
  - `ljp_workflow.py`：法条、罪名、案例索引默认改用 hybrid。
  - `eval_llm_rag_law.py`：法条检索默认改用 hybrid。
  - `eval_llm_rag_penalty.py`：法条和案例检索默认改用 `SimpleHybridIndex`。
  - `ljp_multi_agent.py`、`ljp_eval.py`、`ljp_eval_cail2018.py` 增加检索参数透传。

## Keyword Scoring

keyword 分支使用可复现的确定性公式，不依赖 LLM 或外部 NER：

```text
keyword_score =
  exact_meta_hit * 6.0
  + exact_text_hit * 3.0
  + token_overlap * 4.0
  + fuzzy_bonus
```

- `jieba` 对 query 和候选文本分词，忽略空 token。
- `token_overlap = |query_tokens ∩ doc_tokens| / max(|query_tokens|, 1)`。
- `fuzzy_bonus = rapidfuzz.fuzz.partial_ratio(query, searchable_text[:1000]) / 100`，仅当结果 `>= 0.5` 时加入。
- `exact_text_hit`：query 关键词或实体片段直接出现在候选文本中时计数。
- `exact_meta_hit`：query 关键词或实体片段出现在候选 meta 组成的文本中时计数。
- 关键词来源：
  - `jieba` 分词后长度大于 1 的 token。
  - 正则抽取的连续中文、英文、数字片段：`[\u4e00-\u9fffA-Za-z0-9_()（）-]{2,}`。
  - 法条候选额外把 `article_id`、`chunk_id` 纳入 meta searchable text。
  - 案例候选额外把罪名、相关法条、刑期 meta 纳入 meta searchable text。

## Cache Strategy

- dense 分支继续使用现有 `.npy + .json` embedding cache。
- BM25 和 keyword 分支新增本地缓存，建议放在 `output/retrieval_cache`：
  - `*_bm25.pkl`：保存 BM25 模型、tokenized corpus、`item_id` 列表。
  - `*_keyword.pkl`：保存每个候选的 searchable text、meta searchable text、token set。
  - `*_hybrid_meta.json`：保存缓存签名、构建时间、候选数量、检索参数。
- 缓存签名沿用现有 `build_index_cached()` 的 meta 思路，至少包含：
  - source path、mtime、size。
  - `max_chunks` 或 `max_items`。
  - chunk 参数。
  - embedding model/base url。
  - retrieval mode、BM25/keyword 参数、jieba tokenizer 标记。
- 如果缓存缺失、签名不匹配或候选数量不一致，则重建 BM25/keyword 缓存；embedding 缓存仍按现有逻辑处理。

## Public Interface Changes

新增 CLI 参数，全部有默认值，原命令不需要修改：

- `--retrieval-mode {hybrid,embedding}`，默认 `hybrid`
- `--dense-top-k`，默认 `50`
- `--bm25-top-k`，默认 `50`
- `--keyword-top-k`，默认 `30`
- `--join-top-k`，默认 `50`
- `--rrf-k`，默认 `60`
- `--use-rerank`，显式开启 rerank
- `--no-rerank`，显式关闭 rerank
- `--rerank-top-k`，默认等于最终 `top-k` 或 `8`，rerank 开启时生效

环境依赖安装命令：

```powershell
conda activate Camel
pip install -r requirements-ljp.txt
```

## Test Plan

静态编译检查：

```powershell
conda activate Camel
python -m py_compile ljp_hybrid_retrieval.py ljp_tools.py simple_vector_index.py ljp_workflow.py ljp_multi_agent.py ljp_eval.py ljp_eval_cail2018.py eval_llm_rag_law.py eval_llm_rag_penalty.py
```

CLI smoke test：

```powershell
python ljp_multi_agent.py --help
python ljp_eval.py --help
python ljp_eval_cail2018.py --help
python eval_llm_rag_law.py --help
python eval_llm_rag_penalty.py --help
```

新增测试文件 `test_ljp_hybrid_retrieval.py`，使用 fake embedder，至少覆盖：

- BM25 排序：给定明确 query 和候选文本，包含 query token 更多的候选排在前面。
- keyword 打分：验证 exact meta hit、text hit、token overlap、fuzzy bonus 都能影响排序。
- RRF 融合：构造 dense、BM25、keyword 三个命中列表，断言融合分数和排序符合 `1 / (rrf_k + rank)`。
- `item_id` 去重：同一 `item_id` 被多路召回时只出现一次，并累加 RRF 分数。
- 返回类型兼容：`HybridIndex.search()` 和 `search_index()` 返回 `List[TextItem]`；`SimpleHybridIndex.search()` 返回 `List[str]`。
- 回退兼容：`retrieval-mode=embedding` 时仍能走旧 dense 检索逻辑。

后续手动运行 README 中原评测命令验证实际指标变化。

## Assumptions

- 本次只迁移 KGQA 文本 RAG 主体：`dense + BM25 + keyword + RRF`。
- 不接入 graph 分支。
- rerank 默认开启；如果需要纯 RRF 消融或减少远程 API 调用，命令中添加 `--no-rerank`。
- 原输出 JSONL/XLSX 字段不改，继续兼容 `calc_ours.py` 和 `calc_metrics.py`。
- `ljp_workflow.py` 中硬编码 law path 是既有问题；本次实现可优先改为相对路径 `data/meta/laws.txt`，但不额外扩大为全项目路径治理。
- 计划文件路径：`G:\graduate_1\Code\Camel\LJP_HYBRID_RAG_UPGRADE_PLAN.md`。
