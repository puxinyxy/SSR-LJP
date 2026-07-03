# Experiment Parameter Disclosure

说明：本文件只列出当前项目代码和已生成结果中能够确认的参数；未在代码中出现或无法确认的字段未列入。API key 等敏感信息不披露。

## Retrieval

| 字段 | 数值 | 解释 |
|---|---|---|
| Embedding model | `text-embedding-v4` | 用于法条、罪名示例和类案文本向量化的 embedding 模型。 |
| Embedding API | DashScope OpenAI-compatible API | 项目通过 OpenAI-compatible embedding 接口调用 DashScope 服务。 |
| Embedding base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | embedding 请求的兼容接口地址。 |
| Vector index type | in-memory numpy dense vector matrix | 项目未使用 FAISS/HNSW；向量保存在内存矩阵中，并缓存为 `.npy` 文件。 |
| Lexical index | `rank_bm25.BM25Okapi` | BM25 分支使用 `rank_bm25` 包构建中文分词后的稀疏检索索引。 |
| Similarity metric | cosine similarity | dense 检索计算 query vector 与候选 vector 的余弦相似度；部分简单索引中通过 L2 normalize 后点积实现。 |
| Retrieval mode | `hybrid` | 主配置使用混合检索，而不是纯 dense embedding 检索。 |
| Hybrid branches | dense + BM25 + keyword | 混合检索由语义向量、BM25、关键词/模糊匹配三个分支组成。 |
| Dense top-k | 10 | dense 分支进入融合前保留的候选数量。 |
| BM25 top-k | 10 | BM25 分支进入融合前保留的候选数量。 |
| Keyword top-k | 10 | keyword 分支进入融合前保留的候选数量。 |
| Join top-k | 30 | 三个检索分支融合后的候选池深度。 |
| RRF k | 60.0 | Reciprocal Rank Fusion 的平滑常数。 |
| Dense weight | 2.0 | RRF 模式下 dense 分支权重。 |
| BM25 weight | 0.7 | RRF 模式下 BM25 分支权重。 |
| Keyword weight | 0.3 | RRF 模式下 keyword 分支权重。 |
| Score fusion mode | `score` | 法条检索 V2 配置中使用归一化分数融合。 |
| Dense score weight | 0.65 | score fusion 中 dense 分支的分数权重。 |
| BM25 score weight | 0.25 | score fusion 中 BM25 分支的分数权重。 |
| Keyword score weight | 0.10 | score fusion 中 keyword 分支的分数权重。 |
| Rerank score weight | 0.20 | rerank 分数进入最终融合时的权重。 |
| Retrieval query max chars | 2000 | 检索 query 最长字符数；超长案情保留头尾片段。 |
| Law article retrieval top-k | 3 | 传给 LLM 的法条候选数量。 |
| Precedent retrieval top-k | 3 | 传给 LLM 的相似案例候选数量。 |
| Retrieval ranking record top-k | 30 | 为检索指标保存的 ranking 深度；不等同于传给 LLM 的候选数量。 |
| Reranker model | `qwen3-rerank` | 远程 rerank 使用的模型名称。 |
| Reranker top-n | 30 | rerank 阶段输入的候选池上限。 |
| Reranker URL | `https://dashscope.aliyuncs.com/compatible-api/v1/reranks` | rerank 请求地址。 |
| Reranker timeout | 60s | rerank HTTP 请求超时时间。 |
| Final evidence passed to LLM | top-3 statutes + top-3 precedents | penalty RAG 和 multi-agent workflow 最终给 LLM 的核心证据规模。 |
| Law query construction | dense query 用规范化案情；lexical query 去掉证据/程序性文本并保留犯罪构成子句；rerank query 添加法条选择任务说明 | 法条检索对不同分支使用不同 query，以降低证据噪声和一般量刑情节干扰。 |

## LLM Inference

| 字段 | 数值 | 解释 |
|---|---|---|
| Default backbone model | `qwen3-max` | 项目默认使用的 LLM backbone。 |
| Other model candidates in code | `qwen3-235b-a22b`; `qwen3-8b`; `qwen3-14b`; fine-tuned Qwen variants | 代码注释中保留的候选模型，说明实验曾支持这些模型切换。 |
| Provider/backend | DashScope OpenAI-compatible API | LLM 通过 OpenAI-compatible chat completion 接口调用。 |
| LLM base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM 请求的兼容接口地址。 |
| Temperature | 0 | 推理温度设为 0，用于降低随机性。 |
| Max output tokens | 16000 | 单次 LLM 响应的最大输出 token 数。 |
| Thinking mode | `enable_thinking=False` | Qwen thinking 模式被显式关闭。 |
| Number of runs | 1 run per case | 代码中每个样本只执行一次预测，没有多 seed 多轮投票。 |
| LLM calls per case | 4 calls in multi-agent workflow | multi-agent workflow 每案依次调用 law、acc、precedent、judge 四个 agent。 |
| Retry rule for API failures | no explicit retry; exception leads to empty content in RAG baseline | RAG baseline 捕获调用异常后将 `content` 置为空；未发现显式重试循环。 |
| Malformed prediction parsing | clean Markdown fence -> `json.loads` -> extract first `{...}` and parse again | 对非法 JSON 或带 Markdown 包裹的输出进行容错解析。 |
| Law malformed fallback | `law_articles=[]` | 法条预测解析失败时返回空法条列表。 |
| Penalty malformed fallback | `law_articles=[]`, `imprisonment_months=0` | 刑期预测解析失败时返回空法条和 0 月刑期。 |

## Prompting / Agents

| 字段 | 数值 | 解释 |
|---|---|---|
| Plain prompt output schema | `articles`, `accusations`, `imprisonment_months` | plain baseline 要求模型直接输出三个预测字段。 |
| CoT prompt output schema | `articles`, `accusations`, `imprisonment_months` | CoT baseline 允许模型在内部逐步思考，但最终仍只输出 JSON。 |
| LoT prompt output schema | `articles`, `accusations`, `imprisonment_months` | LoT baseline 使用司法三段论提示，最终输出同一 JSON schema。 |
| RAG law output schema | `law_articles` | law-only RAG 要求只输出一个包含法条数组的 JSON 对象。 |
| RAG penalty output schema | `law_articles`, `imprisonment_months` | penalty RAG 要求输出候选法条和有期徒刑月数。 |
| Ours agent roles | law agent; acc agent; precedent agent; judge agent | 多智能体方法中的四个角色。 |
| Law agent input | case facts + retrieved statute candidates | 法条 agent 根据案情和法条候选预测适用法条。 |
| Acc agent input | case facts + retrieved charge examples | 罪名 agent 根据案情和罪名示例预测罪名。 |
| Precedent agent input | case facts + predicted statutes + predicted charges + candidate precedents | 类案 agent 对候选类案抽取结构化量刑因子。 |
| Judge agent input | case facts + law prediction + charge prediction + structured precedent factors + penalty statistics | 判决 agent 综合前序 agent 输出和类案刑期统计，生成最终判决要素。 |
| Precedent agent output schema | `case_id`, `case_brief`, `penalty_factors`, `sentence_months` | 类案 agent 输出 JSON 数组，每个元素对应一条候选类案。 |
| Judge output schema | `articles`, `accusations`, `imprisonment_months`, `penalty_label`, `reasoning` | 最终判决输出的必需字段。 |
| Optional judge output field | `factor_assessment` | judge 可额外输出本案量刑因子取值，用于溯源。 |

## Data / Evaluation

| 字段 | 数值 | 解释 |
|---|---|---|
| CAIL2018 dataset path | `data/testset/test_cail_sampled_single_seed_42.json` | CAIL2018 主实验使用的采样测试文件。 |
| CAIL2018 sample count | 1000 | 主实验脚本期望并检查 1000 条 CAIL2018 样本。 |
| CAIL2018 sampling seed | 42 | 文件名和实验脚本中体现的采样随机种子。 |
| CJO22 dataset path | `data/testset/testset.json` | CJO22 测试集路径。 |
| CJO22 testset count in code | 1698 | 脚本和数据文件中确认的原始 CJO22 测试样本数。 |
| CJO22 generated combined result count | 1697 | 当前 multi-agent combined 输出中实际生成的样本数。 |
| Missing CJO22 generated caseID | 341 | 当前 combined 输出缺失的样本编号；因此生成结果数比原始测试集少 1。 |
| CJO22 precedent pool path | `data/candidates/precedent_case.json` | CJO22 类案检索池文件。 |
| CJO22 precedent pool size | 5040 | 本地文件中确认的 CJO22 类案池样本数。 |
| CAIL precedent pool path | `data/candidates/precedents_cail.json` | CAIL 类案候选文件。 |
| CAIL precedent pool size | 3030 | 本地文件中确认的 CAIL 类案池样本数。 |
| Penalty bucket 0 | 0 months | 刑期分类第 0 桶，对应无有期徒刑月数。 |
| Penalty bucket 1 | 0-6 months | 刑期分类第 1 桶。 |
| Penalty bucket 2 | 6-9 months | 刑期分类第 2 桶。 |
| Penalty bucket 3 | 9-12 months | 刑期分类第 3 桶。 |
| Penalty bucket 4 | 1-2 years | 刑期分类第 4 桶。 |
| Penalty bucket 5 | 2-3 years | 刑期分类第 5 桶。 |
| Penalty bucket 6 | 3-5 years | 刑期分类第 6 桶。 |
| Penalty bucket 7 | 5-7 years | 刑期分类第 7 桶。 |
| Penalty bucket 8 | 7-10 years | 刑期分类第 8 桶。 |
| Penalty bucket 9 | >10 years | 刑期分类第 9 桶。 |
| Law metrics | Accuracy, Macro-P, Macro-R, Macro-F1 | 法条预测评估 exact match accuracy 和宏平均 P/R/F1。 |
| Penalty metrics | Accuracy, Macro-P, Macro-R, Macro-F1 | 刑期桶分类评估 accuracy 和宏平均 P/R/F1。 |
| Law label normalization | extract article number from `第X条` or pure digits | 法条标签统一为整数编号后比较。 |
| Charge label normalization | remove trailing `罪` | 罪名标签比较前去掉末尾“罪”。 |

## Cost / Runtime

| 字段 | 数值 | 解释 |
|---|---|---|
| Multi-agent LLM calls per case | 4 | 每个样本调用 law、acc、precedent、judge 四个 agent。 |
| CJO22 average input tokens per case | 12972.65 | 从当前 CJO22 combined 输出的 usage 字段统计得到。 |
| CJO22 average output tokens per case | 824.88 | 从当前 CJO22 combined 输出的 usage 字段统计得到。 |
| CJO22 average total tokens per case | 13797.53 | 输入与输出 token 总和的平均值。 |
| CAIL2018 average input tokens per case | 13240.93 | 从当前 CAIL2018 main hybrid rerank 输出的 usage 字段统计得到。 |
| CAIL2018 average output tokens per case | 796.13 | 从当前 CAIL2018 main hybrid rerank 输出的 usage 字段统计得到。 |
| CAIL2018 average total tokens per case | 14037.06 | 输入与输出 token 总和的平均值。 |
