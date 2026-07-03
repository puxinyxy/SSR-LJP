# LJP Hybrid RAG V2 实验修改方案

## 1. 目标与范围

本轮仅优化法条预测的 RAG 检索流程，目标是验证混合检索相对纯
Embedding 检索的有效性，并争取在 CAIL2018 与 CJO22 上稳定提高最终
Accuracy。

本轮修改范围：

1. 默认只输出一个主要法条。
2. 将 RRF 升级为 Dense 锚定的归一化分数融合。
3. 增加法律要件查询重写。
4. 增加法律感知 rerank。
5. 记录完整分支分数，支持离线调权。

暂不修改刑期检索流程。刑期任务继续使用当前实现。

明确排除：

- 不使用 `data/FT_data/cail_2018/train.json`。该文件为早期 Demo 数据。
- 不使用测试集标签建立规则或调整参数。
- 不按测试结果硬编码删除第 67 条等特定法条。
- 不使用当前不完整的 `data/meta/a2i.json` 作为硬白名单。

## 2. 当前问题

现有实验表现为：

- Hybrid/Rerank 提高了 Recall@3、Recall@5 和 Recall@10。
- Hybrid/Rerank 降低了 Recall@1 和 MRR。
- LLM 经常同时输出主要定罪法条和量刑相关法条。
- 最终严格集合 Accuracy 明显下降。

CAIL2018 的 1000 条测试数据中：

- 单法条样本 940 条，占 94%。
- 多法条样本 60 条，占 6%。

CJO22 的 1698 条测试数据全部是单法条样本。

因此，本轮先采用默认单法条输出。CAIL2018 的多法条预测将在后续版本中
通过“主要法条 + 补充法条”的两阶段方法处理。

## 3. 单法条输出策略

### 3.1 配置

在 `ljp_config.py` 中新增：

```python
LAW_MAX_OUTPUT_ARTICLES = 1
```

在 `eval_llm_rag_law.py` 中新增 CLI 参数：

```text
--max-output-articles 1
```

### 3.2 提示词

法条预测提示词调整为：

```text
请选择一个最直接决定本案罪名成立的主要刑法条文。

必须只输出一个法条。
不要输出仅涉及自首、坦白、退赃、谅解、累犯、立功、缓刑、
没收财产、从轻或从重处罚的一般量刑条文。

如果多个条文均相关，选择最直接规定犯罪构成和罪名的条文。
```

保持原有输出结构：

```json
{
  "law_articles": ["刑法第264条"]
}
```

### 3.3 后处理

在预测解析后执行：

```python
law_articles = law_articles[:args.max_output_articles]
```

Embedding、Hybrid、Rerank 三组实验必须使用相同提示词和后处理规则，保证
消融实验公平。

## 4. 法条词表处理

本轮不建立训练标签词表，也不做法条硬过滤。

原因：

- `data/meta/a2i.json` 包含 172 个法条，但缺少 CAIL2018 测试集中的部分
  真实标签。
- 如果将其作为硬白名单，会错误过滤真实法条。
- 当前没有可用于正式统计的独立训练标签数据。

本轮所有完整刑法条文继续参与检索。`a2i.json` 仅可用于诊断，不参与候选
过滤。

## 5. 法律要件查询重写

### 5.1 新增模块

新增：

```text
ljp_law_query.py
```

定义：

```python
@dataclass
class LawRetrievalQueries:
    dense_query: str
    lexical_query: str
    rerank_query: str
    circumstance_query: str
```

### 5.2 分支查询

不同检索分支使用不同查询：

| 分支 | 查询内容 |
|---|---|
| Dense | 规范化后的完整案件事实 |
| BM25 | 犯罪行为、对象、数额、数量、结果、行为方式 |
| Keyword | 与 BM25 相同的犯罪构成要件 |
| Rerank | 法律任务说明 + 犯罪构成要件 |
| Circumstance | 自首、坦白、退赃、谅解等量刑情节 |

`circumstance_query` 本轮只记录，不参与主要法条检索。

### 5.3 文本清理规则

删除或降权以下内容：

```text
上述事实
证据证实
证人证言
辨认笔录
庭审质证
户籍资料
抓获经过
案件来源
审理过程
```

单独提取以下量刑内容：

```text
自首
坦白
退赃
谅解
认罪认罚
累犯
立功
缓刑
从轻处罚
从重处罚
```

示例：

```text
原始事实：
被告人盗窃手机，价值5000元，归案后如实供述并退赃。
上述事实有证人证言、辨认笔录等证据证实。

Dense Query：
保留完整规范化事实。

Lexical Query：
盗窃手机 价值5000元 秘密窃取

Circumstance Query：
如实供述 退赃
```

### 5.4 接口改造

将混合检索接口由：

```python
search(query: str, top_k: int)
```

扩展为兼容：

```python
search(
    query: str | LawRetrievalQueries,
    top_k: int,
)
```

普通字符串输入继续兼容原有调用。

## 6. Dense 锚定分数融合

### 6.1 保留原始分数

当前分支只返回项目 ID。修改为返回项目 ID 和原始分数：

```python
dense_hits: list[tuple[int, float]]
bm25_hits: list[tuple[int, float]]
keyword_hits: list[tuple[int, float]]
```

### 6.2 分数归一化

在单个查询的候选集合内归一化：

```python
dense_norm = min_max(dense_cosine_score)
bm25_norm = min_max(log1p(bm25_score))
keyword_norm = min_max(log1p(keyword_score))
rerank_norm = min_max(rerank_score)
```

如果一个分支所有候选分数相同，归一化值统一设为 0，避免除零。

### 6.3 Rerank 前融合

初始权重：

```python
pre_score = (
    0.65 * dense_norm
    + 0.25 * bm25_norm
    + 0.10 * keyword_norm
)
```

### 6.4 Rerank 后融合

```python
final_score = (
    0.80 * pre_score
    + 0.20 * rerank_norm
)
```

等效初始权重约为：

| 信号 | 权重 |
|---|---:|
| Dense | 0.52 |
| BM25 | 0.20 |
| Keyword | 0.08 |
| Rerank | 0.20 |

上述权重只作为初始值，最终必须通过独立开发集选择。

### 6.5 配置和 CLI

新增配置：

```python
FUSION_MODE = "score"
DENSE_SCORE_WEIGHT = 0.65
BM25_SCORE_WEIGHT = 0.25
KEYWORD_SCORE_WEIGHT = 0.10
RERANK_SCORE_WEIGHT = 0.20
```

新增 CLI：

```text
--fusion-mode rrf|score
--dense-score-weight 0.65
--bm25-score-weight 0.25
--keyword-score-weight 0.10
--rerank-score-weight 0.20
```

保留 `rrf` 模式用于兼容旧实验和消融对照。

## 7. Dense Top-1 保护

### 7.1 置信度

定义 Dense 排名间隔：

```python
dense_margin = dense_top1_score - dense_top2_score
```

### 7.2 初始保护逻辑

```python
if dense_margin >= dense_margin_threshold:
    if fused_top1 != dense_top1:
        fused_advantage = fused_score[fused_top1] - fused_score[dense_top1]
        if fused_advantage < dense_override_threshold:
            final_top1 = dense_top1
```

初始参数：

```text
dense_margin_threshold = 0.02
dense_override_threshold = 0.08
```

### 7.3 CLI

```text
--dense-anchor
--no-dense-anchor
--dense-margin-threshold 0.02
--dense-override-threshold 0.08
```

该机制的目标是：

- Dense 结果置信度高时，防止弱稀疏信号推翻正确 Top-1。
- Dense 结果不确定时，允许 BM25、Keyword 和 Rerank 纠错。

## 8. 法律感知 Rerank

### 8.1 Rerank 查询

Rerank 查询改为：

```text
任务：寻找最直接规定本案犯罪构成和罪名的主要刑法条文。

优先依据：
- 犯罪行为
- 犯罪对象
- 数额或数量
- 行为方式
- 主观目的
- 危害结果

不要因为案件出现自首、坦白、退赃、谅解、累犯、立功、
缓刑、没收、从轻或从重情节，就优先选择一般量刑条文。

案件犯罪要件：
{lexical_query}
```

### 8.2 候选法条压缩

发送给 Reranker 的候选格式：

```text
法条编号：第264条
标题：盗窃罪
核心规定：盗窃公私财物，数额较大的……
```

避免完整长法条中的程序性和量刑性段落干扰排序。

### 8.3 排名策略

Reranker 返回候选池完整排序和分数，但不直接替换融合排名。

流程：

```text
Dense/BM25/Keyword 生成候选
        ↓
归一化分数融合
        ↓
Top-30 进入法律感知 Rerank
        ↓
Rerank 分数与预融合分数再次融合
        ↓
Dense Top-1 保护
        ↓
最终 Top-3 交给 LLM
        ↓
LLM 只输出一个主要法条
```

## 9. 结果记录

每条 JSONL 需要新增：

```json
{
  "retrieval": {
    "queries": {
      "dense": "...",
      "lexical": "...",
      "rerank": "...",
      "circumstance": "..."
    },
    "branch_scores": {
      "dense": [
        {"item_id": 1, "raw_score": 0.82, "normalized_score": 1.0}
      ],
      "bm25": [],
      "keyword": [],
      "rerank": []
    },
    "pre_fusion_scores": [],
    "final_scores": [],
    "dense_margin": 0.034,
    "dense_anchor_applied": true,
    "final_ids": [1, 5, 8]
  }
}
```

记录完整分数后，可以离线调整融合权重和阈值，而不必重新调用 Embedding
或 Rerank API。

## 10. 开发集与参数选择

### 10.1 数据要求

不得使用：

```text
data/FT_data/cail_2018/train.json
```

也不得直接使用最终测试集调参。

需要从与最终测试集不重叠的数据中构建开发集。如果暂时没有可靠的独立数据，
则先保留默认参数，只将当前实验定义为探索性实验，不宣称经过充分调优。

### 10.2 去重

开发集构建时：

1. 规范化案件事实。
2. 计算事实文本哈希。
3. 排除与 CAIL2018 1000 条测试集重复的数据。
4. 排除与 CJO22 1698 条测试集重复的数据。
5. 固定随机种子 42。

### 10.3 调参目标

检索目标函数：

```text
0.60 * Recall@1
+ 0.30 * MRR
+ 0.10 * Recall@3
```

目标重点放在 Top-1，而不是只提高 Top-5 或 Top-10。

## 11. 消融实验矩阵

所有实验统一使用单法条提示词和输出后处理：

| 编号 | 方法 | 查询重写 | Dense Anchor | Rerank |
|---|---|---|---|---|
| E0 | Embedding | 否 | 否 | 否 |
| E1 | 原 Hybrid RRF | 否 | 否 | 否 |
| E2 | 分数融合 | 否 | 是 | 否 |
| E3 | 分数融合 | 是 | 是 | 否 |
| E4 | 分数融合 | 是 | 是 | 法律感知 |

两个数据集均执行：

- CAIL2018：1000 条。
- CJO22：1698 条。

先执行纯检索评测。只有检索指标达到要求的配置再运行完整 LLM 法条预测，
减少 API 调用成本。

## 12. 评价指标

### 12.1 检索指标

```text
Recall@1
Recall@3
Recall@5
Recall@10
MRR
```

### 12.2 最终预测指标

```text
Accuracy
Macro-Precision
Macro-Recall
Macro-F1
Primary Hit Accuracy
单法条样本 Accuracy
多法条样本 Primary Recall
```

因为本轮默认只输出一个法条，必须单独报告 CAIL2018 多法条样本表现，避免
仅用严格集合 Accuracy 掩盖任务限制。

### 12.3 统计检验

```text
配对 McNemar 检验
Accuracy 差值 Bootstrap 95% 置信区间
```

## 13. 成功标准

在实验前固定成功标准：

1. CAIL2018 与 CJO22 的 Recall@1 均高于 Embedding。
2. CAIL2018 与 CJO22 的 MRR 均高于 Embedding。
3. 两个数据集的最终 Accuracy 均高于 Embedding。
4. 至少一个数据集 Accuracy 提升不低于 1 个百分点。
5. 另一个数据集不低于基线。
6. Macro-F1 下降不超过 0.5 个百分点。
7. 报告配对检验和 95% 置信区间，不只报告点估计。

如果只提高 Recall@5/10，但 Recall@1、MRR 或最终 Accuracy 下降，则不能认为
本轮 Hybrid 检索有效。

## 14. 实施顺序

1. 增加单法条提示词和输出限制。
2. 新增 `ljp_law_query.py`。
3. 为 Dense、BM25、Keyword 保留原始分数。
4. 实现分数归一化。
5. 实现分数融合模式，并保留 RRF 模式。
6. 实现 Dense Top-1 保护。
7. 实现法律感知 Rerank 查询。
8. 实现候选法条压缩。
9. 扩展 JSONL 调试信息。
10. 增加融合、查询重写和 Dense Anchor 单元测试。
11. 执行 E0-E4 纯检索消融。
12. 冻结配置。
13. 执行完整 LLM 法条预测。
14. 计算指标、配对检验和置信区间。

## 15. 测试要求

至少增加以下测试：

1. 查询重写能删除证据描述。
2. 查询重写能分离量刑情节。
3. Dense、BM25、Keyword 原始分数排序正确。
4. 分数归一化处理相同分数时不除零。
5. 分数融合结果可复现。
6. Dense Anchor 在达到阈值时保护 Dense Top-1。
7. Dense Anchor 在置信度不足时允许 Hybrid 覆盖。
8. 法律感知 Rerank 使用 `lexical_query`。
9. Rerank 完整排名仍写入 JSONL。
10. LLM 只接收最终 Top-3。
11. 默认预测结果只保留一个法条。
12. 原 `rrf` 模式和旧 CLI 仍可运行。

## 16. 预期文件修改

新增：

```text
ljp_law_query.py
test_ljp_law_query.py
test_ljp_score_fusion.py
```

修改：

```text
ljp_config.py
ljp_hybrid_retrieval.py
ljp_tools.py
simple_vector_index.py
eval_llm_rag_law.py
calc_metrics.py
test_ljp_hybrid_retrieval.py
```

刑期相关脚本只做接口兼容，不改变其检索行为和实验设置。

## 17. 参考依据

- Sparse, Dense, and Attentional Representations for Text Retrieval:
  <https://arxiv.org/abs/2005.00181>
- To Interpolate or not to Interpolate: PRF, Dense and Sparse Retrievers:
  <https://arxiv.org/abs/2205.00235>
- Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
  Methods:
  <https://dl.acm.org/doi/10.1145/1571941.1572114>
