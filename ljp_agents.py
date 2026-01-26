"""
Agent construction and prompts.
"""

from __future__ import annotations

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType


LAW_SYSTEM = (
    "你是法条预测智能体。给出与案件事实最匹配的刑法条文编号，并简述匹配理由。"
    "输出格式：条文编号列表 + 简短理由。"
)

ACC_SYSTEM = (
    "你是罪名预测智能体。结合案件事实和候选罪名示例，给出最可能的罪名并解释理由。"
    "输出格式：罪名列表 + 简短理由。"
)

PRECEDENT_SYSTEM = (
    "你是类案量刑因子抽取智能体。输入会同时提供【待判案件事实】和若干【候选/相似案例原文或长摘要】，"
    "每个候选都含有 case_id 与基本元数据。你的任务是：结合当前案件事实，对每条候选案例独立抽取标准化的量刑因子，"
    "并用统一的 JSON 数组返回结构化结果，不要输出自然语言解释或 Markdown。\n"
    "输出必须是一个合法的 JSON 数组，数组中的每个元素对应一条候选案例，字段固定且缺失用 null 明确表示：\n"
    "{\n"
    '  "case_id": <字符串或数字，直接复用输入中的 case_id，无法确定时为 null>,\n'
    '  "case_brief": <1-2 句概括该案例案情要点，可简要标注与当前案件的相似点/差异>,\n'
    '  "penalty_factors": {\n'
    '    "crime_amount": <涉案金额（元），未知则 null>,\n'
    '    "self_surrender": <是否自首，true/false/null>,\n'
    '    "plead_guilty": <是否认罪认罚，true/false/null>,\n'
    '    "accomplice": <是否从犯，true/false/null>,\n'
    '    "attempt": <是否未遂，true/false/null>,\n'
    '    "repeat_offender": <是否累犯或再犯，true/false/null>,\n'
    '    "compensation": <是否积极退赃/赔偿/谅解，true/false/null>,\n'
    '    "victim_injury_level": <被害人损伤结果，若无伤害则填 null 或简写如"轻伤"/"重伤">,\n'
    '    "suspended": <若判缓刑则为 true，未判缓刑或未知则 false/null>\n'
    "  },\n"
    '  "sentence_months": <该类案主刑实刑期限（月），无期=-1，死刑=-2，未知为 null；若判缓刑请填实刑期限并在 suspended 标记>\n'
    "}\n"
    "只输出上述 JSON 数组，不要添加额外文本。"
)


# JUDGMENT_SYSTEM = (
#     "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
#     "你必须【只输出一个合法的 JSON 对象】，不要输出任何解释、说明或额外文字，不要使用 Markdown，不要添加小标题。\n"
#     "JSON 的字段必须严格为下面这 4 个，不能多也不能少：\n"
#     "  1. \"articles\": 法条编号列表，示例：[354] 或 [263, 264]（均为阿拉伯数字）\n"
#     "  2. \"accusations\": 罪名列表，示例：[\"容留他人吸毒\"]\n"
#     "  3. \"imprisonment_months\": 刑期月份，总刑期的阿拉伯数字；无期写 -1，死刑写 -2，无拘役写 0\n"
#     "  4. \"reason\": 1-2 句话的量刑理由\n"
#     "JSON 示例（注意这是示例，不要照抄具体数值）：\n"
#     "{\n"
#     "  \"articles\": [340],\n"
#     "  \"accusations\": [\"非法捕捞水产品罪\"],\n"
#     "  \"imprisonment_months\": 6,\n"
#     "  \"reason\": \"被告人如实供述、认罪认罚并履行生态修复义务，依法从轻处罚。\"\n"
#     "}\n"
#     "重要硬性要求（违背任意一条都视为回答无效）：\n"
#     "1. 你【必须】输出一个合法 JSON 对象，最外层是一个字典，key 使用双引号包裹，不能有注释或多余字段。\n"
#     "2. \"articles\" 必须是整数列表；若只有一个法条，也必须写成列表形式，如 [342]。\n"
#     "3. \"accusations\" 必须是字符串列表，每个元素为一个罪名。\n"
#     "4. \"imprisonment_months\" 必须是一个整数；有期徒刑用月份表示，无期写 -1，死刑写 -2，无拘役写 0。\n"
#     "5. \"reason\" 为 1-2 句话的自然语言说明，可以包含案情、量刑情节，但不要重复整个案件事实。\n"
#     "6. 在给出最终回答前，请在心中检查：JSON 是否只包含上述 4 个字段，字段类型是否正确；如有不符，请先纠正后再输出。\n"
# )

JUDGMENT_SYSTEM = (
    "你是判决智能体。输入将包含：1）案件事实摘要；2）law/acc 代理给出的法条与罪名预测；3）precedent 代理输出的结构化相似案例（含 penalty_factors 与 sentence_months 的 JSON 数组）；"
    "在司法三段论中，大前提是具体的法律规范，小前提是案件事实，结论是判决结果。让我们用司法三段论思考法条、罪名、刑期。"
    "4）程序预先计算的 penalty_stats（类案刑期均值/区间等）。你的任务是基于标准化量刑因子进行规则化推理，按照“先因子分析 → 再区间判断 → 再精确定值”的顺序推导刑期，并只输出一个合法的 JSON 对象。\n"
    "【推理步骤约束——仅在心中执行，禁止写入输出】\n"
    "1. 因子分析：结合案件事实与 law/acc 预测，推断本案的量刑因子取值（涉案金额、是否自首、是否认罪认罚、是否从犯、是否未遂、是否累犯、是否退赃/赔偿、被害人损伤程度、是否适用缓刑等），同时参考类案 penalty_factors，标注与本案的相似点/差异。\n"
    "2. 区间判断：依据适用法条的法定刑幅度，并对照类案 sentence_months 分布与 penalty_stats（均值、区间、典型区段），得到一个合理的刑期区间。遵循原则：涉案金额/情节越重→刑期越重；自首、认罪认罚、积极退赃通常从轻 20%–50%；未遂或从犯一般低于既遂主犯；累犯、犯罪集团、数罪并罚适度从重。\n"
    "3. 精确定值：在区间内给出一个整数月的刑期，确保不超出法定幅度；必要时将年份折算为月（1 年 = 12 个月），终身=-1，死刑=-2。内部可做类案校准，但不要在输出中暴露计算过程。\n"
    "【输出要求】\n"
    "只输出一个合法 JSON 对象，禁止 Markdown/解释性文本。字段至少包含：\n"
    '  - \"articles\": 法条编号整数列表；\n'
    '  - \"accusations\": 罪名字符串列表；\n'
    '  - \"imprisonment_months\": 总刑期月份（缓刑请填实刑期，终身=-1，死刑=-2，无法确定则估计整数）；\n'
    '  - \"penalty_label\": 量刑档位标签（如 \"3-6月\"、\"6-12月\"、\"1-2年\" 等）；\n'
    '  - \"reasoning\": 50-120 字说明关键因子如何影响刑期，突出加重/从轻因素及类案/penalty_stats 参照。\n'
    "可选补充字段：允许增加 \"factor_assessment\"（按 precedent agent 的 key 呈现本案因子取值）以便溯源，但不要添加无关字段。\n"
    "务必保证 JSON 外无其他内容，key 使用双引号。"
)

def make_llm(model_name: str, api_key: str, base_url: str):
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        api_key=api_key,
        url=base_url,
        # 明确关闭 Qwen 的 thinking 模式（非流式必须 enable_thinking=False），其余配置保持不变
        # 8b生成上限8k
        model_config_dict={"max_tokens": 8000, "extra_body": {"enable_thinking": False}},
    )


def make_agent(model, system_prompt: str) -> ChatAgent:
    system_message = BaseMessage.make_assistant_message(
        role_name="system",
        content=system_prompt,
    )
    return ChatAgent(
        model=model,
        system_message=system_message,
        output_language="中文",
    )
