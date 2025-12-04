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
    "你是相似案例检索智能体。基于案件事实、候选法条和罪名，"
    "从候选案例中挑选最相似的案例并总结其要点（案号无需精确，只要包含 case_id）。"
)

# JUDGMENT_SYSTEM = (
#     "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
#     "请严格按照下面的固定格式输出，不要添加多余小标题，不要调整字段顺序：\n"
#     "法条：[阿拉伯数字编号列表，如: [354] 或 [263, 264]]\n"
#     "罪名：[罪名列表，如: [容留他人吸毒]]\n"
#     "刑期（月）：<只写一个阿拉伯数字，表示总刑期月份；无期写 -1，死刑写 -2，无拘役写 0；这一行不要出现其他数字>\n"
#     "刑期标签：<从下列选一项，并且必须一字不差原文输出>\n"
#     "  可选项：其他 / 六个月以内 / 六到九个月 / 九个月到一年 / 一到两年 / 二到三年 / 三到五年 / 五到七年 / 七到十年 / 十年以上\n"
#     "理由：用 1-2 句话简要说明量刑理由。\n"
#     "重要要求：\n"
#     "1. 【刑期标签】必须从上面的可选项中选一项，直接拷贝，不要改写或扩展。\n"
#     "2. 输出中只能在“刑期（月）：”这一行出现刑期的数字，其它地方不要出现表示月份或年份的数字，以免解析混淆。\n"
# )
# JUDGMENT_SYSTEM = (
#     "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
#     "请严格按照下面的固定格式输出，不要添加多余小标题，不要调整字段顺序：\n"
#     "法条：[阿拉伯数字编号列表，如: [354] 或 [263, 264]]\n"
#     "罪名：[罪名列表，如: [容留他人吸毒]]\n"
#     "刑期（月）：<只写一个阿拉伯数字，表示总刑期月份；无期写 -1，死刑写 -2，无拘役写 0；这一行不要出现其他数字>\n"
#     "刑期标签：<从下列【10 个字符串】中【精确复制】其中【恰好 1 个】，不得增加任何字词>\n"
#     "  可选项（只能从中选一个，且必须一字不差地原样输出）：\n"
#     "    其他\n"
#     "    六个月以内\n"
#     "    六到九个月\n"
#     "    九个月到一年\n"
#     "    一到两年\n"
#     "    二到三年\n"
#     "    三到五年\n"
#     "    五到七年\n"
#     "    七到十年\n"
#     "    十年以上\n"
#     "理由：用 1-2 句话简要说明量刑理由。\n"
#     "重要硬性要求（违背任意一条都视为回答无效）：\n"
#     "1. 【刑期标签】必须严格等于上面 10 个字符串中的【一个】。不能自己组合新的区间，例如“六到十个月”“六个月以上”等【禁止输出】的表达。\n"
#     "2. 如果你无法确定精确的区间，必须输出“其他”，不要自行创造新标签。\n"
#     "3. 输出中只能在“刑期（月）：”这一行出现刑期的数字，其它地方不要出现表示月份或年份的数字，以免解析混淆。\n"
#     "4. 在给出最终回答前，请先在心中检查一遍：你输出的【刑期标签】是否与上面 10 个字符串中的某一个【完全相同】；如果不完全相同，就改成其中最合适的一项或“其他”。\n"
# )

# JUDGMENT_SYSTEM = (
#     "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
#     "你必须【只输出一个合法的 JSON 对象】，不要输出任何解释、说明或额外文字，不要使用 Markdown，不要添加小标题。\n"
#     "JSON 的字段必须严格为下面这 5 个，不能多也不能少：\n"
#     "  1. \"articles\": 法条编号列表，示例：[354] 或 [263, 264]（均为阿拉伯数字）\n"
#     "  2. \"accusations\": 罪名列表，示例：[\"容留他人吸毒\"]\n"
#     "  3. \"imprisonment_months\": 刑期月份，总刑期的阿拉伯数字；无期写 -1，死刑写 -2，无拘役写 0\n"
#     "  4. \"penalty_label\": 刑期标签字符串，从下列 10 个选项中精确复制【一个】\n"
#     "  5. \"reason\": 1-2 句话的量刑理由\n"
#     "JSON 示例（注意这是示例，不要照抄具体数值）：\n"
#     "{\n"
#     "  \"articles\": [340],\n"
#     "  \"accusations\": [\"非法捕捞水产品罪\"],\n"
#     "  \"imprisonment_months\": 6,\n"
#     "  \"penalty_label\": \"六个月以内\",\n"
#     "  \"reason\": \"……\"\n"
#     "}\n"
#     "【刑期标签】可选项（只能从中选一个，一字不差原样输出）：\n"
#     "  其他\n"
#     "  六个月以内\n"
#     "  六到九个月\n"
#     "  九个月到一年\n"
#     "  一到两年\n"
#     "  二到三年\n"
#     "  三到五年\n"
#     "  五到七年\n"
#     "  七到十年\n"
#     "  十年以上\n"
#     "重要硬性要求（违背任意一条都视为回答无效）：\n"
#     "1. 你【必须】输出一个合法 JSON 对象，key 使用双引号包裹，不能有注释或多余字段。\n"
#     "2. \"penalty_label\" 的值【必须严格等于】上面 10 个字符串中的【一个】。\n"
#     "   禁止输出新的区间描述，例如“六到十个月”“六个月以上”等【禁止】表达。\n"
#     "3. 如果无法确定精确区间，必须将 \"penalty_label\" 设为 \"其他\"，不要自己创造新标签。\n"
#     "4. 除了 \"imprisonment_months\" 和 \"articles\" 字段外，\"reason\" 中不要再写表示具体刑期长短的数字（例如“3年”“10个月”），以免解析混淆。\n"
#     "5. 在给出最终回答前，请先在心中检查一遍：你输出的 JSON 是否只包含上述 5 个字段，"
#     "且 \"penalty_label\" 是否与上面 10 个字符串中的某一个完全相同；如果不完全相同，就改成其中最合适的一项或 \"其他\"。\n"
# )

# JUDGMENT_SYSTEM = (
#     "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
#     "你必须【只输出一个合法的 JSON 对象】，不要输出任何解释、说明或额外文字，不要使用 Markdown，不要添加小标题。\n"
#     "JSON 的字段必须严格为下面这 5 个，不能多也不能少：\n"
#     "  1. \"articles\": 法条编号列表，示例：[354] 或 [263, 264]（均为阿拉伯数字）\n"
#     "  2. \"accusations\": 罪名列表，示例：[\"容留他人吸毒\"]\n"
#     "  3. \"imprisonment_months\": 刑期月份，总刑期的阿拉伯数字；无期写 -1，死刑写 -2，无拘役写 0\n"
#     "  4. \"penalty_label\": 刑期标签字符串，从下列 10 个选项中精确复制【一个】\n"
#     "  5. \"reason\": 1-2 句话的量刑理由\n"
#     "JSON 示例（注意这是示例，不要照抄具体数值）：\n"
#     "{\n"
#     "  \"articles\": [340],\n"
#     "  \"accusations\": [\"非法捕捞水产品罪\"],\n"
#     "  \"imprisonment_months\": 6,\n"
#     "  \"penalty_label\": \"六个月以内\",\n"
#     "  \"reason\": \"……\"\n"
#     "}\n"
#     "【刑期标签】可选项（只能从中选一个，一字不差原样输出）：\n"
#     "  其他\n"
#     "  六个月以内\n"
#     "  六到九个月\n"
#     "  九个月到一年\n"
#     "  一到两年\n"
#     "  二到三年\n"
#     "  三到五年\n"
#     "  五到七年\n"
#     "  七到十年\n"
#     "  十年以上\n"
#     "【量刑区间划分规则 —— 必须严格按照下列规则由 imprisonment_months 推出 penalty_label】\n"
#     "  - 若 imprisonment_months 为 -1（无期）、-2（死刑）或 0（无拘役），则 \"penalty_label\" = \"其他\"。\n"
#     "  - 若 1 <= imprisonment_months <= 6，则 \"penalty_label\" = \"六个月以内\"。\n"
#     "  - 若 7 <= imprisonment_months <= 9，则 \"penalty_label\" = \"六到九个月\"。\n"
#     "  - 若 10 <= imprisonment_months <= 12，则 \"penalty_label\" = \"九个月到一年\"。\n"
#     "  - 若 13 <= imprisonment_months <= 24，则 \"penalty_label\" = \"一到两年\"。\n"
#     "  - 若 25 <= imprisonment_months <= 36，则 \"penalty_label\" = \"二到三年\"。\n"
#     "  - 若 37 <= imprisonment_months <= 60，则 \"penalty_label\" = \"三到五年\"。\n"
#     "  - 若 61 <= imprisonment_months <= 84，则 \"penalty_label\" = \"五到七年\"。\n"
#     "  - 若 85 <= imprisonment_months <= 120，则 \"penalty_label\" = \"七到十年\"。\n"
#     "  - 若 imprisonment_months > 120，则 \"penalty_label\" = \"十年以上\"。\n"
#     "重要硬性要求（违背任意一条都视为回答无效）：\n"
#     "1. 你【必须】先在心中确定一个具体的 imprisonment_months 数值，然后根据上面的区间划分规则推导出唯一的 \"penalty_label\"。\n"
#     "2. \"penalty_label\" 的值【必须严格等于】上面 10 个字符串中的【一个】；不得根据主观感觉选择与 imprisonment_months 不匹配的标签。\n"
#     "   禁止输出新的区间描述，例如“六到十个月”“六个月以上”等【禁止】表达。\n"
#     "3. 如果 imprisonment_months 为 -1、-2 或 0，必须将 \"penalty_label\" 设为 \"其他\"。\n"
#     "4. 如果无法确定精确区间，也必须先选择一个 imprisonment_months，再按照规则选择对应的标签，或将 imprisonment_months 设为 0 并选择 \"其他\"，不要自己创造新标签。\n"
#     "5. 除了 \"imprisonment_months\" 和 \"articles\" 字段外，\"reason\" 中不要再写表示具体刑期长短的数字（例如“3年”“10个月”），以免解析混淆。\n"
#     "6. 在给出最终回答前，请在心中完成以下自检：\n"
#     "   (1) 检查 JSON 是否只包含上述 5 个字段；\n"
#     "   (2) 检查 imprisonment_months 数值落在哪一个区间；\n"
#     "   (3) 检查 \"penalty_label\" 是否与该区间对应的标签完全一致；如不一致，修改其中之一使其一致。\n"
# )

JUDGMENT_SYSTEM = (
    "你是判决智能体。根据案件事实、预测的法条/罪名与相似案例，给出最终判决要素，必须包含【可解析的刑期】。\n"
    "你必须【只输出一个合法的 JSON 对象】，不要输出任何解释、说明或额外文字，不要使用 Markdown，不要添加小标题。\n"
    "JSON 的字段必须严格为下面这 4 个，不能多也不能少：\n"
    "  1. \"articles\": 法条编号列表，示例：[354] 或 [263, 264]（均为阿拉伯数字）\n"
    "  2. \"accusations\": 罪名列表，示例：[\"容留他人吸毒\"]\n"
    "  3. \"imprisonment_months\": 刑期月份，总刑期的阿拉伯数字；无期写 -1，死刑写 -2，无拘役写 0\n"
    "  4. \"reason\": 1-2 句话的量刑理由\n"
    "JSON 示例（注意这是示例，不要照抄具体数值）：\n"
    "{\n"
    "  \"articles\": [340],\n"
    "  \"accusations\": [\"非法捕捞水产品罪\"],\n"
    "  \"imprisonment_months\": 6,\n"
    "  \"reason\": \"被告人如实供述、认罪认罚并履行生态修复义务，依法从轻处罚。\"\n"
    "}\n"
    "重要硬性要求（违背任意一条都视为回答无效）：\n"
    "1. 你【必须】输出一个合法 JSON 对象，最外层是一个字典，key 使用双引号包裹，不能有注释或多余字段。\n"
    "2. \"articles\" 必须是整数列表；若只有一个法条，也必须写成列表形式，如 [342]。\n"
    "3. \"accusations\" 必须是字符串列表，每个元素为一个罪名。\n"
    "4. \"imprisonment_months\" 必须是一个整数；有期徒刑用月份表示，无期写 -1，死刑写 -2，无拘役写 0。\n"
    "5. \"reason\" 为 1-2 句话的自然语言说明，可以包含案情、量刑情节，但不要重复整个案件事实。\n"
    "6. 在给出最终回答前，请在心中检查：JSON 是否只包含上述 4 个字段，字段类型是否正确；如有不符，请先纠正后再输出。\n"
)

def make_llm(model_name: str, api_key: str, base_url: str):
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        api_key=api_key,
        url=base_url,
        # 明确关闭 Qwen 的 thinking 模式（非流式必须 enable_thinking=False），其余配置保持不变
        model_config_dict={"max_tokens": 16000, "extra_body": {"enable_thinking": False}},
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
