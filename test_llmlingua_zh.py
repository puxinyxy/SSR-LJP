# -*- coding: utf-8 -*-
"""
LLMLingua-2 中文压缩测试脚本（优先使用 GPU）

使用方法：
1. 确保已安装：
   pip install llmlingua

2. 在 Anaconda 的 camel 环境中运行：
   python test_llmlingua_zh_gpu.py
"""

import torch
from llmlingua import PromptCompressor


def main():
    # 自动选择设备：优先用 GPU，没有就用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("当前使用设备:", device)

    # 初始化 LLMLingua-2 多语言压缩模型
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,   # 使用 LLMLingua-2
        device_map=device,     # 把模型加载到对应设备（cuda / cpu）
    )

    # 一段示例中文长文本，你可以之后改成自己的内容（比如 Camel 的对话上下文 / RAG 段落）
    original_prompt = (
        "你是一名数据分析与乡村经济专家，需要帮助我分析一段关于乡村振兴与数字经济发展的中文材料。"
        "请先阅读下面的背景说明，然后根据材料回答若干问题，并给出清晰的推理过程。\n\n"
        "背景：在乡村振兴战略背景下，数字经济正在成为推动农村社会经济发展的新引擎。"
        "通过电商平台、移动支付、数字基础设施建设等方式，越来越多的农产品可以直接触达城市消费者，"
        "农民的收入结构和生产方式也因此发生了显著变化。然而，不同地区之间在基础设施、人才储备、"
        "制度环境等方面仍存在较大差异，这些差异会导致数字经济赋能效果的不均衡。\n\n"
        "任务：请你用 3–4 点总结数字经济如何具体促进乡村振兴，并指出其中可能存在的两类主要风险，"
        "回答时请保持条理清晰、逻辑严谨。\n"
    )

    print("=== 原始文本 ===")
    print(original_prompt)
    print(f"\n原始字符数: {len(original_prompt)}")

    # 使用 LLMLingua-2 进行压缩
    result = compressor.compress_prompt(
        original_prompt,
        rate=0.5,  # 压缩率：保留大约 50% token，需要更稳妥可以先设 0.7~0.8
        # 强制保留的分隔符（包含常见中文标点，避免句子结构被删得太碎）
        force_tokens=["\n", "。", "？", "！", "："],
    )

    compressed = result["compressed_prompt"]

    print("\n=== 压缩后文本 ===")
    print(compressed)
    print(f"\n压缩后字符数: {len(compressed)}")

    # 一些统计信息（如果对应字段存在）
    origin_tokens = result.get("origin_tokens")
    compressed_tokens = result.get("compressed_tokens")
    ratio = result.get("ratio")
    saving = result.get("saving")

    if origin_tokens is not None:
        print(f"\n原始 token 数: {origin_tokens}")
    if compressed_tokens is not None:
        print(f"压缩后 token 数: {compressed_tokens}")
    if ratio is not None:
        print(f"压缩比例: {ratio}")
    if saving is not None:
        print(f"估算节省比例: {saving}")


if __name__ == "__main__":
    main()
