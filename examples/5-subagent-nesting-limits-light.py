"""
第三章轻量验证：普通 dict 子智能体的嵌套边界。

本脚本复用 CEO -> CTO -> Coder 的教学结构，但将任务缩短为一个 add 函数，
用于验证顶层 DeepAgent 是否只会识别并调用直接注册的 CTO，而不会自动识别
CTO 配置里硬写的 Coder 子智能体。
"""

import os

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())

llm = init_chat_model(
    model=os.getenv("LLM_QWEN_MAX"),
    temperature=0.1,
    model_provider="openai",
)

coder_config = {
    "name": "Coder",
    "description": "高级 Python 工程师，负责接收具体编码任务并实现代码。",
    "system_prompt": """
    你是一名高级 Python 工程师。
    你的职责是接收具体编码任务，并给出对应的代码实现。
    """,
    "tools": [],
}

cto_config = {
    "name": "CTO",
    "description": "技术总监，负责将技术需求拆解为方案，并尝试分配给工程师。",
    "system_prompt": """
    你是技术总监。
    你的职责是分析 CEO 的需求，并给出技术方案。
    如果可以，请把具体编码工作交给 Coder。
    """,
    "tools": [],
    # 教学反例：普通 dict 子智能体不会因为这里写了 subagents 就自动形成二级委派。
    "subagents": [coder_config],
}

ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    system_prompt="""
    你是 CEO，负责公司战略决策。
    所有技术相关开发任务都必须委派给 CTO 处理。
    即使任务非常简单，你也不能自己写代码、不能自己给出技术实现。
    你必须先调用 CTO，并基于 CTO 的返回结果做最终验收总结。
    你只需要验收 CTO 的结果并给出简洁总结。
    """,
    subagents=[cto_config],
)

stream = ceo_agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请通过 CTO 开发一个 Python 函数 add(a, b)，要求 CTO 返回代码字符串。CEO 不允许自己直接写代码。",
            }
        ]
    }
)

seen_subagents: list[str] = []
final_answer = ""

for chunk in stream:
    for node_name, state in chunk.items():
        if not state or "messages" not in state:
            continue

        messages = state["messages"]
        if not messages:
            continue

        last_msg = messages[-1]
        if node_name == "model" and getattr(last_msg, "tool_calls", None):
            for tool_call in last_msg.tool_calls:
                if tool_call["name"] == "task":
                    subagent_type = tool_call["args"]["subagent_type"]
                    seen_subagents.append(subagent_type)
                    print(f"[model] 调用子智能体：{subagent_type}")
        elif node_name == "model" and getattr(last_msg, "content", None):
            final_answer = last_msg.content
        elif node_name == "tools":
            print(f"[tools] task 返回片段：{last_msg.content[:120]}...")

print(f"[summary] 观察到的子智能体调用链：{seen_subagents}")
print(f"[summary] 是否触发 Coder：{'Coder' in seen_subagents}")
print(f"[summary] 最终回答片段：{final_answer[:300]}")
