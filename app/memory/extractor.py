"""候选表达的结构化长期记忆提取器。"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy

from app.agent.llm import model
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.memory.models import MemoryDecision


MEMORY_EXTRACTION_PROMPT = """
你是 DeepSearch 的长期记忆治理器。你的任务不是回答用户，而是判断当前消息是否
包含值得跨会话保存、查看或删除的长期记忆，并返回 MemoryDecision。

允许保存的类别：
1. profile：姓名、角色、职业背景等稳定身份信息；
2. preference：语言、格式、回答风格等长期输出偏好；
3. project：长期目标、技术栈和稳定项目约束。

禁止保存：
- API Key、密码、Token、连接串、密钥和其他敏感凭据；
- 当前库存、价格、销量、销售额、市场规模、最新搜索结果等易失效业务事实；
- 单次任务的中间过程、工具结果、完整聊天记录；
- 模型推测或用户没有明确表达的信息。

规则：
1. 用户明确说“记住”时 explicit=true；“以后、默认、总是、我偏好”等稳定表达
   可以作为保守推断，但 confidence 必须至少 0.9。
2. key 使用稳定 snake_case，例如 preferred_language、response_format、
   user_role、project_goal、tech_stack；同类新偏好必须复用相同 key。
3. “忘记”生成 action=forget，content 填删除目标；“忘记所有/清空记忆”
   设置 clear_requested=true。
4. “你记得我什么/查看记忆”设置 list_requested=true。
5. 如果消息还包含数据库、网络、文件分析或交付任务，把去掉记忆指令后的任务
   放入 remaining_task，并设置 is_memory_only=false。
6. 纯记忆操作设置 is_memory_only=true，不要在 remaining_task 中重复原指令。
7. 不能安全保存时不生成 remember 操作，并填写 rejection_reason。
"""


memory_extraction_agent = create_agent(
    model=model,
    system_prompt=MEMORY_EXTRACTION_PROMPT,
    tools=[],
    response_format=ToolStrategy(
        schema=MemoryDecision,
        handle_errors="记忆结构无效，请只修正 MemoryDecision 字段。",
    ),
    middleware=[
        ToolAllowlistMiddleware(
            set(),
            retry_unknown_tool_calls=True,
            response_tools={"MemoryDecision"},
        ),
        ModelTracingMiddleware("长期记忆提取器"),
        ModelCallLimitMiddleware(run_limit=2, exit_behavior="error"),
    ],
    name="long_term_memory_extractor",
)


async def extract_memory_decision(query: str) -> MemoryDecision:
    """调用一次结构化模型识别候选记忆操作。"""
    result = await memory_extraction_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    structured = result.get("structured_response")
    if isinstance(structured, MemoryDecision):
        return structured
    return MemoryDecision.model_validate(structured)
