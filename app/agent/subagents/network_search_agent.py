"""
网络搜索子智能体配置模块

将 app/prompt/prompts.yml 中的 tavily 配置与 internet_search 工具组装成
DeepAgents 可识别的字典式子智能体。主智能体后续会根据 description
决定是否把公开网络信息查询任务分派给它。
"""

from app.agent.prompts import sub_agents_content
from app.agent.handoff import AgentHandoff, HANDOFF_ERROR_MESSAGE
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.search_governance import SearchGovernanceMiddleware
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from app.tools.tavily_tool import internet_search

# 字典式子智能体的核心字段来自 YAML，便于后续只改配置就能调整路由描述和行为约束
# tools 列表声明该子智能体可以调用的真实外部能力
network_search_agent = {
    "name": sub_agents_content["tavily"]["name"],
    "description": sub_agents_content["tavily"]["description"],
    "system_prompt": sub_agents_content["tavily"]["system_prompt"],
    "response_format": ToolStrategy(
        schema=AgentHandoff,
        handle_errors=HANDOFF_ERROR_MESSAGE,
    ),
    "tools": [internet_search],
    "middleware": [
        SearchGovernanceMiddleware(),
        ToolAllowlistMiddleware(
            {"internet_search"},
            retry_unknown_tool_calls=True,
            response_tools={"AgentHandoff"},
        ),
        ModelTracingMiddleware("网络搜索助手"),
        ModelCallLimitMiddleware(run_limit=10, exit_behavior="error"),
    ],
}
