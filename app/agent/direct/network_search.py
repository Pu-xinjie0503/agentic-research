"""纯网络搜索任务直达 Agent。"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from app.agent.llm import model
from app.agent.middleware.final_response_governance import (
    FinalResponseGovernanceMiddleware,
)
from app.agent.middleware.memory_context import MemoryContextMiddleware
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.search_governance import SearchGovernanceMiddleware
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.agent.prompts import sub_agents_content
from app.memory.runtime import memory_store
from app.tools.tavily_tool import internet_search


network_search_direct_agent = create_agent(
    model=model,
    system_prompt=sub_agents_content["tavily"]["direct_system_prompt"],
    tools=[internet_search],
    middleware=[
        MemoryContextMiddleware(),
        SearchGovernanceMiddleware(direct_response=True),
        ToolAllowlistMiddleware(
            {"internet_search"},
            retry_unknown_tool_calls=True,
        ),
        FinalResponseGovernanceMiddleware(),
        ModelTracingMiddleware("网络搜索助手"),
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="error"),
    ],
    store=memory_store,
    name="network_search_direct_agent",
)
