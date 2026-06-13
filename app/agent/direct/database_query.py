"""纯数据库查询任务直达 Agent。"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from app.agent.llm import model
from app.agent.middleware.database_governance import DatabaseGovernanceMiddleware
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.agent.prompts import sub_agents_content
from app.tools.db_tools import execute_sql_query, get_table_data, list_sql_tables


database_query_direct_agent = create_agent(
    model=model,
    system_prompt=sub_agents_content["db"]["direct_system_prompt"],
    tools=[list_sql_tables, get_table_data, execute_sql_query],
    middleware=[
        ToolAllowlistMiddleware(
            {"list_sql_tables", "get_table_data", "execute_sql_query"},
            retry_unknown_tool_calls=True,
        ),
        DatabaseGovernanceMiddleware(direct_response=True),
        ModelTracingMiddleware("数据库查询助手"),
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="error"),
    ],
    name="database_query_direct_agent",
)
