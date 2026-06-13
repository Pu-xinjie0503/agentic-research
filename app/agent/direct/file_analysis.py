"""纯文件分析任务直达 Agent。"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from app.agent.llm import model
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.agent.prompts import sub_agents_content
from app.tools.upload_file_read_tool import read_file_content


file_analysis_direct_agent = create_agent(
    model=model,
    system_prompt=sub_agents_content["file_analysis"]["direct_system_prompt"],
    tools=[read_file_content],
    middleware=[
        ToolAllowlistMiddleware({"read_file_content"}),
        ModelTracingMiddleware("文件分析助手"),
        ModelCallLimitMiddleware(run_limit=4, exit_behavior="error"),
    ],
    name="file_analysis_direct_agent",
)
