"""数据库子 Agent 决策治理中间件。"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from app.observability.database_state import get_database_snapshot


class DatabaseGovernanceMiddleware(AgentMiddleware):
    """向数据库助手注入预算、缓存和停止状态。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        return handler(self._inject_state(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        return await handler(self._inject_state(request))

    @staticmethod
    def _inject_state(request: ModelRequest) -> ModelRequest:
        """构造数据库助手决策提示。"""
        snapshot = get_database_snapshot()
        if snapshot is None:
            return request
        prompt = f"""

【数据库治理状态】
- SQL 已执行：{snapshot['executed_count']} 次
- SQL 缓存命中：{snapshot['cache_hit_count']} 次
- 已预览表：{snapshot['previewed_table_count']} 张
- 剩余 SQL 预算：{snapshot['remaining_budget']} 次
- 当前决策：{snapshot['decision']}
- 停止原因：{snapshot['stop_reason'] or '无'}

强制规则：
1. list_sql_tables 和同一张表的预览结果会自动缓存，不要重复调用。
2. 只预览回答问题必需的表；通常一次聚合 SQL 即可回答。
3. 完全相同的 SQL 不得重复执行。
4. 第 4、5 次 SQL 必须填写 continuation_reason 和 target_gap。
5. 当前决策为 stop 时禁止继续执行 SQL，应使用已有结果完成回答。
6. 返回给主智能体的结论控制在 4000 字符内，保留查询口径、关键数值和局限。
"""
        tools = list(request.tools or [])
        if snapshot["decision"] == "stop":
            tools = [
                tool
                for tool in tools
                if getattr(tool, "name", None) != "execute_sql_query"
            ]
        current_prompt = request.system_message.text if request.system_message else ""
        return request.override(
            system_message=SystemMessage(content=current_prompt + prompt),
            tools=tools,
        )
