"""
网络搜索子 Agent 的决策提示中间件。

每次模型决策前只注入搜索状态摘要，不把锁、完整查询记录或 URL 集合写入
LangGraph state/checkpoint。
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from app.observability.search_state import get_search_snapshot


class SearchGovernanceMiddleware(AgentMiddleware):
    """把当前搜索预算与信息增益状态注入网络 Agent 的系统提示词。"""

    def __init__(self, *, direct_response: bool = False) -> None:
        """配置预算耗尽后的最终输出形式。"""
        self.direct_response = direct_response

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        return handler(self._inject_search_state(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        return await handler(self._inject_search_state(request))

    def _inject_search_state(self, request: ModelRequest) -> ModelRequest:
        """构造当前模型决策所需的搜索治理摘要。"""
        snapshot = get_search_snapshot()
        if snapshot is None:
            return request
        final_action = (
            "直接输出最终回答"
            if self.direct_response
            else "提交 AgentHandoff"
        )

        extension_rule = (
            f"3. 第 {snapshot['soft_limit'] + 1} 至 "
            f"{snapshot['hard_limit']} 次搜索必须传 "
            "continuation_reason 和 target_gap。"
            if snapshot["hard_limit"] > snapshot["soft_limit"]
            else (
                "3. 本任务不开放软预算后的补搜；达到硬预算后，"
                f"下一条输出必须{final_action}，禁止再次生成 "
                "internet_search 调用。"
            )
        )
        governance_prompt = f"""

【搜索治理状态】
- 当前阶段：{snapshot['phase']}
- 软预算：{snapshot['soft_limit']} 次
- 硬预算：{snapshot['hard_limit']} 次
- 已实际执行：{snapshot['executed_count']} 次
- 已拦截：{snapshot['blocked_count']} 次
- 已发现独立来源域名：{snapshot['unique_domain_count']} 个
- 最近一次新增 URL：{snapshot['last_new_url_count']}
- 剩余硬预算：{snapshot['remaining_budget']} 次
- 当前决策：{snapshot['decision']}
- 停止原因：{snapshot['stop_reason'] or '无'}

决策规则：
1. 前 {snapshot['soft_limit']} 次搜索应覆盖互补角度，禁止仅做同义词改写。
2. 当前决策为 review_required 时，先检查证据缺口；只有证据不足、来源单一或信息冲突才允许定向补搜。
{extension_rule}
4. 当前决策为 stop 时，下一条输出必须{final_action}；禁止继续调用 internet_search，应基于已有证据完成总结并说明局限。
"""
        current_prompt = request.system_message.text if request.system_message else ""
        tools = list(request.tools or [])
        if snapshot["decision"] == "stop":
            tools = [
                tool
                for tool in tools
                if getattr(tool, "name", None) != "internet_search"
            ]
        return request.override(
            system_message=SystemMessage(
                content=current_prompt + governance_prompt,
            ),
            tools=tools,
        )
