"""主 Agent 的专家调度治理中间件。"""

from __future__ import annotations

import json

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage

from app.observability.agent_state import get_agent_run_state, get_agent_snapshot
from app.observability.tracing import record_event, summarize_text


class AgentGovernanceMiddleware(AgentMiddleware):
    """注入专家调用状态，并限制无理由重复调度。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        return handler(self._inject_state(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        return await handler(self._inject_state(request))

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """同步拦截 task 工具。"""
        return self._run_tool(request, handler)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        """异步拦截 task 工具。"""
        if request.tool_call.get("name") != "task":
            return await handler(request)

        state = get_agent_run_state()
        if state is None:
            return await handler(request)

        reservation = state.reserve(
            str(request.tool_call.get("args", {}).get("subagent_type") or "unknown"),
            str(request.tool_call.get("args", {}).get("description") or ""),
        )
        if not reservation.allowed:
            self._record_blocked(reservation, request)
            return ToolMessage(
                content=json.dumps(
                    {
                        "blocked": True,
                        "reason": reservation.blocked_reason,
                        "message": reservation.message,
                        "instruction": "不要再次提交相同专家调用，请使用已有结果完成任务。",
                    },
                    ensure_ascii=False,
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        self._record_reserved(reservation, request)
        try:
            result = await handler(request)
        except BaseException as exc:
            state.complete(reservation, error=exc)
            self._record_completed(reservation, error=exc)
            raise
        state.complete(reservation)
        self._record_completed(reservation)
        return result

    @staticmethod
    def _run_tool(request: ToolCallRequest, handler):
        """同步执行一次受治理的 task 调用。"""
        if request.tool_call.get("name") != "task":
            return handler(request)
        state = get_agent_run_state()
        if state is None:
            return handler(request)
        reservation = state.reserve(
            str(request.tool_call.get("args", {}).get("subagent_type") or "unknown"),
            str(request.tool_call.get("args", {}).get("description") or ""),
        )
        if not reservation.allowed:
            AgentGovernanceMiddleware._record_blocked(reservation, request)
            return ToolMessage(
                content=json.dumps(
                    {
                        "blocked": True,
                        "reason": reservation.blocked_reason,
                        "message": reservation.message,
                        "instruction": "不要再次提交相同专家调用，请使用已有结果完成任务。",
                    },
                    ensure_ascii=False,
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        AgentGovernanceMiddleware._record_reserved(reservation, request)
        try:
            result = handler(request)
        except BaseException as exc:
            state.complete(reservation, error=exc)
            AgentGovernanceMiddleware._record_completed(reservation, error=exc)
            raise
        state.complete(reservation)
        AgentGovernanceMiddleware._record_completed(reservation)
        return result

    @staticmethod
    def _inject_state(request: ModelRequest) -> ModelRequest:
        """把当前专家完成状态注入主 Agent 决策。"""
        snapshot = get_agent_snapshot()
        if snapshot is None:
            return request
        governance_prompt = f"""

【专家调度治理状态】
- 已调用：{snapshot['calls_by_agent']}
- 已完成：{snapshot['completed_by_agent']}
- 已拦截：{snapshot['blocked_count']} 次
- 剩余额度：{snapshot['remaining_budget']} 次
- 当前决策：{snapshot['decision']}

强制规则：
1. 每个专家默认只调用一次，已完成专家的结果应直接复用。
2. 只有已有结果存在明确证据缺口、冲突或错误时，才允许第二次调用同一专家。
3. 第二次调用必须在 description 中包含：
   [continuation_reason=evidence_gap|conflict_resolution|correction]
   [target_gap=需要补齐的具体信息]
4. 不得用同义改写重复调用专家；调用被拦截后必须停止重试并基于已有结果完成任务。
5. 文件、数据库、网络存在依赖时按顺序执行；只有互不依赖的任务才并行。
"""
        current_prompt = request.system_message.text if request.system_message else ""
        tools = list(request.tools or [])
        if snapshot["decision"] == "stop":
            tools = [tool for tool in tools if getattr(tool, "name", None) != "task"]
        return request.override(
            system_message=SystemMessage(content=current_prompt + governance_prompt),
            tools=tools,
        )

    @staticmethod
    def _record_reserved(reservation, request: ToolCallRequest) -> None:
        record_event(
            event_name="agent_call_reserved",
            component="agent_governance",
            message=f"已预留 {reservation.subagent_type} 调用",
            metadata={
                "subagent_type": reservation.subagent_type,
                "call_index": reservation.call_index,
                "agent_call_index": reservation.agent_call_index,
                "is_supplement": reservation.is_supplement,
                "continuation_reason": reservation.continuation_reason,
                "target_gap": summarize_text(reservation.target_gap),
                "description": summarize_text(
                    request.tool_call.get("args", {}).get("description")
                ),
            },
        )

    @staticmethod
    def _record_blocked(reservation, request: ToolCallRequest) -> None:
        record_event(
            event_name="agent_call_blocked",
            component="agent_governance",
            message=reservation.message,
            status="warning",
            metadata={
                "subagent_type": reservation.subagent_type,
                "blocked_reason": reservation.blocked_reason,
                "description": summarize_text(
                    request.tool_call.get("args", {}).get("description")
                ),
            },
        )

    @staticmethod
    def _record_completed(reservation, error: BaseException | None = None) -> None:
        record_event(
            event_name="agent_call_completed",
            component="agent_governance",
            message=f"{reservation.subagent_type} 调用结束",
            status="error" if error else "info",
            metadata={
                "subagent_type": reservation.subagent_type,
                "call_index": reservation.call_index,
                "is_supplement": reservation.is_supplement,
            },
            error=error,
        )
