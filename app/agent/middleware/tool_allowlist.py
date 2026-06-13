"""按 Agent 职责限制模型可见工具。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

from app.observability.tracing import record_event


class ToolAllowlistMiddleware(AgentMiddleware):
    """移除与当前 Agent 职责无关的 DeepAgents 内置工具。"""

    def __init__(
        self,
        allowed_tools: Iterable[str],
        *,
        retry_unknown_tool_calls: bool = False,
        response_tools: Iterable[str] = (),
    ) -> None:
        self.allowed_tools = frozenset(allowed_tools)
        self.retry_unknown_tool_calls = retry_unknown_tool_calls
        self.response_tools = frozenset(response_tools)

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        filtered_request = self._apply(request)
        response = handler(filtered_request)
        unavailable_tools = self.find_unavailable_tool_calls(
            response,
            filtered_request.tools or [],
        )
        if not self.retry_unknown_tool_calls or not unavailable_tools:
            return response
        self._record_retry(unavailable_tools)
        return handler(
            self._build_retry_request(filtered_request, unavailable_tools)
        )

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        filtered_request = self._apply(request)
        response = await handler(filtered_request)
        unavailable_tools = self.find_unavailable_tool_calls(
            response,
            filtered_request.tools or [],
        )
        if not self.retry_unknown_tool_calls or not unavailable_tools:
            return response
        self._record_retry(unavailable_tools)
        return await handler(
            self._build_retry_request(filtered_request, unavailable_tools)
        )

    def _apply(self, request: ModelRequest) -> ModelRequest:
        """仅向模型暴露当前职责允许的工具。"""
        return request.override(
            tools=self.filter_tools(request.tools or []),
        )

    def filter_tools(self, tools: Iterable[Any]) -> list[Any]:
        """过滤工具列表，保留原有顺序。"""
        return [
            tool
            for tool in tools
            if getattr(tool, "name", None) in self.allowed_tools
        ]

    def find_unavailable_tool_calls(
        self,
        response: Any,
        available_tools: Iterable[Any] | None = None,
    ) -> tuple[str, ...]:
        """提取模型响应中未向当前 Agent 开放的工具名。"""
        message = _last_ai_message(response)
        if message is None:
            return ()
        available_names = (
            {
                str(getattr(tool, "name", "") or "")
                for tool in available_tools
            }
            if available_tools is not None
            else set(self.allowed_tools)
        )
        available_names.update(self.response_tools)
        unavailable = {
            str(tool_call.get("name") or "")
            for tool_call in message.tool_calls
            if tool_call.get("name") not in available_names
        }
        return tuple(sorted(name for name in unavailable if name))

    def _build_retry_request(
        self,
        request: ModelRequest,
        unavailable_tools: tuple[str, ...],
    ) -> ModelRequest:
        """构造一次只允许使用当前可见工具的纠正请求。"""
        current_prompt = (
            request.system_message.text if request.system_message else ""
        )
        allowed = "、".join(sorted(self.allowed_tools))
        unavailable = "、".join(unavailable_tools)
        retry_prompt = f"""

【工具调用纠正】
你刚才调用了当前不存在的工具：{unavailable}。
当前唯一可用的工具是：{allowed}。
不得调用 read_file、glob、ls、write_todos 或其他未列出的工具。
已有专家结果应直接用于生成交付物；需要读取上传文件时只能调用 task 委派给文件分析助手。
请立即重新选择一个可用工具，且不要解释这次纠正。"""
        return request.override(
            system_message=SystemMessage(content=current_prompt + retry_prompt),
            tools=self.filter_tools(request.tools or []),
            tool_choice=None,
        )

    @staticmethod
    def _record_retry(unavailable_tools: tuple[str, ...]) -> None:
        """记录一次未授权工具调用纠正。"""
        record_event(
            event_name="unknown_tool_call_retry",
            component="tool_governance",
            message="模型调用了当前不可用工具，已触发一次纠正重试",
            status="warning",
            metadata={"unavailable_tools": list(unavailable_tools)},
        )


def _last_ai_message(response: Any) -> AIMessage | None:
    """兼容 AIMessage 和 ModelResponse。"""
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, ModelResponse):
        for message in reversed(response.result):
            if isinstance(message, AIMessage):
                return message
    return None
