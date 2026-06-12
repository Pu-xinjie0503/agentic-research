"""
大模型调用追踪中间件。

只记录消息数量、字符数量、工具名称、token 和耗时等统计信息，
不记录完整 Prompt、模型正文或工具结果，避免日志泄露业务内容。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage

from app.observability.tracing import next_model_call_index, trace_span


class ModelTracingMiddleware(AgentMiddleware):
    """记录指定 Agent 的每一次同步和异步模型调用。"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def wrap_model_call(self, request: ModelRequest, handler):
        """追踪同步模型调用。"""
        with self._start_span(request) as span:
            try:
                response = handler(request)
            except BaseException:
                span.set_result(end_reason="error")
                raise
            span.set_result(**_build_response_metrics(response))
            return response

    async def awrap_model_call(self, request: ModelRequest, handler):
        """追踪异步模型调用。"""
        with self._start_span(request) as span:
            try:
                response = await handler(request)
            except BaseException:
                span.set_result(end_reason="error")
                raise
            span.set_result(**_build_response_metrics(response))
            return response

    def _start_span(self, request: ModelRequest):
        """根据模型请求构造不含正文的 span 元数据。"""
        messages = list(request.messages or [])
        if request.system_message is not None:
            messages = [request.system_message, *messages]
        tool_names = [_tool_name(tool) for tool in request.tools or []]
        return trace_span(
            "model.call",
            component="model",
            metadata={
                "agent_name": self.agent_name,
                "model_name": _model_name(request.model),
                "call_index": next_model_call_index(),
                "input_message_count": len(messages),
                "input_char_count": sum(_message_char_count(item) for item in messages),
                "tool_count": len(tool_names),
                "tool_names": tool_names,
            },
        )


def _build_response_metrics(response: Any) -> dict[str, Any]:
    """从 LangChain 模型响应中提取输出、工具调用和 token 统计。"""
    messages = _response_messages(response)
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    tool_names = [
        str(tool_call.get("name") or "unknown")
        for message in ai_messages
        for tool_call in message.tool_calls
    ]
    usage = _usage_metrics(ai_messages)
    return {
        "output_message_count": len(messages),
        "output_char_count": sum(_message_char_count(item) for item in messages),
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "end_reason": _end_reason(ai_messages, tool_names),
    }


def _response_messages(response: Any) -> list[BaseMessage]:
    """兼容 AIMessage、ModelResponse 和 ExtendedModelResponse。"""
    if isinstance(response, AIMessage):
        return [response]
    if isinstance(response, ExtendedModelResponse):
        return list(response.model_response.result)
    if isinstance(response, ModelResponse):
        return list(response.result)
    return []


def _usage_metrics(messages: list[AIMessage]) -> dict[str, int | None]:
    """汇总模型返回的 token；供应商未提供时保持 None。"""
    input_values: list[int] = []
    output_values: list[int] = []
    total_values: list[int] = []

    for message in messages:
        usage = message.usage_metadata
        if usage:
            _append_int(input_values, _mapping_value(usage, "input_tokens"))
            _append_int(output_values, _mapping_value(usage, "output_tokens"))
            _append_int(total_values, _mapping_value(usage, "total_tokens"))
            continue

        token_usage = (message.response_metadata or {}).get("token_usage") or {}
        _append_int(input_values, _mapping_value(token_usage, "prompt_tokens"))
        _append_int(output_values, _mapping_value(token_usage, "completion_tokens"))
        _append_int(total_values, _mapping_value(token_usage, "total_tokens"))

    input_tokens = sum(input_values) if input_values else None
    output_tokens = sum(output_values) if output_values else None
    total_tokens = sum(total_values) if total_values else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _end_reason(messages: list[AIMessage], tool_names: list[str]) -> str:
    """提取供应商结束原因，缺失时按工具调用或文本回答推断。"""
    for message in reversed(messages):
        metadata = message.response_metadata or {}
        reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        if reason:
            return str(reason)
    return "tool_calls" if tool_names else "completed"


def _model_name(model: Any) -> str:
    """读取常见 LangChain 模型对象中的模型名。"""
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    return type(model).__name__


def _tool_name(tool: Any) -> str:
    """读取工具对象或 OpenAI 工具定义中的名称。"""
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            return str(function["name"])
        if tool.get("name"):
            return str(tool["name"])
    return type(tool).__name__


def _message_char_count(message: Any) -> int:
    """统计消息内容字符数，不保留具体内容。"""
    return _content_char_count(getattr(message, "content", ""))


def _content_char_count(value: Any) -> int:
    """递归统计字符串、列表和字典内容长度。"""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_content_char_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_content_char_count(item) for item in value)
    return len(str(value))


def _mapping_value(value: Any, key: str) -> Any:
    """兼容字典和属性对象形式的 usage metadata。"""
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _append_int(target: list[int], value: Any) -> None:
    """只收集有效整数 token 值。"""
    if isinstance(value, int):
        target.append(value)
