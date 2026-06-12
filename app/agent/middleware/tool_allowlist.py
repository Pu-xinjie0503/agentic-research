"""按 Agent 职责限制模型可见工具。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest


class ToolAllowlistMiddleware(AgentMiddleware):
    """移除与当前 Agent 职责无关的 DeepAgents 内置工具。"""

    def __init__(self, allowed_tools: Iterable[str]) -> None:
        self.allowed_tools = frozenset(allowed_tools)

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        return handler(self._apply(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        return await handler(self._apply(request))

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
