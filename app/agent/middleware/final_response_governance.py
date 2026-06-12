"""主 Agent 最终响应质量治理。"""

from __future__ import annotations

import re
import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages import ToolMessage

from app.observability.search_state import get_search_snapshot
from app.observability.tracing import record_event


URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+", flags=re.IGNORECASE)


class FinalResponseGovernanceMiddleware(AgentMiddleware):
    """网络任务缺少引用时，最多追加一次纯文本重写。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        response = handler(request)
        if not self.needs_citation_retry(response):
            return response
        self._record_retry()
        return handler(self._build_retry_request(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        response = await handler(request)
        if not self.needs_citation_retry(response):
            return response
        self._record_retry()
        return await handler(self._build_retry_request(request))

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """同步校验 Markdown 产物是否保留网络引用。"""
        error = self.markdown_citation_error(request.tool_call)
        if error is None:
            return handler(request)
        self._record_artifact_block(error)
        return self._artifact_error_message(request, error)

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        """异步校验 Markdown 产物是否保留网络引用。"""
        error = self.markdown_citation_error(request.tool_call)
        if error is None:
            return await handler(request)
        self._record_artifact_block(error)
        return self._artifact_error_message(request, error)

    @staticmethod
    def needs_citation_retry(response: Any) -> bool:
        """判断当前响应是否是缺少来源 URL 的最终网络结论。"""
        snapshot = get_search_snapshot()
        if snapshot is None or int(snapshot.get("executed_count") or 0) == 0:
            return False
        message = _last_ai_message(response)
        if message is None or message.tool_calls:
            return False
        return len(URL_PATTERN.findall(message.text or str(message.content))) < 3

    @staticmethod
    def _build_retry_request(request: ModelRequest) -> ModelRequest:
        """构造不允许继续调用工具的引用修复请求。"""
        current_prompt = request.system_message.text if request.system_message else ""
        retry_prompt = """

【最终响应引用校验未通过】
你上一版最终回答未保留足够的可核验来源 URL。请基于当前消息历史中的网络搜索助手结果，重新输出完整最终回答：
1. 保留原有关键结论、数据和结构；
2. 增加“来源链接”小节，至少列出 3 个真实的 http/https 原始 URL；
3. 不得编造链接，不得只写媒体名称；
4. 不再调用任何工具，只输出修正后的最终正文。
"""
        return request.override(
            system_message=SystemMessage(content=current_prompt + retry_prompt),
            tools=[],
            tool_choice=None,
        )

    @staticmethod
    def _record_retry() -> None:
        """记录一次最终引用修复。"""
        record_event(
            event_name="final_response_retry",
            component="response_governance",
            message="最终响应缺少来源 URL，触发一次无工具重写",
            status="warning",
            metadata={"reason": "missing_citations", "minimum_url_count": 3},
        )

    @staticmethod
    def markdown_citation_error(tool_call: dict[str, Any]) -> str | None:
        """网络任务生成 Markdown 时至少保留 3 个来源 URL。"""
        if tool_call.get("name") != "generate_markdown":
            return None
        snapshot = get_search_snapshot()
        if snapshot is None or int(snapshot.get("executed_count") or 0) == 0:
            return None
        content = str((tool_call.get("args") or {}).get("content") or "")
        url_count = len(URL_PATTERN.findall(content))
        if url_count >= 3:
            return None
        return (
            f"Markdown 正文仅包含 {url_count} 个来源 URL，至少需要 3 个。"
            "请从已有网络搜索助手结果中补充真实 URL 后重新调用 generate_markdown，"
            "不得编造链接。"
        )

    @staticmethod
    def _artifact_error_message(
        request: ToolCallRequest,
        error: str,
    ) -> ToolMessage:
        """返回可供主 Agent 修正产物内容的工具错误。"""
        return ToolMessage(
            content=json.dumps(
                {
                    "blocked": True,
                    "reason": "missing_artifact_citations",
                    "message": error,
                },
                ensure_ascii=False,
            ),
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    @staticmethod
    def _record_artifact_block(error: str) -> None:
        """记录一次产物引用拦截。"""
        record_event(
            event_name="artifact_citation_blocked",
            component="response_governance",
            message=error,
            status="warning",
            metadata={"minimum_url_count": 3},
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
