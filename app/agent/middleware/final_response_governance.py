"""主 Agent 最终响应质量治理。"""

from __future__ import annotations

import re
import json
import math
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages import ToolMessage

from app.agent.evidence import find_unsupported_precision_claims
from app.api.context import get_session_context
from app.observability.search_state import (
    get_search_evidence_records,
    get_search_snapshot,
)
from app.observability.evidence_pack import get_evidence_records
from app.observability.tracing import record_event


URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+", flags=re.IGNORECASE)
TEXT_SOURCE_SUFFIXES = {".md", ".txt", ".csv", ".json", ".yaml", ".yml"}
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)、])\s+")


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
        content = message.text or str(message.content)
        unsupported_claims = find_unsupported_precision_claims(
            content,
            _network_evidence_records(),
        )
        if unsupported_claims:
            return True
        if _has_generated_artifact():
            return False
        return len(URL_PATTERN.findall(content)) < 3

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
4. 删除没有 tier_1_official 或 tier_2_primary 原文数字直接支撑的市场规模、增长率、金额、人口和占比；低等级来源只能保留定性趋势；
5. 不再调用任何工具，只输出修正后的最终正文。
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
        errors = []
        source_coverage_error = _uploaded_source_coverage_error(tool_call)
        if source_coverage_error:
            errors.append(source_coverage_error)
        snapshot = get_search_snapshot()
        if snapshot is None or int(snapshot.get("executed_count") or 0) == 0:
            return "；".join(errors) + "。" if errors else None
        content = str((tool_call.get("args") or {}).get("content") or "")
        cited_urls = URL_PATTERN.findall(content)
        verified_urls = {
            str(item.get("source_url") or "")
            for item in _network_evidence_records()
        }
        verified_citation_count = sum(
            url.rstrip(".,，。；;") in verified_urls
            for url in cited_urls
        )
        if verified_citation_count < 3:
            errors.append(
                "Markdown 正文仅包含 "
                f"{verified_citation_count} 个已通过相关性校验的来源 URL，"
                "至少需要 3 个"
            )
        unsupported = find_unsupported_precision_claims(
            content,
            _network_evidence_records(),
        )
        if unsupported:
            examples = "；".join(unsupported[:3])
            errors.append(
                "以下精确数字缺少高等级来源原文对齐："
                f"{examples}。请删除数字或改为定性趋势"
            )
        if not errors:
            return None
        return "；".join(errors) + "。不得编造链接或数字。"

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


def _network_evidence_records() -> list[dict[str, Any]]:
    """合并搜索治理目录和 Evidence Pack 网络证据。"""
    records = list(get_search_evidence_records())
    seen_urls = {str(item.get("source_url") or "") for item in records}
    for item in get_evidence_records(source_type="network"):
        source_url = str(item.get("source_url") or "")
        if not source_url or source_url in seen_urls:
            continue
        records.append(
            {
                "source_url": source_url,
                "source_title": item.get("source_name") or "",
                "source_tier": (item.get("metadata") or {}).get("source_tier")
                or "unknown",
                "published_at": (item.get("metadata") or {}).get("published_date")
                or "",
                "evidence_excerpt": item.get("content") or "",
            }
        )
        seen_urls.add(source_url)
    return records


def _has_generated_artifact() -> bool:
    """判断当前会话是否已生成 Markdown 或 PDF 交付物。"""
    session_dir = get_session_context()
    if not session_dir:
        return False
    path = Path(session_dir)
    if not path.exists():
        return False
    return any(
        artifact.is_file()
        for pattern in ("*.md", "*.pdf")
        for artifact in path.glob(pattern)
    )


def _uploaded_source_coverage_error(tool_call: dict[str, Any]) -> str | None:
    """校验 Markdown 产物是否覆盖上传文本中的核心事实。"""
    session_dir = get_session_context()
    if not session_dir:
        return None
    root = Path(session_dir)
    if not root.exists():
        return None

    args = tool_call.get("args") or {}
    content = str(args.get("content") or "")
    filename = str(args.get("filename") or "")
    target_name = filename if filename.lower().endswith(".md") else f"{filename}.md"

    for source_path in root.iterdir():
        if (
            not source_path.is_file()
            or source_path.name == target_name
            or source_path.suffix.lower() not in TEXT_SOURCE_SUFFIXES
            or source_path.stat().st_size > 1_000_000
        ):
            continue
        try:
            source_content = source_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue
        evidence_lines = _source_evidence_lines(source_content)
        if not evidence_lines:
            continue
        coverage = [
            _source_line_is_covered(line, content)
            for line in evidence_lines
        ]
        covered_count = sum(coverage)
        required_count = min(
            3,
            max(1, math.ceil(len(evidence_lines) / 2)),
        )
        if covered_count < required_count:
            missing_examples = [
                line[:160]
                for line, covered in zip(evidence_lines, coverage)
                if not covered
            ][:3]
            return (
                f"Markdown 对上传文件 {source_path.name} 的核心事实覆盖不足"
                f"（已覆盖 {covered_count}/{len(evidence_lines)} 条，"
                f"至少需要 {required_count} 条）；"
                "请保留文件中的具体对象、观察结论和信息缺口，"
                "不得用泛化行业描述替代附件事实。"
                "必须补入的源事实示例："
                + "；".join(missing_examples)
            )
    return None


def _source_evidence_lines(content: str) -> list[str]:
    """从上传文本中提取适合做确定性覆盖校验的事实行。"""
    list_items = []
    paragraphs = []
    for raw_line in str(content or "").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if LIST_ITEM_PATTERN.match(stripped):
            item = LIST_ITEM_PATTERN.sub("", stripped).strip()
            if len(_normalize_source_text(item)) >= 8:
                list_items.append(item)
        elif len(_normalize_source_text(stripped)) >= 16:
            paragraphs.append(stripped)
    selected = list_items if len(list_items) >= 2 else paragraphs
    return list(dict.fromkeys(selected))[:10]


def _source_line_is_covered(source_line: str, content: str) -> bool:
    """使用四字符片段判断源事实是否在产物中得到实质覆盖。"""
    source_text = _normalize_source_text(source_line)
    target_text = _normalize_source_text(content)
    if not source_text or not target_text:
        return False
    if len(source_text) < 8:
        return source_text in target_text
    source_shingles = {
        source_text[index : index + 4]
        for index in range(len(source_text) - 3)
    }
    matched_count = sum(shingle in target_text for shingle in source_shingles)
    required_count = min(
        8,
        max(3, math.ceil(len(source_shingles) * 0.2)),
    )
    return matched_count >= required_count


def _normalize_source_text(text: str) -> str:
    """仅保留中英文和数字，降低标点及格式差异的影响。"""
    return "".join(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", str(text or "").lower())
    )
