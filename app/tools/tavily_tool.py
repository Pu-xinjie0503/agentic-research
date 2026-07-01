"""
Tavily 网络搜索工具模块

封装 internet_search 工具，供网络搜索子智能体检索互联网公开信息
工具内部会先通过 monitor 上报调用参数，再请求 Tavily API 返回结构化搜索结果
"""

import os
import re
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

from app.api.monitor import monitor
from app.agent.evidence import classify_source_tier
from app.observability.evidence_pack import record_evidence
from app.observability.search_state import get_search_run_state
from app.observability.tracing import record_event, summarize_text, trace_span

load_dotenv()


# TavilyClient 是实际访问搜索服务的客户端；模块级复用可避免每次工具调用重复初始化
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

GENERIC_QUERY_TERMS = {
    "中国",
    "全球",
    "行业",
    "市场",
    "趋势",
    "最新",
    "政策",
    "发展",
    "变化",
    "研究",
    "分析",
    "相关",
    "公开",
    "信息",
}


def _truncate_text(value: Any, limit: int) -> str:
    """截断外部搜索文本，避免单次工具结果挤占过多上下文。"""
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def compact_search_result(raw_result: Any) -> dict[str, Any]:
    """保留 Tavily 关键结构，并限制结果数量和正文长度。"""
    source = dict(raw_result) if isinstance(raw_result, dict) else {"result": raw_result}
    query = str(source.get("query") or "")
    compact_results = []
    for item in list(source.get("results") or [])[:5]:
        if not isinstance(item, dict):
            continue
        relevance_score = search_result_relevance(query, item)
        if query and relevance_score <= 0:
            continue
        compact_item = {
            "title": _truncate_text(item.get("title"), 300),
            "url": str(item.get("url") or ""),
            "content": _truncate_text(item.get("content"), 600),
            "relevance_score": relevance_score,
        }
        if item.get("score") is not None:
            compact_item["score"] = item["score"]
        if item.get("published_date"):
            compact_item["published_date"] = str(item["published_date"])
        if item.get("raw_content"):
            compact_item["raw_content"] = _truncate_text(item["raw_content"], 600)
        compact_results.append(compact_item)

    result = {
        "query": query,
        "answer": _truncate_text(source.get("answer"), 1000),
        "results": compact_results,
    }
    for key in ("response_time", "request_id"):
        if source.get(key) is not None:
            result[key] = source[key]
    return result


def search_result_relevance(query: str, item: dict[str, Any]) -> float:
    """根据查询核心词与结果文本重合度做保守相关性判断。"""
    normalized_query = str(query or "").lower()
    if not normalized_query:
        return 1.0
    result_text = " ".join(
        str(item.get(field) or "")
        for field in ("title", "content", "raw_content")
    ).lower()
    if not result_text:
        return 0.0

    english_terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9+-]{2,}", normalized_query)
        if token
        not in {"the", "and", "for", "with", "latest", "market", "trend"}
    }
    english_matches = sum(term in result_text for term in english_terms)

    chinese_terms: set[str] = set()
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", normalized_query):
        if chunk not in GENERIC_QUERY_TERMS:
            chinese_terms.add(chunk)
        chinese_terms.update(
            chunk[index : index + 2]
            for index in range(len(chunk) - 1)
            if chunk[index : index + 2] not in GENERIC_QUERY_TERMS
        )
    chinese_matches = sum(term in result_text for term in chinese_terms)
    required_chinese_matches = 2 if len(chinese_terms) >= 4 else 1

    if english_matches > 0 or chinese_matches >= required_chinese_matches:
        denominator = max(1, len(english_terms) + min(6, len(chinese_terms)))
        return round(
            min(1.0, (english_matches + chinese_matches) / denominator),
            4,
        )
    return 0.0


# @tool 会把函数签名和 docstring 暴露给 DeepAgents，模型据此决定是否调用以及如何填参
@tool
def internet_search(
    query: str,
    topic: Literal["news", "finance", "general"] = "general",
    max_results: int = 5,
    include_raw_content: bool = False,
    search_purpose: str = "",
    continuation_reason: Optional[
        Literal["evidence_gap", "source_diversity", "conflict_resolution"]
    ] = None,
    target_gap: str = "",
):
    """
    根据用户问题检索互联网公开信息

    注意：本工具只用于外部公开网页、新闻、政策等信息，不用于查询业务数据库或上传文件内容
    :param query: 搜索关键词或自然语言问题
    :param topic: 搜索主题，可选 news、finance、general
    :param max_results: 返回的最大结果数
    :param include_raw_content: 是否返回网页原文内容；False 返回摘要，True 尝试返回更完整正文
    :param search_purpose: 本次查询相对其他查询的独立目的或角度
    :param continuation_reason: 第 4、5 次补搜原因，只允许证据缺口、来源多样性或冲突核验
    :param target_gap: 第 4、5 次补搜要解决的具体信息缺口
    :return: Tavily 返回的结构化搜索结果
    """
    max_results = max(1, min(int(max_results), 5))
    search_state = get_search_run_state()
    reservation = (
        search_state.reserve(
            query=query,
            continuation_reason=continuation_reason,
            target_gap=target_gap,
        )
        if search_state
        else None
    )

    if reservation is not None and not reservation.allowed:
        control = search_state.control_payload(reservation)
        record_event(
            event_name="search_call_blocked",
            component="search_governance",
            message=reservation.message,
            status="warning",
            metadata={
                "query": summarize_text(query),
                "search_purpose": summarize_text(search_purpose),
                "blocked_reason": reservation.blocked_reason,
                "search_control": control,
            },
        )
        return {
            "query": query,
            "results": [],
            "answer": reservation.message,
            "search_control": control,
        }

    if reservation is not None:
        record_event(
            event_name="search_call_reserved",
            component="search_governance",
            message=f"已预留第 {reservation.call_index} 次搜索预算",
            metadata={
                "call_index": reservation.call_index,
                "query": summarize_text(query),
                "search_purpose": summarize_text(search_purpose),
                "is_extension": reservation.is_extension,
                "continuation_reason": reservation.continuation_reason,
                "target_gap": summarize_text(reservation.target_gap),
            },
        )

    # 工具内部埋点比外层 stream 解析更直接：只要工具被调用，前端就能看到本次搜索参数
    # 这里只上报查询参数，不上报搜索结果正文，避免监控事件体过大
    monitor.report_tool(
        tool_name="网络搜索工具",
        args={
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "search_purpose": search_purpose,
            "continuation_reason": continuation_reason,
            "target_gap": target_gap,
        },
    )

    with trace_span(
        "tool.internet_search",
        component="tool",
        metadata={
            "tool_name": "internet_search",
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "search_purpose": search_purpose,
            "continuation_reason": continuation_reason,
            "target_gap": target_gap,
        },
    ) as span:
        try:
            effective_topic = (
                "general"
                if re.search(r"[\u4e00-\u9fff]", query)
                and re.search(r"政策|行业|市场|经营|集采|医保|药品", query)
                else topic
            )
            # Tavily 返回 query、results、title、url、content 等结构化字段，后续由子智能体阅读并汇总
            raw_result = tavily_client.search(
                query=query,
                topic=effective_topic,
                max_results=max_results,
                include_raw_content=include_raw_content,
            )
            result = compact_search_result(raw_result)
            for item in result.get("results", []):
                if not isinstance(item, dict):
                    continue
                record_evidence(
                    source_type="network",
                    source_name=str(item.get("title") or "网络来源"),
                    source_url=str(item.get("url") or ""),
                    source_locator=f"search:{query}",
                    content=str(item.get("content") or ""),
                    confidence=float(item.get("relevance_score") or 0.8),
                    metadata={
                        "query": query,
                        "search_purpose": search_purpose,
                        "source_tier": classify_source_tier(
                            str(item.get("url") or ""),
                            str(item.get("title") or ""),
                        ),
                        "published_date": item.get("published_date"),
                    },
                )
            control = (
                search_state.complete(reservation, result=result)
                if search_state and reservation
                else {
                    "call_index": None,
                    "new_url_count": None,
                    "unique_domain_count": None,
                    "remaining_budget": None,
                    "decision": "unmanaged",
                    "stop_reason": None,
                    "phase": "UNMANAGED",
                    "blocked": False,
                }
            )
            result["search_control"] = control
            record_event(
                event_name="search_call_completed",
                component="search_governance",
                message="网络搜索执行完成",
                metadata={
                    "call_index": control["call_index"],
                    "new_url_count": control["new_url_count"],
                    "unique_domain_count": control["unique_domain_count"],
                    "remaining_budget": control["remaining_budget"],
                    "decision": control["decision"],
                    "stop_reason": control["stop_reason"],
                },
            )
            span.set_result(
                result_count=len(result.get("results", [])),
                result_summary=summarize_text(result),
                search_control=control,
            )
            return result
        except Exception as exc:
            control = (
                search_state.complete(reservation, error=exc)
                if search_state and reservation
                else None
            )
            record_event(
                event_name="search_call_completed",
                component="search_governance",
                message="网络搜索执行失败",
                status="error",
                metadata={"search_control": control or {}},
                error=exc,
            )
            raise


if __name__ == "__main__":
    from pprint import pprint

    # 本地调试入口：直接运行本文件可验证 TAVILY_API_KEY 和 Tavily API 是否可用
    pprint(
        internet_search.invoke(
            {"query": "2026中国法定节假日放假安排表，我天天都想要放假"}
        )
    )
