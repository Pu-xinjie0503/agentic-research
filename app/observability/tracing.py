"""
轻量级 trace/span 追踪模块

本模块不依赖 OpenTelemetry，先为当前项目提供可落地的链路追踪能力：
- trace_id 串起一次任务执行；
- span 记录关键节点耗时、状态和摘要；
- trace_summary 汇总整条链路，供后续性能分析和准确率评测使用。
"""

from __future__ import annotations

import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Iterator, Optional

from app.api.context import get_thread_context, get_trace_context
from app.observability.logger import write_json_log


MAX_SUMMARY_LENGTH = 500

_trace_state_ctx: ContextVar[Optional["TraceState"]] = ContextVar(
    "trace_state",
    default=None,
)
_current_span_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "current_span_id",
    default=None,
)


@dataclass
class TraceState:
    """单次任务链路的内存统计状态。"""

    trace_id: str
    thread_id: str
    task_query_summary: str
    run_metadata: dict[str, Any]
    started_at: str
    started_perf: float
    spans: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_call_count: int = 0
    model_call_lock: Lock = field(default_factory=Lock, repr=False)


@dataclass
class SpanRecorder:
    """span 执行期间可被业务代码补充的结果信息。"""

    span_id: str
    span_name: str
    component: str
    metadata: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)

    def set_result(self, **kwargs: Any) -> None:
        """补充 span 的执行结果摘要。"""
        self.result.update(kwargs)


def now_iso() -> str:
    """返回当前本地时间的 ISO 字符串。"""
    return datetime.now().isoformat()


def summarize_text(text: Any, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    """
    将任意值压缩成日志摘要，避免日志写入大段正文或工具返回全文

    :param text: 待摘要对象
    :param max_length: 最大保留字符数
    :return: 可写入日志的短文本
    """
    value = "" if text is None else str(text)
    value = value.replace("\r", " ").replace("\n", " ").strip()
    if len(value) <= max_length:
        return value
    return value[:max_length] + "...<truncated>"


def sanitize_for_log(value: Any, max_length: int = MAX_SUMMARY_LENGTH) -> Any:
    """
    递归清洗日志字段

    字典、列表会保留结构；长文本会截断；未知对象降级为字符串。
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return summarize_text(value, max_length)
    if isinstance(value, dict):
        return {
            str(k): sanitize_for_log(v, max_length)
            for k, v in value.items()
            if not _looks_sensitive(str(k))
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_log(item, max_length) for item in list(value)[:20]]
    return summarize_text(value, max_length)


def _looks_sensitive(key: str) -> bool:
    """过滤明显敏感字段，避免密钥进入日志。"""
    lowered = key.lower()
    if lowered in {"input_tokens", "output_tokens", "total_tokens"}:
        return False
    return any(
        word in lowered
        for word in [
            "api_key",
            "password",
            "access_token",
            "refresh_token",
            "authorization",
            "secret",
        ]
    )


def begin_trace(
    trace_id: str,
    thread_id: str,
    task_query: str,
    run_metadata: Optional[dict[str, Any]] = None,
) -> Token[Optional[TraceState]]:
    """
    创建任务级 trace 状态

    :param trace_id: 单次执行链路 ID
    :param thread_id: 会话 ID
    :param task_query: 用户原始任务，日志中只保存摘要
    :return: reset_trace_state 需要使用的 token
    """
    state = TraceState(
        trace_id=trace_id,
        thread_id=thread_id,
        task_query_summary=summarize_text(task_query),
        run_metadata=sanitize_for_log(run_metadata or {}),
        started_at=now_iso(),
        started_perf=time.perf_counter(),
    )
    token = _trace_state_ctx.set(state)
    record_event(
        event_name="trace_start",
        component="trace",
        message="任务链路开始",
        metadata={
            "task_query_summary": state.task_query_summary,
            "run_metadata": state.run_metadata,
        },
    )
    return token


def reset_trace_state(token: Token[Optional[TraceState]]) -> None:
    """恢复 trace 统计状态。"""
    _trace_state_ctx.reset(token)


def get_current_span_id() -> Optional[str]:
    """获取当前 span ID。"""
    return _current_span_id_ctx.get()


def next_model_call_index() -> int:
    """原子生成当前 trace 内递增的模型调用序号。"""
    state = _trace_state_ctx.get()
    if state is None:
        return 0
    with state.model_call_lock:
        state.model_call_count += 1
        return state.model_call_count


def record_event(
    event_name: str,
    component: str,
    message: str = "",
    status: str = "info",
    metadata: Optional[dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> dict[str, Any]:
    """
    记录一个非耗时型事件

    :return: 实际写入的结构化事件
    """
    state = _trace_state_ctx.get()
    event = {
        "timestamp": now_iso(),
        "event": event_name,
        "component": component,
        "status": status,
        "trace_id": get_trace_context(),
        "thread_id": get_thread_context(),
        "span_id": get_current_span_id(),
        "message": message,
        "metadata": sanitize_for_log(metadata or {}),
    }
    if error is not None:
        event["error"] = {
            "type": type(error).__name__,
            "message": summarize_text(str(error)),
        }

    if state is not None:
        state.events.append(event)

    write_json_log(event)
    return event


@contextmanager
def trace_span(
    span_name: str,
    component: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Iterator[SpanRecorder]:
    """
    记录一个带耗时的执行节点

    :param span_name: 节点名称，例如 tool.execute_sql_query
    :param component: 组件类型，例如 api、agent、tool
    :param metadata: 参数摘要
    """
    span_id = str(uuid.uuid4())
    parent_span_id = get_current_span_id()
    span = SpanRecorder(
        span_id=span_id,
        span_name=span_name,
        component=component,
        metadata=sanitize_for_log(metadata or {}),
    )
    span_token = _current_span_id_ctx.set(span_id)
    started_at = now_iso()
    started_perf = time.perf_counter()

    write_json_log(
        {
            "timestamp": started_at,
            "event": "span_start",
            "component": component,
            "span_name": span_name,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "trace_id": get_trace_context(),
            "thread_id": get_thread_context(),
            "metadata": span.metadata,
        }
    )

    try:
        yield span
    except BaseException as exc:
        status = "cancelled" if type(exc).__name__ == "CancelledError" else "error"
        _finish_span(
            span=span,
            parent_span_id=parent_span_id,
            started_at=started_at,
            started_perf=started_perf,
            status=status,
            error=exc,
        )
        raise
    else:
        _finish_span(
            span=span,
            parent_span_id=parent_span_id,
            started_at=started_at,
            started_perf=started_perf,
            status="success",
        )
    finally:
        _current_span_id_ctx.reset(span_token)


def _finish_span(
    span: SpanRecorder,
    parent_span_id: Optional[str],
    started_at: str,
    started_perf: float,
    status: str,
    error: Optional[BaseException] = None,
) -> None:
    """写入 span 结束事件，并加入当前 trace 汇总。"""
    duration_ms = round((time.perf_counter() - started_perf) * 1000, 2)
    record = {
        "timestamp": now_iso(),
        "event": "span_end",
        "component": span.component,
        "span_name": span.span_name,
        "span_id": span.span_id,
        "parent_span_id": parent_span_id,
        "trace_id": get_trace_context(),
        "thread_id": get_thread_context(),
        "status": status,
        "duration_ms": duration_ms,
        "started_at": started_at,
        "metadata": sanitize_for_log(span.metadata),
        "result": sanitize_for_log(span.result),
    }
    if error is not None:
        record["error"] = {
            "type": type(error).__name__,
            "message": summarize_text(str(error)),
            "traceback": summarize_text(traceback.format_exc(), 1000),
        }

    state = _trace_state_ctx.get()
    if state is not None:
        state.spans.append(record)

    write_json_log(record)


def finish_trace(
    status: str,
    error: Optional[BaseException] = None,
) -> Optional[dict[str, Any]]:
    """
    结束当前 trace，并写入汇总事件

    :param status: success、error 或 cancelled
    :param error: 异常对象，可选
    :return: trace_summary 事件；没有 trace 状态时返回 None
    """
    state = _trace_state_ctx.get()
    if state is None:
        return None

    total_duration_ms = round((time.perf_counter() - state.started_perf) * 1000, 2)
    tool_calls: dict[str, int] = {}
    assistant_calls: dict[str, int] = {}

    governed_assistant_events = [
        event
        for event in state.events
        if event.get("event") == "agent_call_reserved"
    ]

    for event in state.events:
        if event.get("event") == "tool_start":
            tool_name = event.get("metadata", {}).get("tool_name", "unknown")
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
        if governed_assistant_events and event.get("event") == "agent_call_reserved":
            assistant_name = event.get("metadata", {}).get(
                "subagent_type",
                "unknown",
            )
            assistant_calls[assistant_name] = assistant_calls.get(assistant_name, 0) + 1
        elif not governed_assistant_events and event.get("event") == "assistant_call":
            assistant_name = event.get("metadata", {}).get(
                "assistant_name",
                "unknown",
            )
            assistant_calls[assistant_name] = assistant_calls.get(assistant_name, 0) + 1

    route_events = [
        event
        for event in state.events
        if event.get("event") == "execution_route_selected"
    ]
    execution_summary = {}
    if route_events:
        route_metadata = route_events[-1].get("metadata") or {}
        execution_summary = {
            "route": route_metadata.get("route"),
            "route_mode": route_metadata.get("route_mode"),
            "route_policy": route_metadata.get("route_policy"),
            "task_scope": route_metadata.get("task_scope"),
        }

    # 延迟导入避免 tracing 与搜索状态模块形成初始化循环。
    from app.observability.search_state import get_search_snapshot
    from app.observability.agent_state import get_agent_snapshot
    from app.observability.database_state import get_database_snapshot
    from app.observability.evidence_pack import get_evidence_pack_snapshot

    search_summary = get_search_snapshot()
    agent_summary = get_agent_snapshot()
    database_summary = get_database_snapshot()
    evidence_summary = get_evidence_pack_snapshot()
    model_summary = _build_model_summary(state.spans)
    performance_summary = _build_performance_summary(
        state.spans,
        total_duration_ms,
        model_summary,
    )

    summary = {
        "timestamp": now_iso(),
        "event": "trace_summary",
        "component": "trace",
        "trace_id": state.trace_id,
        "thread_id": state.thread_id,
        "status": status,
        "started_at": state.started_at,
        "total_duration_ms": total_duration_ms,
        "task_query_summary": state.task_query_summary,
        "run_metadata": state.run_metadata,
        "span_count": len(state.spans),
        "event_count": len(state.events),
        "tool_calls": tool_calls,
        "assistant_calls": assistant_calls,
        "execution": execution_summary,
        "search": search_summary,
        "agent_governance": agent_summary,
        "database": database_summary,
        "evidence": evidence_summary,
        "model": model_summary,
        "performance": performance_summary,
        "spans": [
            {
                "span_name": span.get("span_name"),
                "component": span.get("component"),
                "status": span.get("status"),
                "duration_ms": span.get("duration_ms"),
            }
            for span in state.spans
        ],
    }
    if error is not None:
        summary["error"] = {
            "type": type(error).__name__,
            "message": summarize_text(str(error)),
        }

    write_json_log(summary)
    return summary


def _build_model_summary(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """按 Agent 汇总模型调用次数、耗时、token 和工具调用。"""
    model_spans = [
        span
        for span in spans
        if span.get("component") == "model"
        and span.get("span_name") == "model.call"
    ]
    summary = _empty_model_metrics()
    by_agent: dict[str, dict[str, Any]] = {}
    call_sequence: list[dict[str, Any]] = []

    for span in model_spans:
        metadata = span.get("metadata") or {}
        result = span.get("result") or {}
        agent_name = str(metadata.get("agent_name") or "unknown")
        call_sequence.append(
            {
                "agent_name": agent_name,
                "call_index": int(metadata.get("call_index") or 0),
                "input_message_count": int(
                    metadata.get("input_message_count") or 0
                ),
                "input_char_count": int(metadata.get("input_char_count") or 0),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
                "total_tokens": result.get("total_tokens"),
                "duration_ms": round(float(span.get("duration_ms") or 0), 2),
                "end_reason": result.get("end_reason"),
                "status": span.get("status"),
            }
        )
        agent_metrics = by_agent.setdefault(agent_name, _empty_model_metrics())
        for metrics in (summary, agent_metrics):
            metrics["call_count"] += 1
            metrics["total_duration_ms"] = round(
                metrics["total_duration_ms"]
                + float(span.get("duration_ms") or 0),
                2,
            )
            if span.get("status") == "success":
                metrics["success_count"] += 1
            else:
                metrics["error_count"] += 1
            metrics["tool_call_count"] += int(result.get("tool_call_count") or 0)
            _add_optional_metric(metrics, "input_tokens", result.get("input_tokens"))
            _add_optional_metric(metrics, "output_tokens", result.get("output_tokens"))
            _add_optional_metric(metrics, "total_tokens", result.get("total_tokens"))
            if result.get("total_tokens") is not None:
                metrics["token_reported_call_count"] += 1

    call_sequence.sort(key=lambda item: item["call_index"])
    summary["calls"] = call_sequence
    _add_context_metrics(summary, call_sequence)
    for agent_name, agent_metrics in by_agent.items():
        agent_calls = [
            call for call in call_sequence if call["agent_name"] == agent_name
        ]
        _add_context_metrics(agent_metrics, agent_calls)
    summary["by_agent"] = by_agent
    return summary


def _build_performance_summary(
    spans: list[dict[str, Any]],
    total_duration_ms: float,
    model_summary: dict[str, Any],
) -> dict[str, Any]:
    """计算模型、工具的累计耗时、墙钟耗时和未归因框架开销。"""
    tool_spans = [span for span in spans if span.get("component") == "tool"]
    tool_duration_ms = round(
        sum(float(span.get("duration_ms") or 0) for span in tool_spans),
        2,
    )
    tool_wall_duration_ms = _merged_span_duration_ms(tool_spans)
    model_spans = [span for span in spans if span.get("component") == "model"]
    model_wall_duration_ms = _merged_span_duration_ms(model_spans)
    unattributed_ms = round(
        max(
            0.0,
            total_duration_ms - tool_wall_duration_ms - model_wall_duration_ms,
        ),
        2,
    )
    return {
        "total_duration_ms": total_duration_ms,
        "model_duration_ms": model_summary["total_duration_ms"],
        "model_wall_duration_ms": model_wall_duration_ms,
        "tool_duration_ms": tool_duration_ms,
        "tool_wall_duration_ms": tool_wall_duration_ms,
        "unattributed_duration_ms": unattributed_ms,
    }


def _merged_span_duration_ms(spans: list[dict[str, Any]]) -> float:
    """合并重叠执行区间，计算并行 span 的真实墙钟耗时。"""
    intervals: list[tuple[datetime, datetime]] = []
    for span in spans:
        started_at = span.get("started_at")
        ended_at = span.get("timestamp")
        if not started_at or not ended_at:
            continue
        try:
            start = datetime.fromisoformat(str(started_at))
            end = datetime.fromisoformat(str(ended_at))
        except ValueError:
            continue
        if end >= start:
            intervals.append((start, end))

    if not intervals:
        return 0.0
    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[datetime, datetime]] = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return round(
        sum((end - start).total_seconds() * 1000 for start, end in merged),
        2,
    )


def _empty_model_metrics() -> dict[str, Any]:
    """创建模型汇总计数器。"""
    return {
        "call_count": 0,
        "success_count": 0,
        "error_count": 0,
        "total_duration_ms": 0.0,
        "tool_call_count": 0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "token_reported_call_count": 0,
        "first_input_char_count": None,
        "last_input_char_count": None,
        "max_input_char_count": None,
        "input_char_growth_rate": None,
        "first_input_tokens": None,
        "last_input_tokens": None,
        "max_input_tokens": None,
        "input_token_growth_rate": None,
    }


def _add_context_metrics(
    metrics: dict[str, Any],
    calls: list[dict[str, Any]],
) -> None:
    """补充首次、末次、最大输入规模和增长率。"""
    if not calls:
        return

    char_values = [int(call["input_char_count"]) for call in calls]
    metrics["first_input_char_count"] = char_values[0]
    metrics["last_input_char_count"] = char_values[-1]
    metrics["max_input_char_count"] = max(char_values)
    metrics["input_char_growth_rate"] = _growth_rate(
        char_values[0],
        char_values[-1],
    )

    token_values = [
        int(call["input_tokens"])
        for call in calls
        if isinstance(call.get("input_tokens"), int)
    ]
    if token_values:
        metrics["first_input_tokens"] = token_values[0]
        metrics["last_input_tokens"] = token_values[-1]
        metrics["max_input_tokens"] = max(token_values)
        metrics["input_token_growth_rate"] = _growth_rate(
            token_values[0],
            token_values[-1],
        )


def _growth_rate(first: int, last: int) -> Optional[float]:
    """计算末次相对首次的增长率，首次为零时不计算。"""
    if first <= 0:
        return None
    return round((last - first) / first, 4)


def _add_optional_metric(
    metrics: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    """累加可选整数指标，未返回 token 时继续保持 None。"""
    if not isinstance(value, int):
        return
    metrics[key] = int(metrics[key] or 0) + value
