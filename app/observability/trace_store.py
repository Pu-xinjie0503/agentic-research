"""SQLite trace 持久化与查询。"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DB = APP_ROOT / "data" / "trace.sqlite3"
DECISION_EVENTS = {
    "execution_route_selected",
    "task_scope_inferred",
    "agent_call_reserved",
    "agent_call_blocked",
    "search_call_reserved",
    "search_call_blocked",
    "database_query_reserved",
    "database_query_blocked",
    "memory_recalled",
    "memory_created",
    "memory_updated",
    "memory_deleted",
    "memory_rejected",
}


def utc_now_iso() -> str:
    """返回 UTC ISO 时间，便于跨机器排序。"""
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    """稳定序列化 JSON 字段。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_loads(value: str | None, default: Any) -> Any:
    """解析 JSON 字段，异常时返回默认值。"""
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class SQLiteTraceStore:
    """面向 Agent 任务审计的轻量 SQLite Trace Store。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建短连接，避免跨线程复用连接。"""
        connection = sqlite3.connect(
            self.path,
            timeout=10,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        """创建任务、事件、工具调用和决策审计表。"""
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    trace_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    status TEXT,
                    route TEXT,
                    route_mode TEXT,
                    task_query_summary TEXT,
                    run_metadata_json TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    total_duration_ms REAL,
                    model_call_count INTEGER DEFAULT 0,
                    total_tokens INTEGER,
                    tool_call_count INTEGER DEFAULT 0,
                    assistant_call_count INTEGER DEFAULT 0,
                    search_executed_count INTEGER DEFAULT 0,
                    database_executed_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_task_runs_updated
                ON task_runs(updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_task_runs_route_mode
                ON task_runs(route_mode, updated_at DESC);

                CREATE TABLE IF NOT EXISTS trace_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    thread_id TEXT,
                    timestamp TEXT,
                    event TEXT,
                    component TEXT,
                    status TEXT,
                    span_id TEXT,
                    parent_span_id TEXT,
                    span_name TEXT,
                    duration_ms REAL,
                    message TEXT,
                    metadata_json TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trace_events_trace
                ON trace_events(trace_id, event_id);

                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    thread_id TEXT,
                    span_id TEXT,
                    tool_name TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    status TEXT,
                    duration_ms REAL,
                    args_json TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                DROP INDEX IF EXISTS idx_tool_calls_span;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_calls_span
                ON tool_calls(trace_id, span_id);

                CREATE INDEX IF NOT EXISTS idx_tool_calls_trace
                ON tool_calls(trace_id, call_id);

                CREATE TABLE IF NOT EXISTS agent_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    thread_id TEXT,
                    timestamp TEXT,
                    event TEXT,
                    component TEXT,
                    status TEXT,
                    decision_type TEXT,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_decisions_trace
                ON agent_decisions(trace_id, decision_id);

                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    thread_id TEXT,
                    evidence_id TEXT,
                    source_type TEXT,
                    source_name TEXT,
                    source_url TEXT,
                    source_locator TEXT,
                    content TEXT,
                    confidence REAL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_records_unique
                ON evidence_records(
                    trace_id,
                    source_type,
                    source_name,
                    source_url,
                    source_locator
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_records_trace
                ON evidence_records(trace_id, evidence_row_id);
                """
            )
            connection.commit()

    def record(self, record: dict[str, Any]) -> None:
        """持久化一条结构化 trace 记录。"""
        with self._lock, closing(self._connect()) as connection:
            self._insert_event(connection, record)
            self._upsert_task_run(connection, record)
            self._upsert_tool_call(connection, record)
            self._insert_decision(connection, record)
            self._upsert_evidence_record(connection, record)
            connection.commit()

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        """写入完整事件，用于审计复盘。"""
        connection.execute(
            """
            INSERT INTO trace_events (
                trace_id, thread_id, timestamp, event, component, status,
                span_id, parent_span_id, span_name, duration_ms, message,
                metadata_json, result_json, error_json, record_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("trace_id"),
                record.get("thread_id"),
                record.get("timestamp"),
                record.get("event"),
                record.get("component"),
                record.get("status"),
                record.get("span_id"),
                record.get("parent_span_id"),
                record.get("span_name"),
                record.get("duration_ms"),
                record.get("message"),
                json_dumps(record.get("metadata") or {}),
                json_dumps(record.get("result") or {}),
                json_dumps(record.get("error") or {}),
                json_dumps(record),
                utc_now_iso(),
            ),
        )

    def _upsert_task_run(
        self,
        connection: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        """从 trace_start 和 trace_summary 派生任务运行摘要。"""
        trace_id = record.get("trace_id")
        if not trace_id:
            return
        now = utc_now_iso()
        if record.get("event") == "trace_start":
            metadata = record.get("metadata") or {}
            connection.execute(
                """
                INSERT INTO task_runs (
                    trace_id, thread_id, status, task_query_summary,
                    run_metadata_json, started_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    task_query_summary = excluded.task_query_summary,
                    run_metadata_json = excluded.run_metadata_json,
                    started_at = COALESCE(task_runs.started_at, excluded.started_at),
                    updated_at = excluded.updated_at
                """,
                (
                    trace_id,
                    record.get("thread_id"),
                    "running",
                    metadata.get("task_query_summary"),
                    json_dumps(metadata.get("run_metadata") or {}),
                    record.get("timestamp"),
                    now,
                    now,
                ),
            )
            return

        if record.get("event") != "trace_summary":
            return

        execution = record.get("execution") or {}
        model = record.get("model") or {}
        search = record.get("search") or {}
        database = record.get("database") or {}
        tool_calls = record.get("tool_calls") or {}
        assistant_calls = record.get("assistant_calls") or {}
        connection.execute(
            """
            INSERT INTO task_runs (
                trace_id, thread_id, status, route, route_mode,
                task_query_summary, run_metadata_json, started_at, ended_at,
                total_duration_ms, model_call_count, total_tokens,
                tool_call_count, assistant_call_count,
                search_executed_count, database_executed_count,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                thread_id = excluded.thread_id,
                status = excluded.status,
                route = excluded.route,
                route_mode = excluded.route_mode,
                task_query_summary = excluded.task_query_summary,
                run_metadata_json = excluded.run_metadata_json,
                started_at = COALESCE(task_runs.started_at, excluded.started_at),
                ended_at = excluded.ended_at,
                total_duration_ms = excluded.total_duration_ms,
                model_call_count = excluded.model_call_count,
                total_tokens = excluded.total_tokens,
                tool_call_count = excluded.tool_call_count,
                assistant_call_count = excluded.assistant_call_count,
                search_executed_count = excluded.search_executed_count,
                database_executed_count = excluded.database_executed_count,
                updated_at = excluded.updated_at
            """,
            (
                trace_id,
                record.get("thread_id"),
                record.get("status"),
                execution.get("route"),
                execution.get("route_mode"),
                record.get("task_query_summary"),
                json_dumps(record.get("run_metadata") or {}),
                record.get("started_at"),
                record.get("timestamp"),
                record.get("total_duration_ms"),
                _int_or_zero(model.get("call_count")),
                _optional_int(model.get("total_tokens")),
                _sum_counts(tool_calls),
                _sum_counts(assistant_calls),
                _int_or_zero(search.get("executed_count")),
                _int_or_zero(database.get("executed_count")),
                now,
                now,
            ),
        )

    def _upsert_tool_call(
        self,
        connection: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        """从 tool_start 和 tool span_end 派生工具调用表。"""
        trace_id = record.get("trace_id")
        span_id = record.get("span_id")
        if not trace_id:
            return
        event = record.get("event")
        component = record.get("component")
        now = utc_now_iso()

        if event == "tool_start":
            metadata = record.get("metadata") or {}
            self._insert_or_update_tool(
                connection,
                trace_id=trace_id,
                thread_id=record.get("thread_id"),
                span_id=span_id,
                tool_name=metadata.get("tool_name"),
                started_at=record.get("timestamp"),
                status="running",
                args_json=json_dumps(metadata.get("args") or {}),
                updated_at=now,
            )
            return

        if event == "span_end" and component == "tool":
            metadata = record.get("metadata") or {}
            tool_name = (
                metadata.get("tool_name")
                or _tool_name_from_span(record.get("span_name"))
            )
            self._insert_or_update_tool(
                connection,
                trace_id=trace_id,
                thread_id=record.get("thread_id"),
                span_id=span_id,
                tool_name=tool_name,
                ended_at=record.get("timestamp"),
                status=record.get("status"),
                duration_ms=record.get("duration_ms"),
                result_json=json_dumps(record.get("result") or {}),
                updated_at=now,
            )

    def _insert_or_update_tool(
        self,
        connection: sqlite3.Connection,
        *,
        trace_id: str,
        thread_id: str | None,
        span_id: str | None,
        tool_name: str | None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str | None = None,
        duration_ms: Any = None,
        args_json: str | None = None,
        result_json: str | None = None,
        updated_at: str,
    ) -> None:
        """按 span_id 合并同一次工具调用的开始和结束记录。"""
        if span_id is None:
            connection.execute(
                """
                INSERT INTO tool_calls (
                    trace_id, thread_id, span_id, tool_name, started_at, ended_at,
                    status, duration_ms, args_json, result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    thread_id,
                    span_id,
                    tool_name,
                    started_at,
                    ended_at,
                    status,
                    duration_ms,
                    args_json,
                    result_json,
                    updated_at,
                    updated_at,
                ),
            )
            return

        connection.execute(
            """
            INSERT INTO tool_calls (
                trace_id, thread_id, span_id, tool_name, started_at, ended_at,
                status, duration_ms, args_json, result_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id, span_id) DO UPDATE SET
                thread_id = COALESCE(excluded.thread_id, tool_calls.thread_id),
                tool_name = COALESCE(excluded.tool_name, tool_calls.tool_name),
                started_at = COALESCE(tool_calls.started_at, excluded.started_at),
                ended_at = COALESCE(excluded.ended_at, tool_calls.ended_at),
                status = COALESCE(excluded.status, tool_calls.status),
                duration_ms = COALESCE(excluded.duration_ms, tool_calls.duration_ms),
                args_json = COALESCE(excluded.args_json, tool_calls.args_json),
                result_json = COALESCE(excluded.result_json, tool_calls.result_json),
                updated_at = excluded.updated_at
            """,
            (
                trace_id,
                thread_id,
                span_id,
                tool_name,
                started_at,
                ended_at,
                status,
                duration_ms,
                args_json,
                result_json,
                updated_at,
                updated_at,
            ),
        )

    def _insert_decision(
        self,
        connection: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        """把路由、治理、记忆等关键决策单独落表。"""
        event = str(record.get("event") or "")
        if event not in DECISION_EVENTS:
            return
        metadata = record.get("metadata") or {}
        connection.execute(
            """
            INSERT INTO agent_decisions (
                trace_id, thread_id, timestamp, event, component, status,
                decision_type, details_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("trace_id"),
                record.get("thread_id"),
                record.get("timestamp"),
                event,
                record.get("component"),
                record.get("status"),
                metadata.get("route")
                or metadata.get("blocked_reason")
                or metadata.get("action")
                or event,
                json_dumps(metadata),
                utc_now_iso(),
            ),
        )

    def _upsert_evidence_record(
        self,
        connection: sqlite3.Connection,
        record: dict[str, Any],
    ) -> None:
        """从 evidence_recorded 事件派生证据表。"""
        if record.get("event") != "evidence_recorded":
            return
        metadata = record.get("metadata") or {}
        evidence = metadata.get("evidence")
        if not isinstance(evidence, dict):
            return
        now = utc_now_iso()
        connection.execute(
            """
            INSERT INTO evidence_records (
                trace_id, thread_id, evidence_id, source_type, source_name,
                source_url, source_locator, content, confidence,
                metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                trace_id, source_type, source_name, source_url, source_locator
            ) DO UPDATE SET
                evidence_id = excluded.evidence_id,
                thread_id = excluded.thread_id,
                content = excluded.content,
                confidence = excluded.confidence,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                record.get("trace_id"),
                record.get("thread_id"),
                evidence.get("evidence_id"),
                evidence.get("source_type"),
                evidence.get("source_name"),
                evidence.get("source_url") or "",
                evidence.get("source_locator") or "",
                evidence.get("content") or "",
                evidence.get("confidence"),
                json_dumps(evidence.get("metadata") or {}),
                now,
                now,
            ),
        )

    def list_task_runs(
        self,
        *,
        limit: int = 50,
        route_mode: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """按更新时间倒序列出任务摘要。"""
        limit = max(1, min(int(limit), 200))
        conditions = []
        params: list[Any] = []
        if route_mode:
            conditions.append("route_mode = ?")
            params.append(route_mode)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM task_runs
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [task_run_from_row(row) for row in rows]

    def get_trace(
        self,
        trace_id: str,
        *,
        event_limit: int = 300,
    ) -> dict[str, Any] | None:
        """返回单个 trace 的任务摘要、事件、工具调用和治理决策。"""
        event_limit = max(1, min(int(event_limit), 1000))
        with closing(self._connect()) as connection:
            task_row = connection.execute(
                "SELECT * FROM task_runs WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if task_row is None:
                return None
            event_rows = connection.execute(
                """
                SELECT *
                FROM trace_events
                WHERE trace_id = ?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (trace_id, event_limit),
            ).fetchall()
            tool_rows = connection.execute(
                """
                SELECT *
                FROM tool_calls
                WHERE trace_id = ?
                ORDER BY call_id ASC
                """,
                (trace_id,),
            ).fetchall()
            decision_rows = connection.execute(
                """
                SELECT *
                FROM agent_decisions
                WHERE trace_id = ?
                ORDER BY decision_id ASC
                """,
                (trace_id,),
            ).fetchall()
            evidence_rows = connection.execute(
                """
                SELECT *
                FROM evidence_records
                WHERE trace_id = ?
                ORDER BY evidence_row_id ASC
                """,
                (trace_id,),
            ).fetchall()
        return {
            "task_run": task_run_from_row(task_row),
            "events": [trace_event_from_row(row) for row in event_rows],
            "tool_calls": [tool_call_from_row(row) for row in tool_rows],
            "agent_decisions": [
                decision_from_row(row) for row in decision_rows
            ],
            "evidence_records": [
                evidence_from_row(row) for row in evidence_rows
            ],
        }

    def summarize(
        self,
        *,
        limit: int = 100,
    ) -> dict[str, Any]:
        """汇总最近 N 条任务的运行指标。"""
        runs = self.list_task_runs(limit=limit)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            grouped.setdefault(str(run.get("route_mode") or "unknown"), []).append(
                run
            )
        return {
            "run_count": len(runs),
            "by_route_mode": {
                route_mode: summarize_runs(items)
                for route_mode, items in grouped.items()
            },
            "overall": summarize_runs(runs),
        }


def task_run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """序列化 task_runs 行。"""
    return {
        "trace_id": row["trace_id"],
        "thread_id": row["thread_id"],
        "status": row["status"],
        "route": row["route"],
        "route_mode": row["route_mode"],
        "task_query_summary": row["task_query_summary"],
        "run_metadata": json_loads(row["run_metadata_json"], {}),
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "total_duration_ms": row["total_duration_ms"],
        "model_call_count": row["model_call_count"],
        "total_tokens": row["total_tokens"],
        "tool_call_count": row["tool_call_count"],
        "assistant_call_count": row["assistant_call_count"],
        "search_executed_count": row["search_executed_count"],
        "database_executed_count": row["database_executed_count"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def trace_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """序列化 trace_events 行。"""
    return {
        "event_id": row["event_id"],
        "trace_id": row["trace_id"],
        "thread_id": row["thread_id"],
        "timestamp": row["timestamp"],
        "event": row["event"],
        "component": row["component"],
        "status": row["status"],
        "span_id": row["span_id"],
        "parent_span_id": row["parent_span_id"],
        "span_name": row["span_name"],
        "duration_ms": row["duration_ms"],
        "message": row["message"],
        "metadata": json_loads(row["metadata_json"], {}),
        "result": json_loads(row["result_json"], {}),
        "error": json_loads(row["error_json"], {}),
    }


def tool_call_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """序列化 tool_calls 行。"""
    return {
        "call_id": row["call_id"],
        "trace_id": row["trace_id"],
        "thread_id": row["thread_id"],
        "span_id": row["span_id"],
        "tool_name": row["tool_name"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "status": row["status"],
        "duration_ms": row["duration_ms"],
        "args": json_loads(row["args_json"], {}),
        "result": json_loads(row["result_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def decision_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """序列化 agent_decisions 行。"""
    return {
        "decision_id": row["decision_id"],
        "trace_id": row["trace_id"],
        "thread_id": row["thread_id"],
        "timestamp": row["timestamp"],
        "event": row["event"],
        "component": row["component"],
        "status": row["status"],
        "decision_type": row["decision_type"],
        "details": json_loads(row["details_json"], {}),
        "created_at": row["created_at"],
    }


def evidence_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """序列化 evidence_records 行。"""
    return {
        "evidence_row_id": row["evidence_row_id"],
        "trace_id": row["trace_id"],
        "thread_id": row["thread_id"],
        "evidence_id": row["evidence_id"],
        "source_type": row["source_type"],
        "source_name": row["source_name"],
        "source_url": row["source_url"],
        "source_locator": row["source_locator"],
        "content": row["content"],
        "confidence": row["confidence"],
        "metadata": json_loads(row["metadata_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def summarize_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """计算一组任务摘要统计。"""
    items = list(runs)
    if not items:
        return {
            "run_count": 0,
            "success_rate": 0.0,
            "metrics": {},
        }
    return {
        "run_count": len(items),
        "success_rate": round(
            sum(1 for item in items if item.get("status") == "success")
            / len(items),
            4,
        ),
        "routes": _count_values(item.get("route") for item in items),
        "metrics": {
            "total_duration_ms": number_stats(
                item.get("total_duration_ms") for item in items
            ),
            "model_call_count": number_stats(
                item.get("model_call_count") for item in items
            ),
            "total_tokens": number_stats(item.get("total_tokens") for item in items),
            "tool_call_count": number_stats(
                item.get("tool_call_count") for item in items
            ),
            "assistant_call_count": number_stats(
                item.get("assistant_call_count") for item in items
            ),
            "search_executed_count": number_stats(
                item.get("search_executed_count") for item in items
            ),
            "database_executed_count": number_stats(
                item.get("database_executed_count") for item in items
            ),
        },
    }


def number_stats(values: Iterable[Any]) -> dict[str, float] | None:
    """计算均值、最小值、最大值。"""
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        return None
    return {
        "average": round(sum(numeric) / len(numeric), 4),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
    }


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    """统计非空字符串值。"""
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sum_counts(value: Any) -> int:
    """汇总计数字典。"""
    if not isinstance(value, dict):
        return 0
    return sum(_int_or_zero(item) for item in value.values())


def _int_or_zero(value: Any) -> int:
    """把整数值转成 int，非法值返回 0。"""
    number = _optional_int(value)
    return number if number is not None else 0


def _optional_int(value: Any) -> int | None:
    """读取可选整数。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _tool_name_from_span(span_name: Any) -> str | None:
    """从 tool.xxx span 名称中提取工具名。"""
    if not span_name:
        return None
    text = str(span_name)
    return text.split(".", 1)[1] if text.startswith("tool.") else text


def build_default_trace_store() -> SQLiteTraceStore | None:
    """按环境变量创建默认 Trace Store；禁用时返回 None。"""
    if os.getenv("DEEPSEARCH_TRACE_SQLITE_DISABLED", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    return SQLiteTraceStore(os.getenv("DEEPSEARCH_TRACE_DB", str(DEFAULT_TRACE_DB)))


trace_store = build_default_trace_store()
