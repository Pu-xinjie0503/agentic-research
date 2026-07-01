"""Trace SQLite 持久化和 API 测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.api.server as server_module
from app.observability.trace_store import SQLiteTraceStore


def sample_trace_records(trace_id: str = "trace-store-test") -> list[dict]:
    """构造覆盖任务、路由、工具和汇总的最小 trace。"""
    return [
        {
            "timestamp": "2026-06-30T10:00:00",
            "event": "trace_start",
            "component": "trace",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "metadata": {
                "task_query_summary": "查询数据库库存",
                "run_metadata": {
                    "route_mode": "optimized_router",
                    "case_id": "database_query",
                },
            },
        },
        {
            "timestamp": "2026-06-30T10:00:01",
            "event": "execution_route_selected",
            "component": "agent_governance",
            "status": "info",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "metadata": {
                "route": "database_direct",
                "route_mode": "optimized_router",
            },
        },
        {
            "timestamp": "2026-06-30T10:00:02",
            "event": "tool_start",
            "component": "monitor",
            "status": "info",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "span_id": "span-tool-1",
            "metadata": {
                "tool_name": "execute_sql_query",
                "args": {"query": "SELECT 1"},
            },
        },
        {
            "timestamp": "2026-06-30T10:00:03",
            "event": "span_end",
            "component": "tool",
            "span_name": "tool.execute_sql_query",
            "span_id": "span-tool-1",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "status": "success",
            "duration_ms": 120.5,
            "metadata": {"tool_name": "execute_sql_query"},
            "result": {"row_count": 1},
        },
        {
            "timestamp": "2026-06-30T10:00:03.500000",
            "event": "evidence_recorded",
            "component": "evidence_pack",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "status": "info",
            "metadata": {
                "evidence": {
                    "evidence_id": "ev-001",
                    "source_type": "database",
                    "source_name": "inventory",
                    "source_url": "",
                    "source_locator": "SELECT 1",
                    "content": "result\n1",
                    "confidence": 1.0,
                    "metadata": {"tool_name": "execute_sql_query"},
                }
            },
        },
        {
            "timestamp": "2026-06-30T10:00:04",
            "event": "trace_summary",
            "component": "trace",
            "trace_id": trace_id,
            "thread_id": "thread-store-test",
            "status": "success",
            "started_at": "2026-06-30T10:00:00",
            "total_duration_ms": 4000,
            "task_query_summary": "查询数据库库存",
            "run_metadata": {
                "route_mode": "optimized_router",
                "case_id": "database_query",
            },
            "tool_calls": {"execute_sql_query": 1},
            "assistant_calls": {"数据库查询助手": 1},
            "execution": {
                "route": "database_direct",
                "route_mode": "optimized_router",
            },
            "model": {"call_count": 3, "total_tokens": 900},
            "search": {"executed_count": 0},
            "database": {"executed_count": 1},
        },
    ]


class SQLiteTraceStoreTests(unittest.TestCase):
    """验证 trace 数据能从事件流落到结构化表。"""

    def test_records_task_tool_decision_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTraceStore(Path(directory) / "trace.sqlite3")
            for record in sample_trace_records():
                store.record(record)

            trace = store.get_trace("trace-store-test")
            runs = store.list_task_runs()
            stats = store.summarize()

        self.assertIsNotNone(trace)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["route"], "database_direct")
        self.assertEqual(runs[0]["route_mode"], "optimized_router")
        self.assertEqual(runs[0]["model_call_count"], 3)
        self.assertEqual(runs[0]["tool_call_count"], 1)
        self.assertEqual(trace["tool_calls"][0]["tool_name"], "execute_sql_query")
        self.assertEqual(trace["tool_calls"][0]["duration_ms"], 120.5)
        self.assertEqual(trace["evidence_records"][0]["source_type"], "database")
        self.assertEqual(trace["evidence_records"][0]["source_name"], "inventory")
        self.assertEqual(
            trace["agent_decisions"][0]["decision_type"],
            "database_direct",
        )
        self.assertEqual(
            stats["by_route_mode"]["optimized_router"]["routes"],
            {"database_direct": 1},
        )

    def test_logger_writes_jsonl_and_sqlite_without_breaking(self) -> None:
        from app.observability import logger

        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            store = SQLiteTraceStore(temp_dir / "trace.sqlite3")
            with (
                patch.object(logger, "_trace_log_dir", temp_dir / "logs"),
                patch.object(logger, "trace_store", store),
            ):
                for record in sample_trace_records("trace-logger-test"):
                    logger.write_json_log(record)

            trace = store.get_trace("trace-logger-test")
            log_files = list((temp_dir / "logs").glob("*.jsonl"))

        self.assertIsNotNone(trace)
        self.assertTrue(log_files)


class TraceApiTests(unittest.TestCase):
    """验证 trace 查询接口。"""

    def test_trace_api_lists_detail_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTraceStore(Path(directory) / "trace.sqlite3")
            for record in sample_trace_records("trace-api-test"):
                store.record(record)

            with (
                patch.object(server_module, "trace_store", store),
                TestClient(server_module.app) as client,
            ):
                listed = client.get("/api/traces", params={"limit": 10})
                detail = client.get("/api/traces/trace-api-test")
                stats = client.get("/api/trace-stats", params={"limit": 10})
                missing = client.get("/api/traces/not-exists")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["traces"]), 1)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(
            detail.json()["task_run"]["route"],
            "database_direct",
        )
        self.assertEqual(
            detail.json()["evidence_records"][0]["source_locator"],
            "SELECT 1",
        )
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json()["run_count"], 1)
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
