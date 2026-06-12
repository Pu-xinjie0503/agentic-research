"""Agent、数据库、控制台和质量评测治理测试。"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from app.evaluation.evaluator import TraceView, choose_trace_for_task
from app.evaluation.quality import (
    evaluate_routing_quality,
    evaluate_rule_quality,
)
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.agent.middleware.final_response_governance import (
    FinalResponseGovernanceMiddleware,
)
from app.observability.agent_state import (
    AgentRunState,
    infer_allowed_subagents,
)
from app.observability.database_state import DatabaseRunState
from app.tools.db_tools import validate_read_only_query
from app.tools.tavily_tool import compact_search_result
from app.utils.console import safe_console_print


class AgentGovernanceTests(unittest.TestCase):
    """验证专家默认一次和有理由补充一次。"""

    def test_second_call_requires_explicit_gap(self) -> None:
        state = AgentRunState("trace-agent", soft_limit=3, hard_limit=5)

        first = state.reserve("网络搜索助手", "查询公开资料")
        state.complete(first)
        blocked = state.reserve("网络搜索助手", "再查一次")
        supplement = state.reserve(
            "网络搜索助手",
            "补齐证据 [continuation_reason=evidence_gap]"
            "[target_gap=缺少监管机构来源]",
        )

        self.assertTrue(first.allowed)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.blocked_reason, "missing_supplement_context")
        self.assertTrue(supplement.allowed)
        self.assertTrue(supplement.is_supplement)

    def test_each_agent_has_at_most_two_calls(self) -> None:
        state = AgentRunState("trace-agent", soft_limit=3, hard_limit=5)
        first = state.reserve("数据库查询助手", "首次查询")
        state.complete(first)
        second = state.reserve(
            "数据库查询助手",
            "修正查询 [continuation_reason=correction]"
            "[target_gap=库存聚合口径错误]",
        )
        state.complete(second)
        third = state.reserve(
            "数据库查询助手",
            "再次修正 [continuation_reason=correction]"
            "[target_gap=仍需修改排序]",
        )

        self.assertFalse(third.allowed)
        self.assertEqual(third.blocked_reason, "agent_call_limit")

    def test_unknown_or_out_of_scope_agent_is_blocked(self) -> None:
        state = AgentRunState(
            "trace-agent",
            soft_limit=3,
            hard_limit=5,
            allowed_subagents={"网络搜索助手"},
        )

        file_call = state.reserve("文件分析助手", "读取不存在的附件")
        unknown_call = state.reserve("general-purpose", "尝试通用助手")

        self.assertEqual(file_call.blocked_reason, "unavailable_agent")
        self.assertEqual(unknown_call.blocked_reason, "unavailable_agent")

    def test_task_scope_inference(self) -> None:
        self.assertEqual(
            infer_allowed_subagents(
                "请搜索公开资料并附来源链接",
                has_uploaded_files=False,
            ),
            {"网络搜索助手"},
        )
        self.assertEqual(
            infer_allowed_subagents(
                "分析附件，再用网络搜索补充趋势并生成 PDF",
                has_uploaded_files=True,
            ),
            {"文件分析助手", "网络搜索助手"},
        )
        self.assertEqual(
            infer_allowed_subagents(
                "查询数据库中的库存与销售记录",
                has_uploaded_files=False,
            ),
            {"数据库查询助手"},
        )


class DatabaseGovernanceTests(unittest.TestCase):
    """验证 SQL 只读限制、缓存和弹性预算。"""

    def test_read_only_validation(self) -> None:
        self.assertTrue(validate_read_only_query("SELECT * FROM drugs")[0])
        self.assertTrue(
            validate_read_only_query(
                "WITH totals AS (SELECT drug_id, SUM(quantity_on_hand) total "
                "FROM inventory GROUP BY drug_id) SELECT * FROM totals"
            )[0]
        )
        self.assertFalse(validate_read_only_query("UPDATE drugs SET name='x'")[0])
        self.assertFalse(validate_read_only_query("SELECT 1; SELECT 2")[0])
        self.assertFalse(
            validate_read_only_query(
                "SELECT * FROM drugs INTO OUTFILE '/tmp/drugs.csv'"
            )[0]
        )

    def test_duplicate_sql_uses_cache_without_budget(self) -> None:
        state = DatabaseRunState("trace-db", soft_limit=3, hard_limit=5)
        first = state.reserve_query("SELECT * FROM drugs")
        state.complete_query(first, result="drug_id\n1")
        duplicate = state.reserve_query(" select  *  from drugs; ")

        self.assertFalse(duplicate.allowed)
        self.assertEqual(duplicate.cached_result, "drug_id\n1")
        self.assertEqual(state.snapshot()["reserved_count"], 1)
        self.assertEqual(state.snapshot()["cache_hit_count"], 1)

    def test_fourth_query_requires_gap(self) -> None:
        state = DatabaseRunState("trace-db", soft_limit=3, hard_limit=5)
        for index in range(3):
            reservation = state.reserve_query(f"SELECT {index}")
            state.complete_query(reservation, result=str(index))

        blocked = state.reserve_query("SELECT 4")
        allowed = state.reserve_query(
            "SELECT 4",
            continuation_reason="result_validation",
            target_gap="核验库存并列排序",
        )

        self.assertFalse(blocked.allowed)
        self.assertEqual(
            blocked.blocked_reason,
            "missing_continuation_context",
        )
        self.assertTrue(allowed.allowed)
        self.assertTrue(allowed.is_extension)


class OutputSafetyTests(unittest.TestCase):
    """验证 Windows GBK 控制台不会导致业务假失败。"""

    def test_gbk_console_escapes_unsupported_characters(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="gbk")

        safe_console_print("结果包含 emoji 🚀", stream=stream)
        stream.flush()
        output = buffer.getvalue().decode("gbk")

        self.assertIn("结果包含 emoji", output)
        self.assertIn("\\U0001f680", output)


class ResultCompactionTests(unittest.TestCase):
    """验证外部搜索结果不会无限扩张上下文。"""

    def test_tavily_result_is_compacted(self) -> None:
        compacted = compact_search_result(
            {
                "answer": "a" * 3000,
                "results": [
                    {
                        "title": f"title-{index}",
                        "url": f"https://example.com/{index}",
                        "content": "c" * 2000,
                        "raw_content": "r" * 2000,
                    }
                    for index in range(8)
                ],
            }
        )

        self.assertEqual(len(compacted["results"]), 5)
        self.assertLessEqual(len(compacted["answer"]), 2014)
        self.assertLessEqual(len(compacted["results"][0]["content"]), 1214)


class ToolAllowlistTests(unittest.TestCase):
    """验证内置 todo 和文件系统工具不会暴露给业务 Agent。"""

    def test_only_declared_tools_are_kept(self) -> None:
        class DummyTool:
            def __init__(self, name: str) -> None:
                self.name = name

        middleware = ToolAllowlistMiddleware({"task", "generate_markdown"})
        filtered = middleware.filter_tools(
            [
                DummyTool("write_todos"),
                DummyTool("task"),
                DummyTool("read_file"),
                DummyTool("generate_markdown"),
            ]
        )

        self.assertEqual(
            [tool.name for tool in filtered],
            ["task", "generate_markdown"],
        )


class FinalResponseGovernanceTests(unittest.TestCase):
    """验证引用校验只针对缺少 URL 的最终回答。"""

    def test_final_response_without_urls_needs_retry(self) -> None:
        from langchain.agents.middleware import ModelResponse
        from langchain_core.messages import AIMessage
        from app.observability.search_state import begin_search_run, reset_search_run

        token = begin_search_run("trace-citation")
        try:
            state = __import__(
                "app.observability.search_state",
                fromlist=["get_search_run_state"],
            ).get_search_run_state()
            reservation = state.reserve("测试搜索")
            state.complete(
                reservation,
                {"results": [{"url": "https://example.com/source"}]},
            )
            response = ModelResponse(
                result=[AIMessage(content="只有结论，没有来源链接")]
            )

            self.assertTrue(
                FinalResponseGovernanceMiddleware.needs_citation_retry(response)
            )
        finally:
            reset_search_run(token)

    def test_final_response_with_three_urls_does_not_retry(self) -> None:
        from langchain.agents.middleware import ModelResponse
        from langchain_core.messages import AIMessage
        from app.observability.search_state import begin_search_run, reset_search_run

        token = begin_search_run("trace-citation")
        try:
            state = __import__(
                "app.observability.search_state",
                fromlist=["get_search_run_state"],
            ).get_search_run_state()
            reservation = state.reserve("测试搜索")
            state.complete(
                reservation,
                {"results": [{"url": "https://example.com/source"}]},
            )
            response = ModelResponse(
                result=[
                    AIMessage(
                        content=(
                            "https://a.example.com "
                            "https://b.example.com "
                            "https://c.example.com"
                        )
                    )
                ]
            )

            self.assertFalse(
                FinalResponseGovernanceMiddleware.needs_citation_retry(response)
            )
        finally:
            reset_search_run(token)

    def test_markdown_without_urls_is_blocked_after_search(self) -> None:
        from app.observability.search_state import begin_search_run, reset_search_run

        token = begin_search_run("trace-artifact-citation")
        try:
            state = __import__(
                "app.observability.search_state",
                fromlist=["get_search_run_state"],
            ).get_search_run_state()
            reservation = state.reserve("测试搜索")
            state.complete(
                reservation,
                {"results": [{"url": "https://example.com/source"}]},
            )

            error = FinalResponseGovernanceMiddleware.markdown_citation_error(
                {
                    "name": "generate_markdown",
                    "args": {"content": "# 报告\n只有结论，没有来源链接"},
                }
            )

            self.assertIsNotNone(error)
        finally:
            reset_search_run(token)

    def test_markdown_with_three_urls_is_allowed_after_search(self) -> None:
        from app.observability.search_state import begin_search_run, reset_search_run

        token = begin_search_run("trace-artifact-citation")
        try:
            state = __import__(
                "app.observability.search_state",
                fromlist=["get_search_run_state"],
            ).get_search_run_state()
            reservation = state.reserve("测试搜索")
            state.complete(
                reservation,
                {"results": [{"url": "https://example.com/source"}]},
            )
            content = (
                "# 报告\n"
                "https://a.example.com\n"
                "https://b.example.com\n"
                "https://c.example.com"
            )

            error = FinalResponseGovernanceMiddleware.markdown_citation_error(
                {
                    "name": "generate_markdown",
                    "args": {"content": content},
                }
            )

            self.assertIsNone(error)
        finally:
            reset_search_run(token)


class EvaluationQualityTests(unittest.TestCase):
    """验证精确 trace 匹配和分维度质量。"""

    def test_task_id_matching_wins_over_newer_keyword_trace(self) -> None:
        exact_summary = {
            "timestamp": "2026-06-12T10:00:00",
            "run_metadata": {"evaluation_task_id": "database_low_stock"},
        }
        keyword_summary = {
            "timestamp": "2026-06-12T11:00:00",
            "task_query_summary": "库存数量最低的 5 个药品",
        }
        traces = {
            "exact": TraceView("exact", [exact_summary], exact_summary),
            "keyword": TraceView("keyword", [keyword_summary], keyword_summary),
        }

        selected = choose_trace_for_task(
            {
                "id": "database_low_stock",
                "match_keywords": ["库存数量最低", "5 个药品"],
            },
            traces,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.trace_id, "exact")

    def test_database_quality_requires_name_and_stock(self) -> None:
        truth = [
            {"generic_name": "药品甲", "brand_name": "甲", "total_stock": 12},
            {"generic_name": "药品乙", "brand_name": "乙", "total_stock": 18},
        ]
        quality = evaluate_rule_quality(
            "database_query",
            "| 药品 | 库存 |\n|---|---|\n| 药品甲 | 12 |\n| 药品乙 | 18 |",
            Path(tempfile.gettempdir()) / "missing-artifacts",
            (),
            truth,
        )

        self.assertEqual(
            quality["checks"]["database_fact_accuracy"],
            1.0,
        )
        self.assertEqual(quality["dimensions"]["data"], 1.0)

    def test_routing_quality_reports_missing_and_forbidden(self) -> None:
        result = evaluate_routing_quality(
            {"网络搜索助手": 1, "数据库查询助手": 1},
            ("网络搜索助手", "文件分析助手"),
            ("数据库查询助手",),
        )

        self.assertLess(result["score"], 1.0)
        self.assertEqual(result["missing"], ["文件分析助手"])
        self.assertEqual(result["forbidden_hits"], ["数据库查询助手"])


if __name__ == "__main__":
    unittest.main()
