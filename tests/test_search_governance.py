"""弹性搜索预算与任务级状态测试。"""

from __future__ import annotations

import importlib
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.api.context import (
    reset_thread_context,
    reset_trace_context,
    set_thread_context,
    set_trace_context,
)
from app.observability.search_state import (
    SearchRunState,
    begin_search_run,
    finalize_search_run,
    get_search_run_state,
    reset_search_run,
)
from app.observability.tracing import begin_trace, finish_trace, reset_trace_state


def build_state(trace_id: str = "trace-test") -> SearchRunState:
    """创建固定预算的测试状态。"""
    return SearchRunState(
        trace_id=trace_id,
        soft_limit=3,
        hard_limit=5,
        similarity_threshold=0.82,
    )


def search_result(*urls: str) -> dict:
    """构造 Tavily 风格的模拟结果。"""
    return {
        "results": [
            {"title": f"来源 {index}", "url": url, "content": "模拟摘要"}
            for index, url in enumerate(urls, 1)
        ]
    }


class DummySpan:
    """替代真实 trace span，避免测试写入日志文件。"""

    def set_result(self, **kwargs) -> None:
        self.result = kwargs


@contextmanager
def dummy_trace_span(*args, **kwargs):
    """提供不落盘的 span 上下文。"""
    yield DummySpan()


class FakeTavilyClient:
    """记录调用次数并返回按查询区分的 URL。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fixed_url: str | None = None

    def search(self, query: str, **kwargs) -> dict:
        self.calls.append(query)
        url = self.fixed_url or f"https://source{len(self.calls)}.example.com/article"
        return search_result(url)


class SearchRunStateTests(unittest.TestCase):
    """验证预算、去重、信息增益和并发约束。"""

    def complete_initial_budget(self, state: SearchRunState) -> None:
        """执行三次不同角度的初始搜索。"""
        for index, query in enumerate(
            ["行业市场规模", "行业监管政策", "行业技术趋势"],
            1,
        ):
            reservation = state.reserve(query)
            self.assertTrue(reservation.allowed)
            state.complete(
                reservation,
                search_result(f"https://source{index}.example.com/article"),
            )

    def test_first_three_distinct_queries_are_allowed(self) -> None:
        state = build_state()
        self.complete_initial_budget(state)

        snapshot = state.snapshot()
        self.assertEqual(snapshot["executed_count"], 3)
        self.assertEqual(snapshot["phase"], "REVIEW_REQUIRED")
        self.assertEqual(snapshot["unique_domain_count"], 3)

    def test_duplicate_query_is_blocked(self) -> None:
        state = build_state()
        first = state.reserve("2026 跨境电商 AI 客服市场趋势")
        state.complete(first, search_result("https://a.example.com/1"))

        duplicate = state.reserve("2026跨境电商AI客服市场趋势")

        self.assertFalse(duplicate.allowed)
        self.assertEqual(duplicate.blocked_reason, "duplicate_query")
        self.assertEqual(state.snapshot()["executed_count"], 1)

    def test_fourth_search_requires_reason_and_target_gap(self) -> None:
        state = build_state()
        self.complete_initial_budget(state)

        blocked = state.reserve("行业头部企业案例")
        allowed = state.reserve(
            "行业头部企业案例",
            continuation_reason="evidence_gap",
            target_gap="缺少头部企业落地案例",
        )

        self.assertFalse(blocked.allowed)
        self.assertEqual(
            blocked.blocked_reason,
            "missing_continuation_context",
        )
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.call_index, 4)

    def test_only_one_extension_may_be_in_flight(self) -> None:
        state = build_state()
        self.complete_initial_budget(state)

        fourth = state.reserve(
            "行业企业实践案例",
            continuation_reason="evidence_gap",
            target_gap="缺少企业实践证据",
        )
        fifth = state.reserve(
            "行业协会统计口径",
            continuation_reason="source_diversity",
            target_gap="现有来源只有商业媒体",
        )

        self.assertTrue(fourth.allowed)
        self.assertFalse(fifth.allowed)
        self.assertEqual(fifth.blocked_reason, "extension_in_flight")

    def test_fourth_parallel_call_waits_for_initial_results(self) -> None:
        state = build_state()
        initial_reservations = [
            state.reserve(query)
            for query in ["行业市场规模", "行业监管政策", "行业技术趋势"]
        ]

        fourth = state.reserve(
            "行业企业实践案例",
            continuation_reason="evidence_gap",
            target_gap="缺少企业实践证据",
        )

        self.assertTrue(all(item.allowed for item in initial_reservations))
        self.assertFalse(fourth.allowed)
        self.assertEqual(
            fourth.blocked_reason,
            "initial_searches_in_flight",
        )

    def test_no_gain_extension_stops_further_search(self) -> None:
        state = build_state()
        self.complete_initial_budget(state)

        fourth = state.reserve(
            "行业企业实践案例",
            continuation_reason="evidence_gap",
            target_gap="缺少企业实践证据",
        )
        state.complete(
            fourth,
            search_result("https://source1.example.com/article"),
        )
        fifth = state.reserve(
            "行业协会统计口径",
            continuation_reason="source_diversity",
            target_gap="现有来源只有商业媒体",
        )

        self.assertFalse(fifth.allowed)
        self.assertEqual(fifth.blocked_reason, "no_information_gain")
        self.assertEqual(state.snapshot()["stop_reason"], "no_information_gain")

    def test_fifth_search_reaches_hard_limit(self) -> None:
        state = build_state()
        self.complete_initial_budget(state)

        for call_index, query in [
            (4, "行业企业实践案例"),
            (5, "行业协会统计口径"),
        ]:
            reservation = state.reserve(
                query,
                continuation_reason=(
                    "evidence_gap"
                    if call_index == 4
                    else "source_diversity"
                ),
                target_gap=f"补齐第 {call_index} 个证据缺口",
            )
            self.assertTrue(reservation.allowed)
            state.complete(
                reservation,
                search_result(
                    f"https://source{call_index}.example.com/article"
                ),
            )

        sixth = state.reserve(
            "行业国际对比",
            continuation_reason="evidence_gap",
            target_gap="缺少国际对比",
        )

        self.assertFalse(sixth.allowed)
        self.assertEqual(sixth.blocked_reason, "hard_limit")
        self.assertEqual(state.snapshot()["executed_count"], 5)

    def test_different_traces_have_independent_budgets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "INTERNET_SEARCH_SOFT_LIMIT": "3",
                "INTERNET_SEARCH_HARD_LIMIT": "5",
            },
        ):
            first_token = begin_search_run("trace-one")
            first_state = get_search_run_state()
            self.assertIsNotNone(first_state)
            first_state.reserve("第一个任务查询")
            reset_search_run(first_token)

            second_token = begin_search_run("trace-two")
            second_state = get_search_run_state()
            self.assertIsNotNone(second_state)
            self.assertEqual(second_state.snapshot()["attempted_count"], 0)
            self.assertEqual(second_state.snapshot()["remaining_budget"], 5)
            reset_search_run(second_token)


class InternetSearchToolTests(unittest.TestCase):
    """验证工具拦截不会请求真实 Tavily。"""

    def test_blocked_duplicate_does_not_call_tavily(self) -> None:
        tavily_module = importlib.import_module("app.tools.tavily_tool")
        fake_client = FakeTavilyClient()
        token = begin_search_run("trace-tool-test")

        try:
            with (
                patch.object(tavily_module, "tavily_client", fake_client),
                patch.object(tavily_module, "record_event"),
                patch.object(tavily_module.monitor, "report_tool"),
                patch.object(tavily_module, "trace_span", dummy_trace_span),
            ):
                first = tavily_module.internet_search.invoke(
                    {
                        "query": "2026 跨境电商 AI 客服趋势",
                        "search_purpose": "市场变化",
                    }
                )
                duplicate = tavily_module.internet_search.invoke(
                    {
                        "query": "2026跨境电商AI客服趋势",
                        "search_purpose": "同义改写",
                    }
                )
        finally:
            reset_search_run(token)

        self.assertEqual(len(fake_client.calls), 1)
        self.assertFalse(first["search_control"]["blocked"])
        self.assertTrue(duplicate["search_control"]["blocked"])
        self.assertEqual(
            duplicate["search_control"]["stop_reason"],
            "duplicate_query",
        )

    def test_valid_extension_calls_fake_tavily(self) -> None:
        tavily_module = importlib.import_module("app.tools.tavily_tool")
        fake_client = FakeTavilyClient()
        token = begin_search_run("trace-extension-test")

        try:
            with (
                patch.object(tavily_module, "tavily_client", fake_client),
                patch.object(tavily_module, "record_event"),
                patch.object(tavily_module.monitor, "report_tool"),
                patch.object(tavily_module, "trace_span", dummy_trace_span),
            ):
                for query in ["市场规模", "监管政策", "技术趋势"]:
                    tavily_module.internet_search.invoke(
                        {"query": query, "search_purpose": query}
                    )
                extension = tavily_module.internet_search.invoke(
                    {
                        "query": "头部企业落地案例",
                        "search_purpose": "补齐落地证据",
                        "continuation_reason": "evidence_gap",
                        "target_gap": "缺少企业实践案例",
                    }
                )
        finally:
            reset_search_run(token)

        self.assertEqual(len(fake_client.calls), 4)
        self.assertEqual(extension["search_control"]["call_index"], 4)
        self.assertEqual(extension["search_control"]["phase"], "EXTENDED")

    def test_trace_summary_contains_search_metrics(self) -> None:
        tavily_module = importlib.import_module("app.tools.tavily_tool")
        tracing_module = importlib.import_module("app.observability.tracing")
        fake_client = FakeTavilyClient()
        trace_token = set_trace_context("trace-summary-test")
        thread_token = set_thread_context("thread-summary-test")

        with patch.object(tracing_module, "write_json_log"):
            trace_state_token = begin_trace(
                "trace-summary-test",
                "thread-summary-test",
                "测试搜索摘要",
            )
            search_token = begin_search_run("trace-summary-test")
            try:
                with (
                    patch.object(tavily_module, "tavily_client", fake_client),
                    patch.object(tavily_module.monitor, "report_tool"),
                ):
                    tavily_module.internet_search.invoke(
                        {
                            "query": "搜索摘要测试",
                            "search_purpose": "验证 trace 汇总",
                        }
                    )
                finalize_search_run("evidence_sufficient")
                summary = finish_trace("success")
            finally:
                reset_search_run(search_token)
                reset_trace_state(trace_state_token)
                reset_thread_context(thread_token)
                reset_trace_context(trace_token)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["search"]["executed_count"], 1)
        self.assertEqual(summary["search"]["unique_domain_count"], 1)
        self.assertEqual(
            summary["search"]["stop_reason"],
            "evidence_sufficient",
        )


if __name__ == "__main__":
    unittest.main()
