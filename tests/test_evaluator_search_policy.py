"""离线评测器的搜索治理规则测试。"""

from __future__ import annotations

import unittest

from app.evaluation.evaluator import TraceView, evaluate_task


def build_task() -> dict:
    """构造只关注搜索治理的最小评测任务。"""
    return {
        "id": "search-policy-test",
        "name": "搜索治理测试",
        "query": "测试查询",
        "expected_status": "success",
        "search_policy": {
            "soft_limit": 3,
            "hard_limit": 5,
            "min_search_calls": 2,
            "min_unique_domains": 2,
        },
    }


def build_trace(search: dict) -> TraceView:
    """构造带新版搜索摘要的 trace。"""
    summary = {
        "event": "trace_summary",
        "trace_id": "trace-eval",
        "thread_id": "thread-eval",
        "status": "success",
        "timestamp": "2026-06-10T12:00:00",
        "task_query_summary": "测试查询",
        "total_duration_ms": 1000,
        "tool_calls": {},
        "assistant_calls": {},
        "search": search,
    }
    return TraceView(
        trace_id="trace-eval",
        records=[summary],
        summary=summary,
    )


def check_result(result: dict, name: str) -> bool:
    """按检查项名称读取评测结果。"""
    return next(
        check["passed"]
        for check in result["checks"]
        if check["name"] == name
    )


class EvaluatorSearchPolicyTests(unittest.TestCase):
    """验证评测器能区分四类搜索行为。"""

    def test_valid_extension_passes(self) -> None:
        result = evaluate_task(
            build_task(),
            build_trace(
                {
                    "reserved_count": 4,
                    "executed_count": 4,
                    "blocked_count": 0,
                    "extension_count": 1,
                    "extension_reasons": ["evidence_gap"],
                    "unique_domain_count": 4,
                    "blocked_reasons": {},
                    "stop_reason": "agent_completed",
                }
            ),
        )

        self.assertTrue(check_result(result, "搜索硬预算"))
        self.assertTrue(check_result(result, "弹性补搜理由"))
        self.assertTrue(check_result(result, "重复搜索"))
        self.assertTrue(check_result(result, "避免过早停止"))

    def test_duplicate_attempt_is_reported(self) -> None:
        result = evaluate_task(
            build_task(),
            build_trace(
                {
                    "reserved_count": 3,
                    "executed_count": 3,
                    "blocked_count": 1,
                    "extension_count": 0,
                    "extension_reasons": [],
                    "unique_domain_count": 3,
                    "blocked_reasons": {"duplicate_query": 1},
                    "stop_reason": "agent_completed",
                }
            ),
        )

        self.assertFalse(check_result(result, "重复搜索"))

    def test_over_budget_is_reported(self) -> None:
        result = evaluate_task(
            build_task(),
            build_trace(
                {
                    "reserved_count": 6,
                    "executed_count": 6,
                    "blocked_count": 0,
                    "extension_count": 3,
                    "extension_reasons": [
                        "evidence_gap",
                        "source_diversity",
                        "conflict_resolution",
                    ],
                    "unique_domain_count": 6,
                    "blocked_reasons": {},
                    "stop_reason": "hard_limit",
                }
            ),
        )

        self.assertFalse(check_result(result, "搜索硬预算"))

    def test_premature_stop_is_reported(self) -> None:
        result = evaluate_task(
            build_task(),
            build_trace(
                {
                    "reserved_count": 1,
                    "executed_count": 1,
                    "blocked_count": 0,
                    "extension_count": 0,
                    "extension_reasons": [],
                    "unique_domain_count": 1,
                    "blocked_reasons": {},
                    "stop_reason": "agent_completed",
                }
            ),
        )

        self.assertFalse(check_result(result, "避免过早停止"))


if __name__ == "__main__":
    unittest.main()
