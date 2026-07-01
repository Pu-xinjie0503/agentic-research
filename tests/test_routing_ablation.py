"""路由消融评测测试。"""

from __future__ import annotations

import unittest

from app.agent.main_agent import (
    ROUTE_MODE_BASELINE_MAIN_AGENT,
    ROUTE_MODE_OPTIMIZED,
    select_execution_route,
)
from app.evaluation.routing_ablation import (
    _build_summary,
    build_run_plan,
    extract_comparison_metrics,
)
from app.observability.agent_state import infer_task_scope


class RoutingAblationPlanTests(unittest.TestCase):
    """验证消融运行清单不会访问真实外部服务。"""

    def test_plan_builds_paired_runs_for_each_case(self) -> None:
        runs = build_run_plan(["network_search"], repeat_override=2)

        self.assertEqual(len(runs), 4)
        pair_ids = {run.pair_id for run in runs}
        self.assertEqual(len(pair_ids), 2)
        for pair_id in pair_ids:
            modes = {run.route_mode for run in runs if run.pair_id == pair_id}
            self.assertEqual(
                modes,
                {ROUTE_MODE_BASELINE_MAIN_AGENT, ROUTE_MODE_OPTIMIZED},
            )

    def test_baseline_mode_forces_main_agent_route(self) -> None:
        scope = infer_task_scope(
            "搜索 2026 年跨境电商 AI 客服趋势并附来源，不生成文件",
            has_uploaded_files=False,
        )

        self.assertEqual(
            select_execution_route(scope, [], ROUTE_MODE_OPTIMIZED),
            "network_direct",
        )
        self.assertEqual(
            select_execution_route(scope, [], ROUTE_MODE_BASELINE_MAIN_AGENT),
            "main_agent",
        )


class RoutingAblationMetricTests(unittest.TestCase):
    """验证消融报告的核心指标计算。"""

    def test_extract_metrics_from_baseline_result(self) -> None:
        metrics = extract_comparison_metrics(
            {
                "model": {"call_count": 3, "total_tokens": 1200},
                "performance": {
                    "total_duration_ms": 9000,
                    "model_wall_duration_ms": 5000,
                    "tool_wall_duration_ms": 3000,
                },
                "tool_calls": {"internet_search": 2},
                "assistant_calls": {"网络搜索助手": 1},
                "quality": {"rule_score": 0.875},
            }
        )

        self.assertEqual(metrics["model_calls"], 3.0)
        self.assertEqual(metrics["tool_calls"], 2.0)
        self.assertEqual(metrics["assistant_calls"], 1.0)
        self.assertEqual(metrics["total_tokens"], 1200.0)
        self.assertEqual(metrics["rule_quality_score"], 0.875)

    def test_summary_calculates_pair_reduction(self) -> None:
        results = [
            {
                "pair_id": "pair-network-1",
                "route_mode": ROUTE_MODE_BASELINE_MAIN_AGENT,
                "case_id": "network_search",
                "case_name": "网络公开趋势查询",
                "iteration": 1,
                "trace_id": "trace-baseline",
                "passed": True,
                "execution": {"route": "main_agent"},
                "comparison_metrics": {
                    "model_calls": 4.0,
                    "tool_calls": 2.0,
                    "assistant_calls": 1.0,
                    "total_tokens": 2000.0,
                    "total_duration_ms": 10000.0,
                    "model_wall_duration_ms": 6000.0,
                    "tool_wall_duration_ms": 3000.0,
                    "rule_quality_score": 0.8,
                },
            },
            {
                "pair_id": "pair-network-1",
                "route_mode": ROUTE_MODE_OPTIMIZED,
                "case_id": "network_search",
                "case_name": "网络公开趋势查询",
                "iteration": 1,
                "trace_id": "trace-optimized",
                "passed": True,
                "execution": {"route": "network_direct"},
                "comparison_metrics": {
                    "model_calls": 2.0,
                    "tool_calls": 1.0,
                    "assistant_calls": 1.0,
                    "total_tokens": 1200.0,
                    "total_duration_ms": 7000.0,
                    "model_wall_duration_ms": 4000.0,
                    "tool_wall_duration_ms": 2000.0,
                    "rule_quality_score": 0.9,
                },
            },
        ]

        summary = _build_summary(results)

        self.assertEqual(summary["paired_count"], 1)
        overall = summary["comparison"]["overall"]
        self.assertEqual(
            overall["metrics"]["model_calls"]["reduction_rate"]["average"],
            0.5,
        )
        self.assertEqual(
            overall["metrics"]["total_duration_ms"]["reduction_rate"]["average"],
            0.3,
        )
        self.assertEqual(overall["rule_quality_delta"]["average"], 0.1)
        self.assertEqual(
            summary["by_mode"]["network_search"][ROUTE_MODE_OPTIMIZED][
                "execution_routes"
            ],
            {"network_direct": 1},
        )


if __name__ == "__main__":
    unittest.main()
