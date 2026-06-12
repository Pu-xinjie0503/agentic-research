"""性能基线运行清单测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.evaluation.baseline import (
    BASELINE_CASES,
    DEFAULT_FIXTURE,
    _aggregate_case,
    _quality_content,
    build_run_plan,
)


class BaselinePlanTests(unittest.TestCase):
    """验证 dry-run 使用的运行清单不访问真实服务。"""

    def test_default_balanced_plan_has_twelve_runs(self) -> None:
        runs = build_run_plan()

        self.assertEqual(len(runs), 12)
        counts = {
            case_id: sum(run.case_id == case_id for run in runs)
            for case_id in BASELINE_CASES
        }
        self.assertEqual(counts["network_search"], 3)
        self.assertEqual(counts["database_query"], 3)
        self.assertEqual(counts["file_analysis"], 3)
        self.assertEqual(counts["multi_agent"], 1)
        self.assertEqual(counts["markdown_delivery"], 1)
        self.assertEqual(counts["pdf_delivery"], 1)
        self.assertEqual(len({run.thread_id for run in runs}), 12)
        self.assertEqual(len({run.trace_id for run in runs}), 12)

    def test_selected_tasks_and_repeat_override(self) -> None:
        runs = build_run_plan(
            ["network_search", "database_query"],
            repeat_override=2,
        )

        self.assertEqual(len(runs), 4)
        self.assertEqual(
            {run.case_id for run in runs},
            {"network_search", "database_query"},
        )

    def test_fixture_exists(self) -> None:
        self.assertTrue(DEFAULT_FIXTURE.is_file())

    def test_quality_judge_none_does_not_break_aggregation(self) -> None:
        summary = _aggregate_case(
            [
                {
                    "passed": True,
                    "performance": {},
                    "model": {},
                    "search": {},
                    "agent_governance": {},
                    "database": {},
                    "quality": {"rule_score": 1.0, "dimensions": {"routing": 1.0}},
                    "quality_judge": None,
                }
            ]
        )

        self.assertIsNone(summary["judge_overall_score"])
        self.assertEqual(
            summary["quality_dimensions"]["routing"]["average"],
            1.0,
        )

    def test_aggregation_includes_retry_and_context_metrics(self) -> None:
        summary = _aggregate_case(
            [
                {
                    "passed": True,
                    "performance": {},
                    "model": {
                        "by_agent": {
                            "主智能体": {
                                "max_input_tokens": 100,
                                "input_token_growth_rate": 0.5,
                                "max_input_char_count": 400,
                                "input_char_growth_rate": 0.25,
                            }
                        }
                    },
                    "search": {"post_block_attempt_count": 2},
                    "agent_governance": {"post_block_attempt_count": 1},
                    "database": {"post_block_attempt_count": 0},
                    "quality": {"rule_score": 1.0, "dimensions": {}},
                    "quality_judge": None,
                }
            ]
        )

        self.assertEqual(
            summary["search_post_block_attempt_count"]["average"],
            2.0,
        )
        self.assertEqual(
            summary["model_context"]["主智能体"]["max_input_tokens"]["average"],
            100.0,
        )

    def test_quality_content_uses_latest_generated_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            uploaded = output_path / "industry_brief.md"
            generated = output_path / "2026_report.md"
            uploaded.write_text("上传附件", encoding="utf-8")
            generated.write_text("生成报告 https://example.com", encoding="utf-8")
            os.utime(uploaded, (1, 1))
            os.utime(generated, (2, 2))

            content = _quality_content(
                "最终摘要",
                output_path,
                ("markdown", "pdf"),
            )

        self.assertIn("生成报告", content)


if __name__ == "__main__":
    unittest.main()
