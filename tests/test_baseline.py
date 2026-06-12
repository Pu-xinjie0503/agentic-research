"""性能基线运行清单测试。"""

from __future__ import annotations

import unittest

from app.evaluation.baseline import (
    BASELINE_CASES,
    DEFAULT_FIXTURE,
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


if __name__ == "__main__":
    unittest.main()
