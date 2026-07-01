"""Evidence Pack 运行时状态测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.api.context import (
    reset_thread_context,
    reset_trace_context,
    set_thread_context,
    set_trace_context,
)
from app.observability.evidence_pack import (
    begin_evidence_pack,
    get_evidence_pack_snapshot,
    record_evidence,
    reset_evidence_pack,
)
from app.observability.tracing import begin_trace, finish_trace, reset_trace_state


class EvidencePackTests(unittest.TestCase):
    """验证 Evidence Pack 去重、拒绝和 trace 汇总。"""

    def test_records_are_deduplicated_and_counted_by_source(self) -> None:
        token = begin_evidence_pack("trace-evidence-pack")
        try:
            network = record_evidence(
                source_type="network",
                source_name="监管机构报告",
                source_url="https://www.nmpa.gov.cn/report/2026",
                source_locator="search:药品监管趋势",
                content="监管机构发布年度报告。",
                metadata={"source_tier": "tier_1_official"},
                emit_event=False,
            )
            duplicate = record_evidence(
                source_type="network",
                source_name="监管机构报告",
                source_url="https://www.nmpa.gov.cn/report/2026",
                source_locator="search:药品监管趋势",
                content="重复证据。",
                emit_event=False,
            )
            database = record_evidence(
                source_type="database",
                source_name="drugs, inventory",
                source_locator="SELECT ... SUM(quantity_on_hand)",
                content="药品,总库存\n阿莫西林胶囊,18",
                emit_event=False,
            )
            rejected = record_evidence(
                source_type="network",
                source_name="缺少 URL",
                content="没有来源 URL 的网络证据",
                emit_event=False,
            )
            snapshot = get_evidence_pack_snapshot()
        finally:
            reset_evidence_pack(token)

        self.assertIsNotNone(network)
        self.assertIsNone(duplicate)
        self.assertIsNotNone(database)
        self.assertIsNone(rejected)
        self.assertEqual(snapshot["record_count"], 2)
        self.assertEqual(snapshot["rejected_count"], 1)
        self.assertEqual(snapshot["by_source_type"], {"network": 1, "database": 1})

    def test_trace_summary_includes_evidence_pack_snapshot(self) -> None:
        trace_token = set_trace_context("trace-evidence-summary")
        thread_token = set_thread_context("thread-evidence-summary")
        with patch("app.observability.tracing.write_json_log"):
            trace_state_token = begin_trace(
                "trace-evidence-summary",
                "thread-evidence-summary",
                "测试 Evidence Pack",
            )
            evidence_token = begin_evidence_pack("trace-evidence-summary")
            try:
                record_evidence(
                    source_type="file",
                    source_name="brief.md",
                    source_locator="markdown:提取核心观点",
                    content="阿莫西林胶囊库存周转速度较快。",
                    emit_event=False,
                )
                summary = finish_trace("success")
            finally:
                reset_evidence_pack(evidence_token)
                reset_trace_state(trace_state_token)
                reset_thread_context(thread_token)
                reset_trace_context(trace_token)

        self.assertEqual(summary["evidence"]["record_count"], 1)
        self.assertEqual(summary["evidence"]["by_source_type"], {"file": 1})


if __name__ == "__main__":
    unittest.main()
