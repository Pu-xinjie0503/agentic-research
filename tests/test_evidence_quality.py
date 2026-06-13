"""来源分级和精确数字证据治理测试。"""

from __future__ import annotations

import unittest

from app.agent.evidence import (
    EvidenceRecord,
    classify_source_tier,
    downgrade_precision_claim,
    extract_precision_claims,
    find_unsupported_precision_claims,
)
from app.agent.handoff import HandoffFact


class EvidenceQualityTests(unittest.TestCase):
    """验证来源等级、结构化交接和事实级数字对齐。"""

    def test_source_tier_classification(self) -> None:
        self.assertEqual(
            classify_source_tier("https://www.nmpa.gov.cn/policy/1"),
            "tier_1_official",
        )
        self.assertEqual(
            classify_source_tier(
                "https://example.org/report",
                "中国医药行业协会年度报告",
            ),
            "tier_2_primary",
        )
        self.assertEqual(
            classify_source_tier(
                "https://www.grandviewresearch.com/report/1"
            ),
            "tier_4_commercial",
        )

    def test_commercial_numeric_handoff_is_marked_unsupported(self) -> None:
        fact = HandoffFact(
            statement="该市场规模预计达到 51.9 亿美元。",
            source_type="network",
            source_name="Market Size Forecast",
            source_url="https://www.grandviewresearch.com/report/1",
        )

        self.assertNotIn("51.9", fact.statement)
        self.assertIn("未核验数值", fact.statement)

    def test_official_evidence_supports_matching_numeric_claim(self) -> None:
        evidence = EvidenceRecord(
            claim="官方数据显示相关占比为 28.5%。",
            source_url="https://www.nmpa.gov.cn/report/2026",
            source_title="监管机构年度报告",
            evidence_excerpt="年度报告显示相关占比为 28.5%。",
        )

        self.assertEqual(evidence.source_tier, "tier_1_official")
        self.assertTrue(evidence.supports_numeric_claim)

    def test_precision_claim_extraction_ignores_plain_year(self) -> None:
        claims = extract_precision_claims(
            "2026 年行业继续数字化。市场规模预计达到 51.9 亿美元。"
        )

        self.assertEqual(claims, ["市场规模预计达到 51.9 亿美元"])

    def test_precision_claim_extraction_includes_acquisition_amount(self) -> None:
        claims = extract_precision_claims(
            "某企业宣布以约 106 亿美元收购创新药公司。"
        )

        self.assertEqual(claims, ["某企业宣布以约 106 亿美元收购创新药公司"])

    def test_precision_claim_extraction_ignores_section_number(self) -> None:
        claims = extract_precision_claims(
            "### 3.3 人口结构与药品需求变化\n"
            "人口结构变化将影响慢病用药需求。"
        )

        self.assertEqual(claims, [])

    def test_acquisition_amount_is_downgraded(self) -> None:
        result = downgrade_precision_claim(
            "某企业宣布以约 106 亿美元收购创新药公司。"
        )

        self.assertNotIn("106", result)
        self.assertIn("未核验金额", result)

    def test_markdown_claim_requires_matching_authoritative_evidence(self) -> None:
        content = (
            "市场规模预计达到 51.9 亿美元。\n"
            "来源：https://www.nmpa.gov.cn/report/2026"
        )
        evidence = [
            EvidenceRecord(
                source_url="https://www.nmpa.gov.cn/report/2026",
                source_title="监管机构年度报告",
                evidence_excerpt="市场规模预计达到 51.9 亿美元。",
            )
        ]

        self.assertEqual(
            find_unsupported_precision_claims(content, evidence),
            [],
        )

    def test_markdown_claim_rejects_low_tier_or_mismatched_number(self) -> None:
        content = (
            "市场规模预计达到 51.9 亿美元。\n"
            "来源：https://www.grandviewresearch.com/report/1"
        )
        evidence = [
            EvidenceRecord(
                source_url="https://www.grandviewresearch.com/report/1",
                source_title="Market Size Forecast",
                evidence_excerpt="The market will reach USD 51.9 billion.",
            )
        ]

        self.assertEqual(
            find_unsupported_precision_claims(content, evidence),
            ["市场规模预计达到 51.9 亿美元"],
        )


if __name__ == "__main__":
    unittest.main()
