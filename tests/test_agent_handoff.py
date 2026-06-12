"""结构化子 Agent 交接协议测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.agent.handoff import (
    AgentHandoff,
    HandoffFact,
    MAX_FACT_CHARS,
    MAX_FACTS,
    MAX_SUMMARY_CHARS,
)
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.file_analysis_agent import file_analysis_agent
from app.agent.subagents.network_search_agent import network_search_agent


class AgentHandoffTests(unittest.TestCase):
    """验证交接字段裁剪和来源约束。"""

    def test_valid_handoff_is_serializable(self) -> None:
        handoff = AgentHandoff(
            status="success",
            summary="已完成网络核验",
            facts=[
                {
                    "statement": "监管机构发布了相关政策。",
                    "source_type": "network",
                    "source_name": "监管机构",
                    "source_url": "https://example.com/policy",
                }
            ],
            risks=[],
            gaps=[],
            recommended_next_actions=[],
        )

        payload = handoff.model_dump()

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["facts"][0]["source_type"], "network")

    def test_network_fact_requires_url(self) -> None:
        with self.assertRaisesRegex(ValidationError, "网络事实必须包含"):
            HandoffFact(
                statement="缺少来源的网络结论",
                source_type="network",
                source_name="未知来源",
            )

    def test_database_fact_requires_query_scope(self) -> None:
        with self.assertRaisesRegex(ValidationError, "数据库事实必须包含"):
            HandoffFact(
                statement="库存最低为 10",
                source_type="database",
                source_name="inventory",
            )

    def test_file_fact_requires_filename(self) -> None:
        with self.assertRaisesRegex(ValidationError, "文件事实必须包含"):
            HandoffFact(
                statement="文件指出库存风险",
                source_type="file",
            )

    def test_text_and_list_limits_are_applied(self) -> None:
        handoff = AgentHandoff(
            status="success",
            summary="摘" * (MAX_SUMMARY_CHARS + 20),
            facts=[
                {
                    "statement": "事实" * MAX_FACT_CHARS,
                    "source_type": "file",
                    "source_name": f"file-{index}.md",
                }
                for index in range(MAX_FACTS + 3)
            ],
            risks=["风险"] * 10,
            gaps=["缺口"] * 10,
            recommended_next_actions=["动作"] * 10,
        )

        self.assertLessEqual(len(handoff.summary), MAX_SUMMARY_CHARS)
        self.assertEqual(len(handoff.facts), MAX_FACTS)
        self.assertLessEqual(len(handoff.facts[0].statement), MAX_FACT_CHARS)
        self.assertEqual(len(handoff.risks), 5)
        self.assertEqual(len(handoff.recommended_next_actions), 4)

    def test_all_subagents_use_structured_response_format(self) -> None:
        for agent in (
            database_query_agent,
            file_analysis_agent,
            network_search_agent,
        ):
            self.assertIn("response_format", agent)


if __name__ == "__main__":
    unittest.main()
