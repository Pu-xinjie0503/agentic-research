"""配置化 Agent Governance 测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent.governance_config import (
    budget_float,
    budget_int,
    reload_governance_config,
    route_policy,
    tool_allowlist,
)
from app.observability.database_state import (
    begin_database_run,
    configure_database_run,
    get_database_snapshot,
    reset_database_run,
)
from app.observability.search_state import (
    begin_search_run,
    configure_search_run,
    get_search_snapshot,
    reset_search_run,
)


class GovernanceConfigTests(unittest.TestCase):
    """验证 YAML 配置、环境变量覆盖和路由策略。"""

    def tearDown(self) -> None:
        reload_governance_config()

    def test_default_tool_allowlist_and_route_policy(self) -> None:
        reload_governance_config()

        self.assertIn(
            "internet_search",
            tool_allowlist("network_search_agent", {"fallback_tool"}),
        )
        policy = route_policy("network_direct")

        self.assertEqual(policy.allowed_subagents, frozenset({"网络搜索助手"}))
        self.assertEqual(policy.search_hard_limit, 5)
        self.assertEqual(
            policy.filter_subagents({"网络搜索助手", "数据库查询助手"}),
            {"网络搜索助手"},
        )

    def test_environment_overrides_yaml_budget(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AGENT_CALL_HARD_LIMIT": "7",
                "INTERNET_SEARCH_SIMILARITY_THRESHOLD": "0.5",
            },
            clear=False,
        ):
            reload_governance_config()

            self.assertEqual(budget_int("AGENT_CALL_HARD_LIMIT", 5), 7)
            self.assertEqual(
                budget_float("INTERNET_SEARCH_SIMILARITY_THRESHOLD", 0.82),
                0.5,
            )

    def test_custom_yaml_can_override_route_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "governance.yml"
            config_path.write_text(
                """
budgets:
  internet_search_hard_limit: 2
agents:
  network_search_agent:
    allowed_tools:
      - internet_search
      - custom_search
routes:
  network_direct:
    allowed_subagents:
      - 网络搜索助手
    search_hard_limit: 2
    allow_network: true
""",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"DEEPSEARCH_GOVERNANCE_CONFIG": str(config_path)},
                clear=False,
            ):
                reload_governance_config()

                self.assertEqual(
                    tool_allowlist("network_search_agent", set()),
                    frozenset({"internet_search", "custom_search"}),
                )
                self.assertEqual(
                    budget_int("INTERNET_SEARCH_HARD_LIMIT", 5),
                    2,
                )
                self.assertEqual(route_policy("network_direct").search_hard_limit, 2)

    def test_route_budget_configures_runtime_states(self) -> None:
        search_token = begin_search_run("trace-governance-search")
        database_token = begin_database_run("trace-governance-db")
        try:
            configure_search_run(hard_limit=2)
            configure_database_run(hard_limit=2)

            self.assertEqual(get_search_snapshot()["hard_limit"], 2)
            self.assertEqual(get_database_snapshot()["hard_limit"], 2)
        finally:
            reset_search_run(search_token)
            reset_database_run(database_token)


if __name__ == "__main__":
    unittest.main()
