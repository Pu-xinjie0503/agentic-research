"""Agent Governance 配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOVERNANCE_CONFIG = APP_ROOT / "config" / "governance.yml"

ENV_TO_BUDGET_KEY = {
    "AGENT_CALL_SOFT_LIMIT": "agent_call_soft_limit",
    "AGENT_CALL_HARD_LIMIT": "agent_call_hard_limit",
    "AGENT_MAX_CALLS_PER_AGENT": "agent_max_calls_per_agent",
    "INTERNET_SEARCH_SOFT_LIMIT": "internet_search_soft_limit",
    "INTERNET_SEARCH_HARD_LIMIT": "internet_search_hard_limit",
    "INTERNET_SEARCH_SIMILARITY_THRESHOLD": "internet_search_similarity_threshold",
    "DATABASE_QUERY_SOFT_LIMIT": "database_query_soft_limit",
    "DATABASE_QUERY_HARD_LIMIT": "database_query_hard_limit",
}

DEFAULT_ROUTE_POLICIES: dict[str, dict[str, Any]] = {
    "main_agent": {
        "allowed_subagents": ["网络搜索助手", "数据库查询助手", "文件分析助手"],
        "search_hard_limit": 3,
        "allow_network": True,
        "allow_database": True,
        "allow_file": True,
    },
    "network_direct": {
        "allowed_subagents": ["网络搜索助手"],
        "search_hard_limit": 5,
        "allow_network": True,
    },
    "database_direct": {
        "allowed_subagents": ["数据库查询助手"],
        "database_hard_limit": 5,
        "allow_database": True,
    },
    "file_direct": {
        "allowed_subagents": ["文件分析助手"],
        "allow_file": True,
    },
    "memory_direct": {
        "allowed_subagents": [],
    },
}


@dataclass(frozen=True)
class RoutePolicy:
    """单一路由的治理策略。"""

    name: str
    allowed_subagents: frozenset[str]
    search_hard_limit: int | None = None
    database_hard_limit: int | None = None
    allow_network: bool = False
    allow_database: bool = False
    allow_file: bool = False

    def filter_subagents(self, inferred: set[str]) -> set[str]:
        """用路由策略收紧任务推断出的专家集合。"""
        return set(inferred) & set(self.allowed_subagents)

    def snapshot(self) -> dict[str, Any]:
        """返回可写入 trace 的策略摘要。"""
        return {
            "name": self.name,
            "allowed_subagents": sorted(self.allowed_subagents),
            "search_hard_limit": self.search_hard_limit,
            "database_hard_limit": self.database_hard_limit,
            "allow_network": self.allow_network,
            "allow_database": self.allow_database,
            "allow_file": self.allow_file,
        }


@dataclass(frozen=True)
class GovernanceConfig:
    """完整 Governance 配置。"""

    raw: dict[str, Any]

    def tool_allowlist(self, agent_name: str, fallback: set[str]) -> frozenset[str]:
        """读取指定 Agent 的工具白名单。"""
        configured = (
            self.raw.get("agents", {})
            .get(agent_name, {})
            .get("allowed_tools")
        )
        if not isinstance(configured, list):
            return frozenset(fallback)
        tools = {str(item) for item in configured if str(item).strip()}
        return frozenset(tools or fallback)

    def budget_int(self, env_name: str, default: int) -> int:
        """读取预算整数；环境变量优先于 YAML。"""
        env_value = os.getenv(env_name)
        if env_value:
            return _positive_int(env_value, default)
        key = ENV_TO_BUDGET_KEY.get(env_name)
        configured = (self.raw.get("budgets") or {}).get(key) if key else None
        return _positive_int(configured, default)

    def budget_float(self, env_name: str, default: float) -> float:
        """读取预算浮点数；环境变量优先于 YAML。"""
        env_value = os.getenv(env_name)
        if env_value:
            return _bounded_float(env_value, default)
        key = ENV_TO_BUDGET_KEY.get(env_name)
        configured = (self.raw.get("budgets") or {}).get(key) if key else None
        return _bounded_float(configured, default)

    def route_policy(self, route_name: str) -> RoutePolicy:
        """读取路由策略；缺失时使用保守空策略。"""
        route = {
            **(DEFAULT_ROUTE_POLICIES.get(route_name) or {}),
            **((self.raw.get("routes") or {}).get(route_name) or {}),
        }
        allowed_subagents = route.get("allowed_subagents")
        if not isinstance(allowed_subagents, list):
            allowed_subagents = []
        return RoutePolicy(
            name=route_name,
            allowed_subagents=frozenset(
                str(item) for item in allowed_subagents if str(item).strip()
            ),
            search_hard_limit=_optional_positive_int(route.get("search_hard_limit")),
            database_hard_limit=_optional_positive_int(
                route.get("database_hard_limit")
            ),
            allow_network=bool(route.get("allow_network", False)),
            allow_database=bool(route.get("allow_database", False)),
            allow_file=bool(route.get("allow_file", False)),
        )


@lru_cache(maxsize=1)
def load_governance_config() -> GovernanceConfig:
    """加载 Governance YAML；缺失或格式异常时返回空配置。"""
    path = Path(
        os.getenv("DEEPSEARCH_GOVERNANCE_CONFIG", str(DEFAULT_GOVERNANCE_CONFIG))
    )
    if not path.exists():
        return GovernanceConfig(raw={})
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return GovernanceConfig(raw={})
    return GovernanceConfig(raw=data)


def reload_governance_config() -> GovernanceConfig:
    """清空缓存并重新读取配置，供测试使用。"""
    load_governance_config.cache_clear()
    return load_governance_config()


def tool_allowlist(agent_name: str, fallback: set[str]) -> frozenset[str]:
    """便捷读取工具白名单。"""
    return load_governance_config().tool_allowlist(agent_name, fallback)


def budget_int(env_name: str, default: int) -> int:
    """便捷读取整数预算。"""
    return load_governance_config().budget_int(env_name, default)


def budget_float(env_name: str, default: float) -> float:
    """便捷读取浮点预算。"""
    return load_governance_config().budget_float(env_name, default)


def route_policy(route_name: str) -> RoutePolicy:
    """便捷读取路由策略。"""
    return load_governance_config().route_policy(route_name)


def _positive_int(value: Any, default: int) -> int:
    """读取正整数，非法回退默认值。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _optional_positive_int(value: Any) -> int | None:
    """读取可选正整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bounded_float(value: Any, default: float) -> float:
    """读取 0 到 1 之间的浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if 0 < number <= 1 else default
