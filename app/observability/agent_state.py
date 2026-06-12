"""任务级子 Agent 调用治理状态。"""

from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional


ALLOWED_SUPPLEMENT_REASONS = {
    "evidence_gap",
    "conflict_resolution",
    "correction",
}
KNOWN_SUBAGENTS = {
    "网络搜索助手",
    "数据库查询助手",
    "文件分析助手",
}
_agent_state_ctx: ContextVar[Optional["AgentRunState"]] = ContextVar(
    "agent_run_state",
    default=None,
)


def _read_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量，非法配置回退到默认值。"""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def parse_supplement_context(description: str) -> tuple[Optional[str], str]:
    """从子任务描述中解析补充调用原因和目标缺口。"""
    reason_match = re.search(
        r"\[continuation_reason=(evidence_gap|conflict_resolution|correction)\]",
        description,
        flags=re.IGNORECASE,
    )
    gap_match = re.search(
        r"\[target_gap=([^\]]+)\]",
        description,
        flags=re.IGNORECASE,
    )
    reason = reason_match.group(1).lower() if reason_match else None
    gap = gap_match.group(1).strip() if gap_match else ""
    return reason, gap


@dataclass(frozen=True)
class AgentCallReservation:
    """一次子 Agent 调用的预留结果。"""

    allowed: bool
    subagent_type: str
    call_index: Optional[int] = None
    agent_call_index: Optional[int] = None
    is_supplement: bool = False
    continuation_reason: Optional[str] = None
    target_gap: str = ""
    blocked_reason: Optional[str] = None
    message: str = ""


@dataclass
class AgentRunState:
    """单次任务内的专家调度状态。"""

    trace_id: str
    soft_limit: int
    hard_limit: int
    max_calls_per_agent: int = 2
    allowed_subagents: set[str] = field(
        default_factory=lambda: set(KNOWN_SUBAGENTS)
    )
    attempted_count: int = 0
    reserved_count: int = 0
    completed_count: int = 0
    blocked_count: int = 0
    supplement_count: int = 0
    supplement_in_flight: bool = False
    in_flight_count: int = 0
    stop_reason: Optional[str] = None
    calls_by_agent: dict[str, int] = field(default_factory=dict)
    completed_by_agent: dict[str, int] = field(default_factory=dict)
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    supplement_reasons: list[str] = field(default_factory=list)
    target_gaps: list[str] = field(default_factory=list)
    lock: RLock = field(default_factory=RLock, repr=False)

    def reserve(
        self,
        subagent_type: str,
        description: str,
    ) -> AgentCallReservation:
        """原子预留一次专家调用。"""
        with self.lock:
            self.attempted_count += 1
            current_count = self.calls_by_agent.get(subagent_type, 0)

            if subagent_type not in self.allowed_subagents:
                return self._block(
                    subagent_type,
                    "unavailable_agent",
                    f"当前任务不允许调用 {subagent_type}。",
                )
            if self.stop_reason:
                return self._block(
                    subagent_type,
                    self.stop_reason,
                    f"专家调度已停止：{self.stop_reason}。",
                )
            if self.reserved_count >= self.hard_limit:
                self.stop_reason = "hard_limit"
                return self._block(
                    subagent_type,
                    "hard_limit",
                    f"已达到 {self.hard_limit} 次专家调用硬上限。",
                )
            if current_count >= self.max_calls_per_agent:
                return self._block(
                    subagent_type,
                    "agent_call_limit",
                    f"{subagent_type} 已达到单任务调用上限。",
                )

            is_supplement = current_count > 0
            reason, target_gap = parse_supplement_context(description)
            if is_supplement:
                if self.supplement_in_flight:
                    return self._block(
                        subagent_type,
                        "supplement_in_flight",
                        "同一时间只允许一个专家补充调用。",
                    )
                if reason not in ALLOWED_SUPPLEMENT_REASONS or not target_gap:
                    return self._block(
                        subagent_type,
                        "missing_supplement_context",
                        "第二次调用专家必须声明 continuation_reason 和 target_gap。",
                    )

            self.reserved_count += 1
            self.in_flight_count += 1
            self.calls_by_agent[subagent_type] = current_count + 1
            if is_supplement:
                self.supplement_count += 1
                self.supplement_in_flight = True
                self.supplement_reasons.append(reason or "")
                self.target_gaps.append(target_gap)

            return AgentCallReservation(
                allowed=True,
                subagent_type=subagent_type,
                call_index=self.reserved_count,
                agent_call_index=current_count + 1,
                is_supplement=is_supplement,
                continuation_reason=reason,
                target_gap=target_gap,
                message="专家调用预算预留成功。",
            )

    def complete(
        self,
        reservation: AgentCallReservation,
        error: Optional[BaseException] = None,
    ) -> None:
        """标记专家调用完成。"""
        if not reservation.allowed:
            return
        with self.lock:
            self.in_flight_count = max(0, self.in_flight_count - 1)
            if reservation.is_supplement:
                self.supplement_in_flight = False
            if error is None:
                self.completed_count += 1
                self.completed_by_agent[reservation.subagent_type] = (
                    self.completed_by_agent.get(reservation.subagent_type, 0) + 1
                )
            if self.reserved_count >= self.hard_limit and self.in_flight_count == 0:
                self.stop_reason = "hard_limit"

    def finalize(self, reason: str) -> Optional[dict[str, Any]]:
        """结束本次专家调度状态。"""
        with self.lock:
            if self.attempted_count == 0:
                return None
            if not self.stop_reason:
                self.stop_reason = reason
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """返回可序列化治理摘要。"""
        with self.lock:
            return {
                "trace_id": self.trace_id,
                "soft_limit": self.soft_limit,
                "hard_limit": self.hard_limit,
                "max_calls_per_agent": self.max_calls_per_agent,
                "allowed_subagents": sorted(self.allowed_subagents),
                "attempted_count": self.attempted_count,
                "reserved_count": self.reserved_count,
                "completed_count": self.completed_count,
                "blocked_count": self.blocked_count,
                "supplement_count": self.supplement_count,
                "in_flight_count": self.in_flight_count,
                "calls_by_agent": dict(self.calls_by_agent),
                "completed_by_agent": dict(self.completed_by_agent),
                "blocked_reasons": dict(self.blocked_reasons),
                "supplement_reasons": list(self.supplement_reasons),
                "target_gaps": list(self.target_gaps),
                "remaining_budget": max(0, self.hard_limit - self.reserved_count),
                "stop_reason": self.stop_reason,
                "decision": "stop" if self.stop_reason else "continue_allowed",
            }

    def _block(
        self,
        subagent_type: str,
        reason: str,
        message: str,
    ) -> AgentCallReservation:
        """记录一次被拒绝的专家调用。调用方必须持有锁。"""
        self.blocked_count += 1
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        return AgentCallReservation(
            allowed=False,
            subagent_type=subagent_type,
            blocked_reason=reason,
            message=message,
        )


def begin_agent_run(trace_id: str) -> Token[Optional[AgentRunState]]:
    """创建当前任务的专家调用状态。"""
    soft_limit = _read_positive_int("AGENT_CALL_SOFT_LIMIT", 3)
    hard_limit = max(
        soft_limit,
        _read_positive_int("AGENT_CALL_HARD_LIMIT", 5),
    )
    return _agent_state_ctx.set(
        AgentRunState(
            trace_id=trace_id,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
        )
    )


def get_agent_run_state() -> Optional[AgentRunState]:
    """获取当前任务专家调用状态。"""
    return _agent_state_ctx.get()


def configure_agent_run(allowed_subagents: set[str]) -> None:
    """按用户原始任务和上传文件范围配置本次可用专家。"""
    state = get_agent_run_state()
    if state is None:
        return
    with state.lock:
        state.allowed_subagents = set(allowed_subagents) & KNOWN_SUBAGENTS


def infer_allowed_subagents(
    task_query: str,
    has_uploaded_files: bool,
) -> set[str]:
    """根据明确任务范围推断允许调用的专家集合。"""
    query = task_query.lower()
    allowed: set[str] = set()
    if has_uploaded_files:
        allowed.add("文件分析助手")
    if any(
        keyword in query
        for keyword in (
            "数据库",
            "mysql",
            "库存",
            "销售记录",
            "销量",
            "内部数据",
            "药品数据",
        )
    ):
        allowed.add("数据库查询助手")
    if any(
        keyword in query
        for keyword in (
            "网络",
            "搜索",
            "检索",
            "公开资料",
            "公开信息",
            "市场趋势",
            "最新趋势",
            "来源链接",
        )
    ):
        allowed.add("网络搜索助手")
    return allowed


def get_agent_snapshot() -> Optional[dict[str, Any]]:
    """获取当前专家调用摘要。"""
    state = get_agent_run_state()
    return state.snapshot() if state else None


def finalize_agent_run(reason: str) -> Optional[dict[str, Any]]:
    """结束当前专家调用状态。"""
    state = get_agent_run_state()
    return state.finalize(reason) if state else None


def reset_agent_run(token: Token[Optional[AgentRunState]]) -> None:
    """恢复专家调用上下文。"""
    _agent_state_ctx.reset(token)
