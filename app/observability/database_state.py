"""任务级数据库查询治理状态。"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional

from app.agent.governance_config import budget_int


ALLOWED_QUERY_REASONS = {
    "evidence_gap",
    "query_correction",
    "result_validation",
}
_database_state_ctx: ContextVar[Optional["DatabaseRunState"]] = ContextVar(
    "database_run_state",
    default=None,
)


def _read_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量。"""
    return budget_int(name, default)


def normalize_sql(query: str) -> str:
    """压缩 SQL 空白并统一大小写，用于缓存和去重。"""
    return re.sub(r"\s+", " ", query).strip().rstrip(";").lower()


@dataclass(frozen=True)
class DatabaseQueryReservation:
    """一次 SQL 查询预算预留结果。"""

    allowed: bool
    normalized_query: str
    call_index: Optional[int] = None
    is_extension: bool = False
    continuation_reason: Optional[str] = None
    target_gap: str = ""
    blocked_reason: Optional[str] = None
    cached_result: Optional[str] = None
    message: str = ""


@dataclass
class DatabaseRunState:
    """单次任务的数据库缓存、预算和停止状态。"""

    trace_id: str
    soft_limit: int
    hard_limit: int
    attempted_count: int = 0
    reserved_count: int = 0
    executed_count: int = 0
    blocked_count: int = 0
    post_block_attempt_count: int = 0
    post_stop_attempt_count: int = 0
    cache_hit_count: int = 0
    in_flight_count: int = 0
    extension_in_flight: bool = False
    stop_reason: Optional[str] = None
    table_list_result: Optional[str] = None
    table_names: set[str] = field(default_factory=set)
    table_previews: dict[str, str] = field(default_factory=dict)
    query_results: dict[str, str] = field(default_factory=dict)
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    continuation_reasons: list[str] = field(default_factory=list)
    target_gaps: list[str] = field(default_factory=list)
    truncated_result_count: int = 0
    lock: RLock = field(default_factory=RLock, repr=False)

    def reserve_query(
        self,
        query: str,
        continuation_reason: Optional[str] = None,
        target_gap: str = "",
    ) -> DatabaseQueryReservation:
        """预留一次 SQL 执行预算，重复 SQL 直接返回缓存。"""
        normalized = normalize_sql(query)
        with self.lock:
            self.attempted_count += 1
            if self.blocked_count > 0:
                self.post_block_attempt_count += 1
            if self.stop_reason:
                self.post_stop_attempt_count += 1
            if normalized in self.query_results:
                self.cache_hit_count += 1
                return DatabaseQueryReservation(
                    allowed=False,
                    normalized_query=normalized,
                    blocked_reason="duplicate_query_cache_hit",
                    cached_result=self.query_results[normalized],
                    message="相同 SQL 已执行，直接复用缓存结果。",
                )
            if self.stop_reason:
                return self._block(normalized, self.stop_reason, "数据库查询已停止。")
            if self.reserved_count >= self.hard_limit:
                self.stop_reason = "hard_limit"
                return self._block(
                    normalized,
                    "hard_limit",
                    f"已达到 {self.hard_limit} 次 SQL 查询硬上限。",
                )

            call_index = self.reserved_count + 1
            is_extension = call_index > self.soft_limit
            reason = (continuation_reason or "").strip() or None
            gap = target_gap.strip()
            if is_extension:
                if self.extension_in_flight:
                    return self._block(
                        normalized,
                        "extension_in_flight",
                        "软预算后只允许一个 SQL 补查请求在途。",
                    )
                if self.in_flight_count > 0:
                    return self._block(
                        normalized,
                        "queries_in_flight",
                        "已有 SQL 尚未完成，不能提前补查。",
                    )
                if reason not in ALLOWED_QUERY_REASONS or not gap:
                    return self._block(
                        normalized,
                        "missing_continuation_context",
                        "第 4、5 次 SQL 必须声明补查原因和目标缺口。",
                    )

            self.reserved_count += 1
            self.in_flight_count += 1
            if is_extension:
                self.extension_in_flight = True
                self.continuation_reasons.append(reason or "")
                self.target_gaps.append(gap)
            return DatabaseQueryReservation(
                allowed=True,
                normalized_query=normalized,
                call_index=call_index,
                is_extension=is_extension,
                continuation_reason=reason,
                target_gap=gap,
                message="SQL 查询预算预留成功。",
            )

    def complete_query(
        self,
        reservation: DatabaseQueryReservation,
        result: Optional[str] = None,
        truncated: bool = False,
        error: Optional[BaseException] = None,
    ) -> None:
        """完成 SQL 查询并缓存结果。"""
        if not reservation.allowed:
            return
        with self.lock:
            self.in_flight_count = max(0, self.in_flight_count - 1)
            if reservation.is_extension:
                self.extension_in_flight = False
            self.executed_count += 1
            if error is None and result is not None:
                self.query_results[reservation.normalized_query] = result
            if truncated:
                self.truncated_result_count += 1
            if self.reserved_count >= self.hard_limit and self.in_flight_count == 0:
                self.stop_reason = "hard_limit"

    def snapshot(self) -> dict[str, Any]:
        """返回不含完整查询正文和结果的摘要。"""
        with self.lock:
            return {
                "trace_id": self.trace_id,
                "soft_limit": self.soft_limit,
                "hard_limit": self.hard_limit,
                "attempted_count": self.attempted_count,
                "reserved_count": self.reserved_count,
                "executed_count": self.executed_count,
                "blocked_count": self.blocked_count,
                "post_block_attempt_count": self.post_block_attempt_count,
                "post_stop_attempt_count": self.post_stop_attempt_count,
                "cache_hit_count": self.cache_hit_count,
                "in_flight_count": self.in_flight_count,
                "previewed_table_count": len(self.table_previews),
                "cached_query_count": len(self.query_results),
                "truncated_result_count": self.truncated_result_count,
                "blocked_reasons": dict(self.blocked_reasons),
                "continuation_reasons": list(self.continuation_reasons),
                "target_gaps": list(self.target_gaps),
                "remaining_budget": max(0, self.hard_limit - self.reserved_count),
                "stop_reason": self.stop_reason,
                "decision": (
                    "stop"
                    if self.stop_reason
                    else "review_required"
                    if self.executed_count >= self.soft_limit
                    else "continue_allowed"
                ),
            }

    def finalize(self, reason: str) -> Optional[dict[str, Any]]:
        """结束数据库治理状态。"""
        with self.lock:
            if (
                self.attempted_count == 0
                and self.table_list_result is None
                and not self.table_previews
            ):
                return None
            if not self.stop_reason:
                self.stop_reason = reason
            return self.snapshot()

    def _block(
        self,
        normalized_query: str,
        reason: str,
        message: str,
    ) -> DatabaseQueryReservation:
        """记录一次被拒绝的 SQL 调用。调用方必须持有锁。"""
        self.blocked_count += 1
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        return DatabaseQueryReservation(
            allowed=False,
            normalized_query=normalized_query,
            blocked_reason=reason,
            message=message,
        )


def begin_database_run(trace_id: str) -> Token[Optional[DatabaseRunState]]:
    """创建当前任务数据库治理状态。"""
    soft_limit = _read_positive_int("DATABASE_QUERY_SOFT_LIMIT", 3)
    hard_limit = max(
        soft_limit,
        _read_positive_int("DATABASE_QUERY_HARD_LIMIT", 5),
    )
    return _database_state_ctx.set(
        DatabaseRunState(
            trace_id=trace_id,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
        )
    )


def get_database_run_state() -> Optional[DatabaseRunState]:
    """获取当前数据库治理状态。"""
    return _database_state_ctx.get()


def get_database_snapshot() -> Optional[dict[str, Any]]:
    """获取数据库治理摘要。"""
    state = get_database_run_state()
    return state.snapshot() if state else None


def configure_database_run(
    *,
    hard_limit: int | None = None,
    soft_limit: int | None = None,
) -> None:
    """按路由策略收紧当前 SQL 查询预算。"""
    state = get_database_run_state()
    if state is None:
        return
    with state.lock:
        if hard_limit and hard_limit > 0:
            state.hard_limit = min(state.hard_limit, hard_limit)
        if soft_limit and soft_limit > 0:
            state.soft_limit = min(state.soft_limit, soft_limit)
        state.soft_limit = min(state.soft_limit, state.hard_limit)


def finalize_database_run(reason: str) -> Optional[dict[str, Any]]:
    """结束当前数据库治理状态。"""
    state = get_database_run_state()
    return state.finalize(reason) if state else None


def reset_database_run(token: Token[Optional[DatabaseRunState]]) -> None:
    """恢复数据库治理上下文。"""
    _database_state_ctx.reset(token)
