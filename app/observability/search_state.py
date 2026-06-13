"""
任务级网络搜索治理状态。

状态只存在于当前 trace 的 ContextVar 中，不写入 LangGraph checkpoint。
锁用于保证同一任务内并行工具调用时，预算预留和计数更新保持原子性。
"""

from __future__ import annotations

import os
import re
import unicodedata
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from threading import RLock
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from app.agent.evidence import EvidenceRecord


ALLOWED_CONTINUATION_REASONS = {
    "evidence_gap",
    "source_diversity",
    "conflict_resolution",
}

_search_state_ctx: ContextVar[Optional["SearchRunState"]] = ContextVar(
    "search_run_state",
    default=None,
)


def _read_int_env(name: str, default: int) -> int:
    """读取正整数环境变量，配置非法时使用默认值。"""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _read_float_env(name: str, default: float) -> float:
    """读取 0 到 1 之间的浮点环境变量，配置非法时使用默认值。"""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if 0 < value <= 1 else default


def normalize_query(query: str) -> str:
    """标准化搜索词，用于识别完全重复和高度相似查询。"""
    normalized = unicodedata.normalize("NFKC", query).lower().strip()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def query_similarity(left: str, right: str) -> float:
    """计算两个标准化查询的字符序列和二元组相似度。"""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    sequence_ratio = SequenceMatcher(None, left, right).ratio()
    left_pairs = {left[index : index + 2] for index in range(len(left) - 1)}
    right_pairs = {right[index : index + 2] for index in range(len(right) - 1)}
    if not left_pairs or not right_pairs:
        return sequence_ratio
    pair_ratio = len(left_pairs & right_pairs) / len(left_pairs | right_pairs)
    return max(sequence_ratio, pair_ratio)


def normalize_url(url: str) -> str:
    """规范化 URL，忽略 fragment 和尾部斜杠造成的重复。"""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            "",
        )
    )


@dataclass(frozen=True)
class SearchReservation:
    """一次搜索调用的预留结果。"""

    allowed: bool
    query: str
    normalized_query: str
    call_index: Optional[int] = None
    is_extension: bool = False
    continuation_reason: Optional[str] = None
    target_gap: str = ""
    blocked_reason: Optional[str] = None
    message: str = ""


@dataclass
class SearchRunState:
    """单次任务链路的搜索预算、信息增益和停止状态。"""

    trace_id: str
    soft_limit: int
    hard_limit: int
    similarity_threshold: float
    attempted_count: int = 0
    reserved_count: int = 0
    executed_count: int = 0
    blocked_count: int = 0
    post_block_attempt_count: int = 0
    post_stop_attempt_count: int = 0
    in_flight_count: int = 0
    extension_in_flight_count: int = 0
    extension_count: int = 0
    no_gain_count: int = 0
    phase: str = "INITIAL"
    stop_reason: Optional[str] = None
    normalized_queries: list[str] = field(default_factory=list)
    query_records: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    evidence_urls: list[str] = field(default_factory=list)
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    seen_domains: set[str] = field(default_factory=set)
    blocked_reasons: dict[str, int] = field(default_factory=dict)
    extension_reasons: list[str] = field(default_factory=list)
    last_new_url_count: Optional[int] = None
    last_duplicate_ratio: Optional[float] = None
    last_extension_had_no_gain: bool = False
    lock: RLock = field(default_factory=RLock, repr=False)

    def reserve(
        self,
        query: str,
        continuation_reason: Optional[str] = None,
        target_gap: str = "",
    ) -> SearchReservation:
        """原子预留一次搜索预算，返回允许或拦截决策。"""
        normalized = normalize_query(query)
        with self.lock:
            self.attempted_count += 1
            if self.blocked_count > 0:
                self.post_block_attempt_count += 1
            if self.stop_reason:
                self.post_stop_attempt_count += 1

            if not normalized:
                return self._block(
                    query,
                    normalized,
                    "empty_query",
                    "搜索词为空，已拒绝执行。",
                    call_index=self.reserved_count + 1,
                )

            if self.stop_reason:
                return self._block(
                    query,
                    normalized,
                    self.stop_reason,
                    f"搜索已停止：{self.stop_reason}。",
                    call_index=self.reserved_count + 1,
                )

            if any(
                query_similarity(normalized, previous)
                >= self.similarity_threshold
                for previous in self.normalized_queries
            ):
                return self._block(
                    query,
                    normalized,
                    "duplicate_query",
                    "搜索词与已执行查询完全重复或高度相似，请换一个互补角度。",
                    call_index=self.reserved_count + 1,
                )

            if self.reserved_count >= self.hard_limit:
                self._stop("hard_limit")
                return self._block(
                    query,
                    normalized,
                    "hard_limit",
                    f"已达到 {self.hard_limit} 次搜索硬上限。",
                    call_index=self.reserved_count + 1,
                )

            call_index = self.reserved_count + 1
            is_extension = call_index > self.soft_limit
            reason = (continuation_reason or "").strip() or None
            gap = target_gap.strip()

            if is_extension:
                if self.extension_in_flight_count > 0:
                    return self._block(
                        query,
                        normalized,
                        "extension_in_flight",
                        "软预算后只允许一个定向补搜请求在途。",
                        call_index=call_index,
                        is_extension=True,
                    )
                if self.in_flight_count > 0:
                    return self._block(
                        query,
                        normalized,
                        "initial_searches_in_flight",
                        "前三次搜索尚未全部完成，不能提前提交定向补搜。",
                        call_index=call_index,
                        is_extension=True,
                    )
                if reason not in ALLOWED_CONTINUATION_REASONS or not gap:
                    return self._block(
                        query,
                        normalized,
                        "missing_continuation_context",
                        "软预算后的补搜必须提供 continuation_reason 和 target_gap。",
                        call_index=call_index,
                        is_extension=True,
                    )
                if self.last_extension_had_no_gain:
                    self._stop("no_information_gain")
                    return self._block(
                        query,
                        normalized,
                        "no_information_gain",
                        "上一次定向补搜没有发现新 URL，已停止继续补搜。",
                        call_index=call_index,
                        is_extension=True,
                    )

            self.reserved_count += 1
            self.in_flight_count += 1
            self.normalized_queries.append(normalized)
            if is_extension:
                self.extension_in_flight_count += 1
                self.extension_count += 1
                self.extension_reasons.append(reason or "")
                self.phase = "EXTENDED"
            else:
                self.phase = "INITIAL"

            self.query_records.append(
                {
                    "call_index": call_index,
                    "query": query,
                    "normalized_query": normalized,
                    "continuation_reason": reason,
                    "target_gap": gap,
                    "is_extension": is_extension,
                    "status": "reserved",
                }
            )
            if self.reserved_count >= self.hard_limit:
                self._stop("hard_limit")
            return SearchReservation(
                allowed=True,
                query=query,
                normalized_query=normalized,
                call_index=call_index,
                is_extension=is_extension,
                continuation_reason=reason,
                target_gap=gap,
                message="搜索预算预留成功。",
            )

    def complete(
        self,
        reservation: SearchReservation,
        result: Optional[dict[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> dict[str, Any]:
        """完成一次已预留调用，并根据新 URL 数量计算信息增益。"""
        if not reservation.allowed or reservation.call_index is None:
            return self.control_payload(reservation)

        with self.lock:
            self.in_flight_count = max(0, self.in_flight_count - 1)
            if reservation.is_extension:
                self.extension_in_flight_count = max(
                    0,
                    self.extension_in_flight_count - 1,
                )
            self.executed_count += 1

            urls = self._extract_urls(result or {})
            domains = {
                urlparse(url).netloc.lower()
                for url in urls
                if urlparse(url).netloc
            }
            new_urls = urls - self.seen_urls
            new_domains = domains - self.seen_domains
            duplicate_count = max(0, len(urls) - len(new_urls))
            duplicate_ratio = (
                round(duplicate_count / len(urls), 4) if urls else 0.0
            )

            for url in sorted(new_urls):
                self.evidence_urls.append(url)
            known_evidence_urls = {
                item["source_url"] for item in self.evidence_records
            }
            for item in self._extract_evidence_records(result or {}):
                if item["source_url"] not in known_evidence_urls:
                    self.evidence_records.append(item)
                    known_evidence_urls.add(item["source_url"])
            self.seen_urls.update(urls)
            self.seen_domains.update(domains)
            self.last_new_url_count = len(new_urls)
            self.last_duplicate_ratio = duplicate_ratio

            record = self._find_record(reservation.call_index)
            if record is not None:
                record.update(
                    {
                        "status": "error" if error else "completed",
                        "new_url_count": len(new_urls),
                        "new_domain_count": len(new_domains),
                        "duplicate_ratio": duplicate_ratio,
                    }
                )

            if error is None and len(new_urls) == 0:
                self.no_gain_count += 1
                if reservation.is_extension:
                    self.last_extension_had_no_gain = True
                    self._stop("no_information_gain")
            elif reservation.is_extension:
                self.last_extension_had_no_gain = False

            if self.reserved_count >= self.hard_limit and self.in_flight_count == 0:
                self._stop("hard_limit")
            elif (
                not self.stop_reason
                and self.executed_count >= self.soft_limit
                and self.in_flight_count == 0
            ):
                self.phase = (
                    "EXTENDED" if self.extension_count else "REVIEW_REQUIRED"
                )

            return self.control_payload(reservation)

    def finalize(self, reason: str = "agent_completed") -> Optional[dict[str, Any]]:
        """任务结束时冻结搜索状态；没有搜索行为时不产生停止事件。"""
        with self.lock:
            if self.attempted_count == 0:
                return None
            if not self.stop_reason:
                self._stop(reason)
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """返回可安全写入日志和提示词的摘要，不包含锁和完整 URL 集合。"""
        with self.lock:
            return {
                "trace_id": self.trace_id,
                "phase": self.phase,
                "soft_limit": self.soft_limit,
                "hard_limit": self.hard_limit,
                "attempted_count": self.attempted_count,
                "reserved_count": self.reserved_count,
                "executed_count": self.executed_count,
                "blocked_count": self.blocked_count,
                "post_block_attempt_count": self.post_block_attempt_count,
                "post_stop_attempt_count": self.post_stop_attempt_count,
                "in_flight_count": self.in_flight_count,
                "extension_in_flight_count": self.extension_in_flight_count,
                "extension_count": self.extension_count,
                "extension_reasons": list(self.extension_reasons),
                "unique_url_count": len(self.seen_urls),
                "unique_domain_count": len(self.seen_domains),
                "evidence_tier_counts": self._evidence_tier_counts(),
                "no_gain_count": self.no_gain_count,
                "last_new_url_count": self.last_new_url_count,
                "last_duplicate_ratio": self.last_duplicate_ratio,
                "blocked_reasons": dict(self.blocked_reasons),
                "remaining_budget": max(
                    0,
                    self.hard_limit - self.reserved_count,
                ),
                "stop_reason": self.stop_reason,
                "decision": self._decision(),
            }

    def control_payload(
        self,
        reservation: Optional[SearchReservation] = None,
    ) -> dict[str, Any]:
        """生成返回给网络 Agent 的 search_control 字段。"""
        snapshot = self.snapshot()
        return {
            "call_index": reservation.call_index if reservation else None,
            "new_url_count": (
                0
                if reservation and not reservation.allowed
                else self.last_new_url_count
            ),
            "unique_domain_count": snapshot["unique_domain_count"],
            "remaining_budget": snapshot["remaining_budget"],
            "decision": snapshot["decision"],
            "stop_reason": (
                reservation.blocked_reason
                if reservation and not reservation.allowed
                else snapshot["stop_reason"]
            ),
            "phase": snapshot["phase"],
            "blocked": bool(reservation and not reservation.allowed),
        }

    def _block(
        self,
        query: str,
        normalized_query: str,
        reason: str,
        message: str,
        call_index: Optional[int] = None,
        is_extension: bool = False,
    ) -> SearchReservation:
        """记录一次被治理规则拦截的搜索请求。调用方必须持有锁。"""
        self.blocked_count += 1
        self.blocked_reasons[reason] = self.blocked_reasons.get(reason, 0) + 1
        return SearchReservation(
            allowed=False,
            query=query,
            normalized_query=normalized_query,
            call_index=call_index,
            is_extension=is_extension,
            blocked_reason=reason,
            message=message,
        )

    def _stop(self, reason: str) -> None:
        """设置停止状态，保留最先出现的停止原因。调用方必须持有锁。"""
        if not self.stop_reason:
            self.stop_reason = reason
        self.phase = "STOPPED"

    def _decision(self) -> str:
        """根据当前状态返回模型可理解的继续决策。调用方必须持有锁。"""
        if self.stop_reason:
            return "stop"
        if self.executed_count >= self.soft_limit:
            return "review_required"
        return "continue_allowed"

    def _find_record(self, call_index: int) -> Optional[dict[str, Any]]:
        """查找指定调用序号的内部记录。调用方必须持有锁。"""
        for record in reversed(self.query_records):
            if record["call_index"] == call_index:
                return record
        return None

    def _evidence_tier_counts(self) -> dict[str, int]:
        """统计当前任务网络证据来源等级。调用方必须持有锁。"""
        counts: dict[str, int] = {}
        for item in self.evidence_records:
            tier = str(item.get("source_tier") or "unknown")
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    @staticmethod
    def _extract_evidence_records(result: dict[str, Any]) -> list[dict[str, Any]]:
        """将搜索结果转换为统一证据记录。"""
        records = []
        for item in result.get("results") or []:
            if not isinstance(item, dict):
                continue
            try:
                record = EvidenceRecord(
                    source_url=str(item.get("url") or ""),
                    source_title=str(item.get("title") or ""),
                    published_at=str(item.get("published_date") or ""),
                    evidence_excerpt=str(item.get("content") or ""),
                )
            except ValueError:
                continue
            records.append(record.model_dump())
        return records

    @staticmethod
    def _extract_urls(result: dict[str, Any]) -> set[str]:
        """从 Tavily 结构化结果中提取有效 URL。"""
        urls: set[str] = set()
        for item in result.get("results") or []:
            if not isinstance(item, dict):
                continue
            normalized = normalize_url(str(item.get("url") or ""))
            if normalized:
                urls.add(normalized)
        return urls


def begin_search_run(trace_id: str) -> Token[Optional[SearchRunState]]:
    """为当前 trace 创建独立搜索状态。"""
    soft_limit = _read_int_env("INTERNET_SEARCH_SOFT_LIMIT", 3)
    hard_limit = max(
        soft_limit,
        _read_int_env("INTERNET_SEARCH_HARD_LIMIT", 5),
    )
    state = SearchRunState(
        trace_id=trace_id,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        similarity_threshold=_read_float_env(
            "INTERNET_SEARCH_SIMILARITY_THRESHOLD",
            0.82,
        ),
    )
    return _search_state_ctx.set(state)


def get_search_run_state() -> Optional[SearchRunState]:
    """获取当前任务的搜索治理状态。"""
    return _search_state_ctx.get()


def configure_search_run(*, hard_limit: int) -> None:
    """按任务复杂度收紧当前搜索硬预算，保留环境变量设置的更低上限。"""
    state = get_search_run_state()
    if state is None or hard_limit <= 0:
        return
    with state.lock:
        state.hard_limit = min(state.hard_limit, hard_limit)
        state.soft_limit = min(state.soft_limit, state.hard_limit)


def get_search_snapshot() -> Optional[dict[str, Any]]:
    """获取当前任务可序列化的搜索状态摘要。"""
    state = get_search_run_state()
    return state.snapshot() if state else None


def get_search_evidence_urls(limit: int = 5) -> list[str]:
    """返回已验证搜索结果中的少量 URL，供主智能体引用兜底。"""
    state = get_search_run_state()
    if state is None or limit <= 0:
        return []
    with state.lock:
        return list(state.evidence_urls[:limit])


def get_search_evidence_records(limit: int = 15) -> list[dict[str, Any]]:
    """返回有界网络证据目录，供主智能体和产物校验复用。"""
    state = get_search_run_state()
    if state is None or limit <= 0:
        return []
    with state.lock:
        return [dict(item) for item in state.evidence_records[:limit]]


def finalize_search_run(reason: str = "agent_completed") -> Optional[dict[str, Any]]:
    """结束当前任务的搜索状态。"""
    state = get_search_run_state()
    return state.finalize(reason) if state else None


def reset_search_run(token: Token[Optional[SearchRunState]]) -> None:
    """恢复搜索上下文，确保下一次任务重新计算预算。"""
    _search_state_ctx.reset(token)
