"""任务级 Evidence Pack。"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Literal, Optional


EvidenceSourceType = Literal["network", "database", "file"]
MAX_EVIDENCE_RECORDS = 40
MAX_EVIDENCE_CONTENT_CHARS = 1200
MAX_EVIDENCE_FIELD_CHARS = 500

_evidence_pack_ctx: ContextVar[Optional["EvidencePackState"]] = ContextVar(
    "evidence_pack_state",
    default=None,
)


def _now_iso() -> str:
    """返回本地 ISO 时间。"""
    return datetime.now().isoformat()


def _compact_text(value: Any, limit: int = MAX_EVIDENCE_FIELD_CHARS) -> str:
    """压缩空白并限制证据字段长度。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


@dataclass(frozen=True)
class EvidencePackRecord:
    """一条跨工具统一证据。"""

    evidence_id: str
    source_type: EvidenceSourceType
    source_name: str
    source_url: str = ""
    source_locator: str = ""
    content: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def as_dict(self) -> dict[str, Any]:
        """返回可写入 trace 的结构化证据。"""
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_locator": self.source_locator,
            "content": self.content,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class EvidencePackState:
    """单次任务的证据包状态。"""

    trace_id: str
    records: list[EvidencePackRecord] = field(default_factory=list)
    seen_keys: set[tuple[str, str, str, str]] = field(default_factory=set)
    rejected_count: int = 0
    lock: RLock = field(default_factory=RLock, repr=False)

    def add(
        self,
        *,
        source_type: EvidenceSourceType,
        source_name: str,
        source_url: str = "",
        source_locator: str = "",
        content: str = "",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> EvidencePackRecord | None:
        """新增证据；无可追溯信息或重复证据会被拒绝。"""
        normalized_type = _normalize_source_type(source_type)
        normalized_name = _compact_text(source_name, 240)
        normalized_url = _compact_text(source_url, 1000)
        normalized_locator = _compact_text(source_locator, 500)
        normalized_content = _compact_text(content, MAX_EVIDENCE_CONTENT_CHARS)
        if not _is_traceable(
            normalized_type,
            normalized_name,
            normalized_url,
            normalized_locator,
            normalized_content,
        ):
            with self.lock:
                self.rejected_count += 1
            return None

        key = (
            normalized_type,
            normalized_name,
            normalized_url,
            normalized_locator,
        )
        with self.lock:
            if key in self.seen_keys:
                return None
            if len(self.records) >= MAX_EVIDENCE_RECORDS:
                self.rejected_count += 1
                return None
            evidence_id = f"ev-{len(self.records) + 1:03d}"
            record = EvidencePackRecord(
                evidence_id=evidence_id,
                source_type=normalized_type,
                source_name=normalized_name,
                source_url=normalized_url,
                source_locator=normalized_locator,
                content=normalized_content,
                confidence=_normalize_confidence(confidence),
                metadata=_compact_metadata(metadata or {}),
            )
            self.records.append(record)
            self.seen_keys.add(key)
            return record

    def snapshot(self, limit: int = MAX_EVIDENCE_RECORDS) -> dict[str, Any]:
        """返回有界证据包摘要。"""
        with self.lock:
            records = [record.as_dict() for record in self.records[:limit]]
            counts: dict[str, int] = {}
            for record in self.records:
                counts[record.source_type] = counts.get(record.source_type, 0) + 1
            return {
                "trace_id": self.trace_id,
                "record_count": len(self.records),
                "rejected_count": self.rejected_count,
                "by_source_type": counts,
                "records": records,
            }


def begin_evidence_pack(trace_id: str) -> Token[Optional[EvidencePackState]]:
    """为当前 trace 创建证据包状态。"""
    return _evidence_pack_ctx.set(EvidencePackState(trace_id=trace_id))


def reset_evidence_pack(token: Token[Optional[EvidencePackState]]) -> None:
    """恢复证据包上下文。"""
    _evidence_pack_ctx.reset(token)


def get_evidence_pack_state() -> Optional[EvidencePackState]:
    """获取当前任务证据包状态。"""
    return _evidence_pack_ctx.get()


def record_evidence(
    *,
    source_type: EvidenceSourceType,
    source_name: str,
    source_url: str = "",
    source_locator: str = "",
    content: str = "",
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
    emit_event: bool = True,
) -> dict[str, Any] | None:
    """向当前 Evidence Pack 登记一条证据。"""
    state = get_evidence_pack_state()
    if state is None:
        return None
    record = state.add(
        source_type=source_type,
        source_name=source_name,
        source_url=source_url,
        source_locator=source_locator,
        content=content,
        confidence=confidence,
        metadata=metadata,
    )
    if record is None:
        return None
    payload = record.as_dict()
    if emit_event:
        from app.observability.tracing import record_event

        record_event(
            event_name="evidence_recorded",
            component="evidence_pack",
            message=f"已登记 {record.source_type} 证据：{record.source_name}",
            metadata={"evidence": payload},
        )
    return payload


def get_evidence_pack_snapshot() -> Optional[dict[str, Any]]:
    """获取当前证据包摘要。"""
    state = get_evidence_pack_state()
    return state.snapshot() if state else None


def get_evidence_records(
    source_type: EvidenceSourceType | None = None,
    limit: int = MAX_EVIDENCE_RECORDS,
) -> list[dict[str, Any]]:
    """返回指定来源类型的证据记录。"""
    state = get_evidence_pack_state()
    if state is None or limit <= 0:
        return []
    with state.lock:
        records = [
            record.as_dict()
            for record in state.records
            if source_type is None or record.source_type == source_type
        ]
    return records[:limit]


def finalize_evidence_pack() -> Optional[dict[str, Any]]:
    """记录任务结束时的证据包摘要。"""
    snapshot = get_evidence_pack_snapshot()
    if not snapshot or snapshot["record_count"] == 0:
        return snapshot
    from app.observability.tracing import record_event

    record_event(
        event_name="evidence_pack_finalized",
        component="evidence_pack",
        message="Evidence Pack 已汇总",
        metadata={
            "record_count": snapshot["record_count"],
            "rejected_count": snapshot["rejected_count"],
            "by_source_type": snapshot["by_source_type"],
        },
    )
    return snapshot


def _normalize_source_type(source_type: str) -> EvidenceSourceType:
    """校验并标准化来源类型。"""
    if source_type not in {"network", "database", "file"}:
        raise ValueError(f"不支持的证据来源类型：{source_type}")
    return source_type  # type: ignore[return-value]


def _is_traceable(
    source_type: EvidenceSourceType,
    source_name: str,
    source_url: str,
    source_locator: str,
    content: str,
) -> bool:
    """判断证据是否具备最小可追溯信息。"""
    if not content:
        return False
    if source_type == "network":
        return bool(re.match(r"^https?://", source_url, flags=re.IGNORECASE))
    if source_type == "database":
        return bool(source_name and source_locator)
    return bool(source_name)


def _normalize_confidence(value: float) -> float:
    """限制置信度范围。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return round(min(1.0, max(0.0, number)), 4)


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """压缩证据元数据，避免 trace 过大。"""
    compacted: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (int, float, bool)):
            compacted[str(key)] = value
        elif isinstance(value, (list, tuple)):
            compacted[str(key)] = [
                _compact_text(item, 120) for item in list(value)[:10]
            ]
        elif isinstance(value, dict):
            compacted[str(key)] = {
                str(child_key): _compact_text(child_value, 120)
                for child_key, child_value in list(value.items())[:10]
            }
        else:
            compacted[str(key)] = _compact_text(value, 240)
    return compacted
