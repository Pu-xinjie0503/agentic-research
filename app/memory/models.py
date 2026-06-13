"""长期记忆结构化模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator


MemoryCategory = Literal["profile", "preference", "project"]
MemoryOperationType = Literal["remember", "forget"]


class MemoryOperation(BaseModel):
    """记忆提取器识别出的一次增删操作。"""

    action: MemoryOperationType
    category: Literal["profile", "preference", "project", "any"]
    key: str = Field(default="", description="稳定、可复用的 snake_case 记忆键")
    content: str = Field(default="", description="要保存的事实或要删除的目标")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explicit: bool = Field(
        default=False,
        description="用户是否明确使用了记住或忘记指令",
    )

    @field_validator("key", "content", mode="before")
    @classmethod
    def compact_text(cls, value: object) -> str:
        """压缩空白并限制单条记忆长度。"""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:500]


class MemoryDecision(BaseModel):
    """结构化记忆意图识别结果。"""

    is_memory_only: bool = False
    remaining_task: str = Field(
        default="",
        description="去掉记忆指令后仍需执行的业务任务",
    )
    list_requested: bool = False
    clear_requested: bool = False
    operations: list[MemoryOperation] = Field(default_factory=list)
    rejection_reason: str = ""

    @field_validator("remaining_task", "rejection_reason", mode="before")
    @classmethod
    def compact_fields(cls, value: object) -> str:
        """压缩模型返回的辅助文本。"""
        return re.sub(r"\s+", " ", str(value or "")).strip()[:1000]


@dataclass(frozen=True)
class MemoryRecord:
    """对外暴露的一条长期记忆。"""

    id: str
    category: MemoryCategory
    key: str
    content: str
    confidence: float
    source_thread_id: str
    source_trace_id: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, object]:
        """转换为 API 可序列化字典。"""
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "content": self.content,
            "confidence": self.confidence,
            "source_thread_id": self.source_thread_id,
            "source_trace_id": self.source_trace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MemoryPreparation:
    """一次任务进入 Agent 前的记忆处理结果。"""

    task_query: str
    prompt: str = ""
    direct_response: str = ""
    recalled: tuple[MemoryRecord, ...] = ()
    created_ids: tuple[str, ...] = ()
    updated_ids: tuple[str, ...] = ()
    deleted_ids: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()


@dataclass
class MemoryMutationSummary:
    """内部累积记忆变更结果。"""

    created_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)
