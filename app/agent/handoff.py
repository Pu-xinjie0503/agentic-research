"""子 Agent 向主 Agent 返回的结构化交接协议。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent.evidence import (
    EvidenceRecord,
    downgrade_precision_claim,
    extract_precision_claims,
)


MAX_SUMMARY_CHARS = 500
MAX_FACTS = 8
MAX_FACT_CHARS = 280
MAX_LIST_ITEMS = 5
MAX_LIST_ITEM_CHARS = 220
MAX_ACTIONS = 4


def _compact_text(value: Any, limit: int) -> str:
    """压缩空白并按字符上限截断文本。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return (
        text
        if len(text) <= limit
        else text[: max(0, limit - 3)].rstrip() + "..."
    )


def _compact_list(value: Any, limit: int) -> list[str]:
    """将任意列表压缩为有限数量的非空短文本。"""
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value[:limit]:
        text = _compact_text(item, MAX_LIST_ITEM_CHARS)
        if text:
            result.append(text)
    return result


class HandoffFact(BaseModel):
    """一条可追溯事实。"""

    statement: str = Field(description="不超过 280 字的事实或关键数字")
    source_type: Literal["network", "database", "file"]
    source_name: str = Field(
        default="",
        description="网页标题、数据库表名或上传文件名",
    )
    source_url: str = Field(
        default="",
        description="网络事实对应的真实 HTTP/HTTPS 来源 URL",
    )
    source_locator: str = Field(
        default="",
        description="数据库查询口径、字段说明或文件章节定位",
    )
    @field_validator(
        "statement",
        "source_name",
        "source_url",
        "source_locator",
        mode="before",
    )
    @classmethod
    def compact_fields(cls, value: Any, info) -> str:
        """按字段用途限制文本长度。"""
        limits = {
            "statement": MAX_FACT_CHARS,
            "source_name": 200,
            "source_url": 1000,
            "source_locator": 300,
        }
        return _compact_text(value, limits[info.field_name])

    @model_validator(mode="after")
    def validate_traceability(self) -> "HandoffFact":
        """确保不同来源的事实保留最小可追溯信息。"""
        if self.source_type == "network" and not re.match(
            r"^https?://",
            self.source_url,
            flags=re.IGNORECASE,
        ):
            raise ValueError("网络事实必须包含真实 HTTP/HTTPS 来源 URL")
        if self.source_type == "network":
            evidence = EvidenceRecord(
                claim=self.statement,
                source_url=self.source_url,
                source_title=self.source_name,
            )
            if (
                extract_precision_claims(self.statement)
                and not evidence.supports_numeric_claim
            ):
                self.statement = downgrade_precision_claim(self.statement)
        if self.source_type == "database" and not self.source_locator:
            raise ValueError("数据库事实必须包含查询口径或字段定位")
        if self.source_type == "file" and not self.source_name:
            raise ValueError("文件事实必须包含上传文件名")
        return self


class AgentHandoff(BaseModel):
    """子 Agent 完成任务后返回给主 Agent 的统一短结果。"""

    status: Literal["success", "partial", "error"]
    summary: str = Field(description="不超过 500 字的核心结论")
    facts: list[HandoffFact] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def compact_summary(cls, value: Any) -> str:
        """限制摘要长度。"""
        return _compact_text(value, MAX_SUMMARY_CHARS)

    @field_validator("facts", mode="before")
    @classmethod
    def limit_facts(cls, value: Any) -> list[Any]:
        """限制事实条目数量，保留最重要的前八项。"""
        return list(value[:MAX_FACTS]) if isinstance(value, (list, tuple)) else []

    @field_validator("risks", "gaps", mode="before")
    @classmethod
    def compact_lists(cls, value: Any) -> list[str]:
        """限制风险和缺口列表。"""
        return _compact_list(value, MAX_LIST_ITEMS)

    @field_validator("recommended_next_actions", mode="before")
    @classmethod
    def compact_actions(cls, value: Any) -> list[str]:
        """限制后续动作列表。"""
        return _compact_list(value, MAX_ACTIONS)

    @model_validator(mode="after")
    def validate_content(self) -> "AgentHandoff":
        """成功或部分成功时必须返回摘要和至少一条事实。"""
        if self.status in {"success", "partial"}:
            if not self.summary:
                raise ValueError("成功交接必须包含 summary")
            if not self.facts:
                raise ValueError("成功交接必须包含至少一条 facts")
        return self


HANDOFF_ERROR_MESSAGE = (
    "结构化交接格式无效。请只基于已有工具结果修正字段，不要重复调用业务工具。"
    "网络事实必须包含 URL，数据库事实必须包含查询口径，文件事实必须包含文件名。"
)
