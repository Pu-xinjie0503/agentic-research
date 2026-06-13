"""网络证据分级与精确数字校验。"""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator, model_validator


SourceTier = Literal[
    "tier_1_official",
    "tier_2_primary",
    "tier_3_secondary",
    "tier_4_commercial",
    "unknown",
]

AUTHORITATIVE_TIERS = {"tier_1_official", "tier_2_primary"}
OFFICIAL_DOMAINS = (
    ".gov",
    ".gov.cn",
    ".gov.uk",
    ".go.jp",
    "ec.europa.eu",
    "who.int",
    "worldbank.org",
    "oecd.org",
    "un.org",
)
PRIMARY_DOMAINS = (
    "doi.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
)
COMMERCIAL_DOMAINS = (
    "grandviewresearch.com",
    "marketsandmarkets.com",
    "marketresearch.com",
    "researchandmarkets.com",
    "precedenceresearch.com",
    "fortunebusinessinsights.com",
    "futuremarketinsights.com",
)
PRIMARY_TITLE_KEYWORDS = (
    "协会",
    "学会",
    "association",
    "society",
    "journal",
    "annual report",
    "official report",
)
COMMERCIAL_TITLE_KEYWORDS = (
    "market size",
    "market forecast",
    "market report",
    "市场规模预测",
    "市场研究报告",
)
PRECISION_CONTEXT_PATTERN = re.compile(
    r"市场规模|增长率|复合年增长率|cagr|销售额|营收|人口|患者|"
    r"占比|份额|预计|预测|将达到|增至|降至|同比|"
    r"收购|并购|交易金额|投资|融资",
    flags=re.IGNORECASE,
)
PRECISE_VALUE_PATTERN = re.compile(
    r"(?:[$¥￥]\s*)?\d[\d,.]*\s*"
    r"(?:%|％|个百分点|亿美元|亿元|万元|美元|人民币|元|"
    r"万亿|千亿|百万|千万|亿|万|人|例|家|倍|"
    r"million|billion|trillion)",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")
URL_PATTERN = re.compile(r"https?://[^\s\])}>\"']+", flags=re.IGNORECASE)


def classify_source_tier(url: str, title: str = "") -> SourceTier:
    """根据域名和标题对网络来源做保守分级。"""
    hostname = urlparse(str(url or "")).netloc.lower().split(":", 1)[0]
    normalized_title = str(title or "").lower()
    if not hostname:
        return "unknown"
    if any(
        hostname == domain.lstrip(".") or hostname.endswith(domain)
        for domain in OFFICIAL_DOMAINS
    ):
        return "tier_1_official"
    if any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in PRIMARY_DOMAINS
    ) or any(keyword in normalized_title for keyword in PRIMARY_TITLE_KEYWORDS):
        return "tier_2_primary"
    if any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in COMMERCIAL_DOMAINS
    ) or any(keyword in normalized_title for keyword in COMMERCIAL_TITLE_KEYWORDS):
        return "tier_4_commercial"
    return "tier_3_secondary"


def extract_precision_claims(content: str) -> list[str]:
    """提取需要事实级证据支撑的外部精确数字陈述。"""
    without_urls = URL_PATTERN.sub("", str(content or ""))
    claims: list[str] = []
    for segment in re.split(r"[\r\n。！？；]+", without_urls):
        compact = re.sub(r"\s+", " ", segment).strip(" -*#|")
        compact = re.sub(
            r"^\d+(?:\.\d+)*(?:[.、)])?\s+",
            "",
            compact,
        )
        if (
            compact
            and PRECISION_CONTEXT_PATTERN.search(compact)
            and PRECISE_VALUE_PATTERN.search(compact)
        ):
            claims.append(compact[:500])
    return list(dict.fromkeys(claims))


def numeric_tokens(text: str) -> set[str]:
    """提取数字 Token，忽略单独年份。"""
    result = set()
    for match in NUMBER_PATTERN.findall(str(text or "")):
        normalized = match.replace(",", "")
        try:
            number = float(normalized)
        except ValueError:
            continue
        if number.is_integer() and 1900 <= number <= 2100:
            continue
        result.add(normalized.rstrip("0").rstrip("."))
    return {token for token in result if token}


def downgrade_precision_claim(text: str) -> str:
    """把缺少权威支撑的精确数字降级为定性表述。"""
    value_replacement = "未核验数值"
    if re.search(r"收购|并购|投资|融资|交易金额", text):
        value_replacement = "未核验金额"
    elif re.search(r"增长率|占比|份额|cagr", text, flags=re.IGNORECASE):
        value_replacement = "未核验比例"
    return PRECISE_VALUE_PATTERN.sub(value_replacement, text)


class EvidenceRecord(BaseModel):
    """一条供 Agent 和产物校验复用的网络证据。"""

    claim: str = ""
    source_url: str
    source_title: str = ""
    published_at: str = ""
    source_tier: SourceTier = "unknown"
    evidence_excerpt: str = ""
    supports_numeric_claim: bool = False

    @field_validator(
        "claim",
        "source_url",
        "source_title",
        "published_at",
        "evidence_excerpt",
        mode="before",
    )
    @classmethod
    def compact_text(cls, value: Any, info) -> str:
        """压缩证据字段中的多余空白。"""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        limits = {
            "claim": 500,
            "source_url": 1000,
            "source_title": 300,
            "published_at": 100,
            "evidence_excerpt": 600,
        }
        limit = limits[info.field_name]
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    @model_validator(mode="after")
    def infer_quality(self) -> "EvidenceRecord":
        """统一使用确定性规则推断来源等级和数字支撑能力。"""
        if not re.match(r"^https?://", self.source_url, flags=re.IGNORECASE):
            raise ValueError("网络证据必须包含真实 HTTP/HTTPS 来源 URL")
        self.source_tier = classify_source_tier(
            self.source_url,
            self.source_title,
        )
        claim_tokens = numeric_tokens(self.claim)
        excerpt_tokens = numeric_tokens(self.evidence_excerpt)
        self.supports_numeric_claim = bool(
            self.source_tier in AUTHORITATIVE_TIERS
            and claim_tokens
            and claim_tokens.issubset(excerpt_tokens)
        )
        return self


def find_unsupported_precision_claims(
    content: str,
    evidence_records: list[dict[str, Any] | EvidenceRecord],
) -> list[str]:
    """返回缺少权威原文数字和 URL 对齐的精确陈述。"""
    records = []
    for item in evidence_records:
        try:
            records.append(
                item if isinstance(item, EvidenceRecord) else EvidenceRecord(**item)
            )
        except (TypeError, ValueError):
            continue

    unsupported = []
    for claim in extract_precision_claims(content):
        claim_tokens = numeric_tokens(claim)
        supported = any(
            record.source_tier in AUTHORITATIVE_TIERS
            and record.source_url in content
            and claim_tokens
            and claim_tokens.issubset(numeric_tokens(record.evidence_excerpt))
            for record in records
        )
        if not supported:
            unsupported.append(claim)
    return unsupported
