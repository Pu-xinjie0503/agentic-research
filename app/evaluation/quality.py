"""基线结果质量评测。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


JUDGED_CASES = {"network_search", "multi_agent", "pdf_delivery"}
FILE_FACTS = [
    "阿莫西林胶囊",
    "布洛芬缓释胶囊",
    "盐酸二甲双胍片",
]


def load_low_stock_truth() -> list[dict[str, Any]]:
    """从教学数据库读取低库存 Top 5 真值。"""
    from mysql.connector import connect

    from app.tools.db_tools import get_db_config

    query = """
    SELECT d.generic_name, d.brand_name, SUM(i.quantity_on_hand) AS total_stock
    FROM drugs d
    JOIN inventory i ON d.drug_id = i.drug_id
    GROUP BY d.drug_id, d.generic_name, d.brand_name
    ORDER BY total_stock ASC, d.drug_id ASC
    LIMIT 5
    """
    with connect(**get_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return [
                {
                    "generic_name": str(row[0]),
                    "brand_name": str(row[1]),
                    "total_stock": int(row[2]),
                }
                for row in cursor.fetchall()
            ]


def evaluate_rule_quality(
    case_id: str,
    final_result: str,
    output_path: Path,
    expected_artifacts: tuple[str, ...],
    low_stock_truth: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """按任务类型计算确定性质量指标。"""
    artifact_text, artifact_checks = _read_artifact_content(
        output_path,
        expected_artifacts,
    )
    content = artifact_text or final_result
    checks: dict[str, float] = {}
    details: dict[str, Any] = {}
    dimensions: dict[str, float] = {}

    if case_id in {"database_query", "markdown_delivery"} and low_stock_truth:
        normalized = _normalize_text(content)
        matched = []
        for row in low_stock_truth:
            name_hit = _normalize_text(row["generic_name"]) in normalized
            number = str(row["total_stock"])
            number_hit = number in normalized
            matched.append(name_hit and number_hit)
        checks["database_fact_accuracy"] = round(sum(matched) / len(matched), 4)
        dimensions["data"] = checks["database_fact_accuracy"]
        details["database_matches"] = sum(matched)
        details["database_expected"] = len(matched)

    if case_id in {"file_analysis", "multi_agent", "pdf_delivery"}:
        fact_hits = [fact in content for fact in FILE_FACTS]
        structure_hits = [
            bool(re.search(pattern, content))
            for pattern in ("风险", "核验|待验证", "缺口|不足", "2026|外部")
        ]
        checks["file_fact_coverage"] = round(sum(fact_hits) / len(fact_hits), 4)
        checks["file_analysis_structure"] = round(
            sum(structure_hits) / len(structure_hits),
            4,
        )
        dimensions["data"] = checks["file_fact_coverage"]
        dimensions["content"] = checks["file_analysis_structure"]
        details["file_fact_hits"] = sum(fact_hits)

    if case_id in {
        "network_search",
        "multi_agent",
        "markdown_delivery",
        "pdf_delivery",
    }:
        urls = _extract_urls(content)
        domains = {
            urlparse(url).netloc.lower()
            for url in urls
            if urlparse(url).netloc
        }
        required_sources = 5 if case_id == "network_search" else 3
        checks["citation_coverage"] = round(
            min(1.0, len(urls) / required_sources),
            4,
        )
        checks["source_diversity"] = round(
            min(1.0, len(domains) / max(2, required_sources)),
            4,
        )
        dimensions["citations"] = round(
            (
                checks["citation_coverage"]
                + checks["source_diversity"]
            )
            / 2,
            4,
        )
        details["citation_url_count"] = len(urls)
        details["unique_domain_count"] = len(domains)

    if expected_artifacts:
        checks["artifact_validity"] = round(
            sum(artifact_checks.values()) / len(artifact_checks),
            4,
        )
        dimensions["artifacts"] = checks["artifact_validity"]
        details["artifact_checks"] = artifact_checks

    if "content" not in dimensions:
        dimensions["content"] = _content_completeness(case_id, content)

    available_scores = list(dimensions.values())
    return {
        "rule_score": (
            round(sum(available_scores) / len(available_scores), 4)
            if available_scores
            else None
        ),
        "dimensions": dimensions,
        "checks": checks,
        "details": details,
    }


def evaluate_routing_quality(
    actual_assistants: dict[str, Any],
    expected_assistants: tuple[str, ...],
    forbidden_assistants: tuple[str, ...],
) -> dict[str, Any]:
    """独立评估子 Agent 路由是否完整且没有误调用。"""
    actual = {
        str(name)
        for name, count in actual_assistants.items()
        if int(count or 0) > 0
    }
    expected = set(expected_assistants)
    forbidden = set(forbidden_assistants)
    required_hits = len(actual & expected)
    forbidden_hits = sorted(actual & forbidden)
    denominator = len(expected) + len(forbidden)
    correct = required_hits + len(forbidden - actual)
    score = round(correct / denominator, 4) if denominator else 1.0
    return {
        "score": score,
        "expected": sorted(expected),
        "actual": sorted(actual),
        "missing": sorted(expected - actual),
        "forbidden_hits": forbidden_hits,
    }


async def run_sampled_quality_judge(
    case_id: str,
    query: str,
    content: str,
) -> dict[str, Any] | None:
    """对选定复杂案例执行一次模型裁判。"""
    if case_id not in JUDGED_CASES:
        return None

    from langchain_core.messages import HumanMessage

    from app.agent.llm import model

    prompt = f"""
你是 Agent 结果质量评测员。请根据用户任务评价候选答案，只输出 JSON。

用户任务：
{query}

候选答案：
{content[:12000]}

JSON 格式：
{{
  "relevance": 1到5的整数,
  "evidence_support": 1到5的整数,
  "completeness": 1到5的整数,
  "hallucination_control": 1到5的整数,
  "overall": 1到5的数字,
  "reason": "不超过200字的中文理由"
}}

评分要求：没有可核验来源支撑的具体市场数字应降低 evidence_support 和 hallucination_control。
"""
    response = await model.ainvoke([HumanMessage(content=prompt)])
    text = response.text if hasattr(response, "text") else str(response.content)
    parsed = _parse_json_object(text)
    if parsed is None:
        return {"status": "parse_error", "raw_summary": text[:500]}
    return {"status": "success", **parsed}


def _read_artifact_content(
    output_path: Path,
    expected_artifacts: tuple[str, ...],
) -> tuple[str, dict[str, bool]]:
    """读取交付文件并检查是否可解析。"""
    if not expected_artifacts:
        return "", {}
    checks: dict[str, bool] = {}
    markdown_text = ""
    markdown_files = sorted(output_path.glob("*.md")) if output_path.exists() else []
    pdf_files = sorted(output_path.glob("*.pdf")) if output_path.exists() else []

    if "markdown" in expected_artifacts:
        try:
            markdown_text = markdown_files[-1].read_text(encoding="utf-8")
            checks["markdown"] = len(markdown_text.strip()) >= 200
        except (IndexError, OSError, UnicodeError):
            checks["markdown"] = False

    if "pdf" in expected_artifacts:
        try:
            import pypdf

            reader = pypdf.PdfReader(str(pdf_files[-1]))
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            checks["pdf"] = bool(reader.pages) and len(pdf_text.strip()) >= 100
            if not markdown_text:
                markdown_text = pdf_text
        except (IndexError, OSError, ValueError):
            checks["pdf"] = False

    return markdown_text, checks


def _extract_urls(content: str) -> list[str]:
    """提取答案中的 HTTP 来源链接。"""
    return list(
        dict.fromkeys(
            re.findall(r"https?://[^\s\])}>\"']+", content, flags=re.IGNORECASE)
        )
    )


def _content_completeness(case_id: str, content: str) -> float:
    """按任务类型估算答案结构完整度，不评价事实真伪。"""
    text = content.strip()
    if not text:
        return 0.0
    if case_id == "network_search":
        item_count = len(
            re.findall(
                r"(?m)^\s*(?:#{1,6}\s*)?(?:\d+[.、)\ufe0f\u20e3]*|[-*])\s+\S+",
                text,
            )
        )
        return round(min(1.0, item_count / 5), 4)
    if case_id in {"database_query", "markdown_delivery"}:
        row_like_count = len(
            re.findall(r"(?m)^\s*\|.+\|\s*$", text)
        )
        return round(min(1.0, row_like_count / 7), 4)
    return round(min(1.0, len(text) / 800), 4)


def _normalize_text(value: str) -> str:
    """去除格式符号，便于事实值匹配。"""
    return re.sub(r"[\s,，|:*`_]+", "", value).lower()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取第一个 JSON 对象。"""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
