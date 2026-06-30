"""OCR 与图片理解工具。"""

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from app.api.context import get_session_context
from app.api.monitor import monitor
from app.multimodal.vision import (
    MAX_VISUAL_FILE_BYTES,
    analyze_visual_document,
    is_visual_file,
)
from app.observability.evidence_pack import record_evidence
from app.observability.tracing import summarize_text, trace_span
from app.utils.path_utils import resolve_path


@tool
def analyze_visual_file(
    filename: Annotated[
        str,
        "图片或扫描 PDF 文件名（支持 .png, .jpg, .jpeg, .webp, .pdf）",
    ],
    instruction: Annotated[
        str,
        "OCR 和图片理解要求，例如：提取全部文字并解释图表",
    ] = "执行 OCR，并理解图片中的关键信息",
) -> str:
    """对当前会话中的图片或扫描 PDF 执行 OCR 与视觉理解。"""
    monitor.report_tool(
        "OCR与图片理解工具",
        {"filename": filename, "instruction": instruction},
    )
    with trace_span(
        "tool.analyze_visual_file",
        component="tool",
        metadata={
            "tool_name": "analyze_visual_file",
            "filename": filename,
            "instruction": instruction,
        },
    ) as span:
        session_dir = get_session_context()
        file_path = Path(resolve_path(filename, session_dir))
        if not file_path.exists() or not file_path.is_file():
            result = f"错误：文件 '{filename}' 不存在。"
            span.set_result(file_exists=False, result_summary=summarize_text(result))
            return result
        if not is_visual_file(file_path):
            result = (
                "错误：该工具只支持 PNG、JPG、JPEG、WEBP 图片和扫描 PDF。"
            )
            span.set_result(
                file_exists=True,
                result_summary=summarize_text(result),
            )
            return result
        file_size = file_path.stat().st_size
        if file_size > MAX_VISUAL_FILE_BYTES:
            result = (
                "错误：视觉文件超过 "
                f"{MAX_VISUAL_FILE_BYTES // 1024 // 1024}MB 限制。"
            )
            span.set_result(
                file_exists=True,
                file_size=file_size,
                result_summary=summarize_text(result),
            )
            return result
        try:
            result = analyze_visual_document(
                file_path,
                instruction=instruction,
            )
        except Exception as exc:
            result = (
                "视觉分析失败（不可重试）："
                f"{type(exc).__name__}: {exc}。"
                "请直接向用户说明配置或文件问题，不要再次调用"
                " analyze_visual_file 或 read_file_content。"
            )
            span.set_result(
                file_exists=True,
                file_size=file_size,
                status="error",
                result_summary=summarize_text(result),
            )
            return result
        span.set_result(
            file_exists=True,
            file_size=file_size,
            result_length=len(result),
            result_summary=summarize_text(result),
        )
        record_evidence(
            source_type="file",
            source_name=file_path.name,
            source_locator=f"visual_ocr:{instruction}",
            content=result,
            confidence=0.85,
            metadata={
                "tool_name": "analyze_visual_file",
                "filename": file_path.name,
                "file_extension": file_path.suffix.lower(),
                "file_size": file_size,
                "visual": True,
            },
        )
        return result
