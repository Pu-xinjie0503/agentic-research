"""图片压缩、扫描 PDF 渲染和视觉模型调用。"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image, ImageOps, UnidentifiedImageError

from app.observability.tracing import next_model_call_index, trace_span


load_dotenv(find_dotenv())

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VISUAL_EXTENSIONS = frozenset({*IMAGE_EXTENSIONS, ".pdf"})
MAX_VISUAL_FILE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 6
MAX_IMAGE_EDGE = 2048
JPEG_QUALITY = 88
MAX_VISUAL_RESULT_CHARS = 12_000


@dataclass(frozen=True)
class VisualPage:
    """一张已经压缩并可发送给视觉模型的页面。"""

    page_number: int
    data_url: str
    width: int
    height: int


def is_visual_file(path: str | Path) -> bool:
    """判断文件是否可由视觉模型处理。"""
    return Path(path).suffix.lower() in VISUAL_EXTENSIONS


@lru_cache(maxsize=1)
def get_vision_model() -> ChatOpenAI:
    """延迟创建视觉模型，避免普通文本任务产生额外初始化成本。"""
    model_name = os.getenv("LLM_VISION_MODEL", "qwen-vl-max-latest").strip()
    vision_base_url = os.getenv("VISION_BASE_URL", "").strip()
    vision_api_key = os.getenv("VISION_API_KEY", "").strip()
    base_url = vision_base_url or os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = vision_api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not model_name:
        raise RuntimeError("未配置 LLM_VISION_MODEL")
    if not base_url or not api_key:
        raise RuntimeError(
            "未配置视觉模型连接，请设置 VISION_BASE_URL 和 "
            "VISION_API_KEY"
        )
    if not vision_base_url and "deepseek" in base_url.lower():
        raise RuntimeError(
            "当前 OPENAI_BASE_URL 是仅文本的 DeepSeek 接口；"
            "请单独配置支持 image_url 的 VISION_BASE_URL 和 "
            "VISION_API_KEY"
        )
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_retries=2,
        timeout=90,
    )


def prepare_visual_pages(
    file_path: str | Path,
    max_pages: int = MAX_PDF_PAGES,
) -> tuple[list[VisualPage], int]:
    """将图片或 PDF 页面转换为有界 JPEG Data URL。"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"视觉文件不存在：{path.name}")
    if path.stat().st_size > MAX_VISUAL_FILE_BYTES:
        raise ValueError(
            f"视觉文件超过 {MAX_VISUAL_FILE_BYTES // 1024 // 1024}MB 限制"
        )
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as image:
                return [_encode_page(image, 1)], 1
        except UnidentifiedImageError as exc:
            raise ValueError("图片文件损坏或格式无法识别") from exc
    if suffix != ".pdf":
        raise ValueError(f"不支持的视觉文件格式：{suffix or '无后缀'}")
    return _render_pdf(path, max_pages=max_pages)


def analyze_visual_document(
    file_path: str | Path,
    instruction: str = "执行 OCR，并理解图片中的关键信息",
    model: Any | None = None,
) -> str:
    """对图片或扫描 PDF 同时执行 OCR 和视觉理解。"""
    path = Path(file_path)
    pages, total_pages = prepare_visual_pages(path)
    prompt = _build_prompt(
        filename=path.name,
        instruction=instruction,
        analyzed_pages=len(pages),
        total_pages=total_pages,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for page in pages:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"以下是第 {page.page_number} 页/张图：",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": page.data_url},
                },
            ]
        )

    vision_model = model or get_vision_model()
    with trace_span(
        "model.call",
        component="model",
        metadata={
            "agent_name": "视觉理解助手",
            "model_name": _model_name(vision_model),
            "call_index": next_model_call_index(),
            "input_message_count": 1,
            "input_char_count": len(prompt),
            "image_count": len(pages),
            "tool_count": 0,
            "tool_names": [],
        },
    ) as span:
        try:
            response = vision_model.invoke([HumanMessage(content=content)])
        except Exception:
            span.set_result(end_reason="error")
            raise
        result = _response_text(response)
        usage = getattr(response, "usage_metadata", None) or {}
        span.set_result(
            output_message_count=1,
            output_char_count=len(result),
            tool_call_count=0,
            tool_names=[],
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
            end_reason="completed",
        )
    if not result:
        raise RuntimeError("视觉模型未返回可用内容")
    return result[:MAX_VISUAL_RESULT_CHARS]


def _render_pdf(
    path: Path,
    max_pages: int,
) -> tuple[list[VisualPage], int]:
    """把 PDF 前若干页渲染为图片。"""
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ValueError("PDF 文件损坏或无法打开") from exc
    try:
        if document.needs_pass:
            raise ValueError("加密 PDF 暂不支持视觉分析")
        total_pages = document.page_count
        if total_pages <= 0:
            raise ValueError("PDF 不包含有效页面")
        page_limit = min(max(1, max_pages), total_pages)
        pages = []
        for index in range(page_limit):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                pages.append(_encode_page(image, index + 1))
        return pages, total_pages
    finally:
        document.close()


def _encode_page(image: Image.Image, page_number: int) -> VisualPage:
    """纠正方向、压缩尺寸并转换为 JPEG Data URL。"""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.thumbnail(
        (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
        Image.Resampling.LANCZOS,
    )
    buffer = io.BytesIO()
    normalized.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return VisualPage(
        page_number=page_number,
        data_url=f"data:image/jpeg;base64,{encoded}",
        width=normalized.width,
        height=normalized.height,
    )


def _build_prompt(
    filename: str,
    instruction: str,
    analyzed_pages: int,
    total_pages: int,
) -> str:
    """构造同时覆盖 OCR 与图片理解的受控提示。"""
    truncated_notice = (
        f"文件共有 {total_pages} 页，本次只分析前 {analyzed_pages} 页。"
        if total_pages > analyzed_pages
        else f"本次分析 {analyzed_pages} 页/张图。"
    )
    return f"""
你是 DeepSearch 的视觉理解助手。请只依据图像中实际可见内容分析，不使用外部知识补全。

文件名：{filename}
用户指令：{re.sub(r"\s+", " ", str(instruction or "")).strip()[:500]}
页面范围：{truncated_notice}

请使用中文 Markdown，逐页输出：
1. OCR 文字：尽量保持原顺序；无法辨认处标记为 `[无法辨认]`，不得猜测。
2. 图片理解：说明页面类型、主体、场景、布局、图表和显著视觉关系。
3. 关键对象与数据：列出可见名称、数字、日期、单位、标签及其位置。
4. 风险与不确定项：指出模糊、遮挡、裁切、低分辨率及不能确认的内容。

最后给出跨页综合结论。OCR 文字与视觉推断必须明确区分。
""".strip()


def _response_text(response: Any) -> str:
    """兼容视觉模型返回的字符串或内容块。"""
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _model_name(model: Any) -> str:
    """读取视觉模型名称。"""
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _usage_value(usage: Any, key: str) -> int | None:
    """读取模型 Token 统计。"""
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return value if isinstance(value, int) else None
