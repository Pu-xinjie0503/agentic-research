"""OCR 与图片理解面试演示脚本。"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import uuid
from pathlib import Path
import sys

import pymupdf
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.main_agent import run_deep_agent
from app.multimodal.vision import get_vision_model
from app.utils.console import safe_console_print


APP_ROOT = PROJECT_ROOT / "app"
UPDATED_ROOT = APP_ROOT / "updated"


def parse_args() -> argparse.Namespace:
    """解析图片、扫描 PDF 或双模式演示。"""
    parser = argparse.ArgumentParser(
        description="生成多模态夹具并验证 OCR、图片理解和扫描 PDF。",
    )
    parser.add_argument(
        "--mode",
        choices=("image", "pdf", "both"),
        default="both",
        help="演示模式，默认同时验证图片和扫描 PDF。",
    )
    return parser.parse_args()


def build_demo_image(path: Path) -> None:
    """生成包含中文文字、数字和简单柱状图的图片。"""
    image = Image.new("RGB", (1400, 900), "#f5f8fb")
    draw = ImageDraw.Draw(image)
    font_path = _find_font()
    title_font = ImageFont.truetype(str(font_path), 52)
    body_font = ImageFont.truetype(str(font_path), 34)
    small_font = ImageFont.truetype(str(font_path), 26)

    draw.text((70, 50), "DeepSearch 多模态验收看板", fill="#102a43", font=title_font)
    draw.text((70, 140), "药品：富马酸替诺福韦二吡呋酯片", fill="#243b53", font=body_font)
    draw.text((70, 200), "当前库存：3700", fill="#c53030", font=body_font)
    draw.text((70, 260), "风险等级：高，需要优先补货", fill="#c53030", font=body_font)

    chart_left = 90
    chart_bottom = 760
    values = [3700, 4700, 5700]
    labels = ["替诺福韦", "甘精胰岛素", "恩替卡韦"]
    colors = ["#e53e3e", "#3182ce", "#38a169"]
    for index, (value, label, color) in enumerate(zip(values, labels, colors)):
        left = chart_left + index * 360
        height = int(value / 10)
        top = chart_bottom - height
        draw.rectangle((left, top, left + 180, chart_bottom), fill=color)
        draw.text((left, top - 42), str(value), fill="#102a43", font=small_font)
        draw.text((left, chart_bottom + 18), label, fill="#243b53", font=small_font)
    draw.line((60, chart_bottom, 1250, chart_bottom), fill="#52616b", width=4)
    image.save(path, format="PNG")


def build_scanned_pdf(image_path: Path, pdf_path: Path) -> None:
    """将演示图片嵌入无文本层 PDF，模拟扫描件。"""
    document = pymupdf.open()
    try:
        page = document.new_page(width=1400, height=900)
        page.insert_image(page.rect, filename=str(image_path))
        document.save(pdf_path)
    finally:
        document.close()


async def run_case(mode: str) -> bool:
    """执行单个图片或扫描 PDF 场景。"""
    thread_id = f"multimodal-demo-{mode}-{uuid.uuid4().hex[:8]}"
    upload_dir = UPDATED_ROOT / f"session_{thread_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / "multimodal_dashboard.png"
    build_demo_image(image_path)
    if mode == "pdf":
        target_path = upload_dir / "multimodal_scan.pdf"
        build_scanned_pdf(image_path, target_path)
        image_path.unlink()
    else:
        target_path = image_path

    try:
        safe_console_print(
            f"[MultimodalDemo] 开始验证 {mode}：{target_path.name}"
        )
        result = await run_deep_agent(
            "只分析上传文件，不查询数据库、不调用网络。请执行 OCR 和"
            "图片理解，提取全部可见文字、关键数字、风险等级，并解释柱状图。",
            thread_id,
        )
        safe_console_print(result.final_result)
        normalized = result.final_result.replace(",", "")
        passed = (
            "DeepSearch" in result.final_result
            and "3700" in normalized
            and "风险" in result.final_result
        )
        safe_console_print(
            f"[MultimodalDemo] {mode} 验收={'通过' if passed else '失败'}"
        )
        return passed
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


async def run_demo(mode: str) -> int:
    """按参数运行一项或两项多模态验收。"""
    try:
        get_vision_model()
    except RuntimeError as exc:
        safe_console_print(f"[MultimodalDemo] 视觉配置检查失败：{exc}")
        return 2
    modes = ("image", "pdf") if mode == "both" else (mode,)
    results = [await run_case(item) for item in modes]
    return 0 if all(results) else 1


def _find_font() -> Path:
    """寻找可显示中文的系统字体。"""
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError("未找到可用于生成中文演示图片的字体")


def main() -> int:
    """脚本同步入口。"""
    args = parse_args()
    return asyncio.run(run_demo(args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
