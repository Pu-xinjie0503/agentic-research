"""OCR、图片理解和多模态上传测试。"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from PIL import Image, ImageDraw

import app.api.server as server_module
import app.tools.multimodal_tools as multimodal_tools_module
from app.agent.subagents.file_analysis_agent import file_analysis_agent
from app.api.context import reset_session_context, set_session_context
from app.multimodal.vision import (
    MAX_IMAGE_EDGE,
    analyze_visual_document,
    get_vision_model,
    prepare_visual_pages,
)
from app.tools.multimodal_tools import analyze_visual_file
from app.tools.upload_file_read_tool import read_file_content


def _create_test_image(path: Path, size: tuple[int, int] = (1200, 800)) -> bytes:
    """创建包含文字和简单图形的测试图片。"""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 60), "DeepSearch OCR 2026", fill="black")
    draw.rectangle((60, 150, 600, 500), outline="blue", width=8)
    draw.text((90, 220), "Inventory: 3700", fill="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def _create_scanned_pdf(path: Path, page_count: int = 1) -> None:
    """创建每页只有图片、没有文本层的扫描 PDF。"""
    image_path = path.with_suffix(".png")
    image_data = _create_test_image(image_path, size=(800, 500))
    document = pymupdf.open()
    try:
        for _ in range(page_count):
            page = document.new_page(width=800, height=500)
            page.insert_image(page.rect, stream=image_data)
        document.save(path)
    finally:
        document.close()
        image_path.unlink(missing_ok=True)


class FakeVisionModel:
    """记录视觉请求并返回固定 OCR 结果。"""

    model_name = "fake-vision"

    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(
            content=(
                "## 第 1 页\n"
                "### OCR 文字\nDeepSearch OCR 2026\n"
                "### 图片理解\n包含库存数值和蓝色矩形。"
            ),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 30,
                "total_tokens": 130,
            },
        )


class MultimodalVisionTests(unittest.TestCase):
    """验证视觉预处理和模型请求。"""

    def tearDown(self) -> None:
        get_vision_model.cache_clear()

    def test_deepseek_text_endpoint_requires_separate_vision_config(self) -> None:
        get_vision_model.cache_clear()
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
                    "OPENAI_API_KEY": "text-key",
                    "LLM_VISION_MODEL": "qwen-vl-max-latest",
                },
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "仅文本的 DeepSeek 接口"),
        ):
            get_vision_model()

    def test_vision_endpoint_can_be_configured_independently(self) -> None:
        get_vision_model.cache_clear()
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://api.deepseek.com",
                "OPENAI_API_KEY": "text-key",
                "VISION_BASE_URL": "https://vision.example.com/v1",
                "VISION_API_KEY": "vision-key",
                "LLM_VISION_MODEL": "vision-model",
            },
            clear=True,
        ):
            model = get_vision_model()

        self.assertEqual(model.model_name, "vision-model")
        self.assertEqual(
            str(model.openai_api_base),
            "https://vision.example.com/v1",
        )

    def test_image_is_compressed_to_bounded_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "large.png"
            _create_test_image(image_path, size=(4000, 2400))

            pages, total_pages = prepare_visual_pages(image_path)

        self.assertEqual(total_pages, 1)
        self.assertEqual(len(pages), 1)
        self.assertLessEqual(max(pages[0].width, pages[0].height), MAX_IMAGE_EDGE)
        self.assertTrue(
            pages[0].data_url.startswith("data:image/jpeg;base64,")
        )

    def test_scanned_pdf_is_limited_to_first_six_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "scan.pdf"
            _create_scanned_pdf(pdf_path, page_count=7)

            pages, total_pages = prepare_visual_pages(pdf_path)

        self.assertEqual(total_pages, 7)
        self.assertEqual(len(pages), 6)
        self.assertEqual([page.page_number for page in pages], list(range(1, 7)))

    def test_visual_request_contains_text_and_image_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "receipt.png"
            _create_test_image(image_path)
            fake_model = FakeVisionModel()

            result = analyze_visual_document(
                image_path,
                instruction="提取文字并解释库存图形",
                model=fake_model,
            )

        content = fake_model.messages[0].content
        self.assertIn("OCR 文字", result)
        self.assertTrue(
            any(block.get("type") == "image_url" for block in content)
        )
        self.assertIn("提取文字并解释库存图形", content[0]["text"])

    def test_text_reader_redirects_image_and_scanned_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_test_image(root / "screen.png")
            _create_scanned_pdf(root / "scan.pdf")
            token = set_session_context(temp_dir)
            try:
                image_result = read_file_content.invoke(
                    {"filename": "screen.png"}
                )
                pdf_result = read_file_content.invoke(
                    {"filename": "scan.pdf"}
                )
            finally:
                reset_session_context(token)

        self.assertIn("analyze_visual_file", image_result)
        self.assertIn("analyze_visual_file", pdf_result)

    def test_visual_tool_returns_model_result_and_handles_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "screen.png"
            _create_test_image(image_path)
            token = set_session_context(temp_dir)
            try:
                with patch.object(
                    multimodal_tools_module,
                    "analyze_visual_document",
                    return_value="OCR 成功",
                ):
                    success = analyze_visual_file.invoke(
                        {"filename": "screen.png"}
                    )
                with patch.object(
                    multimodal_tools_module,
                    "analyze_visual_document",
                    side_effect=RuntimeError("模型不可用"),
                ):
                    failure = analyze_visual_file.invoke(
                        {"filename": "screen.png"}
                    )
            finally:
                reset_session_context(token)

        self.assertEqual(success, "OCR 成功")
        self.assertIn("视觉分析失败", failure)
        self.assertIn("不可重试", failure)

    def test_file_subagent_exposes_visual_tool(self) -> None:
        tool_names = {
            getattr(tool, "name", "")
            for tool in file_analysis_agent["tools"]
        }
        self.assertEqual(
            tool_names,
            {"read_file_content", "analyze_visual_file"},
        )


class MultimodalUploadTests(unittest.TestCase):
    """验证图片上传的安全边界。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.updated_dir = Path(self.temp_dir.name) / "updated"
        self.original_updated_dir = server_module.updated_dir
        server_module.updated_dir = self.updated_dir

    def tearDown(self) -> None:
        server_module.updated_dir = self.original_updated_dir
        self.temp_dir.cleanup()

    def test_image_upload_sanitizes_filename(self) -> None:
        with TestClient(server_module.app) as client:
            response = client.post(
                "/api/upload",
                data={"thread_id": "image-thread"},
                files={
                    "files": (
                        "../screen.png",
                        b"image-content",
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["files"], ["screen.png"])
        self.assertTrue(
            (
                self.updated_dir
                / "session_image-thread"
                / "screen.png"
            ).exists()
        )

    def test_upload_rejects_unsupported_extension(self) -> None:
        with TestClient(server_module.app) as client:
            response = client.post(
                "/api/upload",
                data={"thread_id": "image-thread"},
                files={
                    "files": (
                        "payload.exe",
                        b"invalid",
                        "application/octet-stream",
                    )
                },
            )

        self.assertEqual(response.status_code, 415)

    def test_upload_rejects_oversized_file_and_removes_partial(self) -> None:
        with (
            patch.object(server_module, "MAX_UPLOAD_FILE_BYTES", 8),
            TestClient(server_module.app) as client,
        ):
            response = client.post(
                "/api/upload",
                data={"thread_id": "image-thread"},
                files={
                    "files": (
                        "large.png",
                        b"0123456789",
                        "image/png",
                    )
                },
            )

        self.assertEqual(response.status_code, 413)
        self.assertFalse(
            (
                self.updated_dir
                / "session_image-thread"
                / "large.png"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
