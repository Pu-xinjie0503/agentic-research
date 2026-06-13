"""Markdown 转 PDF 布局测试。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from app.utils.word_converter import (
    A4,
    Table,
    _build_styles,
    _build_table,
    cm,
    convert_md_to_pdf,
)


@unittest.skipIf(Table is None, "未安装 reportlab")
class WordConverterTests(unittest.TestCase):
    """验证窄列和长文本不会触发 ReportLab 负宽度异常。"""

    def test_table_with_narrow_index_column_can_wrap(self) -> None:
        styles = _build_styles()
        table = _build_table(
            [
                ["序号", "来源", "说明"],
                [
                    "1",
                    "国家医疗保障局",
                    "这是一段较长的中文说明，用于验证内容能够在固定列宽内换行。",
                ],
            ],
            styles,
        )
        available_width = A4[0] - 4 * cm

        width, height = table.wrap(available_width, A4[1])

        self.assertLessEqual(width, available_width)
        self.assertGreater(height, 0)

    def test_markdown_table_converts_to_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "report.md"
            pdf_path = root / "report.pdf"
            markdown_path.write_text(
                "| 序号 | 来源 | 说明 |\n"
                "|---|---|---|\n"
                "| 1 | 国家医疗保障局 | 一段需要自动换行的较长中文说明。 |\n",
                encoding="utf-8",
            )

            result = convert_md_to_pdf(markdown_path, pdf_path)

            self.assertIn("成功转换", result)
            self.assertTrue(pdf_path.exists())


if __name__ == "__main__":
    unittest.main()
