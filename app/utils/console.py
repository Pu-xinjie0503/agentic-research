"""控制台安全输出工具。"""

from __future__ import annotations

import sys
from typing import Any, TextIO


def safe_console_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    stream: TextIO | None = None,
) -> None:
    """
    输出控制台文本，当前终端编码不支持某些字符时使用转义形式降级。

    控制台仅用于调试，输出失败不能影响 Agent 的业务执行状态。
    """
    target = stream or sys.stdout
    text = sep.join(str(value) for value in values) + end
    try:
        target.write(text)
        target.flush()
        return
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding)
        try:
            target.write(safe_text)
            target.flush()
        except Exception:
            return
    except Exception:
        return
