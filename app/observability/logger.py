"""
结构化日志写入模块

将运行事件按 JSONL 格式追加到 app/logs/traces/YYYY-MM-DD.jsonl。
JSONL 便于本地 grep、后续导入数据库，也能作为评测脚本的原始输入。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.observability.trace_store import trace_store


_write_lock = Lock()
_app_root = Path(__file__).resolve().parents[1]
_trace_log_dir = _app_root / "logs" / "traces"


def _json_default(value: Any) -> str:
    """将不可 JSON 序列化的对象降级为字符串。"""
    return str(value)


def write_json_log(record: dict[str, Any]) -> None:
    """
    追加写入一条结构化日志

    :param record: 已经整理好的事件字典
    """
    _trace_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = _trace_log_dir / f"{datetime.now().date().isoformat()}.jsonl"
    line = json.dumps(record, ensure_ascii=False, default=_json_default)

    with _write_lock:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if trace_store is not None:
        try:
            trace_store.record(record)
        except Exception as exc:
            print(f"[TraceStore] SQLite 写入失败，已保留 JSONL 日志：{exc}")
