"""跨线程长期记忆面试演示脚本。"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.main_agent import run_deep_agent
from app.memory.service import memory_service, validate_user_id
from app.utils.console import safe_console_print


def parse_args() -> argparse.Namespace:
    """解析演示用户和重置选项。"""
    parser = argparse.ArgumentParser(
        description="演示跨线程保存偏好并应用到数据库直达任务。",
    )
    parser.add_argument(
        "--user-id",
        default="interview-demo",
        help="长期记忆用户 ID，默认 interview-demo。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="运行前清空该用户已有长期记忆。",
    )
    return parser.parse_args()


async def run_demo(user_id: str, reset: bool) -> int:
    """执行线程 A 保存、线程 B 召回的完整演示。"""
    normalized_user_id = validate_user_id(user_id)
    if reset:
        memory_service.clear(normalized_user_id)

    thread_a = f"memory-demo-a-{uuid.uuid4().hex[:8]}"
    thread_b = f"memory-demo-b-{uuid.uuid4().hex[:8]}"

    safe_console_print("[MemoryDemo] 线程 A：保存中文表格偏好")
    saved = await run_deep_agent(
        "请记住：以后回答默认使用中文表格",
        thread_a,
        user_id=normalized_user_id,
    )
    safe_console_print(saved.final_result)

    safe_console_print("[MemoryDemo] 线程 B：执行数据库直达查询")
    recalled = await run_deep_agent(
        "查询数据库中库存最低的 5 个药品，不调用网络，不生成文件",
        thread_b,
        user_id=normalized_user_id,
    )
    safe_console_print(recalled.final_result)

    memories = memory_service.list_memories(normalized_user_id)
    passed = bool(memories) and "|" in recalled.final_result
    safe_console_print(
        "[MemoryDemo] "
        f"记忆数量={len(memories)}，跨线程表格回答={'通过' if passed else '失败'}"
    )
    return 0 if passed else 1


def main() -> int:
    """脚本同步入口。"""
    args = parse_args()
    return asyncio.run(run_demo(args.user_id, args.reset))


if __name__ == "__main__":
    raise SystemExit(main())
