"""长期记忆运行时上下文。"""

from contextvars import ContextVar, Token
from typing import Optional


_memory_prompt_ctx: ContextVar[str] = ContextVar(
    "memory_prompt",
    default="",
)


def set_memory_prompt(prompt: str) -> Token[str]:
    """设置当前任务可读取的长期记忆提示。"""
    return _memory_prompt_ctx.set(prompt)


def get_memory_prompt() -> str:
    """获取当前任务已经过治理的长期记忆提示。"""
    return _memory_prompt_ctx.get()


def reset_memory_prompt(token: Token[str]) -> None:
    """恢复长期记忆上下文，避免并发任务串台。"""
    _memory_prompt_ctx.reset(token)
