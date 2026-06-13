"""向所有执行路径注入经过治理的长期记忆。"""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from app.memory.context import get_memory_prompt


class MemoryContextMiddleware(AgentMiddleware):
    """只读注入当前任务的长期记忆提示。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        """同步模型调用入口。"""
        return handler(self._inject(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        """异步模型调用入口。"""
        return await handler(self._inject(request))

    @staticmethod
    def _inject(request: ModelRequest) -> ModelRequest:
        """把有界记忆提示追加到系统消息。"""
        memory_prompt = get_memory_prompt()
        if not memory_prompt:
            return request
        current_prompt = (
            request.system_message.text if request.system_message else ""
        )
        return request.override(
            system_message=SystemMessage(
                content=f"{current_prompt}\n\n{memory_prompt}"
            )
        )
