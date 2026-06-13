"""长期记忆存储、治理和召回测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

import app.agent.main_agent as main_agent_module
import app.api.server as server_module
from app.agent.llm import model
from app.agent.middleware.memory_context import MemoryContextMiddleware
from app.memory.context import reset_memory_prompt, set_memory_prompt
from app.memory.models import MemoryDecision, MemoryOperation
from app.memory.service import (
    MEMORY_PROMPT_LIMIT,
    LongTermMemoryService,
    memory_namespace,
)
from app.memory.store import SQLiteMemoryStore
from app.observability.agent_state import infer_task_scope


def _memory_value(
    category: str,
    key: str,
    content: str,
    thread_id: str = "thread-a",
) -> dict[str, object]:
    """构造测试使用的完整记忆值。"""
    return {
        "category": category,
        "key": key,
        "content": content,
        "confidence": 1.0,
        "source_thread_id": thread_id,
        "source_trace_id": f"trace-{thread_id}",
    }


class QueueExtractor:
    """按顺序返回预设结构化决策。"""

    def __init__(self, *decisions: MemoryDecision) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    async def __call__(self, _query: str) -> MemoryDecision:
        self.calls += 1
        if not self.decisions:
            raise AssertionError("普通任务不应调用记忆提取器")
        return self.decisions.pop(0)


class LongTermMemoryTests(unittest.IsolatedAsyncioTestCase):
    """验证长期记忆的核心行为和边界。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.sqlite3"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sqlite_restart_persistence_and_user_isolation(self) -> None:
        namespace = memory_namespace("user-a")
        first_store = SQLiteMemoryStore(self.db_path)
        first_store.put(
            namespace,
            "preference:preferred_language",
            _memory_value(
                "preference",
                "preferred_language",
                "默认使用中文回答",
            ),
            index=False,
        )

        restarted_store = SQLiteMemoryStore(self.db_path)

        self.assertIsNotNone(
            restarted_store.get(
                namespace,
                "preference:preferred_language",
            )
        )
        self.assertIsNone(
            restarted_store.get(
                memory_namespace("user-b"),
                "preference:preferred_language",
            )
        )

    def test_conflict_update_and_soft_delete_keep_audit(self) -> None:
        store = SQLiteMemoryStore(self.db_path)
        namespace = memory_namespace("user-a")
        memory_id = "preference:response_format"
        store.put(
            namespace,
            memory_id,
            _memory_value("preference", "response_format", "使用列表"),
            index=False,
        )
        created_at = store.get(namespace, memory_id).created_at
        store.put(
            namespace,
            memory_id,
            _memory_value("preference", "response_format", "使用中文表格"),
            index=False,
        )

        updated = store.get(namespace, memory_id)
        self.assertEqual(updated.value["content"], "使用中文表格")
        self.assertEqual(updated.created_at, created_at)
        self.assertEqual(store.audit_count(), 2)

        store.delete(namespace, memory_id)

        self.assertIsNone(store.get(namespace, memory_id))
        self.assertEqual(store.audit_count(), 3)
        self.assertIsNone(SQLiteMemoryStore(self.db_path).get(namespace, memory_id))

    async def test_ordinary_task_skips_extractor_and_recalls_cross_thread(self) -> None:
        store = SQLiteMemoryStore(self.db_path)
        store.put(
            memory_namespace("user-a"),
            "preference:response_format",
            _memory_value(
                "preference",
                "response_format",
                "回答默认使用中文表格",
                thread_id="thread-a",
            ),
            index=False,
        )
        extractor = QueueExtractor()
        service = LongTermMemoryService(store, extractor)
        query = "查询数据库中库存最低的五个药品"

        preparation = await service.prepare(
            "user-a",
            query,
            "thread-b",
            "trace-b",
        )

        self.assertEqual(extractor.calls, 0)
        self.assertEqual(preparation.task_query, query)
        self.assertIn("回答默认使用中文表格", preparation.prompt)
        self.assertEqual(preparation.recalled[0].source_thread_id, "thread-a")
        self.assertEqual(
            infer_task_scope(preparation.task_query, False).allowed_subagents,
            {"数据库查询助手"},
        )

    async def test_explicit_memory_command_creates_then_updates_same_key(self) -> None:
        extractor = QueueExtractor(
            MemoryDecision(
                is_memory_only=True,
                operations=[
                    MemoryOperation(
                        action="remember",
                        category="preference",
                        key="response_format",
                        content="回答使用列表",
                        confidence=1.0,
                        explicit=True,
                    )
                ],
            ),
            MemoryDecision(
                is_memory_only=True,
                operations=[
                    MemoryOperation(
                        action="remember",
                        category="preference",
                        key="response_format",
                        content="回答使用中文表格",
                        confidence=1.0,
                        explicit=True,
                    )
                ],
            ),
        )
        service = LongTermMemoryService(
            SQLiteMemoryStore(self.db_path),
            extractor,
        )

        first = await service.prepare(
            "user-a",
            "记住回答使用列表",
            "thread-a",
            "trace-a",
        )
        second = await service.prepare(
            "user-a",
            "记住以后回答使用中文表格",
            "thread-b",
            "trace-b",
        )
        memories = service.list_memories("user-a")

        self.assertEqual(first.created_ids, ("preference:response_format",))
        self.assertEqual(second.updated_ids, ("preference:response_format",))
        self.assertTrue(first.direct_response)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "回答使用中文表格")
        self.assertEqual(memories[0].source_thread_id, "thread-b")

    async def test_sensitive_and_dynamic_facts_are_rejected(self) -> None:
        extractor = QueueExtractor(
            MemoryDecision(
                is_memory_only=True,
                operations=[
                    MemoryOperation(
                        action="remember",
                        category="project",
                        key="database_password",
                        content="数据库密码是 secret-123",
                        confidence=1.0,
                        explicit=True,
                    ),
                    MemoryOperation(
                        action="remember",
                        category="project",
                        key="current_inventory",
                        content="当前库存为 18",
                        confidence=1.0,
                        explicit=True,
                    ),
                ],
            )
        )
        service = LongTermMemoryService(
            SQLiteMemoryStore(self.db_path),
            extractor,
        )

        preparation = await service.prepare(
            "user-a",
            "记住数据库密码和当前库存",
            "thread-a",
            "trace-a",
        )

        self.assertEqual(service.list_memories("user-a"), [])
        self.assertEqual(len(preparation.rejected_reasons), 2)
        self.assertIn("未保存", preparation.direct_response)

    def test_project_recall_limit_and_prompt_character_limit(self) -> None:
        store = SQLiteMemoryStore(self.db_path)
        namespace = memory_namespace("user-a")
        store.put(
            namespace,
            "profile:user_role",
            _memory_value("profile", "user_role", "软件工程师"),
            index=False,
        )
        for index in range(8):
            content = (
                f"DeepSearch 项目使用 Python LangGraph，模块编号 {index}。"
                + "稳定项目约束" * 80
            )
            store.put(
                namespace,
                f"project:module_{index}",
                _memory_value("project", f"module_{index}", content),
                index=False,
            )
        service = LongTermMemoryService(store, QueueExtractor())

        recalled = service.recall(
            "user-a",
            "DeepSearch Python LangGraph 项目",
        )
        prompt = service.build_prompt(recalled)

        self.assertLessEqual(
            len([item for item in recalled if item.category == "project"]),
            4,
        )
        self.assertLessEqual(len(prompt), MEMORY_PROMPT_LIMIT)
        self.assertIn("当前用户请求始终优先", prompt)

    def test_memory_middleware_injects_only_current_context(self) -> None:
        request = ModelRequest(
            model=model,
            messages=[HumanMessage(content="查询库存")],
            system_message=SystemMessage(content="原系统提示"),
            tools=[],
        )
        token = set_memory_prompt("默认使用中文表格")
        try:
            injected = MemoryContextMiddleware._inject(request)
        finally:
            reset_memory_prompt(token)

        self.assertIn("原系统提示", injected.system_message.text)
        self.assertIn("默认使用中文表格", injected.system_message.text)
        self.assertIs(
            MemoryContextMiddleware._inject(request),
            request,
        )

    async def test_unified_entry_uses_memory_direct_without_business_agent(self) -> None:
        extractor = QueueExtractor(
            MemoryDecision(
                is_memory_only=True,
                operations=[
                    MemoryOperation(
                        action="remember",
                        category="preference",
                        key="preferred_language",
                        content="默认使用中文回答",
                        confidence=1.0,
                        explicit=True,
                    )
                ],
            )
        )
        service = LongTermMemoryService(
            SQLiteMemoryStore(self.db_path),
            extractor,
        )
        runtime_root = Path(self.temp_dir.name) / "app"
        runtime_root.mkdir()

        with (
            patch.object(main_agent_module, "memory_service", service),
            patch.object(main_agent_module, "project_root_path", runtime_root),
            patch.object(
                main_agent_module.monitor,
                "_send_to_websocket",
                return_value=None,
            ),
        ):
            result = await main_agent_module.run_deep_agent(
                "记住以后默认使用中文回答",
                "memory-direct-thread",
                "memory-direct-trace",
                user_id="user-a",
            )

        self.assertIn("[偏好] 默认使用中文回答", result.final_result)
        self.assertEqual(extractor.calls, 1)


class LongTermMemoryApiTests(unittest.TestCase):
    """验证长期记忆管理接口。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_service = server_module.memory_service
        server_module.memory_service = LongTermMemoryService(
            SQLiteMemoryStore(Path(self.temp_dir.name) / "api-memory.sqlite3"),
            QueueExtractor(),
        )
        server_module.memory_service.store.put(
            memory_namespace("api-user"),
            "preference:response_format",
            _memory_value(
                "preference",
                "response_format",
                "默认使用中文表格",
            ),
            index=False,
        )

    def tearDown(self) -> None:
        server_module.memory_service = self.original_service
        self.temp_dir.cleanup()

    def test_list_and_delete_memory(self) -> None:
        with TestClient(server_module.app) as client:
            listed = client.get(
                "/api/memories",
                params={"user_id": "api-user"},
            )
            deleted = client.delete(
                "/api/memories/preference%3Aresponse_format",
                params={"user_id": "api-user"},
            )

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["count"], 1)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            deleted.json()["deleted_ids"],
            ["preference:response_format"],
        )

    def test_clear_memories_and_validate_user_id(self) -> None:
        with TestClient(server_module.app) as client:
            invalid = client.get(
                "/api/memories",
                params={"user_id": "../invalid"},
            )
            cleared = client.delete(
                "/api/memories",
                params={"user_id": "api-user"},
            )
            listed = client.get(
                "/api/memories",
                params={"user_id": "api-user"},
            )

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(listed.json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
