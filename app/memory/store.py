"""基于 SQLite 的 LangGraph BaseStore 实现。"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)


def _now_iso() -> str:
    """生成可排序的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    """解析 SQLite 中保存的 ISO 时间。"""
    return datetime.fromisoformat(value)


def _namespace_text(namespace: tuple[str, ...]) -> str:
    """将命名空间稳定序列化为 JSON。"""
    return json.dumps(namespace, ensure_ascii=False, separators=(",", ":"))


def _normalize_search_text(value: object) -> str:
    """保留中英文和数字，供轻量词法检索使用。"""
    return "".join(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(value or "").lower())
    )


def _search_terms(value: object) -> set[str]:
    """生成适合中文和英文的字符片段集合。"""
    text = _normalize_search_text(value)
    if not text:
        return set()
    terms = {text}
    terms.update(
        text[index : index + 2]
        for index in range(max(0, len(text) - 1))
    )
    terms.update(re.findall(r"[a-z0-9]{2,}", text))
    return {term for term in terms if term}


def lexical_score(query: str, text: str) -> float:
    """计算不依赖向量模型的保守词法相关度。"""
    query_text = _normalize_search_text(query)
    target_text = _normalize_search_text(text)
    if not query_text or not target_text:
        return 0.0
    if query_text in target_text or target_text in query_text:
        return 1.0
    query_terms = _search_terms(query_text)
    target_terms = _search_terms(target_text)
    if not query_terms or not target_terms:
        return 0.0
    overlap = len(query_terms & target_terms)
    return round(overlap / max(1, len(query_terms)), 6)


class SQLiteMemoryStore(BaseStore):
    """支持持久化、软删除、审计和词法搜索的 BaseStore。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建带字典行和超时配置的短连接。"""
        connection = sqlite3.connect(
            self.path,
            timeout=10,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        """创建 Store 和审计表。"""
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_items (
                    namespace TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    PRIMARY KEY (namespace, item_key)
                );

                CREATE INDEX IF NOT EXISTS idx_store_items_active
                ON store_items(namespace, deleted_at, updated_at);

                CREATE TABLE IF NOT EXISTS store_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    value_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.commit()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """在单个事务中执行 LangGraph Store 操作。"""
        operations = list(ops)
        results: list[Result] = []
        with self._lock, closing(self._connect()) as connection:
            try:
                for operation in operations:
                    if isinstance(operation, GetOp):
                        results.append(self._get(connection, operation))
                    elif isinstance(operation, PutOp):
                        self._put(connection, operation)
                        results.append(None)
                    elif isinstance(operation, SearchOp):
                        results.append(self._search(connection, operation))
                    elif isinstance(operation, ListNamespacesOp):
                        results.append(self._list_namespaces(connection, operation))
                    else:
                        raise TypeError(
                            "SQLiteMemoryStore 不支持操作："
                            f"{type(operation).__name__}"
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """在线程池中执行同步 SQLite 事务，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.batch, list(ops))

    def _get(
        self,
        connection: sqlite3.Connection,
        operation: GetOp,
    ) -> Item | None:
        """读取一条未删除记录。"""
        row = connection.execute(
            """
            SELECT namespace, item_key, value_json, created_at, updated_at
            FROM store_items
            WHERE namespace = ? AND item_key = ? AND deleted_at IS NULL
            """,
            (_namespace_text(operation.namespace), operation.key),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def _put(
        self,
        connection: sqlite3.Connection,
        operation: PutOp,
    ) -> None:
        """新增、更新或软删除一条记录。"""
        namespace = _namespace_text(operation.namespace)
        now = _now_iso()
        if operation.value is None:
            row = connection.execute(
                """
                SELECT value_json
                FROM store_items
                WHERE namespace = ? AND item_key = ? AND deleted_at IS NULL
                """,
                (namespace, operation.key),
            ).fetchone()
            if row is None:
                return
            connection.execute(
                """
                UPDATE store_items
                SET deleted_at = ?, updated_at = ?
                WHERE namespace = ? AND item_key = ?
                """,
                (now, now, namespace, operation.key),
            )
            self._audit(
                connection,
                namespace,
                operation.key,
                "delete",
                row["value_json"],
                now,
            )
            return

        value_json = json.dumps(
            operation.value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        existing = connection.execute(
            """
            SELECT created_at
            FROM store_items
            WHERE namespace = ? AND item_key = ?
            """,
            (namespace, operation.key),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO store_items (
                namespace, item_key, value_json,
                created_at, updated_at, deleted_at
            )
            VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(namespace, item_key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at,
                deleted_at = NULL
            """,
            (namespace, operation.key, value_json, now, now),
        )
        self._audit(
            connection,
            namespace,
            operation.key,
            "update" if existing else "create",
            value_json,
            now,
        )

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        namespace: str,
        item_key: str,
        operation: str,
        value_json: str | None,
        timestamp: str,
    ) -> None:
        """写入不可变审计记录。"""
        connection.execute(
            """
            INSERT INTO store_audit (
                namespace, item_key, operation, value_json, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (namespace, item_key, operation, value_json, timestamp),
        )

    def _search(
        self,
        connection: sqlite3.Connection,
        operation: SearchOp,
    ) -> list[SearchItem]:
        """按命名空间、结构化过滤和词法相关度搜索。"""
        rows = connection.execute(
            """
            SELECT namespace, item_key, value_json, created_at, updated_at
            FROM store_items
            WHERE deleted_at IS NULL
            ORDER BY updated_at DESC
            """
        ).fetchall()
        matches: list[SearchItem] = []
        for row in rows:
            namespace = tuple(json.loads(row["namespace"]))
            if namespace[: len(operation.namespace_prefix)] != (
                operation.namespace_prefix
            ):
                continue
            value = json.loads(row["value_json"])
            if not _matches_filter(value, operation.filter):
                continue
            score = None
            if operation.query:
                score = lexical_score(
                    operation.query,
                    json.dumps(value, ensure_ascii=False),
                )
                if score <= 0:
                    continue
            matches.append(
                SearchItem(
                    namespace=namespace,
                    key=row["item_key"],
                    value=value,
                    created_at=_parse_time(row["created_at"]),
                    updated_at=_parse_time(row["updated_at"]),
                    score=score,
                )
            )

        if operation.query:
            matches.sort(
                key=lambda item: (
                    float(item.score or 0),
                    item.updated_at,
                ),
                reverse=True,
            )
        start = max(0, operation.offset)
        end = start + max(0, operation.limit)
        return matches[start:end]

    def _list_namespaces(
        self,
        connection: sqlite3.Connection,
        operation: ListNamespacesOp,
    ) -> list[tuple[str, ...]]:
        """列出满足前后缀条件的活动命名空间。"""
        rows = connection.execute(
            """
            SELECT DISTINCT namespace
            FROM store_items
            WHERE deleted_at IS NULL
            ORDER BY namespace
            """
        ).fetchall()
        namespaces: set[tuple[str, ...]] = set()
        for row in rows:
            namespace = tuple(json.loads(row["namespace"]))
            if not _matches_namespace(namespace, operation.match_conditions):
                continue
            if operation.max_depth is not None:
                namespace = namespace[: operation.max_depth]
            namespaces.add(namespace)
        ordered = sorted(namespaces)
        start = max(0, operation.offset)
        end = start + max(0, operation.limit)
        return ordered[start:end]

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Item:
        """把 SQLite 行转换为 LangGraph Item。"""
        return Item(
            namespace=tuple(json.loads(row["namespace"])),
            key=row["item_key"],
            value=json.loads(row["value_json"]),
            created_at=_parse_time(row["created_at"]),
            updated_at=_parse_time(row["updated_at"]),
        )

    def audit_count(self) -> int:
        """返回审计记录数量，供测试和诊断使用。"""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM store_audit"
            ).fetchone()
            return int(row["count"])


def _matches_filter(
    value: dict[str, Any],
    filters: dict[str, Any] | None,
) -> bool:
    """支持 BaseStore 常用的顶层精确和比较过滤。"""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = value.get(key)
        if not isinstance(expected, dict):
            if actual != expected:
                return False
            continue
        for operator, target in expected.items():
            if operator == "$eq" and actual != target:
                return False
            if operator == "$ne" and actual == target:
                return False
            if operator == "$gt" and not (actual is not None and actual > target):
                return False
            if operator == "$gte" and not (actual is not None and actual >= target):
                return False
            if operator == "$lt" and not (actual is not None and actual < target):
                return False
            if operator == "$lte" and not (actual is not None and actual <= target):
                return False
    return True


def _matches_namespace(
    namespace: tuple[str, ...],
    conditions,
) -> bool:
    """判断命名空间是否满足 LangGraph 前后缀条件。"""
    for condition in conditions or ():
        path = tuple(condition.path)
        if condition.match_type == "prefix":
            candidate = namespace[: len(path)]
        else:
            candidate = namespace[-len(path) :] if path else ()
        if len(candidate) != len(path):
            return False
        if any(
            expected != "*" and expected != actual
            for expected, actual in zip(path, candidate)
        ):
            return False
    return True
