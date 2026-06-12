"""
MySQL 数据库查询工具模块

封装数据库查询助手使用的三个 LangChain 工具：
list_sql_tables 用于发现真实表名，get_table_data 用于预览字段和样例数据，
execute_sql_query 用于在确认结构后执行自定义查询。
"""

import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from mysql.connector import Error, connect

from app.api.monitor import monitor
from app.observability.database_state import get_database_run_state
from app.observability.tracing import record_event
from app.observability.tracing import summarize_text, trace_span

load_dotenv()


READ_ONLY_PREFIXES = {"select", "show", "describe", "desc", "explain", "with"}
FORBIDDEN_SQL_KEYWORDS = {
    "alter",
    "call",
    "create",
    "delete",
    "drop",
    "grant",
    "insert",
    "load",
    "lock",
    "replace",
    "revoke",
    "truncate",
    "update",
}


def _strip_sql_literals_and_comments(query: str) -> str:
    """去除字符串字面量和注释，供只读 SQL 关键字检查使用。"""
    without_comments = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    without_comments = re.sub(r"--[^\r\n]*", " ", without_comments)
    without_comments = re.sub(r"#[^\r\n]*", " ", without_comments)
    return re.sub(
        r"'(?:''|\\.|[^'])*'|\"(?:\"\"|\\.|[^\"])*\"",
        "''",
        without_comments,
        flags=re.DOTALL,
    )


def validate_read_only_query(query: str) -> tuple[bool, str]:
    """验证 SQL 仅包含一条只读语句。"""
    if not query or not query.strip():
        return False, "SQL 不能为空。"
    sanitized = _strip_sql_literals_and_comments(query).strip()
    statements = [part.strip() for part in sanitized.split(";") if part.strip()]
    if len(statements) != 1:
        return False, "只允许执行一条 SQL 语句。"

    tokens = re.findall(r"[a-zA-Z_]+", statements[0].lower())
    if not tokens or tokens[0] not in READ_ONLY_PREFIXES:
        return False, "只允许 SELECT、SHOW、DESCRIBE、EXPLAIN 或只读 WITH 查询。"
    forbidden = sorted(set(tokens) & FORBIDDEN_SQL_KEYWORDS)
    if forbidden:
        return False, f"检测到非只读关键字：{', '.join(forbidden)}。"
    if re.search(r"\binto\s+(outfile|dumpfile)\b", statements[0], flags=re.IGNORECASE):
        return False, "禁止将查询结果写入服务器文件。"
    if re.search(r"\bfor\s+update\b", statements[0], flags=re.IGNORECASE):
        return False, "禁止使用 FOR UPDATE 锁定数据。"
    return True, ""


def _compact_rows(
    columns: list[str],
    rows: list[tuple],
    max_chars: int,
) -> tuple[str, bool]:
    """把查询结果压缩为有界 CSV 文本。"""
    lines = [",".join(columns)]
    truncated = False
    for row in rows:
        line = ",".join("" if value is None else str(value) for value in row)
        if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
            truncated = True
            break
        lines.append(line)
    if truncated:
        lines.append("...结果已按上下文上限截断...")
    return "\n".join(lines), truncated


def _load_table_names(config: dict) -> list[str]:
    """读取真实表名，供列表工具和表名白名单复用。"""
    with connect(**config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            return [str(table[0]) for table in cursor.fetchall()]


# 集中读取数据库配置，后续三个工具都复用这份连接参数
def get_db_config():
    """
    从环境变量读取 MySQL 连接配置

    所有数据库工具都通过此函数拿到同一份连接参数，避免每个工具重复读取环境变量
    :return: mysql.connector.connect 可直接使用的连接参数
    """
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
    }

    # 去掉未配置的可选项，避免把 None 传给 mysql.connector 造成连接参数异常
    config = {k: v for k, v in config.items() if v is not None}

    # user/password/database 是本教程工具能正常查询业务库的最小必要配置
    required_keys = ["user", "password", "database"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"缺失数据库核心配置：{', '.join(missing_keys)}")

    return config


@tool
def list_sql_tables() -> str:
    """
    查询当前数据库中所有可用表

    作用：让模型先识别真实可用的表名，方便后续预览表结构和编写自定义 SQL。
    :return: 有表：可用的表有：表1,表2,表3...
             没有表：没有可用的表
             出现异常：查询出现异常：异常信息
    """

    database_state = get_database_run_state()
    if database_state and database_state.table_list_result is not None:
        with database_state.lock:
            database_state.cache_hit_count += 1
        record_event(
            event_name="database_cache_hit",
            component="database_governance",
            message="复用数据库表名缓存",
            metadata={"cache_type": "table_list"},
        )
        return database_state.table_list_result

    # 埋点：工具一被调用，前端可以展示当前正在查询数据库表名
    monitor.report_tool(tool_name="数据库表名查询工具：list_sql_tables", args={})

    with trace_span(
        "tool.list_sql_tables",
        component="tool",
        metadata={"tool_name": "list_sql_tables"},
    ) as span:
        # 加载数据库连接信息
        config = get_db_config()

        # MySQL 查询的固定步骤：
        # 1. 创建连接
        # 2. 创建 cursor
        # 3. 执行 SQL
        # 4. 获取返回结果
        # 5. 释放连接和 cursor 资源
        # 这里捕获异常并返回中文提示，避免工具报错直接中断 Agent 执行链路
        try:
            table_names = _load_table_names(config)
            if not table_names:
                span.set_result(table_count=0)
                return "没有可用的表"

            result = f"可用的表有：{', '.join(table_names)}"
            if database_state:
                with database_state.lock:
                    database_state.table_list_result = result
                    database_state.table_names.update(table_names)
            span.set_result(
                table_count=len(table_names),
                table_names=table_names,
            )
            return result
        except Error as e:
            span.set_result(error_message=str(e))
            return f"查询出现异常：{str(e)}"


@tool
def get_table_data(table_name: str, row_limit: int = 20) -> str:
    """
    查询指定表的前若干行数据

    当前工具调用之前，应先调用 list_sql_tables 完成表名校验。
    此工具的作用：
    1. 完成单表样例数据查询
    2. 为多表查询提供表结构信息和数据格式参考
    :param table_name: 表名
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             4. 默认查询 20 条，最多查询 50 条表数据
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
                1,张三,18\n
    """
    row_limit = max(1, min(int(row_limit), 50))
    database_state = get_database_run_state()
    if database_state:
        with database_state.lock:
            cached = database_state.table_previews.get(table_name)
            if cached is not None:
                database_state.cache_hit_count += 1
        if cached is not None:
            record_event(
                event_name="database_cache_hit",
                component="database_governance",
                message=f"复用 {table_name} 表预览缓存",
                metadata={"cache_type": "table_preview", "table_name": table_name},
            )
            return cached

    # 埋点：工具二被调用，前端可以展示当前正在预览哪张表
    monitor.report_tool(
        tool_name="数据库表数据查询工具：get_table_data",
        args={"table_name": table_name, "row_limit": row_limit},
    )

    with trace_span(
        "tool.get_table_data",
        component="tool",
        metadata={
            "tool_name": "get_table_data",
            "table_name": table_name,
            "row_limit": row_limit,
        },
    ) as span:
        # 获取数据库参数
        config = get_db_config()

        # 查询流程同样是：连接 -> cursor -> 执行 SQL -> 获取列信息和数据 -> 自动释放资源
        try:
            with connect(**config) as conn:
                with conn.cursor() as cursor:
                    if database_state and database_state.table_names:
                        with database_state.lock:
                            table_names = set(database_state.table_names)
                    else:
                        cursor.execute("SHOW TABLES")
                        table_names = {
                            str(table[0]) for table in cursor.fetchall()
                        }
                    if table_name not in table_names:
                        return f"错误：数据表 {table_name} 不存在或不允许访问。"
                    sql = f"SELECT * FROM `{table_name}` LIMIT {row_limit}"
                    cursor.execute(sql)

                    # cursor.description 保存查询结果的列元信息
                    # 例如：[("id", ...), ("name", ...), ("age", ...)]
                    # 如果 SQL 没有结果集，description 可能为 None
                    description = cursor.description
                    if not description:
                        span.set_result(row_count=0, result_summary="暂无数据")
                        return f"数据表 {table_name} 暂无数据。"

                    # 只取每个列信息元组的第一个元素，也就是列名
                    # 例如：["id", "name", "age"]
                    columns = [desc[0] for desc in description]

                    # fetchall 返回表数据，形如：[(1, "张三", 18), (2, "李四", 20)]
                    rows = cursor.fetchmany(row_limit)
                    result, truncated = _compact_rows(columns, rows, max_chars=8000)
                    if database_state:
                        with database_state.lock:
                            database_state.table_previews[table_name] = result
                            database_state.table_names.update(table_names)
                            if truncated:
                                database_state.truncated_result_count += 1
                    span.set_result(
                        row_count=len(rows),
                        column_count=len(columns),
                        columns=columns,
                        truncated=truncated,
                        result_summary=summarize_text(result),
                    )
                    return result
        except Error as e:
            span.set_result(error_message=str(e))
            return f"查询出现异常：{str(e)}"


@tool
def execute_sql_query(
    query: str,
    continuation_reason: Optional[
        Literal["evidence_gap", "query_correction", "result_validation"]
    ] = None,
    target_gap: str = "",
) -> str:
    """
    执行自定义 SQL 查询

    切记：执行之前，需要通过 list_sql_tables 明确真实表名，
    再通过 get_table_data 明确表结构和数据格式。
    适合多表关联、筛选、聚合、排序等复杂查询。
    :param query: 要执行的自定义 SQL 语句
    :return: CSV 格式数据
             1. 第一行是列信息，列之间使用英文逗号分隔
             2. 第二行开始是表数据，值之间也使用英文逗号分隔
             3. 行和行之间使用 \n 分隔
             例如：
                id,name,age\n -> 列头
                1,张三,18\n
                1,张三,18\n
    """
    valid, validation_error = validate_read_only_query(query)
    if not valid:
        record_event(
            event_name="database_query_blocked",
            component="database_governance",
            message=validation_error,
            status="warning",
            metadata={"blocked_reason": "read_only_validation"},
        )
        return f"SQL 已拒绝执行：{validation_error}"

    database_state = get_database_run_state()
    reservation = (
        database_state.reserve_query(
            query,
            continuation_reason=continuation_reason,
            target_gap=target_gap,
        )
        if database_state
        else None
    )
    if reservation is not None and not reservation.allowed:
        if reservation.cached_result is not None:
            record_event(
                event_name="database_cache_hit",
                component="database_governance",
                message=reservation.message,
                metadata={"cache_type": "sql_result"},
            )
            return reservation.cached_result
        record_event(
            event_name="database_query_blocked",
            component="database_governance",
            message=reservation.message,
            status="warning",
            metadata={
                "blocked_reason": reservation.blocked_reason,
                "remaining_budget": database_state.snapshot()["remaining_budget"],
            },
        )
        return (
            f"SQL 已拒绝执行：{reservation.message}"
            " 请使用已有查询结果完成回答，不要继续重复查询。"
        )

    if reservation is not None:
        record_event(
            event_name="database_query_reserved",
            component="database_governance",
            message=f"已预留第 {reservation.call_index} 次 SQL 查询",
            metadata={
                "call_index": reservation.call_index,
                "is_extension": reservation.is_extension,
                "continuation_reason": reservation.continuation_reason,
                "target_gap": summarize_text(reservation.target_gap),
            },
        )

    # 埋点：记录模型最终生成的 SQL，便于教学时观察是否真的落到了正确表字段上
    monitor.report_tool(
        tool_name="数据库表数据查询工具：execute_sql_query",
        args={
            "query": query,
            "continuation_reason": continuation_reason,
            "target_gap": target_gap,
        },
    )

    with trace_span(
        "tool.execute_sql_query",
        component="tool",
        metadata={
            "tool_name": "execute_sql_query",
            "query": query,
            "continuation_reason": continuation_reason,
            "target_gap": target_gap,
        },
    ) as span:
        # 获取数据库参数
        config = get_db_config()

        # 自定义查询和 get_table_data 的结果处理逻辑一致：
        # 执行 SQL -> 读取 description 得到列名 -> fetchall 得到数据 -> 拼成 CSV 返回
        try:
            with connect(**config) as conn:
                with conn.cursor() as cursor:
                    # 当前章节依赖提示词约束模型生成只读查询；生产环境建议在工具层限制 SELECT/SHOW
                    cursor.execute(query)

                    # 非查询类 SQL 没有结果集描述，这里统一返回提示，避免工具调用直接抛错给模型
                    description = cursor.description
                    if not description:
                        result = f"执行自定义 SQL 语句没有查询结果，SQL 为：{query}"
                        if database_state and reservation:
                            database_state.complete_query(
                                reservation,
                                result=result,
                            )
                        span.set_result(row_count=0, result_summary=summarize_text(result))
                        return result
                    # description => [("列1", ...), ("列2", ...)]
                    columns = [desc[0] for desc in description]

                    # rows => [(值1, 值2), (值1, 值2)]
                    rows = cursor.fetchmany(101)
                    row_limit_truncated = len(rows) > 100
                    rows = rows[:100]
                    result, char_limit_truncated = _compact_rows(
                        columns,
                        rows,
                        max_chars=12000,
                    )
                    truncated = row_limit_truncated or char_limit_truncated
                    if row_limit_truncated and not result.endswith(
                        "...结果已按上下文上限截断..."
                    ):
                        result += "\n...结果已限制为前 100 行..."
                    if database_state and reservation:
                        database_state.complete_query(
                            reservation,
                            result=result,
                            truncated=truncated,
                        )
                    span.set_result(
                        row_count=len(rows),
                        column_count=len(columns),
                        columns=columns,
                        truncated=truncated,
                        result_summary=summarize_text(result),
                    )
                    record_event(
                        event_name="database_query_completed",
                        component="database_governance",
                        message="SQL 查询执行完成",
                        metadata={
                            "call_index": (
                                reservation.call_index if reservation else None
                            ),
                            "row_count": len(rows),
                            "truncated": truncated,
                        },
                    )
                    return result
        except Exception as e:
            if database_state and reservation:
                database_state.complete_query(reservation, error=e)
            span.set_result(error_message=str(e))
            return f"查询出现异常：{str(e)}"


if __name__ == "__main__":
    # 本地调试入口：直接运行本文件可验证 .env 中的 MySQL 连接配置是否可用
    print(
        execute_sql_query.invoke(
            {
                "query": "SELECT * FROM `drugs` dgs join sales_records srd on dgs.drug_id = srd.drug_id"
            }
        )
    )
