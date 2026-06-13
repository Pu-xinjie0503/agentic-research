"""
主智能体组装与异步执行模块

负责把模型、主提示词、交付类工具和三个专家子智能体组装成 DeepAgent，
并提供 run_deep_agent 作为后续 API 层调用的统一入口。运行时还会为每个
session_id 创建独立工作目录，并把工具调用、子智能体调用和最终结果推送给前端。
"""

import asyncio
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.direct.file_analysis import file_analysis_direct_agent
from app.agent.llm import model
from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.agent.middleware.agent_governance import AgentGovernanceMiddleware
from app.agent.middleware.final_response_governance import (
    FinalResponseGovernanceMiddleware,
)
from app.agent.middleware.tool_allowlist import ToolAllowlistMiddleware
from app.agent.prompts import main_agent_content
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.file_analysis_agent import file_analysis_agent
from app.agent.subagents.network_search_agent import network_search_agent
from app.api.context import (
    reset_session_context,
    reset_thread_context,
    reset_trace_context,
    set_session_context,
    set_thread_context,
    set_trace_context,
)
from app.api.monitor import monitor
from app.observability.tracing import (
    begin_trace,
    finish_trace,
    record_event,
    reset_trace_state,
    summarize_text,
    trace_span,
)
from app.observability.search_state import (
    begin_search_run,
    configure_search_run,
    finalize_search_run,
    reset_search_run,
)
from app.observability.agent_state import (
    begin_agent_run,
    configure_agent_run,
    finalize_agent_run,
    infer_task_scope,
    reset_agent_run,
)
from app.observability.database_state import (
    begin_database_run,
    finalize_database_run,
    reset_database_run,
)

# 交付类工具由主智能体直接掌握，负责生成最终 Markdown/PDF 文档
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.utils.console import safe_console_print

# 主智能体是调度中心：
# 1. tools 只放最终交付相关的文件生成工具
# 2. subagents 放网络、数据库、文件分析三类信息获取助手
# 3. checkpointer 通过 thread_id 保存同一会话中的执行上下文
main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[generate_markdown, convert_md_to_pdf],
    middleware=[
        AgentGovernanceMiddleware(),
        ToolAllowlistMiddleware(
            {"task", "generate_markdown", "convert_md_to_pdf"},
            retry_unknown_tool_calls=True,
        ),
        FinalResponseGovernanceMiddleware(),
        ModelTracingMiddleware("主智能体"),
        ModelCallLimitMiddleware(run_limit=14, exit_behavior="error"),
    ],
    checkpointer=InMemorySaver(),
    subagents=[database_query_agent, network_search_agent, file_analysis_agent],
)

# 当前文件位于 app/agent/main_agent.py，parents[1] 即 app 目录
project_root_path = Path(__file__).parents[1].resolve()


@dataclass(frozen=True)
class AgentRunResult:
    """一次 Agent 执行的结构化返回值。"""

    trace_summary: dict[str, Any] | None
    final_result: str
    artifacts: list[dict[str, Any]]


def _collect_artifacts(session_dir: Path | None) -> list[dict[str, Any]]:
    """收集当前会话实际生成的文件元数据。"""
    if session_dir is None or not session_dir.exists():
        return []
    return [
        {
            "filename": path.name,
            "path": str(path),
            "suffix": path.suffix.lower(),
            "size": path.stat().st_size,
        }
        for path in sorted(session_dir.rglob("*"))
        if path.is_file()
    ]


def should_use_file_direct_path(
    task_scope,
    uploaded_files: list[str],
) -> bool:
    """仅让无产物、无其他能力依赖的上传文件任务走直达路径。"""
    return bool(uploaded_files) and (
        task_scope.allowed_subagents == frozenset({"文件分析助手"})
        and task_scope.artifact_type is None
    )


def build_file_direct_request(
    task_query: str,
    uploaded_files: list[str],
) -> str:
    """构造文件直达 Agent 的最小运行时请求。"""
    file_list = "\n".join(f"- {filename}" for filename in uploaded_files)
    return f"""用户任务：
{task_query}

当前会话上传文件：
{file_list}

请逐个读取上述文件，只根据文件内容完成分析并直接回答用户。"""


def extract_final_agent_text(result: dict[str, Any]) -> str:
    """从 Agent 最终状态中提取最后一条非工具调用文本。"""
    for message in reversed(result.get("messages") or []):
        if getattr(message, "tool_calls", None):
            continue
        text = getattr(message, "text", None)
        if text:
            return str(text)
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return ""


async def run_deep_agent(
    task_query,
    session_id,
    trace_id=None,
    run_metadata: dict[str, Any] | None = None,
) -> AgentRunResult:
    """
    异步流式执行主智能体

    API 层会为每次任务传入用户问题和 session_id。本函数负责准备会话目录、
    复制上传文件、写入 ContextVar，并在流式执行过程中把关键事件上报给前端。
    :param task_query: 前端提交的原始任务问题
    :param session_id: 当前任务 ID，同时用于 thread_id、输出目录和 WebSocket 定向推送
    :param trace_id: 当前任务执行链路 ID；不传时自动生成
    """
    trace_id = trace_id or str(uuid.uuid4())
    trace_token = set_trace_context(trace_id)
    session_id_token = set_thread_context(session_id)
    trace_state_token = begin_trace(
        trace_id,
        session_id,
        task_query,
        run_metadata=run_metadata,
    )
    search_state_token = begin_search_run(trace_id)
    agent_state_token = begin_agent_run(trace_id)
    database_state_token = begin_database_run(trace_id)
    session_dir_token = None
    final_status = "success"
    final_error = None
    trace_summary = None
    final_result = ""
    session_dir: Path | None = None

    safe_console_print(
        f"[MainAgent] 开始执行会话，session_id={session_id}, trace_id={trace_id}"
    )

    try:
        with trace_span(
            "agent.session.prepare",
            component="agent",
            metadata={"session_id": session_id},
        ) as prepare_span:
            # 每个会话独立使用 output/session_{session_id}，避免不同用户的产物互相覆盖
            session_dir = project_root_path / "output" / f"session_{session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)

            # 前端和工具使用绝对路径；提示词里只给模型相对路径，降低模型误用系统绝对路径的概率
            session_dir_str = str(session_dir).replace("\\", "/")
            relative_session_dir_str = str(
                session_dir.relative_to(project_root_path)
            ).replace("\\", "/")

            # 上传文件先落在 updated/session_{session_id}，执行前复制到本次 output 工作目录
            # 这样读文件工具和生成文件工具都只需要围绕同一个 session_dir 工作
            updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
            updated_info_prompt = ""
            uploaded_files = []
            if updated_dir_path.exists():
                files = [f.name for f in updated_dir_path.iterdir() if f.is_file()]
                if files:
                    for filename in files:
                        # copy2 会保留上传文件的修改时间、权限等元数据，便于后续排查文件来源
                        shutil.copy2(updated_dir_path / filename, session_dir / filename)
                        uploaded_files.append(filename)

                    # 把上传文件列表注入用户消息，提醒模型先调用 read_file_content 获取附件内容
                    updated_info_prompt = (
                        "\n    [已上传文件] 已加载到工作目录:\n"
                        + "\n".join([f"    - {f}" for f in files])
                        + "\n    请优先调用文件分析助手读取并分析这些文件。"
                    )

            prepare_span.set_result(
                session_dir=relative_session_dir_str,
                uploaded_file_count=len(uploaded_files),
                uploaded_files=uploaded_files,
            )

        task_scope = infer_task_scope(
            str(task_query),
            has_uploaded_files=bool(uploaded_files),
        )
        allowed_subagents = set(task_scope.allowed_subagents)
        configure_agent_run(allowed_subagents)
        if task_scope.artifact_type or len(allowed_subagents) > 1:
            configure_search_run(hard_limit=3)
        record_event(
            event_name="task_scope_inferred",
            component="agent_governance",
            message="已推断本次任务的能力范围",
            metadata=task_scope.snapshot(),
        )

        # ContextVar 让深层工具无需显式传参，也能拿到当前会话目录和 WebSocket thread_id
        session_dir_token = set_session_context(session_dir_str)

        # 前端拿到工作目录后，可以展示本次任务生成的 Markdown/PDF 等产物
        monitor.report_session_dir(session_dir_str)

        # checkpointer 依赖 thread_id 区分会话记忆；同一 session_id 会复用同一条执行上下文
        config = {"configurable": {"thread_id": session_id}}
        use_file_direct_path = should_use_file_direct_path(
            task_scope,
            uploaded_files,
        )
        execution_route = "file_direct" if use_file_direct_path else "main_agent"
        record_event(
            event_name="execution_route_selected",
            component="agent_governance",
            message=f"已选择 {execution_route} 执行路径",
            metadata={
                "route": execution_route,
                "uploaded_files": uploaded_files,
                "task_scope": task_scope.snapshot(),
            },
        )

        # 工作环境指令是运行时动态补充的，约束模型只在当前会话目录读写文件
        evidence_policy = (
            "本次属于多源组合或文档交付任务。外部趋势只保留可核验的"
            "定性结论、政策变化和技术方向；禁止写入第三方市场规模、"
            "CAGR、预测金额、并购或融资金额，以及没有权威来源支撑的"
            "人口等精确数字。"
            if task_scope.artifact_type or len(allowed_subagents) > 1
            else "按用户任务保留必要事实，并确保精确数字可由来源 URL 核验。"
        )
        path_instruction = f"""
        【工作环境指令】
        工作目录: {relative_session_dir_str}
        {updated_info_prompt}

        规则：
        1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
        2. 读取已上传的文件时，请调用文件分析助手，并要求它直接将文件名（例如：'开篇.txt'）作为 filename 参数传入读取工具，不要带上任何目录前缀。
        3. 使用相对路径，禁止使用绝对路径
        4. 若存在上传文件，请先分析内容
        5. 本次允许调用的专家仅限：{', '.join(sorted(allowed_subagents)) or '无'}。不得调用列表之外的专家。
        6. 证据策略：{evidence_policy}
        7. 主智能体不存在 read_file、glob、ls、write_todos 等文件系统工具。专家返回后直接调用 generate_markdown；上传文件只能通过 task 委派给文件分析助手读取。
        8. 生成报告时必须保留上传文件中的具体对象、观察结论和信息缺口，不得把附件事实替换成泛化行业描述。
        9. 报告必须包含“待核验/信息缺口”和“风险与边界”两类说明；网络引用只能使用证据目录中的相关 URL，不得引用与任务主题无关的真实链接。
        10. 通用行业或政策证据不得外推为某个具体药品已经出现价格、需求、份额或集采影响；只有证据标题或摘要直接点名该药品时才可形成品种级外部结论，否则只写行业层趋势和待核验项。
        """

        with trace_span(
            "agent.file_direct.run" if use_file_direct_path else "agent.main.run",
            component="agent",
            metadata={
                "session_id": session_id,
                "task_query": summarize_text(task_query),
                "execution_route": execution_route,
            },
        ) as agent_span:
            chunk_count = 0
            assistant_calls = []
            if use_file_direct_path:
                assistant_calls.append("文件分析助手")
                monitor.report_assistant(
                    "文件分析助手",
                    {
                        "mode": "direct",
                        "files": uploaded_files,
                    },
                )
                direct_result = await file_analysis_direct_agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": build_file_direct_request(
                                    str(task_query),
                                    uploaded_files,
                                ),
                            }
                        ]
                    }
                )
                final_result = extract_final_agent_text(direct_result)
                if not final_result:
                    raise RuntimeError("文件直达 Agent 未返回最终文本")
                safe_console_print(
                    f"文件分析直达结果：{final_result[:100]}"
                )
                monitor.report_task_result(final_result)
            else:
                # astream 会持续产出模型节点、工具节点和子智能体节点的状态片段
                async for chunk in main_agent.astream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": task_query + path_instruction,
                            }
                        ]
                    },
                    config=config,
                ):
                    chunk_count += 1
                    # chunk 形如 {"model": {"messages": [...]}}，这里主要关心模型最新消息
                    for node_name, state in chunk.items():
                        if not state or "messages" not in state:
                            continue
                        messages = state["messages"]
                        if messages and isinstance(messages, list):
                            last_msg = messages[-1]
                            if node_name == "model":
                                if last_msg.tool_calls:
                                    # DeepAgents 调用子智能体时，本质上会产生名为 task 的工具调用
                                    for tool_call in last_msg.tool_calls:
                                        if tool_call["name"] == "task":
                                            assistant_name = tool_call["args"][
                                                "subagent_type"
                                            ]
                                            assistant_calls.append(assistant_name)
                                            # 子智能体调用单独上报，前端可以展示“正在调用哪个专家助手”
                                            monitor.report_assistant(
                                                assistant_name,
                                                {
                                                    "description": tool_call["args"][
                                                        "description"
                                                    ]
                                                },
                                            )
                                elif last_msg.content:
                                    # 模型没有继续调用工具时，最新文本内容就是本轮可反馈给前端的结果
                                    final_result = last_msg.text or str(
                                        last_msg.content
                                    )
                                    safe_console_print(
                                        "主智能体执行结果，最终结果："
                                        f"{final_result[:100]}"
                                    )
                                    monitor.report_task_result(final_result)

            agent_span.set_result(
                chunk_count=chunk_count,
                assistant_calls=assistant_calls,
                execution_route=execution_route,
                final_result_summary=summarize_text(final_result),
            )

    except asyncio.CancelledError:
        final_status = "cancelled"
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        final_status = "error"
        final_error = e
        # 异步执行异常也走 monitor，保证前端能收到明确错误事件
        monitor._emit("error", f"执行主智能发生异常信息：{str(e)}")
    finally:
        search_stop_reason = {
            "success": "evidence_sufficient",
            "error": "task_error",
            "cancelled": "task_cancelled",
        }[final_status]
        search_summary = finalize_search_run(search_stop_reason)
        finalize_agent_run(search_stop_reason)
        finalize_database_run(search_stop_reason)
        if search_summary:
            record_event(
                event_name="search_stop",
                component="search_governance",
                message="本次任务的网络搜索阶段已结束",
                metadata=search_summary,
            )
        trace_summary = finish_trace(final_status, final_error)
        if trace_summary:
            monitor._emit(
                "trace_summary",
                "任务链路汇总已生成",
                trace_summary,
                write_log=False,
            )
        # 任务结束后恢复 ContextVar，避免后续请求复用到本次会话目录或 thread_id
        if session_dir_token is not None:
            reset_session_context(session_dir_token)
        reset_search_run(search_state_token)
        reset_database_run(database_state_token)
        reset_agent_run(agent_state_token)
        reset_trace_state(trace_state_token)
        reset_thread_context(session_id_token)
        reset_trace_context(trace_token)

    return AgentRunResult(
        trace_summary=trace_summary,
        final_result=final_result,
        artifacts=_collect_artifacts(session_dir),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001")
    )
