"""模型调用中间件与性能汇总测试。"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.middleware.model_tracing import ModelTracingMiddleware
from app.api.context import (
    reset_thread_context,
    reset_trace_context,
    set_thread_context,
    set_trace_context,
)
from app.evaluation.evaluator import merge_span_intervals_ms
from app.observability.tracing import begin_trace, finish_trace, reset_trace_state


class FakeModel:
    """提供可被中间件识别的模型名称。"""

    model_name = "fake-chat-model"


class FakeTool:
    """提供最小工具名称。"""

    name = "fake_tool"


class CapturedSpan:
    """收集中间件写入的 span 结果。"""

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.result: dict = {}

    def set_result(self, **kwargs) -> None:
        self.result.update(kwargs)


def build_request() -> ModelRequest:
    """构造不依赖真实模型的请求。"""
    return ModelRequest(
        model=FakeModel(),
        messages=[HumanMessage(content="请查询库存")],
        system_message=SystemMessage(content="你是数据库助手"),
        tools=[FakeTool()],
    )


class ModelTracingMiddlewareTests(unittest.TestCase):
    """验证同步模型调用统计。"""

    def setUp(self) -> None:
        self.captured_spans: list[CapturedSpan] = []

    @contextmanager
    def capture_span(self, _name, component=None, metadata=None):
        span = CapturedSpan(metadata)
        self.captured_spans.append(span)
        yield span

    def test_sync_response_with_tokens_and_tool_calls(self) -> None:
        middleware = ModelTracingMiddleware("数据库查询助手")
        response = ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "execute_sql_query",
                            "args": {"query": "SELECT 1"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                    usage_metadata={
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150,
                    },
                    response_metadata={"finish_reason": "tool_calls"},
                )
            ]
        )

        with (
            patch(
                "app.agent.middleware.model_tracing.trace_span",
                self.capture_span,
            ),
            patch(
                "app.agent.middleware.model_tracing.next_model_call_index",
                return_value=2,
            ),
        ):
            actual = middleware.wrap_model_call(
                build_request(),
                lambda _request: response,
            )

        self.assertIs(actual, response)
        span = self.captured_spans[0]
        self.assertEqual(span.metadata["agent_name"], "数据库查询助手")
        self.assertEqual(span.metadata["model_name"], "fake-chat-model")
        self.assertEqual(span.metadata["call_index"], 2)
        self.assertEqual(span.metadata["input_message_count"], 2)
        self.assertEqual(span.metadata["tool_names"], ["fake_tool"])
        self.assertEqual(span.result["total_tokens"], 150)
        self.assertEqual(span.result["tool_names"], ["execute_sql_query"])
        self.assertEqual(span.result["end_reason"], "tool_calls")

    def test_missing_usage_keeps_tokens_none(self) -> None:
        middleware = ModelTracingMiddleware("文件分析助手")
        response = ModelResponse(result=[AIMessage(content="分析完成")])

        with (
            patch(
                "app.agent.middleware.model_tracing.trace_span",
                self.capture_span,
            ),
            patch(
                "app.agent.middleware.model_tracing.next_model_call_index",
                return_value=1,
            ),
        ):
            middleware.wrap_model_call(build_request(), lambda _request: response)

        result = self.captured_spans[0].result
        self.assertIsNone(result["input_tokens"])
        self.assertIsNone(result["output_tokens"])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["output_char_count"], 4)

    def test_error_is_recorded_and_rethrown(self) -> None:
        middleware = ModelTracingMiddleware("主智能体")

        def fail(_request):
            raise RuntimeError("模拟模型失败")

        with (
            patch(
                "app.agent.middleware.model_tracing.trace_span",
                self.capture_span,
            ),
            patch(
                "app.agent.middleware.model_tracing.next_model_call_index",
                return_value=1,
            ),
            self.assertRaisesRegex(RuntimeError, "模拟模型失败"),
        ):
            middleware.wrap_model_call(build_request(), fail)

        self.assertEqual(self.captured_spans[0].result["end_reason"], "error")


class AsyncModelTracingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """验证异步模型调用统计。"""

    async def test_async_response_is_returned(self) -> None:
        captured: list[CapturedSpan] = []

        @contextmanager
        def capture_span(_name, component=None, metadata=None):
            span = CapturedSpan(metadata)
            captured.append(span)
            yield span

        async def handler(_request):
            return ModelResponse(result=[AIMessage(content="异步完成")])

        middleware = ModelTracingMiddleware("网络搜索助手")
        with (
            patch(
                "app.agent.middleware.model_tracing.trace_span",
                capture_span,
            ),
            patch(
                "app.agent.middleware.model_tracing.next_model_call_index",
                return_value=3,
            ),
        ):
            response = await middleware.awrap_model_call(build_request(), handler)

        self.assertEqual(response.result[0].content, "异步完成")
        self.assertEqual(captured[0].metadata["agent_name"], "网络搜索助手")
        self.assertEqual(captured[0].result["end_reason"], "completed")


class TraceModelSummaryTests(unittest.TestCase):
    """验证不同 Agent 的模型统计不会混淆。"""

    def test_trace_summary_groups_model_spans_by_agent(self) -> None:
        from app.observability import tracing

        trace_token = set_trace_context("trace-model-summary")
        thread_token = set_thread_context("thread-model-summary")
        with patch.object(tracing, "write_json_log"):
            state_token = begin_trace(
                "trace-model-summary",
                "thread-model-summary",
                "测试模型汇总",
            )
            try:
                with tracing.trace_span(
                    "model.call",
                    component="model",
                    metadata={"agent_name": "主智能体", "call_index": 1},
                ) as span:
                    span.set_result(
                        tool_call_count=1,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                    )
                with tracing.trace_span(
                    "model.call",
                    component="model",
                    metadata={"agent_name": "数据库查询助手", "call_index": 2},
                ) as span:
                    span.set_result(
                        tool_call_count=0,
                        input_tokens=20,
                        output_tokens=10,
                        total_tokens=30,
                    )
                summary = finish_trace("success")
            finally:
                reset_trace_state(state_token)
                reset_thread_context(thread_token)
                reset_trace_context(trace_token)

        self.assertEqual(summary["model"]["call_count"], 2)
        self.assertEqual(summary["model"]["total_tokens"], 45)
        self.assertEqual(
            [call["call_index"] for call in summary["model"]["calls"]],
            [1, 2],
        )
        self.assertEqual(summary["model"]["by_agent"]["主智能体"]["call_count"], 1)
        self.assertEqual(
            summary["model"]["by_agent"]["数据库查询助手"]["total_tokens"],
            30,
        )

    def test_trace_summary_calculates_context_growth(self) -> None:
        from app.observability import tracing

        spans = [
            {
                "component": "model",
                "span_name": "model.call",
                "status": "success",
                "duration_ms": 10,
                "metadata": {
                    "agent_name": "主智能体",
                    "call_index": 2,
                    "input_message_count": 4,
                    "input_char_count": 300,
                },
                "result": {
                    "input_tokens": 150,
                    "output_tokens": 10,
                    "total_tokens": 160,
                },
            },
            {
                "component": "model",
                "span_name": "model.call",
                "status": "success",
                "duration_ms": 10,
                "metadata": {
                    "agent_name": "主智能体",
                    "call_index": 1,
                    "input_message_count": 2,
                    "input_char_count": 100,
                },
                "result": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                },
            },
        ]

        summary = tracing._build_model_summary(spans)

        self.assertEqual(summary["first_input_tokens"], 50)
        self.assertEqual(summary["last_input_tokens"], 150)
        self.assertEqual(summary["max_input_char_count"], 300)
        self.assertEqual(summary["input_token_growth_rate"], 2.0)


class PerformanceIntervalTests(unittest.TestCase):
    """验证并行工具区间不会重复累计。"""

    def test_parallel_intervals_are_merged(self) -> None:
        spans = [
            {
                "started_at": "2026-06-11T10:00:00.000",
                "timestamp": "2026-06-11T10:00:03.000",
            },
            {
                "started_at": "2026-06-11T10:00:00.500",
                "timestamp": "2026-06-11T10:00:04.000",
            },
            {
                "started_at": "2026-06-11T10:00:05.000",
                "timestamp": "2026-06-11T10:00:06.000",
            },
        ]

        self.assertEqual(merge_span_intervals_ms(spans), 5000.0)


if __name__ == "__main__":
    unittest.main()
