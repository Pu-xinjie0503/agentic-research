"""
DeepSearch 离线评测器。

评测器不直接调用大模型，而是读取 app/logs/traces 下已经持久化的 JSONL
链路日志，并根据任务集中的预期行为判断 Agent 路由、工具调用、生成物和耗时
是否符合要求。这样可以低成本复盘每次真实运行结果。
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


CURRENT_FILE = Path(__file__).resolve()
APP_ROOT = CURRENT_FILE.parents[1]
DEFAULT_TRACE_DIR = APP_ROOT / "logs" / "traces"
DEFAULT_OUTPUT_DIR = APP_ROOT / "logs" / "evaluations"
DEFAULT_TASK_SET = CURRENT_FILE.parent / "task_sets" / "default.json"


@dataclass
class TraceView:
    """单条 trace 的评测视图。"""

    trace_id: str
    records: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None

    @property
    def task_query(self) -> str:
        if self.summary:
            value = self.summary.get("task_query_summary")
            if value:
                return str(value)
        for record in self.records:
            if record.get("event") == "trace_start":
                metadata = record.get("metadata") or {}
                value = metadata.get("task_query_summary")
                if value:
                    return str(value)
            if record.get("event") == "task_submitted":
                metadata = record.get("metadata") or {}
                value = metadata.get("query")
                if value:
                    return str(value)
        return ""

    @property
    def timestamp(self) -> str:
        if self.summary and self.summary.get("timestamp"):
            return str(self.summary["timestamp"])
        for record in reversed(self.records):
            if record.get("timestamp"):
                return str(record["timestamp"])
        return ""

    @property
    def status(self) -> str:
        if self.summary:
            return str(self.summary.get("status") or "unknown")
        return "unknown"

    @property
    def duration_ms(self) -> float | None:
        if self.summary and self.summary.get("total_duration_ms") is not None:
            return float(self.summary["total_duration_ms"])
        return None

    def assistant_names(self) -> set[str]:
        names: set[str] = set()
        if self.summary:
            names.update((self.summary.get("assistant_calls") or {}).keys())
        for record in self.records:
            if record.get("event") == "assistant_call":
                metadata = record.get("metadata") or {}
                name = metadata.get("assistant_name")
                if name:
                    names.add(str(name))
        return names

    def tool_signals(self) -> set[str]:
        """返回可用于匹配的工具信号，包含中文展示名、内部工具名和 span 名。"""
        signals: set[str] = set()
        if self.summary:
            signals.update((self.summary.get("tool_calls") or {}).keys())

        for record in self.records:
            event = record.get("event")
            metadata = record.get("metadata") or {}
            if event == "tool_start":
                tool_name = metadata.get("tool_name")
                if tool_name:
                    signals.add(str(tool_name))
            if event == "span_end" and record.get("component") == "tool":
                span_name = record.get("span_name")
                if span_name:
                    signals.add(str(span_name))
                    if str(span_name).startswith("tool."):
                        signals.add(str(span_name).split(".", 1)[1])
                tool_name = metadata.get("tool_name")
                if tool_name:
                    signals.add(str(tool_name))
        return signals

    def tool_counts(self) -> dict[str, int]:
        """统计工具调用次数，内部工具名和中文展示名分开计数。"""
        counts: dict[str, int] = {}

        def increase_once(keys: set[str], key: str | None) -> None:
            """同一条记录里相同工具名只计一次，避免内部名重复累加。"""
            if not key or key in keys:
                return
            keys.add(key)
            counts[key] = counts.get(key, 0) + 1

        for record in self.records:
            metadata = record.get("metadata") or {}
            if record.get("event") == "tool_start":
                tool_name = metadata.get("tool_name")
                if tool_name:
                    counts[str(tool_name)] = counts.get(str(tool_name), 0) + 1
            if record.get("event") == "span_end" and record.get("component") == "tool":
                counted_keys: set[str] = set()
                span_name = str(record.get("span_name") or "")
                if span_name:
                    increase_once(counted_keys, span_name)
                    if span_name.startswith("tool."):
                        internal_name = span_name.split(".", 1)[1]
                        increase_once(counted_keys, internal_name)
                tool_name = metadata.get("tool_name")
                if tool_name:
                    increase_once(counted_keys, str(tool_name))
        return counts

    def artifacts(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for record in self.records:
            if record.get("event") != "span_end" or record.get("component") != "tool":
                continue
            result = record.get("result") or {}
            artifact_type = result.get("artifact_type")
            if artifact_type:
                results.append(
                    {
                        "type": artifact_type,
                        "filename": result.get("artifact_filename"),
                        "exists": result.get("artifact_exists", True),
                        "size": result.get("artifact_size"),
                    }
                )
        return results

    def tool_duration_ms(self) -> float:
        total = 0.0
        for record in self.records:
            if record.get("event") == "span_end" and record.get("component") == "tool":
                total += float(record.get("duration_ms") or 0)
        return round(total, 2)

    def performance_metrics(self) -> dict[str, Any]:
        """读取或回算模型、工具和框架耗时分解。"""
        if self.summary and isinstance(self.summary.get("performance"), dict):
            return dict(self.summary["performance"])

        total_duration_ms = float(self.duration_ms or 0)
        tool_spans = self._spans_by_component("tool")
        model_spans = self._spans_by_component("model")
        tool_duration_ms = round(
            sum(float(span.get("duration_ms") or 0) for span in tool_spans),
            2,
        )
        model_duration_ms = round(
            sum(float(span.get("duration_ms") or 0) for span in model_spans),
            2,
        )
        tool_wall_duration_ms = merge_span_intervals_ms(tool_spans)
        model_wall_duration_ms = merge_span_intervals_ms(model_spans)
        return {
            "total_duration_ms": total_duration_ms,
            "model_duration_ms": model_duration_ms,
            "model_wall_duration_ms": model_wall_duration_ms,
            "tool_duration_ms": tool_duration_ms,
            "tool_wall_duration_ms": tool_wall_duration_ms,
            "unattributed_duration_ms": round(
                max(
                    0.0,
                    total_duration_ms
                    - model_wall_duration_ms
                    - tool_wall_duration_ms,
                ),
                2,
            ),
        }

    def model_metrics(self) -> dict[str, Any] | None:
        """读取 trace_summary 中的模型调用汇总。"""
        if self.summary and isinstance(self.summary.get("model"), dict):
            return dict(self.summary["model"])
        return None

    def _spans_by_component(self, component: str) -> list[dict[str, Any]]:
        """获取指定组件的完整 span_end 记录。"""
        return [
            record
            for record in self.records
            if record.get("event") == "span_end"
            and record.get("component") == component
        ]

    def search_metrics(self) -> dict[str, Any] | None:
        """读取搜索治理摘要；兼容只有事件、没有新版 trace_summary 的日志。"""
        if self.summary and isinstance(self.summary.get("search"), dict):
            return dict(self.summary["search"])

        reserved_count = 0
        executed_count = 0
        blocked_count = 0
        unique_domain_count = 0
        blocked_reasons: dict[str, int] = {}
        stop_reason = None
        found_event = False

        for record in self.records:
            event = record.get("event")
            metadata = record.get("metadata") or {}
            if event == "search_call_reserved":
                found_event = True
                reserved_count += 1
            elif event == "search_call_completed":
                found_event = True
                executed_count += 1
                unique_domain_count = max(
                    unique_domain_count,
                    int(metadata.get("unique_domain_count") or 0),
                )
            elif event == "search_call_blocked":
                found_event = True
                blocked_count += 1
                reason = str(metadata.get("blocked_reason") or "unknown")
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
            elif event == "search_stop":
                found_event = True
                stop_reason = metadata.get("stop_reason")

        if not found_event:
            return None
        return {
            "reserved_count": reserved_count,
            "executed_count": executed_count,
            "blocked_count": blocked_count,
            "unique_domain_count": unique_domain_count,
            "blocked_reasons": blocked_reasons,
            "extension_count": 0,
            "extension_reasons": [],
            "stop_reason": stop_reason,
        }


def read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def iter_log_files(trace_dir: Path, date: str | None) -> list[Path]:
    """按日期筛选 trace 日志文件。"""
    if date:
        path = trace_dir / f"{date}.jsonl"
        return [path] if path.exists() else []
    return sorted(trace_dir.glob("*.jsonl"))


def load_traces(trace_dir: Path, date: str | None) -> dict[str, TraceView]:
    """读取 JSONL 并按 trace_id 聚合。"""
    traces: dict[str, TraceView] = {}
    for log_file in iter_log_files(trace_dir, date):
        for line_number, line in enumerate(log_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            trace_id = record.get("trace_id")
            if not trace_id:
                continue
            trace = traces.setdefault(str(trace_id), TraceView(trace_id=str(trace_id)))
            record["_source_file"] = str(log_file)
            record["_source_line"] = line_number
            trace.records.append(record)
            if record.get("event") == "trace_summary":
                trace.summary = record
    return traces


def matches_keywords(trace: TraceView, keywords: list[str]) -> bool:
    """判断 trace 的任务摘要是否包含全部关键字。"""
    text = trace.task_query
    return bool(text) and all(keyword in text for keyword in keywords)


def choose_trace_for_task(task: dict[str, Any], traces: dict[str, TraceView]) -> TraceView | None:
    """为任务选择最新的匹配 trace。"""
    keywords = task.get("match_keywords") or []
    candidates = [
        trace
        for trace in traces.values()
        if trace.summary is not None and matches_keywords(trace, keywords)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.timestamp)[-1]


def signal_exists(expected: str, signals: set[str]) -> bool:
    """支持精确匹配和包含匹配，兼容中文展示名与内部工具名。"""
    return any(expected == signal or expected in signal for signal in signals)


def count_for_signal(expected: str, counts: dict[str, int]) -> int:
    """按工具名读取调用次数。"""
    if expected in counts:
        return counts[expected]
    total = 0
    for key, value in counts.items():
        if expected in key:
            total += value
    return total


def merge_span_intervals_ms(spans: list[dict[str, Any]]) -> float:
    """合并并行 span 时间区间，避免把并行工具耗时重复相加。"""
    intervals: list[tuple[datetime, datetime]] = []
    for span in spans:
        started_at = span.get("started_at")
        ended_at = span.get("timestamp")
        if not started_at or not ended_at:
            continue
        try:
            start = datetime.fromisoformat(str(started_at))
            end = datetime.fromisoformat(str(ended_at))
        except ValueError:
            continue
        if end >= start:
            intervals.append((start, end))

    if not intervals:
        return 0.0
    intervals.sort(key=lambda item: item[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return round(
        sum((end - start).total_seconds() * 1000 for start, end in merged),
        2,
    )


def evaluate_task(task: dict[str, Any], trace: TraceView | None) -> dict[str, Any]:
    """评测单个任务。"""
    result: dict[str, Any] = {
        "task_id": task["id"],
        "task_name": task["name"],
        "query": task["query"],
        "status": "not_run",
        "passed": False,
        "checks": [],
        "issues": ["没有找到匹配的 trace 日志，说明该任务还没有实际运行或任务描述差异过大。"],
    }
    if trace is None:
        return result

    assistants = trace.assistant_names()
    tools = trace.tool_signals()
    tool_counts = trace.tool_counts()
    artifacts = trace.artifacts()
    search_metrics = trace.search_metrics()
    performance = trace.performance_metrics()
    model_metrics = trace.model_metrics()
    issues: list[str] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            issues.append(detail)

    expected_status = task.get("expected_status", "success")
    add_check(
        "任务状态",
        trace.status == expected_status,
        f"期望状态 {expected_status}，实际状态 {trace.status}",
    )

    for assistant in task.get("expected_assistants", []):
        has_assistant = assistant in assistants
        add_check(
            f"应调用子 Agent：{assistant}",
            has_assistant,
            f"已调用预期子 Agent：{assistant}"
            if has_assistant
            else f"缺少预期子 Agent：{assistant}",
        )

    for assistant in task.get("forbidden_assistants", []):
        forbidden_hit = assistant in assistants
        add_check(
            f"不应调用子 Agent：{assistant}",
            not forbidden_hit,
            f"误调用了子 Agent：{assistant}"
            if forbidden_hit
            else f"未调用禁用子 Agent：{assistant}",
        )

    for tool_name in task.get("expected_tools", []):
        has_tool = signal_exists(tool_name, tools)
        add_check(
            f"应调用工具：{tool_name}",
            has_tool,
            f"已调用预期工具：{tool_name}" if has_tool else f"缺少预期工具：{tool_name}",
        )

    for tool_name in task.get("forbidden_tools", []):
        forbidden_hit = signal_exists(tool_name, tools)
        add_check(
            f"不应调用工具：{tool_name}",
            not forbidden_hit,
            f"误调用了工具：{tool_name}" if forbidden_hit else f"未调用禁用工具：{tool_name}",
        )

    for tool_name, max_calls in (task.get("max_tool_calls") or {}).items():
        actual_calls = count_for_signal(tool_name, tool_counts)
        within_limit = actual_calls <= int(max_calls)
        add_check(
            f"工具调用次数上限：{tool_name}",
            within_limit,
            f"工具 {tool_name} 调用了 {actual_calls} 次，未超过上限 {max_calls}"
            if within_limit
            else f"工具 {tool_name} 调用了 {actual_calls} 次，超过上限 {max_calls}",
        )

    search_policy = task.get("search_policy")
    if search_policy:
        has_search_metrics = search_metrics is not None
        add_check(
            "搜索治理遥测",
            has_search_metrics,
            "已记录搜索预算、信息增益和停止原因"
            if has_search_metrics
            else "缺少搜索治理遥测，无法判断弹性补搜是否合理",
        )

        if search_metrics:
            soft_limit = int(search_policy.get("soft_limit", 3))
            hard_limit = int(search_policy.get("hard_limit", 5))
            min_search_calls = int(search_policy.get("min_search_calls", 1))
            min_unique_domains = int(search_policy.get("min_unique_domains", 1))
            executed_count = int(search_metrics.get("executed_count") or 0)
            reserved_count = int(search_metrics.get("reserved_count") or 0)
            extension_count = int(search_metrics.get("extension_count") or 0)
            extension_reasons = search_metrics.get("extension_reasons") or []
            blocked_reasons = search_metrics.get("blocked_reasons") or {}
            unique_domain_count = int(
                search_metrics.get("unique_domain_count") or 0
            )

            within_budget = (
                executed_count <= hard_limit and reserved_count <= hard_limit
            )
            add_check(
                "搜索硬预算",
                within_budget,
                f"实际执行 {executed_count} 次、预留 {reserved_count} 次，"
                f"硬上限为 {hard_limit} 次",
            )

            expected_extensions = max(0, executed_count - soft_limit)
            valid_extension = (
                expected_extensions == extension_count
                and len(extension_reasons) == extension_count
                and all(
                    reason
                    in {
                        "evidence_gap",
                        "source_diversity",
                        "conflict_resolution",
                    }
                    for reason in extension_reasons
                )
            )
            add_check(
                "弹性补搜理由",
                valid_extension,
                "未超过软预算，无需补搜理由"
                if expected_extensions == 0 and valid_extension
                else (
                    f"软预算后执行 {expected_extensions} 次补搜，"
                    f"已记录 {extension_count} 个有效理由"
                    if valid_extension
                    else (
                        f"软预算后执行 {expected_extensions} 次补搜，"
                        f"但有效补搜记录为 {extension_count} 次"
                    )
                ),
            )

            duplicate_count = int(blocked_reasons.get("duplicate_query") or 0)
            add_check(
                "重复搜索",
                duplicate_count == 0,
                "未出现完全重复或高度相似查询"
                if duplicate_count == 0
                else f"模型尝试了 {duplicate_count} 次重复搜索，已被工具层拦截",
            )

            sufficient_search = (
                executed_count >= min_search_calls
                and unique_domain_count >= min_unique_domains
            )
            add_check(
                "避免过早停止",
                sufficient_search,
                f"执行 {executed_count} 次搜索，覆盖 {unique_domain_count} 个独立域名；"
                f"最低要求为 {min_search_calls} 次、{min_unique_domains} 个域名",
            )

    artifact_types = {str(item.get("type")) for item in artifacts}
    for artifact_type in task.get("expected_artifacts", []):
        has_artifact = artifact_type in artifact_types
        add_check(
            f"应生成产物：{artifact_type}",
            has_artifact,
            f"已生成预期产物：{artifact_type}"
            if has_artifact
            else f"缺少预期产物：{artifact_type}",
        )

    passed = all(check["passed"] for check in checks)
    return {
        **result,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "trace_id": trace.trace_id,
        "thread_id": trace.summary.get("thread_id") if trace.summary else None,
        "trace_status": trace.status,
        "matched_query": trace.task_query,
        "total_duration_ms": trace.duration_ms,
        "tool_duration_ms": trace.tool_duration_ms(),
        "performance": performance,
        "model_metrics": model_metrics,
        "assistants": sorted(assistants),
        "tools": sorted(tools),
        "tool_counts": tool_counts,
        "search_metrics": search_metrics,
        "artifacts": artifacts,
        "checks": checks,
        "issues": issues,
    }


def build_report(task_set: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """构造总体评测报告。"""
    evaluated = [item for item in results if item["status"] != "not_run"]
    passed = [item for item in evaluated if item["passed"]]
    failed = [item for item in evaluated if not item["passed"]]
    not_run = [item for item in results if item["status"] == "not_run"]
    durations = [
        float(item["total_duration_ms"])
        for item in evaluated
        if item.get("total_duration_ms") is not None
    ]

    return {
        "generated_at": datetime.now().isoformat(),
        "task_set": {
            "name": task_set.get("name"),
            "version": task_set.get("version"),
            "description": task_set.get("description"),
        },
        "summary": {
            "total_tasks": len(results),
            "evaluated_tasks": len(evaluated),
            "passed_tasks": len(passed),
            "failed_tasks": len(failed),
            "not_run_tasks": len(not_run),
            "pass_rate": round(len(passed) / len(evaluated), 4) if evaluated else 0,
            "average_duration_ms": round(sum(durations) / len(durations), 2)
            if durations
            else None,
        },
        "results": results,
    }


def markdown_bool(value: bool) -> str:
    """把布尔值转成报告中的中文状态。"""
    return "通过" if value else "未通过"


def render_markdown(report: dict[str, Any]) -> str:
    """渲染 Markdown 评测报告。"""
    summary = report["summary"]
    lines = [
        "# DeepSearch 离线评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 任务集：{report['task_set']['name']} ({report['task_set']['version']})",
        f"- 已评测任务：{summary['evaluated_tasks']}/{summary['total_tasks']}",
        f"- 通过任务：{summary['passed_tasks']}",
        f"- 失败任务：{summary['failed_tasks']}",
        f"- 未运行任务：{summary['not_run_tasks']}",
        f"- 通过率：{summary['pass_rate'] * 100:.2f}%",
    ]
    if summary["average_duration_ms"] is not None:
        lines.append(f"- 平均耗时：{summary['average_duration_ms'] / 1000:.2f}s")

    lines.extend(
        [
            "",
            "## 任务总览",
            "",
            "| 任务 ID | 状态 | Trace | 耗时 | 主要问题 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in report["results"]:
        trace_id = item.get("trace_id") or "-"
        duration = (
            f"{float(item['total_duration_ms']) / 1000:.2f}s"
            if item.get("total_duration_ms") is not None
            else "-"
        )
        issue = "；".join(item.get("issues") or ["无"])
        lines.append(
            f"| {item['task_id']} | {item['status']} | {trace_id} | {duration} | {issue} |"
        )

    lines.extend(["", "## 任务详情", ""])
    for item in report["results"]:
        lines.extend(
            [
                f"### {item['task_id']}：{item['task_name']}",
                "",
                f"- 状态：{item['status']}",
                f"- Trace：{item.get('trace_id') or '未匹配'}",
                f"- 任务：{item['query']}",
            ]
        )
        if item.get("total_duration_ms") is not None:
            lines.append(f"- 总耗时：{float(item['total_duration_ms']) / 1000:.2f}s")
        if item.get("performance"):
            performance = item["performance"]
            lines.append(
                "- 性能分解："
                f"模型墙钟 {float(performance.get('model_wall_duration_ms') or 0) / 1000:.2f}s，"
                f"工具墙钟 {float(performance.get('tool_wall_duration_ms') or 0) / 1000:.2f}s，"
                f"工具累计 {float(performance.get('tool_duration_ms') or 0) / 1000:.2f}s，"
                f"未归因 {float(performance.get('unattributed_duration_ms') or 0) / 1000:.2f}s"
            )
        if item.get("model_metrics"):
            model_metrics = item["model_metrics"]
            lines.append(
                "- 模型调用："
                f"{model_metrics.get('call_count', 0)} 次，"
                f"累计 {float(model_metrics.get('total_duration_ms') or 0) / 1000:.2f}s，"
                f"总 token {model_metrics.get('total_tokens')}"
            )
        if item.get("assistants"):
            lines.append(f"- 子 Agent：{', '.join(item['assistants'])}")
        if item.get("tool_counts"):
            tool_counts = ", ".join(
                f"{name}={count}" for name, count in sorted(item["tool_counts"].items())
            )
            lines.append(f"- 工具调用次数：{tool_counts}")
        if item.get("search_metrics"):
            search = item["search_metrics"]
            lines.append(
                "- 搜索治理："
                f"执行 {search.get('executed_count', 0)} 次，"
                f"拦截 {search.get('blocked_count', 0)} 次，"
                f"独立域名 {search.get('unique_domain_count', 0)} 个，"
                f"停止原因 {search.get('stop_reason') or '未记录'}"
            )
        if item.get("artifacts"):
            artifacts = ", ".join(
                f"{artifact.get('type')}:{artifact.get('filename')}"
                for artifact in item["artifacts"]
            )
            lines.append(f"- 生成产物：{artifacts}")

        lines.extend(["", "| 检查项 | 结果 | 说明 |", "| --- | --- | --- |"])
        for check in item.get("checks", []):
            lines.append(
                f"| {check['name']} | {markdown_bool(check['passed'])} | {check['detail']} |"
            )
        if not item.get("checks"):
            lines.append("| 未执行 | 未通过 | 没有找到可评测 trace |")
        lines.append("")

    lines.extend(
        [
            "## 结论",
            "",
            "- `not_run` 表示任务集里定义了案例，但当前日志中还没有对应真实运行记录。",
            "- `failed` 表示找到了对应 trace，但至少一个预期行为不满足。",
            "- 当前评测重点是流程正确性、工具路由和性能，不替代人工对最终内容质量的判断。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """写入 JSON 和 Markdown 报告，并维护 latest 快捷文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"evaluation_{stamp}.json"
    md_path = output_dir / f"evaluation_{stamp}.md"
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return latest_json, latest_md


def run_evaluation(
    task_set_path: Path,
    trace_dir: Path,
    output_dir: Path,
    date: str | None,
) -> dict[str, Any]:
    """执行完整离线评测流程。"""
    task_set = read_json(task_set_path)
    traces = load_traces(trace_dir, date)
    results = [
        evaluate_task(task, choose_trace_for_task(task, traces))
        for task in task_set.get("tasks", [])
    ]
    report = build_report(task_set, results)
    latest_json, latest_md = write_report(report, output_dir)
    report["output"] = {"json": str(latest_json), "markdown": str(latest_md)}
    return report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="DeepSearch 离线 trace 评测器")
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", type=str, default=None, help="只评测指定日期，例如 2026-06-10")
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    args = parse_args()
    report = run_evaluation(args.task_set, args.trace_dir, args.output_dir, args.date)
    summary = report["summary"]
    print(
        "评测完成："
        f"{summary['passed_tasks']}/{summary['evaluated_tasks']} 个已运行任务通过，"
        f"{summary['not_run_tasks']} 个任务未运行。"
    )
    if summary["average_duration_ms"] is not None:
        print(f"平均耗时：{summary['average_duration_ms'] / 1000:.2f}s")
    print(f"JSON 结果：{report['output']['json']}")
    print(f"Markdown 报告：{report['output']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
