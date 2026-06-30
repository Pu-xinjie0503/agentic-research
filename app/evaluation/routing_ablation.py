"""
路由消融评测运行器。

同一批任务分别运行两种模式：
- baseline_main_agent：所有非记忆任务强制进入主 Agent 编排。
- optimized_router：使用当前意图直达路由。

报告会按任务配对比较模型调用、工具调用、Token、端到端耗时和规则质量，
用于回答“直达路由到底减少了多少编排成本”。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agent.main_agent import (
    ROUTE_MODE_BASELINE_MAIN_AGENT,
    ROUTE_MODE_OPTIMIZED,
    normalize_execution_route_mode,
)
from app.evaluation.baseline import (
    APP_ROOT,
    BASELINE_CASES,
    DEFAULT_FIXTURE,
    BaselineRun,
    PreflightError,
    _build_run_result,
    run_preflight,
)


DEFAULT_OUTPUT_DIR = APP_ROOT / "logs" / "routing_ablation"
DEFAULT_ROUTE_MODES = (
    ROUTE_MODE_BASELINE_MAIN_AGENT,
    ROUTE_MODE_OPTIMIZED,
)
COMPARISON_METRICS = (
    "model_calls",
    "tool_calls",
    "assistant_calls",
    "total_tokens",
    "total_duration_ms",
    "model_wall_duration_ms",
    "tool_wall_duration_ms",
)


@dataclass(frozen=True)
class RoutingAblationRun:
    """一次路由消融运行。"""

    run_id: str
    pair_id: str
    route_mode: str
    case_id: str
    case_name: str
    iteration: int
    query: str
    thread_id: str
    trace_id: str
    requires_file: bool
    expected_artifacts: tuple[str, ...]
    evaluation_task_id: str

    def to_baseline_run(self) -> BaselineRun:
        """转换为复用现有质量评测逻辑所需的基线运行对象。"""
        return BaselineRun(
            run_id=self.run_id,
            case_id=self.case_id,
            case_name=self.case_name,
            iteration=self.iteration,
            query=self.query,
            thread_id=self.thread_id,
            trace_id=self.trace_id,
            requires_file=self.requires_file,
            expected_artifacts=self.expected_artifacts,
            evaluation_task_id=self.evaluation_task_id,
        )


def build_run_plan(
    selected_cases: list[str] | None = None,
    repeat_override: int | None = None,
    route_modes: tuple[str, ...] = DEFAULT_ROUTE_MODES,
) -> list[RoutingAblationRun]:
    """构造 baseline/optimized 成对运行清单。"""
    case_ids = selected_cases or list(BASELINE_CASES)
    unknown = [case_id for case_id in case_ids if case_id not in BASELINE_CASES]
    if unknown:
        raise ValueError(f"未知消融任务：{', '.join(unknown)}")
    if repeat_override is not None and repeat_override < 1:
        raise ValueError("repeat 必须大于等于 1")

    normalized_modes = tuple(
        normalize_execution_route_mode(mode) for mode in route_modes
    )
    runs: list[RoutingAblationRun] = []
    for case_id in case_ids:
        case = BASELINE_CASES[case_id]
        repeats = repeat_override or case.default_repeats
        for iteration in range(1, repeats + 1):
            suffix = uuid.uuid4().hex[:12]
            pair_id = f"{case_id}-{iteration}-{suffix}"
            for route_mode in normalized_modes:
                runs.append(
                    RoutingAblationRun(
                        run_id=f"{pair_id}-{route_mode}",
                        pair_id=pair_id,
                        route_mode=route_mode,
                        case_id=case_id,
                        case_name=case.name,
                        iteration=iteration,
                        query=case.query,
                        thread_id=f"ablation-{route_mode}-{case_id}-{suffix}",
                        trace_id=str(uuid.uuid4()),
                        requires_file=case.requires_file,
                        expected_artifacts=case.expected_artifacts,
                        evaluation_task_id=case.evaluation_task_id,
                    )
                )
    return runs


async def execute_ablation(
    runs: list[RoutingAblationRun],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixture_path: Path = DEFAULT_FIXTURE,
    quality_judge: bool = False,
) -> dict[str, Any]:
    """执行消融运行清单并生成报告。"""
    selected_cases = list(dict.fromkeys(run.case_id for run in runs))
    checks = await run_preflight(selected_cases, fixture_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat()
    manifest_path = output_dir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "started_at": started_at,
                "benchmark_type": "routing_ablation",
                "route_modes": sorted({run.route_mode for run in runs}),
                "preflight": checks,
                "runs": [asdict(run) for run in runs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for run in runs:
        try:
            results.append(
                await _execute_run(
                    run,
                    fixture_path,
                    output_dir,
                    quality_judge=quality_judge,
                )
            )
        except Exception as exc:
            results.append(_runner_error_result(run, exc))

    report = {
        "generated_at": datetime.now().isoformat(),
        "benchmark_type": "routing_ablation",
        "manifest": str(manifest_path),
        "preflight": checks,
        "summary": _build_summary(results),
        "results": results,
    }
    _write_report(report, output_dir)
    return report


async def _execute_run(
    run: RoutingAblationRun,
    fixture_path: Path,
    output_dir: Path,
    quality_judge: bool,
) -> dict[str, Any]:
    """执行一条消融任务并复用基线质量评测。"""
    from app.agent.main_agent import run_deep_agent

    staging_dir = APP_ROOT / "updated" / f"session_{run.thread_id}"
    if run.requires_file:
        staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture_path, staging_dir / fixture_path.name)

    try:
        agent_result = await run_deep_agent(
            run.query,
            run.thread_id,
            run.trace_id,
            run_metadata={
                "run_id": run.run_id,
                "case_id": run.case_id,
                "evaluation_task_id": run.evaluation_task_id,
                "benchmark_type": "routing_ablation",
                "ablation_pair_id": run.pair_id,
                "route_mode": run.route_mode,
            },
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    summary = agent_result.trace_summary or {}
    result = await _build_run_result(
        run=run.to_baseline_run(),
        summary=summary,
        final_result=agent_result.final_result,
        output_dir=output_dir,
        quality_judge=quality_judge,
    )
    result.update(
        {
            "pair_id": run.pair_id,
            "route_mode": run.route_mode,
            "execution": summary.get("execution") or {},
        }
    )
    result["comparison_metrics"] = extract_comparison_metrics(result)
    return result


def _runner_error_result(
    run: RoutingAblationRun,
    exc: BaseException,
) -> dict[str, Any]:
    """构造运行器异常结果，保证报告仍可生成。"""
    return {
        **asdict(run),
        "status": "runner_error",
        "passed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "performance": {},
        "model": {},
        "search": {},
        "tool_calls": {},
        "assistant_calls": {},
        "agent_governance": {},
        "database": {},
        "artifacts": {},
        "quality": {},
        "quality_judge": None,
        "execution": {},
        "comparison_metrics": {},
    }


def extract_comparison_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    """提取用于成对比较的核心指标。"""
    model = result.get("model") or {}
    performance = result.get("performance") or {}
    quality = result.get("quality") or {}
    tool_calls = result.get("tool_calls") or {}
    assistant_calls = result.get("assistant_calls") or {}
    return {
        "model_calls": _optional_number(model.get("call_count")),
        "tool_calls": _sum_counter(tool_calls),
        "assistant_calls": _sum_counter(assistant_calls),
        "total_tokens": _optional_number(model.get("total_tokens")),
        "total_duration_ms": _optional_number(
            performance.get("total_duration_ms")
        ),
        "model_wall_duration_ms": _optional_number(
            performance.get("model_wall_duration_ms")
        ),
        "tool_wall_duration_ms": _optional_number(
            performance.get("tool_wall_duration_ms")
        ),
        "rule_quality_score": _optional_number(quality.get("rule_score")),
    }


def _sum_counter(value: Any) -> float:
    """汇总字典计数，缺失时返回 0。"""
    if not isinstance(value, dict):
        return 0.0
    total = 0.0
    for item in value.values():
        number = _optional_number(item)
        if number is not None:
            total += number
    return total


def _optional_number(value: Any) -> float | None:
    """把 int/float 转成 float，None 或非法值保持缺失。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """构造模式聚合、任务聚合和配对对比。"""
    comparisons = _compare_pairs(results)
    return {
        "run_count": len(results),
        "paired_count": len(comparisons),
        "by_mode": _aggregate_by_mode(results),
        "comparison": _aggregate_comparisons(comparisons),
        "pairs": comparisons,
    }


def _aggregate_by_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按任务和路由模式聚合基础指标。"""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for result in results:
        case_id = str(result.get("case_id") or "unknown")
        route_mode = str(result.get("route_mode") or "unknown")
        grouped.setdefault(case_id, {}).setdefault(route_mode, []).append(result)

    summary: dict[str, Any] = {}
    for case_id, mode_items in grouped.items():
        summary[case_id] = {}
        for route_mode, items in mode_items.items():
            summary[case_id][route_mode] = {
                "run_count": len(items),
                "success_rate": round(
                    sum(1 for item in items if item.get("passed")) / len(items),
                    4,
                ),
                "execution_routes": _count_values(
                    (item.get("execution") or {}).get("route") for item in items
                ),
                "metrics": _aggregate_metric_values(items),
            }
    return summary


def _count_values(values) -> dict[str, int]:
    """统计字符串值出现次数。"""
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _aggregate_metric_values(items: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合一组运行的比较指标。"""
    return {
        metric: _stats(
            [
                (item.get("comparison_metrics") or {}).get(metric)
                for item in items
            ]
        )
        for metric in (*COMPARISON_METRICS, "rule_quality_score")
    }


def _compare_pairs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 pair_id 成对比较 baseline 和 optimized。"""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        pair_id = result.get("pair_id")
        route_mode = result.get("route_mode")
        if not pair_id or not route_mode:
            continue
        grouped.setdefault(str(pair_id), {})[str(route_mode)] = result

    comparisons: list[dict[str, Any]] = []
    for pair_id, by_mode in sorted(grouped.items()):
        baseline = by_mode.get(ROUTE_MODE_BASELINE_MAIN_AGENT)
        optimized = by_mode.get(ROUTE_MODE_OPTIMIZED)
        if baseline is None or optimized is None:
            continue
        baseline_metrics = baseline.get("comparison_metrics") or {}
        optimized_metrics = optimized.get("comparison_metrics") or {}
        comparisons.append(
            {
                "pair_id": pair_id,
                "case_id": baseline.get("case_id"),
                "case_name": baseline.get("case_name"),
                "iteration": baseline.get("iteration"),
                "baseline_trace_id": baseline.get("trace_id"),
                "optimized_trace_id": optimized.get("trace_id"),
                "baseline_route": (baseline.get("execution") or {}).get("route"),
                "optimized_route": (optimized.get("execution") or {}).get("route"),
                "baseline_passed": bool(baseline.get("passed")),
                "optimized_passed": bool(optimized.get("passed")),
                "metrics": {
                    metric: _metric_delta(
                        baseline_metrics.get(metric),
                        optimized_metrics.get(metric),
                    )
                    for metric in COMPARISON_METRICS
                },
                "rule_quality_score": _quality_delta(
                    baseline_metrics.get("rule_quality_score"),
                    optimized_metrics.get("rule_quality_score"),
                ),
            }
        )
    return comparisons


def _metric_delta(
    baseline_value: Any,
    optimized_value: Any,
) -> dict[str, float | None]:
    """计算优化模式相对 baseline 的节省量和降幅。"""
    baseline_number = _optional_number(baseline_value)
    optimized_number = _optional_number(optimized_value)
    if baseline_number is None or optimized_number is None:
        return {
            "baseline": baseline_number,
            "optimized": optimized_number,
            "saved": None,
            "reduction_rate": None,
        }
    saved = round(baseline_number - optimized_number, 4)
    reduction_rate = (
        round(saved / baseline_number, 4)
        if baseline_number > 0
        else None
    )
    return {
        "baseline": baseline_number,
        "optimized": optimized_number,
        "saved": saved,
        "reduction_rate": reduction_rate,
    }


def _quality_delta(
    baseline_value: Any,
    optimized_value: Any,
) -> dict[str, float | None]:
    """质量分使用 optimized - baseline，正数表示优化模式更好。"""
    baseline_number = _optional_number(baseline_value)
    optimized_number = _optional_number(optimized_value)
    if baseline_number is None or optimized_number is None:
        return {
            "baseline": baseline_number,
            "optimized": optimized_number,
            "delta": None,
        }
    return {
        "baseline": baseline_number,
        "optimized": optimized_number,
        "delta": round(optimized_number - baseline_number, 4),
    }


def _aggregate_comparisons(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """按任务和 overall 聚合配对差异。"""
    grouped: dict[str, list[dict[str, Any]]] = {"overall": comparisons}
    for comparison in comparisons:
        case_id = str(comparison.get("case_id") or "unknown")
        grouped.setdefault(case_id, []).append(comparison)

    return {
        case_id: _aggregate_comparison_group(items)
        for case_id, items in grouped.items()
    }


def _aggregate_comparison_group(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """聚合一个任务组的成对对比结果。"""
    metric_summary = {}
    for metric in COMPARISON_METRICS:
        deltas = [
            comparison["metrics"][metric]
            for comparison in comparisons
            if metric in comparison.get("metrics", {})
        ]
        metric_summary[metric] = {
            "baseline": _stats(delta.get("baseline") for delta in deltas),
            "optimized": _stats(delta.get("optimized") for delta in deltas),
            "saved": _stats(delta.get("saved") for delta in deltas),
            "reduction_rate": _stats(
                delta.get("reduction_rate") for delta in deltas
            ),
        }

    quality_deltas = [
        comparison.get("rule_quality_score", {}).get("delta")
        for comparison in comparisons
    ]
    return {
        "paired_count": len(comparisons),
        "both_passed_count": sum(
            1
            for item in comparisons
            if item.get("baseline_passed") and item.get("optimized_passed")
        ),
        "metrics": metric_summary,
        "rule_quality_delta": _stats(quality_deltas),
    }


def _stats(values) -> dict[str, float] | None:
    """计算均值、中位数、最小值和最大值，自动跳过缺失值。"""
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        return None
    return {
        "average": round(statistics.fmean(numeric), 4),
        "median": round(statistics.median(numeric), 4),
        "min": round(min(numeric), 4),
        "max": round(max(numeric), 4),
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    """写入带时间戳和 latest 快捷文件的消融报告。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_content = json.dumps(report, ensure_ascii=False, indent=2)
    markdown_content = render_markdown(report)
    (output_dir / f"routing_ablation_{stamp}.json").write_text(
        json_content,
        encoding="utf-8",
    )
    (output_dir / f"routing_ablation_{stamp}.md").write_text(
        markdown_content,
        encoding="utf-8",
    )
    (output_dir / "latest.json").write_text(json_content, encoding="utf-8")
    (output_dir / "latest.md").write_text(markdown_content, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    """渲染面向面试复盘的 Markdown 消融报告。"""
    comparison = report["summary"]["comparison"]
    lines = [
        "# DeepSearch 路由消融评测",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 运行清单：{report['manifest']}",
        f"- 总运行数：{report['summary']['run_count']}",
        f"- 有效配对数：{report['summary']['paired_count']}",
        "",
        "## 核心结论",
        "",
        "| 任务 | 配对数 | 模型调用降幅 | 工具调用降幅 | Token 降幅 | 总耗时降幅 | 质量变化 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case_id, item in comparison.items():
        metrics = item.get("metrics") or {}
        lines.append(
            f"| {case_id} | {item['paired_count']} | "
            f"{_format_rate(metrics.get('model_calls'))} | "
            f"{_format_rate(metrics.get('tool_calls'))} | "
            f"{_format_rate(metrics.get('total_tokens'))} | "
            f"{_format_rate(metrics.get('total_duration_ms'))} | "
            f"{_format_delta(item.get('rule_quality_delta'))} |"
        )

    lines.extend(
        [
            "",
            "## 运行模式",
            "",
            "- `baseline_main_agent`：所有非记忆任务强制进入主 Agent，由主 Agent 再调度专家。",
            "- `optimized_router`：纯网络、数据库、文件任务优先直达专业 Agent，组合和交付任务仍进入主 Agent。",
            "",
            "## 配对明细",
            "",
            "| 任务 | Baseline 路由 | Optimized 路由 | 模型调用 | 工具调用 | 总耗时 | 是否双通过 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pair in report["summary"]["pairs"]:
        metrics = pair.get("metrics") or {}
        lines.append(
            f"| {pair.get('case_id')} | {pair.get('baseline_route') or '-'} | "
            f"{pair.get('optimized_route') or '-'} | "
            f"{_format_pair_metric(metrics.get('model_calls'))} | "
            f"{_format_pair_metric(metrics.get('tool_calls'))} | "
            f"{_format_pair_metric(metrics.get('total_duration_ms'), scale=1000, suffix='s')} | "
            f"{'是' if pair.get('baseline_passed') and pair.get('optimized_passed') else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 降幅计算公式：`(baseline - optimized) / baseline`，正数表示直达路由更省。",
            "- 多能力组合和 Markdown/PDF 交付任务通常仍走主 Agent，降幅可能接近 0，这是预期结果。",
            "- Token 取决于模型供应商是否返回 usage；未返回时报告中显示为 `-`。",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_rate(metric: dict[str, Any] | None) -> str:
    """格式化聚合降幅。"""
    if not metric:
        return "-"
    rate = metric.get("reduction_rate")
    if not rate:
        return "-"
    average = rate.get("average")
    return "-" if average is None else f"{average * 100:.2f}%"


def _format_delta(metric: dict[str, Any] | None) -> str:
    """格式化质量变化。"""
    if not metric or metric.get("average") is None:
        return "-"
    return f"{metric['average']:+.4f}"


def _format_pair_metric(
    metric: dict[str, Any] | None,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    """格式化单个配对指标。"""
    if not metric:
        return "-"
    baseline = metric.get("baseline")
    optimized = metric.get("optimized")
    saved = metric.get("saved")
    rate = metric.get("reduction_rate")
    if baseline is None or optimized is None or saved is None:
        return "-"
    rate_text = "-" if rate is None else f"{rate * 100:.2f}%"
    return (
        f"{baseline / scale:.2f}{suffix} -> "
        f"{optimized / scale:.2f}{suffix}，省 {saved / scale:.2f}{suffix}"
        f" ({rate_text})"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="DeepSearch 路由消融评测")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(BASELINE_CASES),
        default=None,
        help="只运行指定任务；不传时使用全部基线任务",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="统一覆盖所选任务的重复次数",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(DEFAULT_ROUTE_MODES),
        default=list(DEFAULT_ROUTE_MODES),
        help="运行的路由模式；默认 baseline 与 optimized 都跑",
    )
    parser.add_argument(
        "--quality-judge",
        action="store_true",
        help="对复杂案例首轮结果执行抽样模型裁判",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印运行清单，不执行真实 API",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只执行依赖预检，不运行任务",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    """消融脚本异步入口。"""
    runs = build_run_plan(
        selected_cases=args.tasks,
        repeat_override=args.repeat,
        route_modes=tuple(args.modes),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_count": len(runs),
                    "runs": [asdict(run) for run in runs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    selected_cases = list(dict.fromkeys(run.case_id for run in runs))
    if args.preflight_only:
        checks = await run_preflight(selected_cases, args.fixture)
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0

    report = await execute_ablation(
        runs,
        args.output_dir,
        args.fixture,
        quality_judge=args.quality_judge,
    )
    print(f"路由消融完成，共执行 {len(report['results'])} 次。")
    print(f"有效配对：{report['summary']['paired_count']} 组。")
    print(f"报告目录：{args.output_dir}")
    return 0


def main() -> int:
    """同步命令行入口。"""
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except PreflightError as exc:
        print(json.dumps(exc.checks, ensure_ascii=False, indent=2))
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
