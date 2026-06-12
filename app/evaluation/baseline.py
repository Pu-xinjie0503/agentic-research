"""
DeepSearch 多能力性能基线运行器。

真实执行会调用 DeepSeek、Tavily 和 MySQL。默认先完成全部依赖预检，
再按运行清单逐项执行，避免依赖缺失时产生无意义的部分基线。
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


CURRENT_FILE = Path(__file__).resolve()
APP_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_OUTPUT_DIR = APP_ROOT / "logs" / "baselines"
DEFAULT_FIXTURE = CURRENT_FILE.parent / "fixtures" / "industry_brief.md"


@dataclass(frozen=True)
class BaselineCase:
    """一类性能基线任务。"""

    case_id: str
    name: str
    query: str
    default_repeats: int
    requires_web: bool = False
    requires_database: bool = False
    requires_file: bool = False
    expected_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineRun:
    """一次独立基线运行。"""

    run_id: str
    case_id: str
    case_name: str
    iteration: int
    query: str
    thread_id: str
    trace_id: str
    requires_file: bool
    expected_artifacts: tuple[str, ...]


BASELINE_CASES: dict[str, BaselineCase] = {
    "network_search": BaselineCase(
        case_id="network_search",
        name="网络公开趋势查询",
        query="请检索 2026 年跨境电商 AI 客服趋势，列出 5 条关键变化并附来源链接，不生成文件。",
        default_repeats=3,
        requires_web=True,
    ),
    "database_query": BaselineCase(
        case_id="database_query",
        name="数据库低库存查询",
        query="请查询当前数据库中库存数量最低的 5 个药品，按库存从低到高输出表格，不调用网络搜索，不生成文件。",
        default_repeats=3,
        requires_database=True,
    ),
    "file_analysis": BaselineCase(
        case_id="file_analysis",
        name="上传文件分析",
        query="请读取我上传的药品经营简报，提取核心观点、需要数据库核验的数据、风险点和信息缺口，不调用网络搜索，不生成文件。",
        default_repeats=3,
        requires_file=True,
    ),
    "multi_agent": BaselineCase(
        case_id="multi_agent",
        name="文件数据库网络协作",
        query="请先分析我上传的药品经营简报，再查询数据库核验相关药品的库存和销售情况，最后搜索公开资料补充 2026 年市场趋势，汇总结论，不生成文件。",
        default_repeats=1,
        requires_web=True,
        requires_database=True,
        requires_file=True,
    ),
    "markdown_delivery": BaselineCase(
        case_id="markdown_delivery",
        name="数据库网络 Markdown 交付",
        query="请查询数据库中库存数量最低的 5 个药品，再搜索相关治疗领域的 2026 年市场趋势，最后生成一份 Markdown 报告。",
        default_repeats=1,
        requires_web=True,
        requires_database=True,
        expected_artifacts=("markdown",),
    ),
    "pdf_delivery": BaselineCase(
        case_id="pdf_delivery",
        name="文件网络 PDF 交付",
        query="请分析我上传的药品经营简报，再用网络搜索补充 2026 年最新趋势，最后生成一份 PDF 报告。",
        default_repeats=1,
        requires_web=True,
        requires_file=True,
        expected_artifacts=("markdown", "pdf"),
    ),
}


class PreflightError(RuntimeError):
    """基线依赖预检失败。"""

    def __init__(self, checks: list[dict[str, Any]]) -> None:
        self.checks = checks
        failed = [check["name"] for check in checks if not check["passed"]]
        super().__init__(f"基线预检失败：{', '.join(failed)}")


def build_run_plan(
    selected_cases: list[str] | None = None,
    repeat_override: int | None = None,
) -> list[BaselineRun]:
    """构造带独立 thread_id 和 trace_id 的运行清单。"""
    case_ids = selected_cases or list(BASELINE_CASES)
    unknown = [case_id for case_id in case_ids if case_id not in BASELINE_CASES]
    if unknown:
        raise ValueError(f"未知基线任务：{', '.join(unknown)}")
    if repeat_override is not None and repeat_override < 1:
        raise ValueError("repeat 必须大于等于 1")

    runs: list[BaselineRun] = []
    for case_id in case_ids:
        case = BASELINE_CASES[case_id]
        repeats = repeat_override or case.default_repeats
        for iteration in range(1, repeats + 1):
            suffix = uuid.uuid4().hex[:12]
            runs.append(
                BaselineRun(
                    run_id=f"{case_id}-{iteration}-{suffix}",
                    case_id=case_id,
                    case_name=case.name,
                    iteration=iteration,
                    query=case.query,
                    thread_id=f"baseline-{case_id}-{suffix}",
                    trace_id=str(uuid.uuid4()),
                    requires_file=case.requires_file,
                    expected_artifacts=case.expected_artifacts,
                )
            )
    return runs


async def run_preflight(
    selected_cases: list[str],
    fixture_path: Path = DEFAULT_FIXTURE,
) -> list[dict[str, Any]]:
    """验证所选任务真正需要的模型、搜索、数据库和附件依赖。"""
    cases = [BASELINE_CASES[case_id] for case_id in selected_cases]
    checks: list[dict[str, Any]] = []

    if any(case.requires_file for case in cases):
        checks.append(
            {
                "name": "测试附件",
                "passed": fixture_path.is_file(),
                "detail": str(fixture_path),
            }
        )

    try:
        from langchain_core.messages import HumanMessage

        from app.agent.llm import model

        response = await model.ainvoke([HumanMessage(content="仅回复 OK")])
        checks.append(
            {
                "name": "DeepSeek",
                "passed": bool(getattr(response, "content", None)),
                "detail": "模型最小请求成功",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "DeepSeek",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    if any(case.requires_web for case in cases):
        try:
            from app.tools.tavily_tool import tavily_client

            result = await asyncio.to_thread(
                tavily_client.search,
                query="DeepSearch baseline connectivity check",
                max_results=1,
                include_raw_content=False,
            )
            checks.append(
                {
                    "name": "Tavily",
                    "passed": isinstance(result, dict),
                    "detail": "搜索最小请求成功",
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "Tavily",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    if any(case.requires_database for case in cases):
        try:
            await asyncio.to_thread(_check_database)
            checks.append(
                {
                    "name": "MySQL",
                    "passed": True,
                    "detail": "SELECT 1 成功",
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "MySQL",
                    "passed": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    if any(not check["passed"] for check in checks):
        raise PreflightError(checks)
    return checks


async def execute_baseline(
    runs: list[BaselineRun],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    """预检通过后执行运行清单并生成 JSON/Markdown 报告。"""
    selected_cases = list(dict.fromkeys(run.case_id for run in runs))
    checks = await run_preflight(selected_cases, fixture_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat()
    manifest_path = output_dir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "started_at": started_at,
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
            results.append(await _execute_run(run, fixture_path))
        except Exception as exc:
            results.append(
                {
                    **asdict(run),
                    "status": "runner_error",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "performance": {},
                    "model": {},
                    "search": {},
                    "tool_calls": {},
                    "assistant_calls": {},
                    "artifacts": {},
                }
            )

    report = {
        "generated_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "preflight": checks,
        "summary": _aggregate_results(results),
        "results": results,
    }
    _write_baseline_report(report, output_dir)
    return report


async def _execute_run(
    run: BaselineRun,
    fixture_path: Path,
) -> dict[str, Any]:
    """执行单条任务并提取性能指标。"""
    from app.agent.main_agent import run_deep_agent

    staging_dir = APP_ROOT / "updated" / f"session_{run.thread_id}"
    if run.requires_file:
        staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture_path, staging_dir / fixture_path.name)

    try:
        summary = await run_deep_agent(
            run.query,
            run.thread_id,
            run.trace_id,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    summary = summary or {}
    output_path = APP_ROOT / "output" / f"session_{run.thread_id}"
    artifact_status = _artifact_status(output_path, run.expected_artifacts)
    return {
        **asdict(run),
        "status": summary.get("status", "missing_summary"),
        "passed": (
            summary.get("status") == "success"
            and all(artifact_status.values())
        ),
        "performance": summary.get("performance") or {},
        "model": summary.get("model") or {},
        "search": summary.get("search") or {},
        "tool_calls": summary.get("tool_calls") or {},
        "assistant_calls": summary.get("assistant_calls") or {},
        "artifacts": artifact_status,
    }


def _check_database() -> None:
    """执行 MySQL 最小连通性检查。"""
    from mysql.connector import connect

    from app.tools.db_tools import get_db_config

    with connect(**get_db_config()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()


def _artifact_status(
    output_path: Path,
    expected_artifacts: tuple[str, ...],
) -> dict[str, bool]:
    """检查交付任务是否生成预期扩展名文件。"""
    suffix_map = {"markdown": ".md", "pdf": ".pdf"}
    files = list(output_path.rglob("*")) if output_path.exists() else []
    return {
        artifact: any(
            path.is_file() and path.suffix.lower() == suffix_map[artifact]
            for path in files
        )
        for artifact in expected_artifacts
    }


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """按任务类别聚合成功率和性能指标。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["case_id"], []).append(result)
    return {
        case_id: _aggregate_case(case_results)
        for case_id, case_results in grouped.items()
    }


def _aggregate_case(results: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合一个任务类别的多次运行结果。"""
    total_durations = [
        float(item["performance"].get("total_duration_ms") or 0)
        for item in results
    ]
    model_calls = [
        int(item["model"].get("call_count") or 0)
        for item in results
    ]
    model_durations = [
        float(item["performance"].get("model_wall_duration_ms") or 0)
        for item in results
    ]
    tool_wall_durations = [
        float(item["performance"].get("tool_wall_duration_ms") or 0)
        for item in results
    ]
    total_tokens = [
        item["model"].get("total_tokens")
        for item in results
        if item["model"].get("total_tokens") is not None
    ]
    by_agent: dict[str, list[float]] = {}
    for item in results:
        for agent_name, metrics in (item["model"].get("by_agent") or {}).items():
            by_agent.setdefault(agent_name, []).append(
                float(metrics.get("total_duration_ms") or 0)
            )
    return {
        "run_count": len(results),
        "success_count": sum(1 for item in results if item["passed"]),
        "success_rate": round(
            sum(1 for item in results if item["passed"]) / len(results),
            4,
        ),
        "total_duration_ms": _number_stats(total_durations),
        "model_call_count": _number_stats(model_calls),
        "model_wall_duration_ms": _number_stats(model_durations),
        "tool_wall_duration_ms": _number_stats(tool_wall_durations),
        "total_tokens": _number_stats(total_tokens) if total_tokens else None,
        "search_executed_count": _number_stats(
            [
                int(item["search"].get("executed_count") or 0)
                for item in results
            ]
        ),
        "search_blocked_count": _number_stats(
            [
                int(item["search"].get("blocked_count") or 0)
                for item in results
            ]
        ),
        "agent_duration_ms": {
            agent_name: _number_stats(values)
            for agent_name, values in by_agent.items()
        },
    }


def _number_stats(values: list[float | int]) -> dict[str, float]:
    """计算均值、中位数、最小值和最大值。"""
    numeric = [float(value) for value in values]
    return {
        "average": round(statistics.fmean(numeric), 2),
        "median": round(statistics.median(numeric), 2),
        "min": round(min(numeric), 2),
        "max": round(max(numeric), 2),
    }


def _write_baseline_report(report: dict[str, Any], output_dir: Path) -> None:
    """写入带时间戳和 latest 快捷文件的基线报告。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_content = json.dumps(report, ensure_ascii=False, indent=2)
    markdown_content = _render_markdown(report)
    (output_dir / f"baseline_{stamp}.json").write_text(
        json_content,
        encoding="utf-8",
    )
    (output_dir / f"baseline_{stamp}.md").write_text(
        markdown_content,
        encoding="utf-8",
    )
    (output_dir / "latest.json").write_text(json_content, encoding="utf-8")
    (output_dir / "latest.md").write_text(markdown_content, encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    """渲染便于人工比较的 Markdown 基线报告。"""
    lines = [
        "# DeepSearch 多能力性能基线",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 运行清单：{report['manifest']}",
        "",
        "| 任务 | 成功率 | 总耗时均值 | 模型调用均值 | 模型墙钟均值 | 工具墙钟均值 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case_id, summary in report["summary"].items():
        lines.append(
            f"| {case_id} | {summary['success_rate'] * 100:.2f}% | "
            f"{summary['total_duration_ms']['average'] / 1000:.2f}s | "
            f"{summary['model_call_count']['average']:.2f} | "
            f"{summary['model_wall_duration_ms']['average'] / 1000:.2f}s | "
            f"{summary['tool_wall_duration_ms']['average'] / 1000:.2f}s |"
        )

    lines.extend(["", "## 任务统计", ""])
    for case_id, summary in report["summary"].items():
        total = summary["total_duration_ms"]
        tokens = summary.get("total_tokens")
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- 总耗时：均值 {total['average'] / 1000:.2f}s，"
                f"中位数 {total['median'] / 1000:.2f}s，"
                f"最小 {total['min'] / 1000:.2f}s，最大 {total['max'] / 1000:.2f}s",
                f"- 模型调用：平均 {summary['model_call_count']['average']:.2f} 次",
                f"- 搜索调用：平均执行 {summary['search_executed_count']['average']:.2f} 次，"
                f"平均拦截 {summary['search_blocked_count']['average']:.2f} 次",
                (
                    f"- Token：平均 {tokens['average']:.2f}，"
                    f"中位数 {tokens['median']:.2f}"
                    if tokens
                    else "- Token：供应商未返回"
                ),
                "",
            ]
        )

    lines.extend(["## 分 Agent 模型耗时", ""])
    for case_id, summary in report["summary"].items():
        lines.append(f"### {case_id}")
        agent_metrics = summary.get("agent_duration_ms") or {}
        if not agent_metrics:
            lines.append("- 无模型调用数据")
        for agent_name, metrics in agent_metrics.items():
            lines.append(
                f"- {agent_name}：平均 {metrics['average'] / 1000:.2f}s，"
                f"中位数 {metrics['median'] / 1000:.2f}s"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """解析基线命令行参数。"""
    parser = argparse.ArgumentParser(description="DeepSearch 多能力性能基线")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(BASELINE_CASES),
        default=None,
        help="只运行指定任务；不传时使用全部平衡基线任务",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="统一覆盖所选任务的重复次数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印运行清单，不执行预检或真实 API",
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
    """基线脚本异步入口。"""
    runs = build_run_plan(args.tasks, args.repeat)
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

    report = await execute_baseline(runs, args.output_dir, args.fixture)
    print(f"基线完成，共执行 {len(report['results'])} 次。")
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
