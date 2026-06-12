# Repository Guidelines

## 项目结构与模块组织

后端位于 `app/`：`agent/` 管理主/子智能体与治理中间件，`api/` 提供 FastAPI 和 WebSocket，`tools/` 封装搜索、数据库及文件工具，`observability/` 与 `evaluation/` 负责追踪和评测。React 前端位于 `frontend/src/`，测试位于 `tests/`，MySQL 容器与初始化 SQL 位于 `docker/`。`app/logs/`、`app/output/`、`app/updated/` 和 `agent_workspace/` 是运行时目录，不应提交。

## 构建、测试与开发命令

- `uv sync`：安装 Python 3.12 后端依赖。
- `docker compose --env-file .env -f docker/docker-compose.yaml up -d`：启动本地 MySQL（宿主机默认端口 `3307`）。
- `uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload`：启动后端开发服务。
- `cd frontend && corepack pnpm install && corepack pnpm dev`：安装并启动 Vite 前端。
- `cd frontend && corepack pnpm build`：执行 TypeScript 检查并构建前端。
- `uv run python -m unittest discover -s tests -q`：运行全部单元测试。
- `uv run python -m compileall -q app tests`：检查 Python 语法与导入编译。
- `uv run python app/evaluation/run_offline_evaluation.py`：读取已有日志执行离线评测。

## 编码风格与命名约定

Python 使用 4 空格缩进、类型标注和中文 Docstring；模块、函数、变量采用 `snake_case`，类采用 `PascalCase`。按 `.pre-commit-config.yaml` 使用 Ruff 检查并格式化。React/TypeScript 使用 2 空格缩进；组件采用 `PascalCase.tsx`，Hooks 采用 `useXxx.ts`。保持智能体、工具、治理状态和 API 层的职责边界。

## 测试规范

测试使用标准库 `unittest`；文件命名为 `test_<主题>.py`，方法以 `test_` 开头。外部模型、Tavily、MySQL 和文件 I/O 应使用 mock 或最小夹具。项目未设固定覆盖率阈值，但修改路由、预算、数据库只读限制、引用治理或 trace 汇总时必须增加回归测试。

## 提交与 Pull Request

提交遵循 Conventional Commits，如 `feat: ...`、`test: ...`、`chore: ...`，且每个提交聚焦一个可验证改动。PR 需说明行为变化、验证命令和配置影响；界面改动附截图，治理或性能改动附基线对比，并关联相关 Issue。

## 安全与配置

从 `.env.example` 创建 `.env`，禁止提交密钥、密码、真实业务数据或内网地址。数据库工具必须只读。真实性能基线会调用模型、Tavily 和 MySQL；运行前先执行 `uv run python app/evaluation/run_baseline.py --preflight-only`。
