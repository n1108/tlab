# RAG-Agent

**自包含**的 **工具调用式根因分析（RCA）** 流程：由大模型决定何时运行内置的 **指标 / 链路 / 日志** 检测器（代码从 `exp/` 拷贝并放在本目录内），最后调用 `submit_root_cause_analysis` 提交结论。若工具循环未正常结束（API 限制、不支持 tools、达到最大轮数等），会 **回退** 为与 `exp/main.py` 相同的单次 `JudgeAgent.analyze` 路径。

## 目录结构

```
RAG-Agent/
  pyproject.toml
  requirements.txt
  README.md
  运行说明.md          # 更详细的中文运行步骤
  rag_agent/
    __init__.py
    __main__.py          # 命令行入口
    orchestrator.py      # 多轮 OpenAI tools 循环 + 回退
    tool_runner.py       # 封装 MetricAgent / TraceAgent / LogAgent
    prompts.py           # HWLYYZC_SYSTEM_PROMPT + 工具协议
    bundled/             # 自 tlab/exp 拷贝，import 仅限 rag_agent.bundled
      utils/
      prompt/
      agent/
      template/drain3_log.ini
```

**Python 代码不会 import `RAG-Agent` 以外的路径。** **数据**（parquet、`input.json`）通过你传入的路径读取（`--dataset-root`），一般为上级仓库中的 `dataset/` 目录。

## 安装

```bash
cd RAG-Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 或: pip install -e .
```

配置大模型凭证（默认走 Yuzo）：

- `YUZO_API_KEY` / `YUZO_API_URL`，或使用 `--llm-api-key`、`--llm-api-url`。

## 运行

在 **tlab** 仓库根目录（`RAG-Agent` 的上一级）执行时，默认 `--dataset-root` 会解析为 **`tlab/dataset`**：

```bash
PYTHONPATH=RAG-Agent python -m rag_agent --limit 1 --max-workers 4
```

在 **RAG-Agent** 目录内执行：

```bash
PYTHONPATH=. python -m rag_agent --dataset-root ../dataset --limit 1 --uuid <UUID> --llm-provider yuzo
```

输出：

- `RAG-Agent/output/rag_agent_run.jsonl`（可用 `--output` 指定其它路径）
- `RAG-Agent/output/rag_agent.log`

每条结果的 JSON 中含 `rag_meta`：`mode` 为 `rag_tools`（走通工具提交）或 `fallback_judge_analyze`（回退单次 Judge）。

## 数据目录要求

检测器要求与 `exp` 一致的目录结构：在每个日期目录下包含 `metric-parquet/`、`trace-parquet/`、`log-parquet/`，供 `MetricAgent` / `TraceAgent` / `LogAgent` 读取。

## 说明

- 工具调用需要支持 **`tools` 参数** 的 **OpenAI 兼容** API。若在 `--max-turns` 内模型始终未调用 `submit_root_cause_analysis`，编排器会使用 **回退** 的单次分析。
- 即使只做最小冒烟测试，只要会触发检测器，该时间窗内仍需存在有效的 parquet 数据路径。
- 提速默认项：
  - case 级并行：`--max-workers`（默认 `4`）
  - 单 case 内 metric/trace/log 预取并行：环境变量 `RAG_PREFETCH_TOOLS=1`（默认开启）

更多中文步骤见 **`运行说明.md`**。
