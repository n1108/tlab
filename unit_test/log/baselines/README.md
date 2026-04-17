# Log Baselines - 清理后版本

本目录已**大幅整理**，现在结构清晰：

- **orchestrator.py** 是**唯一主要入口**，负责运行所有 baseline、生成预计算结果文件，并汇总成 `results/log_summary.txt` 供 JudgeAgent 使用。
- 每个 baseline 都有独立实现（放在 `baselines/` 下）。
- `neural_log/` 是完全独立的 NeuralLog 项目复现文件夹。
- 旧的分散预计算脚本（precompute_*.py、generate_log_summary.py、log_summary_common.py）已删除。

## 当前文件结构

```
baselines/
├── orchestrator.py              ← **唯一推荐主入口**（运行 baseline + 生成 summary）
├── loader.py                    ← 共享日志加载工具
├── input_cases.py               ← 从 input.json 加载时间窗（核心共享工具）
├── baselines/
│   ├── __init__.py
│   ├── lightad.py               ← KNN / DT / SLFN
│   ├── neural_log.py
│   └── log_agent.py
├── neural_log/                  ← NeuralLog 原项目代码（独立）
│   ├── baseline.py
│   ├── data_loader.py
│   └── models/
├── run_comparison.py            ← 独立对比报告（可选）
├── evaluate_log_agent.py        ← LogAgent 评估工具（可选）
└── README.md
```

## 使用方式（推荐）

```bash
# 运行所有 baseline 并生成最新 summary（推荐）
python -m unit_test.log.baselines.orchestrator --dataset-root dataset --quiet

# 只跑特定 baseline（更快）
python -m unit_test.log.baselines.orchestrator --baselines neural_log,log_agent --limit-uuids 50 --quiet

# 查看帮助
python -m unit_test.log.baselines.orchestrator --help
```

生成的 `results/log_summary.txt` 会包含 `[LOG_AGENT]`、`[LIGHTAD_KNN_BASELINE]`、`[NEURAL_LOG_BASELINE]` 等段落，供 `exp/agent/judge.py` 使用。

---

**当前状态**：
- `input_cases.py` **核心保留** — orchestrator 和 score 脚本的共享依赖。
- `evaluate_log_agent.py` **已重构** — 现在以 `score()` 函数为主（类似 `unit_test/metric/score.py`），专注 per-uuid anomaly/component 统计，输出 `log_score.json`。
- `run_comparison.py` **标记为可选/Deprecated** — 功能与 orchestrator 大量重合。如果不再需要手动对比报告，可以删除。
- `generate_log_test_data.py` **已删除**。

**推荐使用**：
```bash
python -m unit_test.log.baselines.orchestrator --dataset-root dataset
```

如果你想**彻底清理**评估脚本（run_comparison.py / evaluate_log_agent.py），告诉我，我可以删除它们并更新依赖。
