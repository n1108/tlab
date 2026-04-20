"""
Log Baselines Orchestrator - 单一入口文件

功能：
1. 运行多个 baseline（LogAgent, LightAD KNN/DT/SLFN, NeuralLog）
2. 为每个 baseline 生成独立的预计算结果文件（results/log_precompute/*.jsonl）
3. 汇总所有结果成一个统一的 log_summary.txt，供 JudgeAgent 使用
4. 每个 baseline 逻辑封装在独立类中，结构清晰

用法：
    python -m unit_test.log.baselines.orchestrator --dataset-root dataset
    # 或指定 baseline: --baselines log_agent,lightad_knn,neural_log
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.agent.log import LogAgent
from unit_test.log.baselines.loader import load_window_lines
from unit_test.log.input_cases import load_input_json_cases
from unit_test.log.baselines.baselines.lightad import LightADBaseline
from unit_test.log.baselines.baselines.neural_log import NeuralLogBaseline
from unit_test.log.baselines.baselines.log_agent import LogAgentBaseline
from unit_test.log.baselines.run_comparison import (
    _neural_log_has_positive_on_gt_pods,
    load_log_unit_test_ground_truth,
)

LOG_UNIT_TEST_GT_PATH = PROJECT_ROOT / "unit_test" / "log" / "log_unit_test_dataset.json"

logger = logging.getLogger(__name__)

PRECOMPUTE_DIR = PROJECT_ROOT / "results" / "log_precompute"
PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)

BASELINES = {
    "log_agent": LogAgentBaseline,
    "lightad_knn": lambda: LightADBaseline("knn"),
    "lightad_dt": lambda: LightADBaseline("dt"),
    "lightad_slfn": lambda: LightADBaseline("slfn"),
    "neural_log": NeuralLogBaseline,
}


class BaselineOrchestrator:
    """统一管理所有 log baseline 的运行和汇总"""

    def __init__(self, dataset_root: Path, normal_window_minutes: int = 30, max_normal_lines: int = 800, seed: int = 42):
        self.dataset_root = dataset_root
        self.normal_window_minutes = normal_window_minutes
        self.max_normal_lines = max_normal_lines
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cases = load_input_json_cases(PROJECT_ROOT)

    def run_baseline(self, name: str, limit: int | None = None) -> dict[str, Any]:
        """运行单个 baseline 并返回结果"""
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline: {name}")

        baseline_cls = BASELINES[name]
        baseline = baseline_cls() if callable(baseline_cls) else baseline_cls

        results = {}
        cases_to_run = self.cases[:limit] if limit else self.cases
        nl_gt_map: dict[str, dict[str, object]] = {}
        if name == "neural_log":
            nl_gt_map = load_log_unit_test_ground_truth(LOG_UNIT_TEST_GT_PATH)

        for idx, item in enumerate(cases_to_run, 1):
            uuid = str(item.get("uuid", ""))
            if not uuid or not item.get("parse_ok", True):
                continue

            try:
                start = self._parse_time(item["start_time"])
                end = self._parse_time(item["end_time"])
            except Exception:
                continue

            # 加载窗口数据
            normal_start = start - pd.Timedelta(minutes=self.normal_window_minutes)
            normal_end = start
            fault_df = load_window_lines(str(self.dataset_root), start, end)
            normal_df = load_window_lines(str(self.dataset_root), normal_start, normal_end)

            normal_texts = normal_df["text_line"].astype(str).tolist() if not normal_df.empty else []
            fault_texts = fault_df["text_line"].astype(str).tolist() if not fault_df.empty else []

            if len(normal_texts) > self.max_normal_lines:
                normal_texts = self.rng.sample(normal_texts, self.max_normal_lines)

            result = baseline.score(fault_texts, normal_texts)
            row: dict[str, Any] = {
                "idx": idx,
                "uuid": uuid,
                "text": result.get("text", "- no anomaly"),
                "count": result.get("count", 0),
            }
            if name == "neural_log" and nl_gt_map:
                gt_entry = nl_gt_map.get(uuid)
                comps = set(gt_entry.get("components") or []) if isinstance(gt_entry, dict) else set()
                has_flag = False
                if comps and not fault_df.empty:
                    pos_nl = np.zeros(len(fault_df), dtype=bool)
                    try:
                        impl = getattr(baseline, "_impl", baseline)
                        impl.fit(normal_texts)
                        pos_nl = impl.predict(fault_texts).astype(bool)
                    except Exception:
                        pass
                    has_flag = bool(_neural_log_has_positive_on_gt_pods(fault_df, pos_nl, comps))
                row["has_if_positive_on_gt_pod"] = has_flag
            results[uuid] = row

            if idx % 20 == 0:
                logger.info(f"[{name}] processed {idx}/{len(cases_to_run)}")

        # 保存预计算结果
        out_path = PRECOMPUTE_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in results.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Saved {name} results to {out_path}")
        return results

    def _parse_time(self, time_str: str):
        from unit_test.log.evaluate_log_agent import _parse_iso_utc
        return _parse_iso_utc(time_str)

    def run_all(self, selected_baselines: list[str] | None = None, limit: int | None = None) -> None:
        """运行所有（或指定）baseline 并生成最终 summary"""
        if not selected_baselines:
            selected_baselines = list(BASELINES.keys())

        logger.info(f"Running baselines: {selected_baselines}")

        all_results = {}
        for name in selected_baselines:
            all_results[name] = self.run_baseline(name, limit)

        self._generate_summary(all_results)
        logger.info("All baselines completed. Summary written to results/log_summary.txt")

    def _generate_summary(self, all_results: dict) -> None:
        """将所有 baseline 结果汇总成一个统一的 log_summary.txt"""
        summary_path = PROJECT_ROOT / "results" / "log_summary.txt"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with summary_path.open("w", encoding="utf-8") as f:
            f.write("# Log Summary - Aggregated from Multiple Baselines\n")
            f.write("# Generated by baselines/orchestrator.py\n\n")

            for uuid, results in all_results.get("log_agent", {}).items():
                f.write(f"# UUID: {uuid}\n")
                f.write("[LOG_PRECOMPUTED_NOTE]\n")
                f.write("- Aggregated from multiple baselines (LogAgent, LightAD, NeuralLog)\n\n")

                f.write("[LOG_AGENT]\n")
                log_text = results.get("text", "- no anomaly")
                f.write(log_text + "\n\n")

                for baseline_name in ["lightad_knn", "lightad_dt", "lightad_slfn", "neural_log"]:
                    if baseline_name in all_results and uuid in all_results[baseline_name]:
                        name_display = baseline_name.replace("lightad_", "LightAD-").upper()
                        f.write(f"[{name_display}_BASELINE]\n")
                        baseline_text = all_results[baseline_name][uuid].get("text", "- no anomaly")
                        f.write(baseline_text + "\n\n")

                f.write("[LOG_SUMMARY_FOR_JUDGE]\n")
                f.write("This section provides multimodal log evidence for root cause analysis.\n")
                f.write("Focus on components with repeated errors, connection issues, or abnormal patterns.\n")
                f.write("Cross-reference with metrics and traces for final judgment.\n\n")
                f.write("=" * 80 + "\n\n")

        logger.info(f"Final summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Baselines Orchestrator")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--baselines", type=str, default="all",
                       help="Comma-separated list of baselines or 'all'")
    parser.add_argument("--limit-uuids", type=int, default=0, help="Limit number of cases (0 = all)")
    parser.add_argument("--normal-window-minutes", type=int, default=30)
    parser.add_argument("--max-normal-lines", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce verbose 'Searching records' messages (recommended for large runs)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Reduce noise from parquet search (very verbose when many time windows)
    log_level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    for noisy_logger in ("exp.utils.input", "exp.agent.log", "exp.agent.trace"):
        logging.getLogger(noisy_logger).setLevel(log_level)

    # Reduce noise from parquet search (very verbose when many time windows)
    for noisy_logger in ("exp.utils.input", "exp.agent.log", "exp.agent.trace"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    orchestrator = BaselineOrchestrator(
        dataset_root=args.dataset_root,
        normal_window_minutes=args.normal_window_minutes,
        max_normal_lines=args.max_normal_lines,
        seed=args.seed,
    )

    selected = args.baselines.split(",") if args.baselines != "all" else None
    orchestrator.run_all(selected_baselines=selected, limit=args.limit_uuids or None)


if __name__ == "__main__":
    main()
