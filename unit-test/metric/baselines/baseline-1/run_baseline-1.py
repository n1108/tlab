# metric 异常检测 baseline-1 运行脚本 (IF+HBOS+IQR)
# 输入：dataset 目录为 metric 原始数据，unit-test/metric/data/metric_dataset.json 读取故障 uuid 和持续时间段
# 运行测试后，在 unit-test/metric/results 目录下生成结果文件 result_baseline_1.csv
# 结果文件为所有故障时间段（uuid）内检测到的所有指标异常（组件 + 指标）
# 文件格式为 uuid, component, metric

import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# 计算项目根目录：当前文件位于 unit-test/metric/baselines/baseline-1/
# 需要向上 4 级到达项目根目录 (/home/tyt21/tlab/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加 unit-test 目录到路径，以便导入 metric 模块
UNIT_TEST_DIR = PROJECT_ROOT / "unit-test"
if str(UNIT_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_TEST_DIR))

from metric import MetricAgent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_iso_utc(time_str: str) -> datetime:
    """解析 ISO 格式时间字符串为 datetime 对象（UTC 时间）"""
    if not time_str:
        raise ValueError("empty time string")
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt


def _load_test_cases() -> list:
    """加载测试用例数据集"""
    dataset_path = PROJECT_ROOT / "unit-test/metric/data/metric_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"test dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid test dataset format: {dataset_path}")

    return data


def _run_baseline_1(metric_agent: MetricAgent, start_time: datetime, end_time: datetime) -> list:
    """运行 baseline-1 异常检测算法"""
    analysis = metric_agent.query_metrics(start_time, end_time)
    events = analysis.get("events", []) if isinstance(analysis, dict) else []

    rows = []
    for event in events:
        component = event.get("pod")
        metric = event.get("kpi")
        if component and metric:
            rows.append((str(component), str(metric)))
    return rows


def run_tests(limit=None, uuid=None):
    """运行 baseline-1 测试"""
    dataset_root = PROJECT_ROOT / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    test_cases = _load_test_cases()

    # 如果指定了 uuid，只运行该 uuid 的测试用例
    if uuid:
        test_cases = [case for case in test_cases if str(case.get("uuid", "")) == uuid]
        if not test_cases:
            logger.warning("UUID %s not found in test dataset", uuid)
            return

    # 如果指定了 limit，只运行前 limit 个测试用例
    if limit is not None:
        test_cases = test_cases[:limit]

    metric_agent = MetricAgent(str(dataset_root))
    records = []
    uuid_order = {
        str(case.get("uuid", "")).strip(): idx
        for idx, case in enumerate(test_cases)
        if str(case.get("uuid", "")).strip()
    }

    for case in tqdm(test_cases, desc="Running baseline-1"):
        case_uuid = str(case.get("uuid", "")).strip()
        start_str = case.get("start_time")
        end_str = case.get("end_time")

        if not case_uuid or not start_str or not end_str:
            logger.warning("Skip invalid test case: %s", case)
            continue

        try:
            start_time = _parse_iso_utc(str(start_str))
            end_time = _parse_iso_utc(str(end_str))
        except Exception as exc:
            logger.warning("Skip %s due to invalid time range: %s", case_uuid, exc)
            continue

        anomalies = _run_baseline_1(metric_agent, start_time, end_time)
        for component, metric in anomalies:
            records.append({"uuid": case_uuid, "component": component, "metric": metric})

    result_df = pd.DataFrame(records, columns=["uuid", "component", "metric"])
    if not result_df.empty:
        result_df = result_df.drop_duplicates()
        result_df["_uuid_order"] = result_df["uuid"].map(uuid_order).fillna(len(uuid_order))
        result_df = (
            result_df
            .sort_values(["_uuid_order", "component", "metric"])
            .drop(columns=["_uuid_order"])
            .reset_index(drop=True)
        )

    output_dir = PROJECT_ROOT / "unit-test/metric/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "result_baseline_1.csv"
    result_df.to_csv(output_file, index=False)

    logger.info("Saved %d anomaly rows to %s", len(result_df), output_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()
    
    run_tests(limit=args.limit, uuid=args.uuid)
