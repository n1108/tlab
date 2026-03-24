# metric 异常检测 baseline 运行脚本
# 输入：dataset 目录为 metric 原始数据，unit-test/metric/data/metric_dataset.json 读取故障 uuid 和持续时间段
# 运行测试后，在 unit-test/metric/results 目录下生成结果文件 result_baseline_[method].csv
# 结果文件为所有故障时间段（uuid）内检测到的所有指标异常（组件+指标）
# 文件格式为 uuid, component(node/service/pod), metric

# baseline-1: IF+HBOS+IQR, exp/agent/metric.py
# baseline-2: BOCPD（待写）


import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE1_DIR = PROJECT_ROOT / "unit-test/metric/baselines/baseline-1"
BASELINE1_FILE = BASELINE1_DIR / "metric.py"
if not BASELINE1_FILE.exists():
    raise FileNotFoundError(f"baseline-1 metric file not found: {BASELINE1_FILE}")
if str(BASELINE1_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE1_DIR))

from metric import MetricAgent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Using MetricAgent from %s", BASELINE1_FILE)


def _parse_iso_utc(time_str: str) -> datetime:
    if not time_str:
        raise ValueError("empty time string")
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt


def _load_test_cases() -> list:
    dataset_path = PROJECT_ROOT / "unit-test/metric/data/metric_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"test dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid test dataset format: {dataset_path}")

    return data


def _run_baseline_1(metric_agent: MetricAgent, start_time: datetime, end_time: datetime) -> list:
    analysis = metric_agent.query_metrics(start_time, end_time)
    events = analysis.get("events", []) if isinstance(analysis, dict) else []

    rows = []
    for event in events:
        component = event.get("pod")
        metric = event.get("kpi")
        if component and metric:
            rows.append((str(component), str(metric)))
    return rows

def run_tests(limit=None, method="1", uuid=None):
    dataset_root = PROJECT_ROOT / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    test_cases = _load_test_cases()

    if uuid:
        test_cases = [case for case in test_cases if str(case.get("uuid", "")) == uuid]
        if not test_cases:
            logger.warning("UUID %s not found in test dataset", uuid)
            return

    if limit is not None:
        test_cases = test_cases[:limit]

    if method == "2":
        raise NotImplementedError("baseline-2 (BOCPD) is not implemented yet")

    if method != "1":
        raise ValueError(f"Unsupported method: {method}")

    metric_agent = MetricAgent(str(dataset_root))
    records = []
    uuid_order = {
        str(case.get("uuid", "")).strip(): idx
        for idx, case in enumerate(test_cases)
        if str(case.get("uuid", "")).strip()
    }

    for case in tqdm(test_cases, desc=f"Running baseline-{method}"):
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
    output_file = output_dir / f"result_baseline_{method}.csv"
    result_df.to_csv(output_file, index=False)

    logger.info("Saved %d anomaly rows to %s", len(result_df), output_file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--method", type=str, default="1", choices=["1", "2"], help="Anomaly detection method to use")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()
    
    run_tests(limit=args.limit, method=args.method, uuid=args.uuid)

# python3 unit-test/metric/run_baseline.py --limit=5 --method=1