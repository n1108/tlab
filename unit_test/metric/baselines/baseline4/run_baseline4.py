import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_iso_utc(time_str: str) -> datetime:
    if not time_str:
        raise ValueError("empty time string")
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt


def _load_test_cases() -> list:
    dataset_path = PROJECT_ROOT / "unit_test/metric/data/metric_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"test dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"invalid test dataset format: {dataset_path}")

    return data


def _trim_extremes(values: np.ndarray) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return clean
    if clean.size <= 4:
        return clean
    sorted_vals = np.sort(clean)
    return sorted_vals[2:-2]


def _calc_p50_p99(values: np.ndarray) -> Tuple[float, float] | None:
    trimmed = _trim_extremes(values)
    if trimmed.size == 0:
        return None
    p50 = float(np.percentile(trimmed, 50))
    p99 = float(np.percentile(trimmed, 99))
    return p50, p99


def _symmetric_ratio(a: float, b: float) -> float:
    denom = (abs(a) + abs(b)) / 2.0
    if denom <= 1e-12:
        return 0.0
    return abs(a - b) / denom


def _build_stats(df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    stats_map: Dict[Tuple[str, str], Tuple[float, float]] = {}
    if df.empty:
        return stats_map

    for (component, metric), group in df.groupby(["pod", "kpi_key"]):
        values = group["value"].to_numpy(dtype=float)
        stat = _calc_p50_p99(values)
        if stat is None:
            continue
        stats_map[(str(component), str(metric))] = stat

    return stats_map


def run_tests(limit=None, uuid=None, threshold=0.05):
    sys.path.insert(0, str(PROJECT_ROOT))
    from unit_test.metric.baselines.baseline1.metric import MetricAgent

    dataset_root = PROJECT_ROOT / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_root}")

    test_cases = _load_test_cases()
    if uuid:
        test_cases = [case for case in test_cases if str(case.get("uuid", "")) == uuid]
        if not test_cases:
            logger.warning("UUID %s not found in test dataset", uuid)
            return

    if limit is not None:
        test_cases = test_cases[:limit]

    uuid_order = {
        str(case.get("uuid", "")).strip(): idx
        for idx, case in enumerate(test_cases)
        if str(case.get("uuid", "")).strip()
    }

    metric_agent = MetricAgent(str(dataset_root))
    records = []

    for case in tqdm(test_cases, desc="Running baseline-4"):
        case_uuid = str(case.get("uuid", "")).strip()
        start_str = case.get("start_time")
        end_str = case.get("end_time")

        if not case_uuid or not start_str or not end_str:
            logger.warning("skip invalid test case: %s", case)
            continue

        try:
            fault_start = _parse_iso_utc(str(start_str))
            fault_end = _parse_iso_utc(str(end_str))
        except Exception as exc:
            logger.warning("skip %s due to invalid time range: %s", case_uuid, exc)
            continue

        if fault_end <= fault_start:
            logger.warning("skip %s due to non-positive fault window", case_uuid)
            continue

        duration = fault_end - fault_start
        normal_end = fault_start
        normal_start = normal_end - duration

        try:
            normal_df = metric_agent.load_data(normal_start, normal_end)
            fault_df = metric_agent.load_data(fault_start, fault_end)
        except Exception as exc:
            logger.warning("load_data failed for %s: %s", case_uuid, exc)
            continue

        normal_stats = _build_stats(normal_df)
        fault_stats = _build_stats(fault_df)
        common_keys = set(normal_stats.keys()) & set(fault_stats.keys())

        for component, metric in common_keys:
            normal_p50, normal_p99 = normal_stats[(component, metric)]
            fault_p50, fault_p99 = fault_stats[(component, metric)]

            p50_ratio = _symmetric_ratio(fault_p50, normal_p50)
            p99_ratio = _symmetric_ratio(fault_p99, normal_p99)

            # 双重验证：P50 + P99 均超过阈值才判异常
            if p50_ratio >= threshold and p99_ratio >= threshold:
                records.append(
                    {
                        "uuid": case_uuid,
                        "component": component,
                        "metric": metric,
                    }
                )

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
    output_file = output_dir / "result_baseline_4.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    parser.add_argument("--threshold", type=float, default=0.05, help="Symmetric ratio threshold")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid, threshold=args.threshold)

# python3 unit-test/metric/baselines/baseline-4/run_baseline-4.py --limit=5
