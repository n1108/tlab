import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASELINE2_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE2_DIR))


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


def _load_bocpd_func():
    try:
        from anomaly_detection import bocpd
    except Exception as exc:
        raise ImportError("failed to import bocpd from anomaly_detection.py") from exc

    logger.info("using bocpd from %s", BASELINE2_DIR / "anomaly_detection.py")
    return bocpd


def _run_series_bocpd(bocpd_func, series: pd.Series, metric_name: str) -> bool:
    if len(series) < 5:
        return False

    if series.std() == 0:
        return False

    input_df = pd.DataFrame({
        "time": np.arange(len(series), dtype=np.int64),
        str(metric_name): series.values,
    })

    try:
        anomalies = bocpd_func(input_df)
        return anomalies is not None and len(anomalies) > 0
    except Exception as exc:
        logger.debug("bocpd failed on metric=%s: %s", metric_name, exc)
        return False


def run_tests(limit=None, uuid=None):
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

    bocpd_func = _load_bocpd_func()
    loader_agent = MetricAgent(str(dataset_root))

    records = []
    for case in tqdm(test_cases, desc="Running baseline-2"):
        case_uuid = str(case.get("uuid", "")).strip()
        start_str = case.get("start_time")
        end_str = case.get("end_time")

        if not case_uuid or not start_str or not end_str:
            logger.warning("skip invalid test case: %s", case)
            continue

        try:
            start_time = _parse_iso_utc(str(start_str))
            end_time = _parse_iso_utc(str(end_str))
        except Exception as exc:
            logger.warning("skip %s due to invalid time range: %s", case_uuid, exc)
            continue

        try:
            df = loader_agent.load_data(start_time, end_time)
        except Exception as exc:
            logger.warning("load_data failed for %s: %s", case_uuid, exc)
            continue

        if df.empty:
            continue

        for (component, metric), group in df.groupby(["pod", "kpi_key"]):
            try:
                series = group.set_index("time")["value"].sort_index()
                series = series.resample("1min").max().fillna(0)
            except Exception:
                continue

            if _run_series_bocpd(bocpd_func, series, str(metric)):
                records.append(
                    {
                        "uuid": case_uuid,
                        "component": str(component),
                        "metric": str(metric),
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
    output_file = output_dir / "result_baseline_2.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid)

# python3 unit-test/metric/baselines/baseline-2/run_baseline-2.py --limit=5
