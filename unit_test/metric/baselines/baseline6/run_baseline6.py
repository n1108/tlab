import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pate_utils import convert_vector_to_events_PATE, generate_buffer_points


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE6_DIR = Path(__file__).resolve().parent
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


def _robust_scores(series: pd.Series) -> np.ndarray:
    vals = series.to_numpy(dtype=float)
    if vals.size == 0:
        return np.array([], dtype=float)

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad < 1e-8:
        std = float(np.std(vals))
        if std < 1e-8:
            return np.zeros_like(vals, dtype=float)
        return np.abs((vals - med) / (std + 1e-8))
    return np.abs(0.6745 * (vals - med) / mad)


def _expand_binary_mask(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or binary.size == 0:
        return binary
    idx = np.where(binary > 0)[0]
    if idx.size == 0:
        return binary
    out = np.zeros_like(binary, dtype=np.int8)
    n = binary.size
    for i in idx:
        l = max(0, int(i) - radius)
        r = min(n - 1, int(i) + radius)
        out[l : r + 1] = 1
    return out


def _is_series_anomaly(series: pd.Series) -> bool:
    if len(series) < 8:
        return False
    if float(series.std()) < 1e-10:
        return False

    scores = _robust_scores(series)
    if scores.size == 0:
        return False

    # PATE-like point->range flow: threshold points, then evaluate contiguous events.
    p97 = float(np.percentile(scores, 97))
    threshold = max(2.8, p97)
    binary = (scores >= threshold).astype(np.int8)

    # Use a small proximity buffer (in points) inspired by PATE buffer concept.
    max_buf = max(1, int(0.03 * len(series)))
    buf_points = generate_buffer_points(max_buffer_size=max_buf, num_splits=1, include_zero=False)
    radius = int(buf_points[-1]) if len(buf_points) else 0
    binary = _expand_binary_mask(binary, radius=radius)

    events = convert_vector_to_events_PATE(binary)
    if not events:
        return False

    # Keep meaningful events only: length >= 2 or strong peak z.
    peak = float(scores.max())
    for s, e in events:
        if (e - s + 1) >= 2:
            return True
    return peak >= 4.0


def run_tests(limit=None, uuid=None):
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

    loader_agent = MetricAgent(str(dataset_root))
    records = []

    for case in tqdm(test_cases, desc="Running baseline6 (PATE-inspired)"):
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

            if _is_series_anomaly(series):
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
            result_df.sort_values(["_uuid_order", "component", "metric"])
            .drop(columns=["_uuid_order"])
            .reset_index(drop=True)
        )

    output_dir = PROJECT_ROOT / "unit_test/metric/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "result_baseline6.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid)
