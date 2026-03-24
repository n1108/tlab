import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    STL = None
    HAS_STL = False

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class MethodResult:
    stat_test: bool
    threshold: bool
    zvar: bool
    stl: bool
    weighted_score: int


class HybridMetricDetector:
    """
    baseline-3: 4-method fusion
    1) Mann-Whitney U / Welch t-test
    2) Mean ± k*std threshold exceedance
    3) Standardized bias + variance ratio
    4) STL residual anomaly
    """

    def __init__(self):
        self.k_sigma = 3.0
        self.p_value_thresh = 0.01
        self.cohen_d_thresh = 0.8
        self.z_bias_thresh = 2.5
        self.var_ratio_thresh = 2.0
        self.tail_ratio_thresh = 0.10

        if not HAS_STL:
            warnings.warn(
                "statsmodels is not installed. STL method is disabled in baseline-3."
            )
            logger.warning("statsmodels missing: STL method disabled. Install statsmodels for full baseline-3.")

    def _split_windows(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        values = series.values.astype(float)
        n = len(values)
        split = max(5, int(n * 0.4))
        baseline = values[:split]
        detect = values[split:]
        return baseline, detect

    @staticmethod
    def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1)
        if pooled <= 1e-12:
            return 0.0
        return abs(np.mean(y) - np.mean(x)) / np.sqrt(pooled)

    def _method_stat_test(self, baseline: np.ndarray, detect: np.ndarray) -> bool:
        try:
            p_u = stats.mannwhitneyu(baseline, detect, alternative="two-sided").pvalue
            p_t = stats.ttest_ind(baseline, detect, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            return False

        effect = self._cohen_d(baseline, detect)
        return ((p_u < self.p_value_thresh) or (p_t < self.p_value_thresh)) and effect >= self.cohen_d_thresh

    def _method_threshold(self, baseline: np.ndarray, detect: np.ndarray) -> bool:
        mean_b = float(np.mean(baseline))
        std_b = float(np.std(baseline))
        if std_b <= 1e-12:
            return False

        upper = mean_b + self.k_sigma * std_b
        lower = mean_b - self.k_sigma * std_b
        outlier_mask = (detect > upper) | (detect < lower)
        outlier_ratio = float(np.mean(outlier_mask)) if len(outlier_mask) > 0 else 0.0
        return outlier_ratio >= self.tail_ratio_thresh or np.any(outlier_mask)

    def _method_zvar(self, baseline: np.ndarray, detect: np.ndarray) -> bool:
        mean_b = float(np.mean(baseline))
        mean_d = float(np.mean(detect))
        std_b = float(np.std(baseline))
        var_b = float(np.var(baseline))
        var_d = float(np.var(detect))

        z_bias = abs(mean_d - mean_b) / max(std_b, 1e-6)
        var_ratio = max(var_d, var_b) / max(min(var_d, var_b), 1e-6)
        return (z_bias >= self.z_bias_thresh) or (var_ratio >= self.var_ratio_thresh)

    def _method_stl(self, series: pd.Series) -> bool:
        if not HAS_STL:
            return False

        if len(series) < 24:
            return False

        period = max(4, min(12, len(series) // 4))
        if period < 4:
            return False

        try:
            result = STL(series.values.astype(float), period=period, robust=True).fit()
            resid = pd.Series(result.resid)
            tail = resid.iloc[max(1, int(0.7 * len(resid))):]
            if tail.empty:
                return False
            z = (tail - tail.mean()) / max(tail.std(ddof=0), 1e-6)
            return bool((np.abs(z) > 3.0).any())
        except Exception:
            return False

    def detect(self, series: pd.Series) -> MethodResult:
        if len(series) < 12:
            return MethodResult(False, False, False, False, 0)

        if float(series.std()) <= 1e-12:
            return MethodResult(False, False, False, False, 0)

        baseline, detect = self._split_windows(series)
        if len(detect) < 5:
            return MethodResult(False, False, False, False, 0)

        s1 = self._method_stat_test(baseline, detect)
        s2 = self._method_threshold(baseline, detect)
        s3 = self._method_zvar(baseline, detect)
        s4 = self._method_stl(series)

        weighted = (2 if s1 else 0) + (1 if s2 else 0) + (1 if s3 else 0) + (2 if s4 else 0)
        return MethodResult(s1, s2, s3, s4, weighted)

    @staticmethod
    def is_anomaly(result: MethodResult) -> bool:
        votes = sum([result.stat_test, result.threshold, result.zvar, result.stl])
        if result.stat_test and result.stl:
            return True
        return votes >= 2 and result.weighted_score >= 3


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
        raise ValueError(f"invalid test dataset format: {dataset_path}")

    return data


def run_tests(limit=None, uuid=None):
    from exp.agent.metric import MetricAgent

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

    detector = HybridMetricDetector()
    loader_agent = MetricAgent(str(dataset_root))

    records = []
    for case in tqdm(test_cases, desc="Running baseline-3"):
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

            result = detector.detect(series)
            if detector.is_anomaly(result):
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
    output_file = output_dir / "result_baseline_3.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid)

# python3 unit-test/metric/baselines/baseline-3/run_baseline.py --limit=5
