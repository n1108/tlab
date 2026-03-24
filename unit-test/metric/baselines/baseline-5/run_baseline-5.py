import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASELINE5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POD_METRICS_LIST = [
    "pod_cpu_usage",
    "pod_fs_writes_bytes",
    "pod_memory_working_set_bytes",
    "pod_network_receive_bytes",
    "pod_network_receive_packets",
    "pod_network_transmit_bytes",
    "pod_network_transmit_packets",
    "pod_processes",
    "request",
    "response",
    "client_error",
    "client_error_ratio",
    "error",
    "error_ratio",
]

PODS_LIST = [
    "adservice-0", "adservice-1", "adservice-2",
    "cartservice-0", "cartservice-1", "cartservice-2",
    "currencyservice-0", "currencyservice-1", "currencyservice-2",
    "productcatalogservice-0", "productcatalogservice-1", "productcatalogservice-2",
    "checkoutservice-0", "checkoutservice-1", "checkoutservice-2",
    "recommendationservice-0", "recommendationservice-1", "recommendationservice-2",
    "shippingservice-0", "shippingservice-1", "shippingservice-2",
    "emailservice-0", "emailservice-1", "emailservice-2",
    "paymentservice-0", "paymentservice-1", "paymentservice-2",
]

RULE1_IGNORE_METRICS = [
    "memory_usage",
    "pod_memory_working_set_bytes",
    "node_disk_written_bytes_total",
    "pod_fs_writes_bytes",
]


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


def _get_service_name(pod_name: str) -> str:
    if not isinstance(pod_name, str):
        return "unknown"
    parts = pod_name.rsplit("-", 1)
    if len(parts) > 1 and parts[-1].isdigit():
        return parts[0]
    return pod_name


def _detect_missing_data_events(df: pd.DataFrame) -> list[dict]:
    raw_events: list[dict] = []

    if df.empty:
        return raw_events

    for metric in POD_METRICS_LIST:
        metric_df = df[df["kpi_key"] == metric]
        if metric_df.empty:
            continue

        pod_counts = metric_df["pod"].value_counts()
        max_count = pod_counts.max() if not pod_counts.empty else 0
        pods_with_data = set(metric_df["pod"].unique())

        for pod in PODS_LIST:
            svc = _get_service_name(pod)

            if pod not in pods_with_data:
                raw_events.append(
                    {
                        "pod": pod,
                        "service": svc,
                        "kpi": metric,
                        "pattern": "missing_data",
                        "timestamps": [],
                    }
                )
                continue

            # 部分数据缺失：同窗口下该 pod 数据点远少于其他同指标 pod
            count = int(pod_counts.get(pod, 0))
            if max_count > 5 and count < 0.5 * max_count:
                ts = metric_df[metric_df["pod"] == pod]["time"].astype(str).tolist()
                raw_events.append(
                    {
                        "pod": pod,
                        "service": svc,
                        "kpi": metric,
                        "pattern": "missing_data",
                        "timestamps": ts,
                    }
                )

    return raw_events


def _detect_rule1_mean_outlier_events(df: pd.DataFrame) -> list[dict]:
    raw_events: list[dict] = []

    if df.empty:
        return raw_events

    for kpi, kpi_df in df.groupby("kpi_key"):
        if kpi in RULE1_IGNORE_METRICS:
            continue

        if "node_" in kpi or "max" in kpi:
            continue

        rule1_whitelist = ["cpu", "memory", "request", "error", "rrt", "response", "processes", "network"]
        if not any(token in kpi for token in rule1_whitelist):
            continue

        kpi_events: list[dict] = []

        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)

            same_service_pods = kpi_df[kpi_df["pod"].str.startswith(service_name)]
            other_pods_internal = same_service_pods[same_service_pods["pod"] != pod]

            use_global = False
            if other_pods_internal.empty:
                use_global = True
            else:
                vals_int = other_pods_internal["value"]
                mad_int = (vals_int - vals_int.median()).abs().median()
                if mad_int < 1e-4:
                    global_others = kpi_df[kpi_df["pod"] != pod]
                    if not global_others.empty:
                        global_median = global_others["value"].median()
                        peer_median = vals_int.median()
                        if abs(peer_median - global_median) > max(global_median, 0.1) * 2.0:
                            use_global = True

            if use_global:
                other_pods = kpi_df[kpi_df["pod"] != pod]
            else:
                other_pods = other_pods_internal

            if other_pods.empty:
                continue

            vals = other_pods["value"]
            median_val = vals.median()
            mad = (vals - median_val).abs().median()

            if "error" in kpi:
                val = pod_df["value"].max()
            else:
                val = pod_df["value"].median()

            is_outlier = False
            epsilon = 0.05

            if mad > 1e-4:
                z_score = 0.6745 * (val - median_val) / mad
                if abs(z_score) > 3.5:
                    if median_val > 1e-4:
                        if (val > 3.0 * median_val or val < median_val / 3.0) and abs(val - median_val) > epsilon:
                            is_outlier = True
                    else:
                        if abs(val) > epsilon:
                            is_outlier = True
            else:
                if median_val > 1e-4:
                    if (val > 3.0 * median_val or val < median_val / 3.0) and abs(val - median_val) > epsilon:
                        is_outlier = True
                else:
                    if abs(val) > epsilon:
                        is_outlier = True

            if not is_outlier:
                is_network = "network" in kpi and ("bytes" in kpi or "packets" in kpi)
                if is_network and abs(val) < 1e-4:
                    global_median = kpi_df["value"].median()
                    if global_median > 0.05:
                        is_outlier = True

            if is_outlier:
                kpi_events.append(
                    {
                        "pod": pod,
                        "service": service_name,
                        "kpi": kpi,
                        "pattern": "mean_outlier",
                        "timestamps": pod_df["time"].astype(str).tolist(),
                    }
                )

        total_pods_in_kpi = kpi_df["pod"].nunique()
        unique_anomalous_pods = {e["pod"] for e in kpi_events if e["pattern"] == "mean_outlier"}

        is_error_metric = "error" in kpi
        is_network_metric = "network" in kpi
        if (
            not is_error_metric
            and not is_network_metric
            and total_pods_in_kpi > 0
            and (len(unique_anomalous_pods) / total_pods_in_kpi) > 0.25
        ):
            continue

        raw_events.extend(kpi_events)

    return raw_events


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

    metric_agent = MetricAgent(str(dataset_root))
    records: list[dict] = []

    for case in tqdm(test_cases, desc="Running baseline-5"):
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
            df = metric_agent.load_data(start_time, end_time)
        except Exception as exc:
            logger.warning("load_data failed for %s: %s", case_uuid, exc)
            continue

        events_missing = _detect_missing_data_events(df)
        events_rule1 = _detect_rule1_mean_outlier_events(df)
        events = events_missing + events_rule1
        for ev in events:
            records.append(
                {
                    "uuid": case_uuid,
                    "component": str(ev["pod"]),
                    "metric": str(ev["kpi"]),
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
    output_file = output_dir / "result_baseline_5.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid)

# python3 unit-test/metric/baselines/baseline-5/run_baseline-5.py --limit=5
