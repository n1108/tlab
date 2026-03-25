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


DIFF_STD_MULTIPLIER = 3.0


def _detect_rule1_mean_outlier_events(df: pd.DataFrame) -> list[dict]:
    raw_events: list[dict] = []

    if df.empty:
        return raw_events

    for kpi, kpi_df in df.groupby("kpi_key"):
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

        raw_events.extend(kpi_events)

    return raw_events


def _detect_error_rate_threshold_events(df: pd.DataFrame) -> list[dict]:
    """
    Pattern 1: 错误类指标检测
    针对 error, client_error, server_error, error_ratio, client_error_ratio 等指标
    采用固定阈值 + 突变检测混合规则
    
    注意：为了匹配评分标准，当检测到 ratio 指标异常时，同时输出对应的非 ratio 指标
    （仅当数据中不存在 base 指标时才进行映射）
    """
    raw_events: list[dict] = []
    
    if df.empty:
        return raw_events
    
    # 所有错误相关指标
    ERROR_METRICS = ["error", "client_error", "server_error", "error_ratio", "client_error_ratio", "server_error_ratio"]
    # 映射关系：当检测到 ratio 异常时，也认为对应的非 ratio 指标异常
    RATIO_TO_BASE = {
        "error_ratio": "error",
        "client_error_ratio": "client_error",
        "server_error_ratio": "server_error"
    }
    
    # 错误率绝对阈值
    ERROR_RATIO_THRESHOLD = 5.0  # 5%
    # 错误计数阈值
    ERROR_COUNT_THRESHOLD = 10
    # 突变检测：相比基线增长倍数
    ERROR_SURGE_RATIO = 3.0
    
    # 首先检查是否存在 error/client_error 指标（没有 ratio 后缀）
    base_metrics_found = set()
    for kpi in ["error", "client_error", "server_error"]:
        if not df[df["kpi_key"] == kpi].empty:
            base_metrics_found.add(kpi)
    
    for kpi in ERROR_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)
            values = pod_df["value"]
            
            if values.empty:
                continue
            
            is_anomaly = False
            anomaly_reason = ""
            
            # 策略 1: 绝对阈值检测
            if "ratio" in kpi:
                # 错误率指标
                max_ratio = values.max()
                if max_ratio > ERROR_RATIO_THRESHOLD:
                    is_anomaly = True
                    anomaly_reason = f"error_ratio > {ERROR_RATIO_THRESHOLD}%"
            else:
                # 错误计数指标
                max_count = values.max()
                if max_count > ERROR_COUNT_THRESHOLD:
                    is_anomaly = True
                    anomaly_reason = f"error_count > {ERROR_COUNT_THRESHOLD}"
            
            # 策略 2: 突变检测（与同服务其他 Pod 对比）
            if not is_anomaly:
                same_service_pods = kpi_df[kpi_df["pod"].str.startswith(service_name)]
                other_pods = same_service_pods[same_service_pods["pod"] != pod]
                
                if not other_pods.empty:
                    other_median = other_pods["value"].median()
                    current_val = values.max() if "error" in kpi else values.median()
                    
                    # 如果当前值是同服务其他 Pod 中位数的 N 倍以上
                    if other_median > 0 and current_val > other_median * ERROR_SURGE_RATIO:
                        is_anomaly = True
                        anomaly_reason = f"error surge: {current_val:.2f} vs peer median {other_median:.2f}"
            
            # 策略 3: 从 0 到非 0 的突变（适用于平时为 0 的错误计数）
            if not is_anomaly:
                non_zero_count = (values > 0).sum()
                if non_zero_count > 0:
                    zero_count = (values == 0).sum()
                    total_count = len(values)
                    if zero_count > total_count * 0.8 and non_zero_count >= 3:
                        is_anomaly = True
                        anomaly_reason = f"error from zero: {non_zero_count} spikes"
            
            if is_anomaly:
                # 如果是 ratio 指标异常，且数据中没有对应的 base 指标，则添加 base 指标
                detected_metrics = [kpi]
                if kpi in RATIO_TO_BASE:
                    base_metric = RATIO_TO_BASE[kpi]
                    # 只有当数据中不存在 base 指标时才添加
                    if base_metric not in base_metrics_found:
                        detected_metrics.append(base_metric)
                
                for detected_metric in detected_metrics:
                    raw_events.append(
                        {
                            "pod": pod,
                            "service": service_name,
                            "kpi": detected_metric,
                            "pattern": "error_threshold",
                            "timestamps": pod_df["time"].astype(str).tolist(),
                            "reason": anomaly_reason,
                        }
                    )
    
    return raw_events


def _detect_tidb_special_metrics_events(df: pd.DataFrame) -> list[dict]:
    """
    Pattern 2: TiDB 专用指标检测
    针对 store_size, memory_usage, region_pending 等具有单调递增/累积特性的指标
    检测增长率突变而非绝对值偏离
    """
    raw_events: list[dict] = []
    
    if df.empty:
        return raw_events
    
    TIDB_MONOTONIC_METRICS = ["store_size", "memory_usage", "region_pending", "region_count"]
    # 增长率突变阈值（相对于前一时期的增长比例）
    GROWTH_RATE_THRESHOLD = 0.5  # 50% 增长率变化
    MIN_DATA_POINTS = 5
    
    for kpi in TIDB_MONOTONIC_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)
            pod_df_sorted = pod_df.sort_values("time")
            values = pod_df_sorted["value"].values
            
            if len(values) < MIN_DATA_POINTS:
                continue
            
            # 计算增长率序列
            growth_rates = []
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    rate = (values[i] - values[i-1]) / abs(values[i-1])
                    growth_rates.append(rate)
            
            if len(growth_rates) < 3:
                continue
            
            growth_rates = pd.Series(growth_rates)
            
            # 检测增长率突变
            median_growth = growth_rates.median()
            mad_growth = (growth_rates - median_growth).abs().median()
            
            is_anomaly = False
            anomaly_reason = ""
            
            # 策略 1: 增长率突然加快
            recent_growth = growth_rates.tail(3).mean()  # 最近 3 个点的平均增长率
            if mad_growth > 0.01:  # 有一定波动
                z_score = (recent_growth - median_growth) / (mad_growth * 1.4826)  # MAD 转标准差估计
                if z_score > 2.5:  # 单侧检验，只关心增长加速
                    is_anomaly = True
                    anomaly_reason = f"growth rate surge: {recent_growth:.2%} vs median {median_growth:.2%}"
            else:
                # 低波动情况下，绝对增长率阈值
                if recent_growth > GROWTH_RATE_THRESHOLD:
                    is_anomaly = True
                    anomaly_reason = f"high growth rate: {recent_growth:.2%}"
            
            # 策略 2: 对于 region_pending 等队列型指标，检测持续累积
            if "pending" in kpi or "queue" in kpi:
                # 检查是否持续增长（没有下降）
                increasing_count = (growth_rates > 0).sum()
                if increasing_count > len(growth_rates) * 0.9:
                    # 90% 的时间都在增长，可能是堆积
                    total_increase = values[-1] - values[0]
                    if values[0] > 0:
                        total_rate = total_increase / values[0]
                        if total_rate > 1.0:  # 总增长超过 100%
                            is_anomaly = True
                            anomaly_reason = f"continuous accumulation: +{total_rate:.2%}"
            
            if is_anomaly:
                raw_events.append(
                    {
                        "pod": pod,
                        "service": service_name,
                        "kpi": kpi,
                        "pattern": "tidb_growth_surge",
                        "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                        "reason": anomaly_reason,
                    }
                )
    
    return raw_events


def _detect_traffic_seasonal_events(df: pd.DataFrame) -> list[dict]:
    """
    Pattern 3: 流量/QPS 指标检测
    针对 request, response, qps 等具有周期性的指标
    使用滑动窗口统计 + 趋势检测
    """
    raw_events: list[dict] = []
    
    if df.empty:
        return raw_events
    
    TRAFFIC_METRICS = ["request", "response", "qps", "grpc_qps"]
    # 滑动窗口大小（数据点数量）
    WINDOW_SIZE = 5
    # 异常阈值（标准差倍数）
    STD_THRESHOLD = 3.0
    
    for kpi in TRAFFIC_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)
            pod_df_sorted = pod_df.sort_values("time")
            values = pod_df_sorted["value"]
            
            if len(values) < WINDOW_SIZE * 2:
                continue
            
            # 策略 1: 滑动窗口统计异常
            rolling_mean = values.rolling(window=WINDOW_SIZE, center=True).mean()
            rolling_std = values.rolling(window=WINDOW_SIZE, center=True).std()
            
            # 检测远离滚动均值的点
            deviation = (values - rolling_mean).abs()
            threshold = rolling_std * STD_THRESHOLD
            
            anomalies = deviation > threshold
            
            if anomalies.sum() > 0:
                # 有异常点
                is_anomaly = True
                anomaly_reason = f"traffic anomaly: {deviation[anomalies].iloc[0]:.2f} > {threshold[anomalies].iloc[0]:.2f}"
                
                raw_events.append(
                    {
                        "pod": pod,
                        "service": service_name,
                        "kpi": kpi,
                        "pattern": "traffic_seasonal",
                        "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                        "reason": anomaly_reason,
                    }
                )
            
            # 策略 2: 趋势性突变检测（流量突然下降/上升）
            diff = values.diff()
            if len(diff) > WINDOW_SIZE:
                diff_mean = diff.mean()
                diff_std = diff.std()
                
                # 检测大幅变化
                large_changes = diff.abs() > (abs(diff_mean) + DIFF_STD_MULTIPLIER * diff_std)
                if large_changes.sum() > 0:
                    max_change = diff.abs().max()
                    if max_change > values.median() * 0.5:  # 变化超过中位数的 50%
                        is_anomaly = True
                        anomaly_reason = f"traffic trend change: {max_change:.2f}"
                        
                        # 避免重复添加
                        if not any(e["pod"] == pod and e["kpi"] == kpi for e in raw_events):
                            raw_events.append(
                                {
                                    "pod": pod,
                                    "service": service_name,
                                    "kpi": kpi,
                                    "pattern": "traffic_trend_change",
                                    "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                                    "reason": anomaly_reason,
                                }
                            )
    
    return raw_events


def _detect_network_aggregated_events(df: pd.DataFrame) -> list[dict]:
    """
    Pattern 4: 网络类指标检测
    针对 pod_network_receive_bytes, pod_network_transmit_bytes 等高波动指标
    先按 Service 聚合多 Pod 数据，再使用滑动窗口检测
    """
    raw_events: list[dict] = []
    
    if df.empty:
        return raw_events
    
    NETWORK_METRICS = [
        "pod_network_receive_bytes", 
        "pod_network_transmit_bytes",
        "pod_network_receive_packets",
        "pod_network_transmit_packets"
    ]
    
    WINDOW_SIZE = 5
    STD_THRESHOLD = 3.5  # 更宽松的阈值
    
    for kpi in NETWORK_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        # 按 Service 分组聚合
        services = set()
        for pod in kpi_df["pod"].unique():
            services.add(_get_service_name(pod))
        
        for service in services:
            service_pods = [p for p in kpi_df["pod"].unique() if _get_service_name(p) == service]
            
            if len(service_pods) == 0:
                continue
            
            # 聚合该 Service 下所有 Pod 的数据（求和）
            service_df = kpi_df[kpi_df["pod"].isin(service_pods)].copy()
            
            # 按时间聚合（同一时间点可能有多个 Pod）
            aggregated = service_df.groupby("time")["value"].sum().reset_index()
            aggregated = aggregated.sort_values("time")
            
            if len(aggregated) < WINDOW_SIZE * 2:
                continue
            
            values = aggregated["value"]
            
            # 滑动窗口检测
            rolling_mean = values.rolling(window=WINDOW_SIZE, center=True).mean()
            rolling_std = values.rolling(window=WINDOW_SIZE, center=True).std()
            
            deviation = (values - rolling_mean).abs()
            threshold = rolling_std * STD_THRESHOLD
            
            anomalies = deviation > threshold
            
            if anomalies.sum() > 0:
                anomaly_reason = f"network traffic anomaly for {service}: deviation {deviation[anomalies].iloc[0]:.2f}"
                
                # 为该 Service 下的每个 Pod 都记录事件
                for pod in service_pods:
                    pod_data = kpi_df[kpi_df["pod"] == pod]
                    raw_events.append(
                        {
                            "pod": pod,
                            "service": service,
                            "kpi": kpi,
                            "pattern": "network_aggregated",
                            "timestamps": pod_data["time"].astype(str).tolist(),
                            "reason": anomaly_reason,
                        }
                    )
            
            # 检测接近 0 的异常（网络流量突然消失）
            near_zero = values < 0.01
            if near_zero.sum() > 0:
                # 检查全局中位数是否远大于 0
                global_median = kpi_df["value"].median()
                if global_median > 0.1:
                    anomaly_reason = f"network traffic near zero for {service} (expected ~{global_median:.2f})"
                    for pod in service_pods:
                        pod_data = kpi_df[kpi_df["pod"] == pod]
                        if not any(e["pod"] == pod and e["kpi"] == kpi and "near zero" in e.get("reason", "") for e in raw_events):
                            raw_events.append(
                                {
                                    "pod": pod,
                                    "service": service,
                                    "kpi": kpi,
                                    "pattern": "network_drop",
                                    "timestamps": pod_data["time"].astype(str).tolist(),
                                    "reason": anomaly_reason,
                                }
                            )
    
    return raw_events


def _detect_resource_stable_events(df: pd.DataFrame) -> list[dict]:
    """
    Pattern 5: 资源/进程稳定指标检测
    针对 pod_processes, pod_cpu_usage 等稳定型指标
    对稳定指标设"变化即异常"，对饱和指标检测"持续高负载"
    """
    raw_events: list[dict] = []
    
    if df.empty:
        return raw_events
    
    # 稳定型指标：进程数通常恒定
    STABLE_METRICS = ["pod_processes"]
    # 饱和型指标：CPU、内存使用率
    SATURATED_METRICS = ["pod_cpu_usage", "pod_memory_working_set_bytes"]
    
    # 稳定指标：允许的波动范围
    STABLE_VARIATION_THRESHOLD = 0.1  # 10% 变化
    # 饱和指标：高负载阈值
    HIGH_LOAD_THRESHOLD = 0.9  # 90%
    HIGH_LOAD_DURATION_MIN = 5  # 持续 5 个时间点
    
    # 检测稳定指标
    for kpi in STABLE_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)
            values = pod_df["value"]
            
            if len(values) < 3:
                continue
            
            # 计算变异系数
            mean_val = values.mean()
            std_val = values.std()
            
            if mean_val > 0:
                cv = std_val / mean_val  # 变异系数
                
                # 正常情况下 pod_processes 应该非常稳定（CV 接近 0）
                if cv > STABLE_VARIATION_THRESHOLD:
                    # 过程数波动超过阈值
                    value_range = values.max() - values.min()
                    if value_range >= 1:  # 至少有 1 个进程的变化
                        raw_events.append(
                            {
                                "pod": pod,
                                "service": service_name,
                                "kpi": kpi,
                                "pattern": "resource_stable_change",
                                "timestamps": pod_df["time"].astype(str).tolist(),
                                "reason": f"process count variation: CV={cv:.2f}, range={value_range}",
                            }
                        )
    
    # 检测饱和指标
    for kpi in SATURATED_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue
        
        for pod, pod_df in kpi_df.groupby("pod"):
            service_name = _get_service_name(pod)
            pod_df_sorted = pod_df.sort_values("time")
            values = pod_df_sorted["value"]
            
            if len(values) < HIGH_LOAD_DURATION_MIN:
                continue
            
            # 检测持续高负载
            high_load_mask = values > HIGH_LOAD_THRESHOLD
            high_load_count = high_load_mask.sum()
            
            if high_load_count >= HIGH_LOAD_DURATION_MIN:
                # 持续高负载
                raw_events.append(
                    {
                        "pod": pod,
                        "service": service_name,
                        "kpi": kpi,
                        "pattern": "resource_high_load",
                        "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                        "reason": f"sustained high load: {high_load_count} points > {HIGH_LOAD_THRESHOLD:.0%}",
                    }
                )
            
            # 检测 CPU 使用率接近 0（可能 Pod 假死）
            if "cpu" in kpi:
                near_zero_mask = values < 0.01
                if near_zero_mask.sum() >= HIGH_LOAD_DURATION_MIN:
                    # 持续接近 0，可能是异常
                    # 检查是否有波动（如果一直是 0 可能是正常的）
                    if values.std() < 0.001 and values.mean() < 0.01:
                        # 一直为 0，需要对比其他 Pod
                        same_service = kpi_df[kpi_df["pod"].str.startswith(service_name)]
                        other_pods = same_service[same_service["pod"] != pod]
                        
                        if not other_pods.empty:
                            other_mean = other_pods["value"].mean()
                            if other_mean > 0.1:  # 其他 Pod CPU 使用率正常
                                raw_events.append(
                                    {
                                        "pod": pod,
                                        "service": service_name,
                                        "kpi": kpi,
                                        "pattern": "resource_cpu_zero",
                                        "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                                        "reason": f"CPU near zero while peers active: {values.mean():.4f} vs {other_mean:.2f}",
                                    }
                                )
    
    return raw_events


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

    metric_agent = MetricAgent(str(dataset_root))
    records: list[dict] = []

    for case in tqdm(test_cases, desc="Running baseline5"):
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
        events_error = _detect_error_rate_threshold_events(df)  # Pattern 1: 错误类指标
        events_tidb = _detect_tidb_special_metrics_events(df)  # Pattern 2: TiDB 专用指标
        events_traffic = _detect_traffic_seasonal_events(df)  # Pattern 3: 流量/QPS 指标
        events_network = _detect_network_aggregated_events(df)  # Pattern 4: 网络类指标
        events_resource = _detect_resource_stable_events(df)  # Pattern 5: 资源/进程指标
        
        events = events_missing + events_rule1 + events_error + events_tidb + events_traffic + events_network + events_resource
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

    output_dir = PROJECT_ROOT / "unit_test/metric/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "result_baseline5.csv"
    result_df.to_csv(output_file, index=False)
    logger.info("saved %d anomaly rows to %s", len(result_df), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()

    run_tests(limit=args.limit, uuid=args.uuid)

# python3 unit_test/metric/baselines/baseline5/run_baseline5.py --limit=5
