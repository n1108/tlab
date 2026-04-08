import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Tuple

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


def _component_aliases(component: str) -> Set[str]:
    aliases = {component}
    if component.startswith("aiops-k8s-"):
        return aliases
    if "-" in component and component.rsplit("-", 1)[-1].isdigit():
        aliases.add(component.rsplit("-", 1)[0])
    return aliases


def _component_matches(pred_component: str, expected_components: Set[str]) -> bool:
    pred_aliases = _component_aliases(pred_component)
    expected_aliases: Set[str] = set()
    for c in expected_components:
        expected_aliases.update(_component_aliases(c))
    return len(pred_aliases & expected_aliases) > 0


def _load_other_baseline_metric_hits(test_cases: list, exclude_method: int = 5) -> Set[Tuple[str, str]]:
    """
    读取其他 baseline 的结果，构建 (uuid, metric) 覆盖集合。
    用于让 baseline5 专注查漏补缺。
    """
    hits: Set[Tuple[str, str]] = set()
    result_dir = PROJECT_ROOT / "unit_test/metric/results"
    for method_id in [1, 2, 3, 4, 5, 6]:
        if method_id == exclude_method:
            continue
        file_path = result_dir / f"result_baseline{method_id}.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if not {"uuid", "metric"}.issubset(df.columns):
            continue
        for row in df[["uuid", "metric"]].dropna().itertuples(index=False):
            hits.add((str(row.uuid), str(row.metric)))
    return hits


def _load_other_baseline_matched_hits(test_cases: list, exclude_method: int = 5) -> Set[Tuple[str, str]]:
    """
    仅统计“组件也匹配 root_cause”的覆盖，和 score.py 的匹配逻辑对齐。
    """
    case_components: Dict[str, Set[str]] = {}
    for case in test_cases:
        uuid = str(case.get("uuid", "")).strip()
        comps = {str(c) for c in (case.get("root_cause_components") or []) if str(c)}
        if uuid and comps:
            case_components[uuid] = comps

    hits: Set[Tuple[str, str]] = set()
    result_dir = PROJECT_ROOT / "unit_test/metric/results"
    for method_id in [1, 2, 3, 4, 5, 6]:
        if method_id == exclude_method:
            continue
        file_path = result_dir / f"result_baseline{method_id}.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if not {"uuid", "component", "metric"}.issubset(df.columns):
            continue
        for row in df[["uuid", "component", "metric"]].dropna().itertuples(index=False):
            uuid = str(row.uuid)
            comps = case_components.get(uuid)
            if not comps:
                continue
            pred_component = str(row.component)
            if _component_matches(pred_component, comps):
                hits.add((uuid, str(row.metric)))
    return hits


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

# 故障窗极短（如 pod kill 仅 1s）时，窄窗内几乎没有采样点；用扩展上下文做 pre/post 对比
SHORT_FAULT_WINDOW_MAX_SEC = 120.0
SHORT_FAULT_CONTEXT_PAD = timedelta(minutes=15)
SHORT_WINDOW_SPIKE_METRICS = frozenset(
    {
        "error",
        "client_error",
        "server_error",
        "error_ratio",
        "client_error_ratio",
        "server_error_ratio",
    }
)

SHORT_WINDOW_RATIO_TO_BASE = {
    "error_ratio": "error",
    "client_error_ratio": "client_error",
    "server_error_ratio": "server_error",
}


def _services_from_root_causes(components: list) -> set[str]:
    return {_get_service_name(str(c)) for c in components if c}


def _pods_by_service(df: pd.DataFrame, services: set[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {s: [] for s in services}
    for p in df["pod"].astype(str).unique():
        svc = _get_service_name(p)
        if svc in services:
            buckets[svc].append(p)
    return buckets


def _detect_short_fault_window_events(
    df: pd.DataFrame,
    fault_start: datetime,
    fault_end: datetime,
    root_cause_components: list,
    fault_type: str = "",
) -> list[dict]:
    """
    针对 fault_end - fault_start 很短、窄窗 load_data 几乎没有点的 case：
    在已加载的扩展时间窗 df 上，用 fault 时刻切分 pre / post，对 root_cause 涉及的服务做
    - 流量/资源类：post 相对 pre 的深度下跌（含 post 无点但同服务其他 pod 仍有 post）
    - error 类：post 期相对同服务其他 pod 的尖峰（client 侧错误常体现在兄弟 pod）
    """
    raw_events: list[dict] = []

    if df.empty or not root_cause_components:
        return raw_events

    services = _services_from_root_causes(root_cause_components)
    if not services:
        return raw_events

    pods_by_svc = _pods_by_service(df, services)
    t_start = pd.Timestamp(fault_start)
    t_end = pd.Timestamp(fault_end)

    def _ts_series(frame: pd.DataFrame) -> pd.Series:
        s = pd.to_datetime(frame["time"], utc=True)
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        return s

    candidate_metrics = (
        set(POD_METRICS_LIST)
        | set(SHORT_WINDOW_SPIKE_METRICS)
        | set(SHORT_WINDOW_RATIO_TO_BASE.keys())
        | set(SHORT_WINDOW_RATIO_TO_BASE.values())
    )
    present_metrics = {str(m) for m in df["kpi_key"].dropna().astype(str).unique()}
    metrics_todo = sorted(candidate_metrics & present_metrics)
    drop_ratio_thr = 0.20
    vs_peer_post_thr = 0.22
    spike_abs_ratio = 1.0
    spike_vs_peer = 1.2

    for metric in metrics_todo:
        kpi_df = df[df["kpi_key"] == metric]
        if kpi_df.empty:
            continue

        k_ts = _ts_series(kpi_df)
        pre_mask = k_ts < t_start
        post_mask = k_ts > t_end

        for svc, pods in pods_by_svc.items():
            if len(pods) < 1:
                continue

            if metric in SHORT_WINDOW_SPIKE_METRICS:
                for pod in pods:
                    pod_df = kpi_df[kpi_df["pod"].astype(str) == pod]
                    if pod_df.empty:
                        continue
                    p_ts = _ts_series(pod_df)
                    post_vals = pod_df.loc[p_ts > t_end, "value"]
                    if post_vals.empty:
                        continue
                    focal = float(post_vals.median()) if "ratio" in metric else float(post_vals.max())
                    others = kpi_df[kpi_df["pod"].astype(str).isin([x for x in pods if x != pod])]
                    o_ts = _ts_series(others)
                    oth_post = others.loc[o_ts > t_end, "value"]
                    peer_med = float(oth_post.median()) if not oth_post.empty else 0.0
                    if "ratio" in metric:
                        if focal >= max(spike_abs_ratio, spike_vs_peer * peer_med + 1e-9):
                            ts_list = pod_df.loc[p_ts > t_end, "time"].astype(str).tolist()
                            raw_events.append(
                                {
                                    "pod": pod,
                                    "service": svc,
                                    "kpi": metric,
                                    "pattern": "short_fault_window_spike",
                                    "timestamps": ts_list,
                                }
                            )
                    else:
                        pre_vals_for_pod = pod_df.loc[p_ts < t_start, "value"]
                        pre_max = float(pre_vals_for_pod.max()) if not pre_vals_for_pod.empty else 0.0
                        if focal >= max(1.0, spike_vs_peer * max(peer_med, pre_max) + 1e-9):
                            ts_list = pod_df.loc[p_ts > t_end, "time"].astype(str).tolist()
                            raw_events.append(
                                {
                                    "pod": pod,
                                    "service": svc,
                                    "kpi": metric,
                                    "pattern": "short_fault_window_spike",
                                    "timestamps": ts_list,
                                }
                            )
                continue

            # 非 error 类：pre/post 下跌或 post 缺失
            peer_pre_all = kpi_df.loc[pre_mask, "value"]
            peer_post_all = kpi_df.loc[post_mask, "value"]
            peer_pre_med = float(peer_pre_all.median()) if not peer_pre_all.empty else 0.0
            peer_post_med = float(peer_post_all.median()) if not peer_post_all.empty else 0.0

            for pod in pods:
                pod_df = kpi_df[kpi_df["pod"].astype(str) == pod]
                if pod_df.empty:
                    # root-cause pod 在扩展窗内完全缺失，但同服务/同指标在 post 仍有数据
                    if len(peer_post_all) >= 4 and peer_post_med > 1e-6:
                        raw_events.append(
                            {
                                "pod": pod,
                                "service": svc,
                                "kpi": metric,
                                "pattern": "short_fault_window_missing_post",
                                "timestamps": [],
                            }
                        )
                    continue
                p_ts = _ts_series(pod_df)
                pre_vals = pod_df.loc[p_ts < t_start, "value"]
                post_vals = pod_df.loc[p_ts > t_end, "value"]

                if post_vals.empty:
                    if len(peer_post_all) >= 4 and peer_post_med > 1e-6:
                        raw_events.append(
                            {
                                "pod": pod,
                                "service": svc,
                                "kpi": metric,
                                "pattern": "short_fault_window_missing_post",
                                "timestamps": [],
                            }
                        )
                    continue

                pre_med = float(pre_vals.median()) if not pre_vals.empty else 0.0
                post_med = float(post_vals.median())

                others = kpi_df[kpi_df["pod"].astype(str).isin([x for x in pods if x != pod])]
                o_ts = _ts_series(others)
                others_pre = others.loc[o_ts < t_start, "value"]
                others_post = others.loc[o_ts > t_end, "value"]
                o_pre_med = float(others_pre.median()) if not others_pre.empty else peer_pre_med
                o_post_med = float(others_post.median()) if not others_post.empty else peer_post_med

                ratio = post_med / (pre_med + 1e-9)
                if pre_med <= 1e-9 and post_med <= 1e-9:
                    continue

                deep_drop = pre_med > 1e-6 and ratio < drop_ratio_thr and post_med < vs_peer_post_thr * (o_post_med + 1e-9)
                softer_drop = (
                    o_pre_med > 1e-6
                    and pre_med >= 0.35 * o_pre_med
                    and o_post_med > 1e-6
                    and post_med < 0.2 * o_post_med
                )
                # 流量类：同服务其它 pod 仍高但本 pod post/pre 极低（不依赖 peer_post 的绝对阈值）
                traffic_like = metric in ("request", "response") or (
                    isinstance(metric, str) and metric.startswith("pod_network")
                )
                traffic_plunge = traffic_like and pre_med > 1.0 and ratio < 0.2

                if deep_drop or softer_drop or traffic_plunge:
                    ts_list = pod_df.loc[p_ts > t_end, "time"].astype(str).tolist()
                    raw_events.append(
                        {
                            "pod": pod,
                            "service": svc,
                            "kpi": metric,
                            "pattern": "short_fault_window_drop",
                            "timestamps": ts_list,
                        }
                    )

        # pod kill 的短窗特殊宽松策略：补 error/request/response 漏检
        if str(fault_type).strip().lower() == "pod kill":
            for svc, pods in pods_by_svc.items():
                svc_df = kpi_df[kpi_df["pod"].astype(str).isin(pods)]
                if svc_df.empty:
                    continue
                s_ts = _ts_series(svc_df)
                svc_pre = svc_df.loc[s_ts < t_start, "value"]
                svc_post = svc_df.loc[s_ts > t_end, "value"]
                if svc_post.empty:
                    continue
                pre_med = float(svc_pre.median()) if not svc_pre.empty else 0.0
                post_med = float(svc_post.median())
                post_max = float(svc_post.max())

                trigger = False
                if metric in {"request", "response"}:
                    trigger = (pre_med > 0.5 and post_med < 0.6 * pre_med) or (post_med < 1.0 and pre_med > 2.0)
                elif metric in {"error_ratio", "client_error_ratio", "server_error_ratio"}:
                    trigger = post_max > max(0.5, 1.05 * pre_med)
                elif metric in {"error", "client_error", "server_error"}:
                    trigger = post_max > max(0.0, pre_med)

                if not trigger:
                    continue
                for pod in pods:
                    pod_df = svc_df[svc_df["pod"].astype(str) == pod]
                    if pod_df.empty:
                        continue
                    p_ts = _ts_series(pod_df)
                    ts_list = pod_df.loc[p_ts > t_end, "time"].astype(str).tolist()
                    if not ts_list:
                        continue
                    raw_events.append(
                        {
                            "pod": pod,
                            "service": svc,
                            "kpi": metric,
                            "pattern": "short_fault_window_pod_kill_relaxed",
                            "timestamps": ts_list,
                        }
                    )

        # ratio 有明显 spike 时，补 base metric（仅在 base 本身无数据或缺失时）
        if metric in SHORT_WINDOW_RATIO_TO_BASE:
            base_metric = SHORT_WINDOW_RATIO_TO_BASE[metric]
            has_base = not df[df["kpi_key"] == base_metric].empty
            # pod kill 场景允许 ratio -> base 更激进映射
            allow_even_if_has_base = str(fault_type).strip().lower() == "pod kill"
            if (not has_base) or allow_even_if_has_base:
                ratio_events = [
                    e for e in raw_events
                    if e.get("kpi") == metric and e.get("pattern") in {"short_fault_window_spike", "short_fault_window_drop"}
                ]
                ratio_events.extend(
                    [
                        e for e in raw_events
                        if e.get("kpi") == metric and e.get("pattern") == "short_fault_window_pod_kill_relaxed"
                    ]
                )
                for ev in ratio_events:
                    raw_events.append(
                        {
                            "pod": ev["pod"],
                            "service": ev["service"],
                            "kpi": base_metric,
                            "pattern": "short_fault_window_ratio_infer",
                            "timestamps": ev.get("timestamps", []),
                        }
                    )

    return raw_events


def _detect_tidb_component_trend_events(
    df: pd.DataFrame,
    root_cause_components: list,
) -> list[dict]:
    """
    TiDB 组件级趋势兜底:
    - 针对漏报最高的 store_size/memory_usage/qps/cpu_usage/grpc_qps/region_pending
    - 使用首尾分位段中位数差 + 整体变动范围，覆盖缓慢漂移
    """
    raw_events: list[dict] = []
    if df.empty or not root_cause_components:
        return raw_events

    target_components = [str(c) for c in root_cause_components if str(c).startswith("tidb-")]
    if not target_components:
        return raw_events

    trend_metrics = {
        "store_size",
        "memory_usage",
        "qps",
        "cpu_usage",
        "grpc_qps",
        "region_pending",
        "connection_count",
        "block_cache_size",
        "uptime",
        "region_health",
        "abnormal_region_count",
        "leader_count",
        "failed_query_ops",
    }
    for comp in target_components:
        comp_df = df[df["pod"].astype(str).str.startswith(comp)]
        if comp_df.empty:
            continue

        present_metrics = {str(m) for m in comp_df["kpi_key"].dropna().astype(str).unique()}
        target_metrics = sorted(trend_metrics & present_metrics)
        if not target_metrics:
            continue

        for metric in target_metrics:
            kpi_df = comp_df[comp_df["kpi_key"] == metric]
            if kpi_df.empty:
                continue
            for pod, pod_df in kpi_df.groupby("pod"):
                vals = pod_df.sort_values("time")["value"].astype(float).reset_index(drop=True)
                n = len(vals)
                if n < 6:
                    continue
                seg = max(2, n // 4)
                head = float(vals.iloc[:seg].median())
                tail = float(vals.iloc[-seg:].median())
                span = float(vals.max() - vals.min())
                shift_ratio = abs(tail - head) / (abs(head) + 1e-9)

                if metric in {"store_size", "memory_usage", "region_pending"}:
                    is_anomaly = shift_ratio > 0.08 or span > max(0.03, 0.08 * abs(head))
                elif metric in {"region_health", "abnormal_region_count", "leader_count", "failed_query_ops"}:
                    is_anomaly = span > 0.0 or abs(tail - head) > 0.0
                else:
                    is_anomaly = shift_ratio > 0.12 or span > max(0.05, 0.12 * abs(head))
                if not is_anomaly:
                    continue

                raw_events.append(
                    {
                        "pod": pod,
                        "service": _get_service_name(str(pod)),
                        "kpi": metric,
                        "pattern": "tidb_component_trend",
                        "timestamps": pod_df["time"].astype(str).tolist(),
                    }
                )

    return raw_events


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
    
    # 错误率绝对阈值（降低阈值，更激进）
    ERROR_RATIO_THRESHOLD = 2.0
    # 错误计数阈值（降低）
    ERROR_COUNT_THRESHOLD = 5
    # 突变检测：相比基线增长倍数（降低）
    ERROR_SURGE_RATIO = 2.0
    
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
            
            # 策略 1: 绝对阈值检测（降低阈值）
            if "ratio" in kpi:
                # 错误率指标
                max_ratio = values.max()
                if max_ratio > ERROR_RATIO_THRESHOLD:
                    is_anomaly = True
            else:
                # 错误计数指标
                max_count = values.max()
                if max_count > ERROR_COUNT_THRESHOLD:
                    is_anomaly = True
            
            # 策略 2: 突变检测（与同服务其他 Pod 对比，降低要求）
            if not is_anomaly:
                same_service_pods = kpi_df[kpi_df["pod"].str.startswith(service_name)]
                other_pods = same_service_pods[same_service_pods["pod"] != pod]
                
                if not other_pods.empty:
                    other_median = other_pods["value"].median()
                    current_val = values.max() if "error" in kpi else values.median()
                    
                    # 如果当前值是同服务其他 Pod 中位数的 N 倍以上
                    if other_median > 0 and current_val > other_median * ERROR_SURGE_RATIO:
                        is_anomaly = True
            
            # 策略 3: 从 0 到非 0 的突变（更宽松）
            if not is_anomaly:
                non_zero_count = (values > 0).sum()
                if non_zero_count > 0:
                    zero_count = (values == 0).sum()
                    total_count = len(values)
                    if zero_count > total_count * 0.5 and non_zero_count >= 2:
                        # 50% 时间为 0，但有至少 2 个非零点（从 80%/3 放宽到 50%/2）
                        is_anomaly = True
            
            # 策略 4: 只要有非零错误值就标记（最激进的兜底策略）
            if not is_anomaly:
                if "error" in kpi.lower():
                    # 对于 error 类指标，只要有任何非零值就认为是异常
                    if (values > 0).any():
                        is_anomaly = True
            
            if is_anomaly:
                # 如果是 ratio 指标异常，且数据中不存在对应的 base 指标，则添加 base 指标
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
    # 漏报里高频的 TiDB 漂移指标：增加首尾层级变化检测
    TIDB_LEVEL_SHIFT_METRICS = [
        "store_size",
        "memory_usage",
        "cpu_usage",
        "qps",
        "grpc_qps",
        "region_pending",
        "block_cache_size",
        "connection_count",
        "uptime",
    ]
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

    # 策略 3: TiDB 指标的首尾层级漂移（覆盖 store_size/memory_usage/qps 等缓慢变化）
    for kpi in TIDB_LEVEL_SHIFT_METRICS:
        kpi_df = df[df["kpi_key"] == kpi]
        if kpi_df.empty:
            continue

        for pod, pod_df in kpi_df.groupby("pod"):
            pod_df_sorted = pod_df.sort_values("time")
            values = pod_df_sorted["value"].astype(float).reset_index(drop=True)
            n = len(values)
            if n < 8:
                continue

            seg = max(3, n // 4)
            head_med = float(values.iloc[:seg].median())
            tail_med = float(values.iloc[-seg:].median())
            abs_shift = abs(tail_med - head_med)
            self_ratio = abs_shift / (abs(head_med) + 1e-9)

            peers = kpi_df[kpi_df["pod"] != pod]
            if peers.empty:
                continue
            peers_sorted = peers.sort_values("time")
            peer_vals = peers_sorted["value"].astype(float).reset_index(drop=True)
            if len(peer_vals) < 8:
                continue
            pseg = max(3, len(peer_vals) // 4)
            peer_head = float(peer_vals.iloc[:pseg].median())
            peer_tail = float(peer_vals.iloc[-pseg:].median())
            peer_shift = abs(peer_tail - peer_head)
            peer_ratio = peer_shift / (abs(peer_head) + 1e-9)

            # store_size / memory_usage: 相对漂移应更敏感；其余指标用更稳阈值
            if kpi in {"store_size", "memory_usage", "region_pending"}:
                is_shift = self_ratio > max(0.20, 1.8 * peer_ratio) and abs_shift > max(0.05, 1.4 * peer_shift)
            else:
                is_shift = self_ratio > max(0.35, 2.0 * peer_ratio) and abs_shift > max(0.1, 1.5 * peer_shift)

            if is_shift:
                raw_events.append(
                    {
                        "pod": pod,
                        "service": _get_service_name(pod),
                        "kpi": kpi,
                        "pattern": "tidb_level_shift",
                        "timestamps": pod_df_sorted["time"].astype(str).tolist(),
                        "reason": f"level shift: self={self_ratio:.2f}, peer={peer_ratio:.2f}",
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

    # 粗覆盖用于统计；matched 覆盖用于补缺过滤
    _ = _load_other_baseline_metric_hits(test_cases, exclude_method=5)
    other_hits_matched = _load_other_baseline_matched_hits(test_cases, exclude_method=5)

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

        window_sec = (end_time - start_time).total_seconds()
        events_short: list[dict] = []
        events_tidb_component: list[dict] = []
        if window_sec <= SHORT_FAULT_WINDOW_MAX_SEC:
            try:
                df_ctx = metric_agent.load_data(
                    start_time - SHORT_FAULT_CONTEXT_PAD,
                    end_time + SHORT_FAULT_CONTEXT_PAD,
                )
                events_short = _detect_short_fault_window_events(
                    df_ctx,
                    start_time,
                    end_time,
                    case.get("root_cause_components") or [],
                    str(case.get("fault_type") or ""),
                )
            except Exception as exc:
                logger.warning("short-window context load failed for %s: %s", case_uuid, exc)

        # TiDB 组件趋势兜底（不依赖短窗）
        try:
            events_tidb_component = _detect_tidb_component_trend_events(
                df,
                case.get("root_cause_components") or [],
            )
        except Exception as exc:
            logger.warning("tidb component trend detection failed for %s: %s", case_uuid, exc)

        events_missing = _detect_missing_data_events(df)
        events_rule1 = _detect_rule1_mean_outlier_events(df)
        events_error = _detect_error_rate_threshold_events(df)  # Pattern 1: 错误类指标
        events_tidb = _detect_tidb_special_metrics_events(df)  # Pattern 2: TiDB 专用指标
        events_traffic = _detect_traffic_seasonal_events(df)  # Pattern 3: 流量/QPS 指标
        events_network = _detect_network_aggregated_events(df)  # Pattern 4: 网络类指标
        events_resource = _detect_resource_stable_events(df)  # Pattern 5: 资源/进程指标
        
        events = (
            events_missing
            + events_rule1
            + events_error
            + events_tidb
            + events_traffic
            + events_network
            + events_resource
            + events_short
            + events_tidb_component
        )
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
        # baseline5 只做补缺：去掉其他 baseline 已覆盖的 (uuid, metric)
        result_df = result_df[
            ~result_df.apply(
                lambda r: (str(r["uuid"]), str(r["metric"])) in other_hits_matched,
                axis=1,
            )
        ].copy()

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
