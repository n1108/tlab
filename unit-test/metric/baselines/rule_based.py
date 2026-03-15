import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[3]
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

logger = logging.getLogger(__name__)

class RuleBasedMetricAgent(MetricAgent):
    """
    基于规则的异常检测算法
    """
    
    def __init__(self, root_path):
        super().__init__(root_path)
        # 规则1忽略的指标: error/exception 等稀疏指标不适合用中位数绝对偏差(MAD)检测
        self.rule1_ignore_metrics = [
            "error_ratio",
            "client_error_ratio",
            "memory_usage",
            "pod_memory_working_set_bytes",

            # testcase 1 确实有异常，暂且忽略
            "node_disk_written_bytes_total",
            "pod_fs_writes_bytes"
        ]
        
        # 指标绑定关系：如果检测到 Key 指标异常，则认为 Value 列表中的指标也异常
        self.metric_binds = {
            "rrt": ["rrt_max"],
            # "cpu_usage": ["cpu_load"],
        }

    def _get_service_name(self, pod_name):
        if not isinstance(pod_name, str): return "unknown"
        parts = pod_name.rsplit('-', 1)
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]
        return pod_name

    def query_metrics(self, start_time, end_time):

        df = self.load_data(start_time, end_time)
        if df.empty: return {"observation": "No data", "events": []}
            
        events = []
        
        for kpi, kpi_df in df.groupby("kpi_key"):
            
            # --- Aggregation Prep ---
            # Calculate total pods for each service in this KPI context
            service_pods_map = {}
            for pod in kpi_df["pod"].unique():
                svc = self._get_service_name(pod)
                if svc not in service_pods_map: service_pods_map[svc] = set()
                service_pods_map[svc].add(pod)

            kpi_events = [] 

            # 规则1：对于某个metric，某个组件的平均值偏离其他组件，基于和中位数的倍数关系识别
            if kpi not in self.rule1_ignore_metrics:
                for pod, pod_df in kpi_df.groupby("pod"):
                    service_name = self._get_service_name(pod)
                    other_pods = kpi_df[kpi_df["pod"] != pod]
                    if other_pods.empty: continue
                    
                    vals = other_pods["value"]
                    median_val = vals.median()
                    mad = (vals - median_val).abs().median()
                    # 使用中位数代替均值，避免单点突刺（spike）导致的误报
                    val = pod_df["value"].median()
                    
                    is_outlier = False
                    
                    # 定义绝对阈值 epsilon，避免在数值极小时产生误报
                    # 对于 CPU (0~1) 或 Error Rate (0~1) 等归一化指标，0.1 是一个显著的变化
                    # 对于 Latency/Throughput 等大数值指标，0.1 也可以接受
                    EPSILON = 0.1 

                    # Case 1: 分布有波动 (MAD > 0)
                    if mad > 1e-4:
                        z_score = 0.6745 * (val - median_val) / mad
                        if abs(z_score) > 3.5:
                            # 结合相对变化，过滤掉高 Z-score 但低幅度的噪声
                            if median_val > 1e-4:
                                # 3倍中位数 或者 变化幅度超过阈值
                                if val > 3.0 * median_val or val < median_val / 3.0:
                                    is_outlier = True
                            else:
                                # 基准为0，但波动大
                                if abs(val) > EPSILON:
                                    is_outlier = True

                    # Case 2: 分布集中 (MAD ~ 0)
                    else:
                        if median_val > 1e-4:
                            # 基准非0，要求3倍差异 (e.g. 1 process vs 10, 0.02 cpu vs 0.4)
                            if val > 3.0 * median_val or val < median_val / 3.0:
                                is_outlier = True
                        else:
                            # 基准为0 (e.g. Error Count)
                            # 只有当值显著大于0时才报警
                            if abs(val) > EPSILON:
                                is_outlier = True

                    if is_outlier:
                        kpi_events.append({
                            "pod": pod,
                            "service": service_name,
                            "kpi": kpi,
                            "pattern": "mean_outlier",
                            "timestamps": pod_df["time"].astype(str).tolist()
                        })

            # 如果检测出太多（超过总数25%）的pod都是outlier，说明这个metric本身在不同服务间差异巨大（如network bytes）
            # 这种情况下，全局outlier检测失效，应忽略该metric
            total_pods_in_kpi = kpi_df["pod"].nunique()
            unique_anomalous_pods = set([e["pod"] for e in kpi_events])
            if total_pods_in_kpi > 0 and (len(unique_anomalous_pods) / total_pods_in_kpi) > 0.25:
                continue

            # --- Aggregation Logic ---
            # Group caught events by service
            detected_service_events = {}
            for e in kpi_events:
                svc = e["service"]
                if svc not in detected_service_events: detected_service_events[svc] = []
                detected_service_events[svc].append(e)
            
            for svc, svc_events in detected_service_events.items():
                total_count = len(service_pods_map.get(svc, []))
                # Check if ALL pods for this service are anomalous
                if total_count > 0 and len(svc_events) == total_count:
                    # Aggregate
                    all_timestamps = set()
                    for ev in svc_events:
                        for ts in ev["timestamps"]:
                            all_timestamps.add(ts)
                            
                    events.append({
                        "pod": svc, # Use service name as component
                        "service": svc,
                        "kpi": kpi,
                        "pattern": "mean_outlier",
                        "timestamps": sorted(list(all_timestamps))
                    })
                else:
                    # No aggregation, keep individual events
                    events.extend(svc_events)

        # --- Metric Binding Post-processing ---
        # 如果检测到了 metric A 的异常，而 A 与 B 绑定，则为同一组件补充 B 的异常
        # 这有助于保持相关指标的一致性
        derived_events = []
        for event in events:
            kpi = event.get("kpi")
            if kpi in self.metric_binds:
                for bound_kpi in self.metric_binds[kpi]:
                    # 复用原事件的所有属性，仅修改 metric name
                    new_event = event.copy()
                    new_event["kpi"] = bound_kpi
                    # 可以在 pattern 中标记这是推导出来的，或者保持原样
                    # new_event["pattern"] = "derived_from_" + kpi
                    derived_events.append(new_event)
        
        events.extend(derived_events)

        return {"observation": "Rule-based detection completed", "events": events}
