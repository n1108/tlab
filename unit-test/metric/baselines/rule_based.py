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
        # 调试配置：手动修改此列表以只运行特定规则 [1, 2, 3]
        self.rules = [1, 2, 3] 
        
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
            if 1 in self.rules and kpi not in self.rule1_ignore_metrics:
                # Explicitly exclude noisy metrics
                if "client_error" in kpi or "node_" in kpi or "max" in kpi:
                    pass
                else:
                    # 过滤: 仅对 Golden Signals 和 资源指标 做 Rule 1 检测
                    rule1_whitelist = ["cpu", "memory", "request", "error", "rrt", "response"]
                    should_check = False
                    for w in rule1_whitelist:
                        if w in kpi: 
                            should_check = True
                            break
                    
                    if should_check:
                        pass
                    
                    for pod, pod_df in kpi_df.groupby("pod"):
                        if not should_check: break
                        service_name = self._get_service_name(pod)
                        
                        # Compare against ALL other pods (Global Outlier)
                        # This works well if most services are healthy/fast
                        other_pods = kpi_df[kpi_df["pod"] != pod]
                        if other_pods.empty: continue
                        
                        vals = other_pods["value"]
                        median_val = vals.median()
                        mad = (vals - median_val).abs().median()
                        # 使用中位数代替均值，避免单点突刺（spike）导致的误报
                        val = pod_df["value"].median()
                        
                        is_outlier = False
                    
                        # Avoid small noise triggering alerts
                        EPSILON = 0.05

                        # Case 1: 分布有波动 (MAD > 0)
                        if mad > 1e-4:
                            z_score = 0.6745 * (val - median_val) / mad
                            if abs(z_score) > 3.5:
                                # 结合相对变化，过滤掉高 Z-score 但低幅度的噪声
                                if median_val > 1e-4:
                                    # 3倍中位数 或者 变化幅度超过阈值 AND 绝对差异显著
                                    if (val > 3.0 * median_val or val < median_val / 3.0) and abs(val - median_val) > EPSILON:
                                        is_outlier = True
                                else:
                                    # 基准为0，但波动大
                                    if abs(val) > EPSILON:
                                        is_outlier = True

                        # Case 2: 分布集中 (MAD ~ 0)
                        else:
                            if median_val > 1e-4:
                                # 基准非0，要求3倍差异 (e.g. 1 process vs 10, 0.02 cpu vs 0.4)
                                # AND 绝对差异显著
                                if (val > 3.0 * median_val or val < median_val / 3.0) and abs(val - median_val) > EPSILON:
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

            # 规则2：数据缺失 - 某组件在一段时间内中断上报 (Gap Detection)
            gap_check_iter = kpi_df.groupby("pod") if 2 in self.rules else []
            for pod, pod_df in gap_check_iter:
                # 过滤关键指标，避免在非关键指标上报 Missing Data
                # 仅检测核心黄金指标的缺失
                if not any(x in kpi for x in ["cpu", "memory", "request", "error", "latency", "rrt"]):
                    continue
                pod_df = pod_df.sort_values("time")
                times = pod_df["time"].values
                if len(times) < 2: continue

                diffs = np.diff(times)
                
                # 兼容 timestamp (int/float) 和 datetime64[ns]
                if np.issubdtype(diffs.dtype, np.timedelta64):
                    diffs_sec = diffs / np.timedelta64(1, 's')
                else:
                    diffs_sec = diffs
                
                median_interval = np.median(diffs_sec)
                if median_interval < 1.0: 
                    # 采样间隔过小，可能是脏数据或非周期性数据，忽略
                    continue
                
                # 缺失数据的判定阈值：3倍采样间隔
                threshold = 3.0 * median_interval
                gap_indices = np.where(diffs_sec > threshold)[0]
                
                if len(gap_indices) > 0:
                    svc = self._get_service_name(pod)
                    for idx in gap_indices:
                        t_start = times[idx]
                        t_end = times[idx+1]
                        
                        kpi_events.append({
                            "pod": pod,
                            "service": svc,
                            "kpi": kpi, 
                            "pattern": "missing_data",
                            "timestamps": [str(t_start), str(t_end)]
                        })
            
            # 规则3：更加细粒度的类别特定检测逻辑
            # Instead of a single "Waveform Spike", we define specialized detectors.
            if 3 in self.rules:
                for pod, pod_df in kpi_df.groupby("pod"):
                    series = pod_df["value"]
                    if len(series) < 5: continue
                    
                    # Ignore client-side errors (4xx) and noisy node metrics and max metrics
                    if "client_error" in kpi or "node_" in kpi or "max" in kpi: continue

                    is_anomaly = False
                    timestamps = []
                    
                    # --- Sub-Rule 3.1: Error Metrics (Golden Signal: Errors) ---
                    # 严格判定：任何 > 0 的错误率都被视为异常，如果有一定持续性
                    # Include timeout as error type
                    if "error" in kpi or "timeout" in kpi:
                        # 忽略极小的偶然错误 (e.g. 1/1000 requests)
                        # Increase threshold to 2% to reduce FP
                        threshold = 0.02 # 2% error rate
                        anoms = series > threshold
                        # Require persistence: at least 3 points over threshold (more strict)
                        if anoms.sum() >= 3:
                            timestamps = pod_df.loc[anoms, "time"].astype(str).tolist()
                            is_anomaly = True

                    # --- Sub-Rule 3.2: Latency Metrics (Golden Signal: Latency) ---
                    # 延迟指标的 IQR 检测需要非常宽容，因为延迟本身方差大
                    elif "rrt" in kpi or "latency" in kpi or "time" in kpi:
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr = q3 - q1
                        median = series.median()
                        
                        # Use massive margin (15.0 IQR) or absolute threshold (e.g. > 300ms baseline increase)
                        if iqr > 1e-5:
                             # 20.0 IQR to be very very loose on variance
                            upper = q3 + 20.0 * iqr
                            anoms = series > upper
                        else:
                            # Baseline calm -> Spike > 300ms is anomalous
                            anoms = series > (median + 0.3)
                        
                        if anoms.any():
                            timestamps = pod_df.loc[anoms, "time"].astype(str).tolist()
                            is_anomaly = True

                    # --- Sub-Rule 3.3: Traffic/Throughput (Golden Signal: Traffic) ---
                    # 流量突降 (Drop) 或突增 (Spike)
                    # 通常由 Rule 2 (Missing Data) 覆盖 Drop to 0。这里检测显著变化。
                    elif "qps" in kpi or "bps" in kpi or "request" in kpi or "response" in kpi or "packet" in kpi:
                        median = series.median()
                        if median > 10.0: # Only analyze if there is meaningful traffic
                             # Drop significantly: < 10% of median (Severe Drop)
                             # Spike significantly: > 5x median
                             anoms = (series < median * 0.1) | (series > median * 5.0)
                             if anoms.any():
                                 timestamps = pod_df.loc[anoms, "time"].astype(str).tolist()
                                 is_anomaly = True

                    # --- Sub-Rule 3.4: Saturation (CPU/Memory) ---
                    # 资源使用率通常比较平稳。
                    elif "cpu" in kpi or "memory" in kpi: 
                        # Identify system components to reduce noise
                        is_system = pod.startswith("k8s-master") or pod.startswith("tidb") or pod.startswith("aiops-k8s")
                        
                        # CPU/Mem Usage Rate (0-1 or 0-100)
                        # Only flag simple overload or massive spike
                        if "usage_rate" in kpi or "util" in kpi: 
                            # If usage > 95% (Saturation). Stricter (99%) for system nodes.
                            sat_thresh = 0.99 if is_system else 0.95
                            saturation = series > sat_thresh
                            
                            if saturation.any():
                                timestamps = pod_df.loc[saturation, "time"].astype(str).tolist()
                                is_anomaly = True
                            else:
                                # Detection of sudden jump (e.g. 10% -> 80%)
                                # Stricter jump for system nodes
                                diff = series.diff().abs()
                                jump_thresh = 0.8 if is_system else 0.5
                                jump = diff > jump_thresh
                                if jump.any():
                                    timestamps = pod_df.loc[jump, "time"].astype(str).tolist()
                                    is_anomaly = True
                        
                        # Handle raw bytes/cores if needed (usually less critical for saturation alarms than rate)
                        # We skip raw value spikes here to avoid noise, assuming usage_rate covers saturation.

                    # --- Sub-Rule 3.5: Fallback Generic (Strict) ---
                    # 对于未命名的其他指标（如 I/O, Count等），忽略！
                    # 为了提高 Precision，在 Rule-based 方法中我们只关注具备明确业务含义的黄金指标。
                    else:
                        pass

                    if is_anomaly:
                        svc = self._get_service_name(pod)
                        kpi_events.append({
                            "pod": pod,
                            "service": svc,
                            "kpi": kpi, 
                            "pattern": "waveform_spike",
                            "timestamps": timestamps
                        })

            # 如果检测出太多（超过总数25%）的pod都是outlier，说明这个metric本身在不同服务间差异巨大（如network bytes）
            # 这种情况下，全局outlier检测失效，应忽略该metric
            total_pods_in_kpi = kpi_df["pod"].nunique()
            # NOTICE: 仅统计 mean_outlier 类型的 pod，避免 missing_data 事件被误报为 metric 不可用
            unique_anomalous_pods = set([e["pod"] for e in kpi_events if e["pattern"] == "mean_outlier"])
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
                        "pattern": svc_events[0]["pattern"],
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
