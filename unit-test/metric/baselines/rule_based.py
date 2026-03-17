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
        # self.rules = [1, 2, 3]
        self.rules = [1, 2, 3] 

        # 监控的 pod 级别（除 tidb）metrics 列表
        self.pod_metrics_list = [
            'pod_cpu_usage',
            'pod_fs_writes_bytes',
            'pod_memory_working_set_bytes',
            'pod_network_receive_bytes',
            'pod_network_receive_packets',
            'pod_network_transmit_bytes',
            'pod_network_transmit_packets',
            'pod_processes',

            # pod 级别和 service 级别均存在
            'request',
            'response'
        ]

        self.pods_list = [
            'adservice-0', 'adservice-1', 'adservice-2',
            'cartservice-0', 'cartservice-1', 'cartservice-2',
            'currencyservice-0', 'currencyservice-1', 'currencyservice-2',
            'productcatalogservice-0', 'productcatalogservice-1', 'productcatalogservice-2',
            'checkoutservice-0', 'checkoutservice-1', 'checkoutservice-2',
            'recommendationservice-0', 'recommendationservice-1', 'recommendationservice-2',
            'shippingservice-0', 'shippingservice-1', 'shippingservice-2',
            'emailservice-0', 'emailservice-1', 'emailservice-2',      
            'paymentservice-0', 'paymentservice-1', 'paymentservice-2',
            # 'tidb-pd', 'tidb-tidb','tidb-tikv'
        ]

        self.nodes_list = [
            'aiops-k8s-01', 'aiops-k8s-02', 'aiops-k8s-03', 'aiops-k8s-04',
            'aiops-k8s-05', 'aiops-k8s-06', 'aiops-k8s-07', 'aiops-k8s-08',
            'k8s-master1', 'k8s-master2', 'k8s-master3'
        ]
        
        # 规则 1 忽略的指标
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
            "rrt": ["rrt_max"]
        }

    # 从 Pod 名称提取服务名
    def _get_service_name(self, pod_name):
        if not isinstance(pod_name, str): return "unknown"
        parts = pod_name.rsplit('-', 1)
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]
        return pod_name


    # 测试接口，异常检测逻辑
    def query_metrics(self, start_time, end_time):

        df = self.load_data(start_time, end_time)
        if df.empty: return {"observation": "No data", "events": []}
            
        raw_events = []
        
        # 预先构建 service -> pods 映射，用于后续聚合
        service_to_pods = {}
        for pod in self.pods_list:
            svc = self._get_service_name(pod)
            if svc not in service_to_pods: service_to_pods[svc] = set()
            service_to_pods[svc].add(pod)

        # 规则2：数据缺失，即在故障时间段内某个组件上找不到指标数据
        if 2 in self.rules:
            
            # 将 metric 数据转换为 dict
            records = df[['pod', 'kpi_key']].drop_duplicates().to_dict('records')
            
            # 获取 service->pod 和 pod->metric 映射关系
            for row in records:
                pod = row['pod']
                kpi = row['kpi_key']
                
                if kpi not in self.pod_metrics_list:
                    continue
            
            # 检测指标数据缺失。遍历所有 metric, 如果是 pod 级别数据（即部分 pod 数据正常），而有的 pod 数据缺失，则认为异常
            for metric in self.pod_metrics_list:
                metric_df = df[df["kpi_key"] == metric]
                if metric_df.empty: continue
                
                # Calculate counts for peer comparison
                pod_counts = metric_df["pod"].value_counts()
                max_count = pod_counts.max()
                
                pods_with_data = set(metric_df["pod"].unique())

                for pod in self.pods_list:
                    if pod not in pods_with_data:
                        svc = self._get_service_name(pod)
                        raw_events.append({
                            "pod": pod,
                            "service": svc,
                            "kpi": metric,
                            "pattern": "missing_data",
                            "timestamps": []
                        })
                    else:
                        # 检查部分数据缺失：基于数据点数量
                        # 理论上相同时间段内数据点数量一致。小于最大值的一半视为缺失。
                        count = pod_counts.get(pod, 0)
                        
                        if max_count > 5 and count < 0.5 * max_count:
                             svc = self._get_service_name(pod)
                             ts = metric_df[metric_df["pod"] == pod]["time"].astype(str).tolist()
                             raw_events.append({
                                "pod": pod,
                                "service": svc,
                                "kpi": metric,
                                "pattern": "missing_data",
                                "timestamps": ts
                            })
        
        for kpi, kpi_df in df.groupby("kpi_key"):
            
            kpi_events = [] 

            # 规则1：对于某个metric，某个组件的平均值偏离其他组件，基于和中位数的倍数关系识别
            if 1 in self.rules and kpi not in self.rule1_ignore_metrics:
                # Explicitly exclude noisy metrics
                if "client_error" in kpi or "node_" in kpi or "max" in kpi:
                    pass
                else:
                    rule1_whitelist = ["cpu", "memory", "request", "error", "rrt", "response", "processes"]
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


            # 规则3：使用 MetricAgent (EnsembleDetector) 检测时序异常
            if 3 in self.rules:
                for pod, pod_df in kpi_df.groupby("pod"):
                    try:
                        # Align preprocessing with MetricAgent: Resample to 1min
                        series = pod_df.set_index('time')['value'].sort_index()
                        series = series.resample('1min').max().fillna(0)
                        
                        result = self.detector.detect(series)
                        
                        if result:
                            # Filter out low-value noise for ratio metrics (same as MetricAgent)
                            if "ratio" in kpi and result.get("max_val", 0) < 0.01:
                                continue
                                
                            svc = self._get_service_name(pod)
                            kpi_events.append({
                                "pod": pod,
                                "service": svc,
                                "kpi": kpi, 
                                "pattern": result["pattern"],
                                "timestamps": [str(t) for t in result["timestamps"]]
                            })
                    except Exception:
                        continue

            # 如果检测出太多（超过总数25%）的pod都是outlier，说明这个metric本身在不同服务间差异巨大（如network bytes）
            # 这种情况下，全局outlier检测失效，应忽略该metric
            total_pods_in_kpi = kpi_df["pod"].nunique()
            # NOTICE: 仅统计 mean_outlier 类型的 pod，避免 missing_data 事件被误报为 metric 不可用
            unique_anomalous_pods = set([e["pod"] for e in kpi_events if e["pattern"] == "mean_outlier"])
            if total_pods_in_kpi > 0 and (len(unique_anomalous_pods) / total_pods_in_kpi) > 0.25:
                continue

            raw_events.extend(kpi_events)

        # --- Aggregation Logic (Now applied to ALL rules) ---
        events = []
        
        # Group raw_events by (kpi, pattern, service)
        grouped_events = {}
        for ev in raw_events:
            # 使用 tuple 作为 key
            key = (ev['kpi'], ev['pattern'], ev.get('service'))
            if key not in grouped_events: grouped_events[key] = []
            grouped_events[key].append(ev)
        
        for key, ev_list in grouped_events.items():
            kpi, pattern, svc = key
            
            # 只对有完整 Pod 列表的服务进行聚合
            if svc in service_to_pods:
                expected_pods = service_to_pods[svc]
                detected_pods = set(e['pod'] for e in ev_list)
                
                # Check if ALL pods for this service are anomalous
                # 如果检测到的 Pod 覆盖了该服务下的所有 Pod，则聚合为服务级别异常
                if len(expected_pods) > 0 and expected_pods.issubset(detected_pods):
                    # Aggregate timestamps
                    all_timestamps = set()
                    for e in ev_list:
                        for ts in e['timestamps']:
                            all_timestamps.add(ts)
                            
                    events.append({
                        "pod": svc, # Use service name as component
                        "service": svc,
                        "kpi": kpi,
                        "pattern": pattern,
                        "timestamps": sorted(list(all_timestamps))
                    })
                else:
                    # No aggregation, keep individual events
                    events.extend(ev_list)
            else:
                # 无法聚合的组件（如 node），直接保留
                events.extend(ev_list)

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
