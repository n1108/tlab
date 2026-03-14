import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[3]
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

logger = logging.getLogger(__name__)

class RuleBasedHighPrecisionAgent(MetricAgent):
    """
    High-Precision Rule-based metric anomaly detection.
    V7.1: Hybrid Filtering with Aggregator Exclusion (Corrected).
    """
    
    def __init__(self, root_path):
        super().__init__(root_path)

    def _get_service_name(self, pod_name):
        if not isinstance(pod_name, str): return "unknown"
        parts = pod_name.rsplit('-', 1)
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]
        return pod_name

    def _post_process_events(self, events):
        def get_val(e): return e.get("max_value", 0)

        # 1. Separation
        resource_events = [e for e in events if e["pattern"] in ["cpu_saturation", "process_overflow"]]
        other_events = [e for e in events if e["pattern"] not in ["cpu_saturation", "process_overflow"]]
        
        # Rule 1: Resource Priority
        if resource_events:
            resource_services = set(self._get_service_name(e["pod"]) for e in resource_events)
            final_events = resource_events[:]
            for ev in other_events:
                if self._get_service_name(ev["pod"]) in resource_services:
                    final_events.append(ev)
            return final_events

        # Rule 2: Score-Based Filtering
        if not other_events: return []
        
        service_stats = {} 
        global_max_rrt = 0.0
        global_max_error = 0.0
        
        for ev in other_events:
            svc = self._get_service_name(ev["pod"])
            val = get_val(ev)
            if svc not in service_stats:
                service_stats[svc] = {'rrt': 0.0, 'error': 0.0, 'events': []}
            service_stats[svc]['events'].append(ev)
            
            if ev["kpi"] in ["rrt", "rrt_max"]:
                if val > service_stats[svc]['rrt']: service_stats[svc]['rrt'] = val
            elif "error" in ev["kpi"]:
                if val > service_stats[svc]['error']: service_stats[svc]['error'] = val

        for svc, stats in service_stats.items():
            if "frontend" in svc: continue
            if stats['rrt'] > global_max_rrt: global_max_rrt = stats['rrt']
            if stats['error'] > global_max_error: global_max_error = stats['error']
            
        if global_max_rrt == 0: global_max_rrt = 1.0
        if global_max_error == 0: global_max_error = 1.0
        
        final_events = []
        candidates = []
        
        for svc, stats in service_stats.items():
            l_score = stats['rrt'] / global_max_rrt
            e_score = stats['error'] / global_max_error
            combined_score = (l_score + e_score) / 2.0
            
            if combined_score > 0.4:
                candidates.append(svc)
                final_events.extend(stats['events'])

        has_others = any("frontend" not in c for c in candidates)
        if has_others:
            final_events = [e for e in final_events if "frontend" not in self._get_service_name(e["pod"])]

        return final_events

    def query_metrics(self, start_time, end_time):
        adj_start = start_time - pd.Timedelta(minutes=5)
        adj_end = end_time + pd.Timedelta(minutes=5)
        
        df = self.load_data(adj_start, adj_end)
        if df.empty: return {"observation": "No data", "events": []}
            
        events = []
        
        for kpi, kpi_df in df.groupby("kpi_key"):
            try:
                kpi_df['value'] = pd.to_numeric(kpi_df['value'], errors='coerce')
                pivoted = kpi_df.pivot_table(index="time", columns="pod", values="value", aggfunc='max')
                pivoted = pivoted.resample('1min').max()
                if pivoted.empty: continue
            except Exception: continue
            
            pod_to_service = {col: self._get_service_name(col) for col in pivoted.columns}
            services = set(pod_to_service.values())
            
            for service in services:
                service_pods = [p for p in pivoted.columns if pod_to_service[p] == service]
                if not service_pods: continue
                
                service_df = pivoted[service_pods]
                spatial_median = None
                if len(service_pods) > 2:
                    spatial_median = service_df.median(axis=1)
                
                for pod in service_pods:
                    series_window = service_df[pod].loc[start_time:end_time]
                    if series_window.empty: continue
                    
                    anomaly_mask = np.zeros(len(series_window), dtype=bool)
                    pattern_name = None
                    max_val = series_window.max()

                    # 1. CPU
                    if 'pod_cpu_usage' in kpi:
                        mask = (series_window > 0.35)
                        if spatial_median is not None:
                             med = spatial_median.reindex(series_window.index).fillna(0)
                             mask &= ((series_window > 5 * med.replace(0, 1e-6)) | (med > 0.35))
                        if mask.any():
                            anomaly_mask |= mask
                            pattern_name = "cpu_saturation"

                    # 2. Process
                    elif 'pod_process' in kpi:
                        mask = (series_window >= 9)
                        if mask.any():
                            anomaly_mask |= mask
                            pattern_name = "process_overflow"
                            
                    # 3. Errors
                    elif 'error' in kpi: 
                        mask = (series_window > 1.0) 
                        if mask.any():
                            anomaly_mask |= mask
                            pattern_name = "error_spike"

                    # 4. Latency
                    elif 'rrt' in kpi and 'max' not in kpi: 
                        mask = series_window > 20000 
                        if spatial_median is not None:
                             med = spatial_median.reindex(series_window.index).fillna(0).replace(0, 1)
                             mask &= ((series_window > 3 * med) | (med > 20000))
                        if mask.any():
                            anomaly_mask |= mask
                            pattern_name = "latency_spike"
                    
                    # 5. RRT_MAX
                    elif 'rrt_max' in kpi:
                        if "frontend" in service: continue 
                        mask = series_window > 50000 
                        if spatial_median is not None:
                           med = spatial_median.reindex(series_window.index).fillna(0).replace(0, 1)
                           mask &= ((series_window > 5 * med) | (med > 50000))
                        if mask.any():
                            anomaly_mask |= mask
                            pattern_name = "latency_spike"

                    if anomaly_mask.any():
                        ev = {
                            "pod": pod, "kpi": kpi, 
                            "pattern": pattern_name if pattern_name else "anomaly",
                            "timestamps": series_window.index[anomaly_mask].astype(str).tolist(),
                            "max_value": float(max_val)
                        }
                        events.append(ev)

        final_events = self._post_process_events(events)
        return {"observation": "High precision analysis V7.1 (Best Known).", "events": final_events}
