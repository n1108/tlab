import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add workspace root to sys.path to import exp.agent.metric
workspace_root = Path(__file__).resolve().parents[3]
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

logger = logging.getLogger(__name__)

class RuleBasedMetricAgent(MetricAgent):
    """
    Rule-based metric anomaly detection baselines.
    Implements specific heuristic rules based on observed patterns in test cases.
    """
    
    def __init__(self, root_path):
        super().__init__(root_path)

    def _detect_pattern_type(self, series, anomaly_mask):
        """
        Classify the anomaly pattern based on shape.
        types: spike, drop, level_shift_up, level_shift_down, surge, dip
        """
        if not anomaly_mask.any():
            return None
            
        anom_indices = np.where(anomaly_mask)[0]
        if len(anom_indices) == 0:
            return None
            
        # Get values
        anom_values = series.iloc[anom_indices]
        normal_mask = ~anomaly_mask
        if normal_mask.any():
            baseline = series[normal_mask].median()
            if np.isnan(baseline): baseline = 0
        else:
            baseline = 0 # Fallback
            
        mean_anom = anom_values.mean()
        is_high = mean_anom > baseline
        
        # Check duration
        duration = len(anom_indices)
        total_len = len(series)
        
        # Check position (start, end)
        starts_at_end = anom_indices[-1] == (total_len - 1)
        
        if is_high:
            if duration <= 2:
                return "spike"
            elif starts_at_end:
                 return "level_shift_up"
            else:
                return "surge"
        else:
            if duration <= 2:
                return "drop"
            elif starts_at_end:
                return "level_shift_down"
            else:
                return "dip"

    def query_metrics(self, start_time, end_time):
        # Extend window by 10 mins to catch edge cases and context
        adj_start = start_time - pd.Timedelta(minutes=10)
        adj_end = end_time + pd.Timedelta(minutes=10)
        
        # Load data
        df = self.load_data(adj_start, adj_end)
        
        if df.empty:
            return {"observation": "No data", "events": []}
            
        events = []
        
        # Group by KPI to process cross-component logic where applicable, or just efficient iteration
        # pivot: index=time, columns=pod, values=value
        for kpi, kpi_df in df.groupby("kpi_key"):
            
            # Pivot
            try:
                pivoted = kpi_df.pivot_table(index="time", columns="pod", values="value", aggfunc='max')
            except Exception:
                continue
                
            # Resample to 1min to align and expose missing data
            # Use 'max' for resampling to preserve spikes
            pivoted = pivoted.resample('1min').max()
            
            # Slice to the requested detection window (plus maybe 1-2 mins context if needed for boundary checks)
            # But strictly we should detect anomalies falling into the requested window.
            # However, for "missing data", we need to check if it's missing INSIDE the window.
            
            # Analyze each component
            for pod in pivoted.columns:
                series_full = pivoted[pod]
                
                # Slice to strict window for reporting
                series_window = series_full.loc[start_time:end_time]
                
                # --- Rule 1: Missing Data (Testcase 3) ---
                if series_window.isna().any():
                     # Only report if it's not empty (completely missing vs partially missing)
                     # If the pod completely doesn't exist in time range, it might just be not running. 
                     # But here we have columns, so it existed at some point in the extended window.
                     mask = series_window.isna()
                     timestamps = series_window.index[mask].astype(str).tolist()
                     events.append({
                         "pod": pod, "kpi": kpi, "pattern": "missing", "timestamps": timestamps
                     })
                     # Fill NaNs for further checks (fill with 0 usually safe for metrics like error/request)
                     series_window = series_window.fillna(0)
                
                if series_window.empty: continue
                
                # Prepare baseline (median of the series itself in extended window or cross-component)
                # Cross-component baseline
                others = pivoted.drop(columns=[pod], errors='ignore')
                if not others.empty:
                    spatial_median = others.median(axis=1).loc[start_time:end_time]
                else:
                    spatial_median = None

                anomaly_mask = np.zeros(len(series_window), dtype=bool)
                
                # --- Rule 2: Error Metrics (Testcase 2, 3) ---
                if any(x in kpi for x in ['error', 'fail']):
                    # Threshold > 0
                    anomaly_mask = series_window > 0
                    
                # --- Rule 3: Pod Process (Testcase 1) ---
                elif kpi == 'pod_processes':
                    # Threshold > 5 (Normal is 1)
                    anomaly_mask = series_window > 5
                    
                # --- Rule 4: Pod CPU (Testcase 1) ---
                elif kpi == 'pod_cpu_usage':
                    # Threshold > 0.2 (User said 0.4 anomaly vs 0.02 normal)
                    anomaly_mask = series_window > 0.2
                    
                # --- Rule 5: Node Memory Usage (Testcase 4) ---
                elif kpi == 'node_memory_usage_rate':
                    # Spike > 50 (Normal 22, Anomaly 68)
                    anomaly_mask = series_window > 50
                    
                # --- Rule 6: Node Memory Available (Testcase 4) ---
                elif kpi == 'node_memory_MemAvailable_bytes':
                    # Dip < 20 GB (Normal 26GB, Anomaly 10GB)
                    # 20 GB = 2e10
                    anomaly_mask = series_window < 2.0e10
                    
                # --- Rule 7: Network Bytes/Packets Drop to 0 (Testcase 2, 5) ---
                elif 'network' in kpi and ('bytes' in kpi or 'packets' in kpi):
                    # Check for drop to 0. 
                    # Only if baseline was not 0? 
                    # Simple heuristic: strictly 0 is suspicious if it wasn't 0 before.
                    # Or compare to cross-component?
                    # User says "Normal components fluctuate, abnormal is 0".
                    # Let's use Cross-Component Contrast: Self is 0 AND Others > 0.
                    if spatial_median is not None:
                        # If self is 0 and median of others is significant (> 100 bytes/packets)
                        is_zero = series_window <= 1e-6
                        others_active = spatial_median > 100
                        anomaly_mask = is_zero & others_active
                    else:
                        # Fallback: just check for 0 if max > 100 (it was active)
                        if series_full.max() > 100:
                            anomaly_mask = series_window <= 1e-6
                            
                # --- Rule 8: RRT / Latency (Testcase 1, 2, 5) ---
                elif 'rrt' in kpi or 'latency' in kpi:
                    # Massive increase.
                    # Testcase 1: 30000 vs 3000.
                    # Testcase 2: 3 orders of magnitude.
                    # Rule: Value > 5 * spatial_median (if available) OR Value > 3 * self_median
                    if spatial_median is not None:
                        # Avoid div by zero
                        safe_median = spatial_median.replace(0, 1) # assuming ms/us
                        ratio = series_window / safe_median
                        anomaly_mask = ratio > 5
                    else:
                        # Self comparison
                        self_median = series_full.median()
                        if self_median > 0:
                            anomaly_mask = series_window > (self_median * 5)
                            
                # --- Rule 9: Request/Response (Throughput) (Testcase 2, 5) ---
                elif kpi in ['request', 'response', 'qps']:
                    if spatial_median is not None:
                        # Compare deviations? Hard because different services have different loads.
                        # Better to match temporal change: Spike or Drop.
                        # Use Z-Score on self_history (extended window)
                        
                        # Calculate rolling stats on full series to get context
                        rolling_mean = series_full.rolling(window=10, min_periods=1, center=True).mean()
                        rolling_std = series_full.rolling(window=10, min_periods=1, center=True).std()
                        
                        # Align to window
                        mu = rolling_mean.loc[start_time:end_time]
                        sig = rolling_std.loc[start_time:end_time]
                        
                        # Threshold: 3 sigma
                        z_score = (series_window - mu) / (sig + 1e-6) # avoid div 0
                        
                        anomaly_mask = z_score.abs() > 3
                        
                        # Refinement for Testcase 2 (Spike 80->400) -> Z score approx (400-80)/std. High.
                        # Refinement for Testcase 5 (Drop) -> Z score negative.

                # --- Final: Classify and Report ---
                if anomaly_mask.any():
                    pattern = self._detect_pattern_type(series_window, anomaly_mask)
                    if pattern:
                        # Additional Check: Cross Component Contrast (as requested pattern name)
                        # If the anomaly is defined by difference from others, we can tag it.
                        # But user listed it as a pattern name. 
                        # Only add if we used spatial logic? Let's just stick to shape patterns (+ missing)
                        
                        # Filter timestamps
                        timestamps = series_window.index[anomaly_mask].astype(str).tolist()
                        events.append({
                            "pod": pod,
                            "kpi": kpi,
                            "pattern": pattern,
                            "timestamps": timestamps
                        })

        return {
            "observation": "Rule-based analysis completed.",
            "events": events
        }
