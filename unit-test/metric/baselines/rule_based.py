import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parents[3]
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

logger = logging.getLogger(__name__)

class RuleBasedMetricAgent(MetricAgent):
    """
    Rule-based metric anomaly detection (Dynamic Semantic Version).
    
    Philosophy:
    - Anomalies are relative (Spatial or Temporal), not absolute numbers.
    - Spatial: "I am significantly different from my peers." (e.g., CPU Stress, Net Drop)
    - Temporal: "I have changed significantly from my past." (e.g., Node Memory Spike)
    """
    
    def __init__(self, root_path):
        super().__init__(root_path)

    def _calculate_spatial_stats(self, df_pivot):
        """
        Calculate robust spatial statistics: Median and MAD (Median Absolute Deviation).
        """
        if df_pivot.shape[1] < 2:
            return None, None
            
        median = df_pivot.median(axis=1)
        # MAD = median(|x - median|)
        # Constant 1.4826 makes MAD consistent with StdDev for normal distribution
        mad = (df_pivot.sub(median, axis=0)).abs().median(axis=1) * 1.4826
        return median, mad

    def _detect_pattern_type(self, series, anomaly_mask):
        """Analyze the shape of the anomaly."""
        if not anomaly_mask.any(): return None
        
        # Simple shape detection
        anom_vals = series[anomaly_mask]
        duration = len(anom_vals)
        
        if duration <= 2:
            return "spike"
        elif duration > len(series) * 0.8:
            anom_mean = anom_vals.mean()
            # If mask is mostly true, compare to non-existent baseline? 
            # Assume trend.
            return "level_shift"
        else:
             return "surge" 

    def query_metrics(self, start_time, end_time):
        adj_start = start_time - pd.Timedelta(minutes=10)
        adj_end = end_time + pd.Timedelta(minutes=10)
        
        df = self.load_data(adj_start, adj_end)
        if df.empty: return {"observation": "No data", "events": []}
            
        events = []
        
        for kpi, kpi_df in df.groupby("kpi_key"):
            try:
                # pivot and resample
                kpi_df['value'] = pd.to_numeric(kpi_df['value'], errors='coerce')
                pivoted = kpi_df.pivot_table(index="time", columns="pod", values="value", aggfunc='max')
                pivoted = pivoted.resample('1min').max() # 1min Granularity
                
                if pivoted.empty: continue
            except Exception: continue
            
            # --- 1. Spatial Analysis (Cross-Component) ---
            spatial_median, spatial_mad = self._calculate_spatial_stats(pivoted)
            
            for pod in pivoted.columns:
                series_full = pivoted[pod]
                try:
                    # Target window
                    series_window = series_full.loc[start_time:end_time]
                except KeyError: continue
                
                if series_window.empty: continue
                anomaly_mask = np.zeros(len(series_window), dtype=bool)
                pattern_name = None
                
                # --- Strategy A: Missing Data ---
                if series_window.isna().any():
                     # If it's a critical resource metric
                     if any(x in kpi for x in ['cpu', 'memory', 'disk', 'net', 'request']):
                         events.append({
                             "pod": pod, "kpi": kpi, "pattern": "missing",
                             "timestamps": series_window.index[series_window.isna()].astype(str).tolist()
                         })
                     series_window = series_window.fillna(0)

                # --- Strategy B: Absolute Errors (Domain Knowledge) ---
                # Errors are always bad. No relative logic needed.
                if any(x in kpi for x in ['error', 'fail']):
                     mask = series_window > 0
                     if mask.any():
                         anomaly_mask |= mask
                         pattern_name = "error"

                # --- Strategy C: Spatial Outlier (The "High Contrast" Logic) ---
                # Used for: CPU Stress, High Latency, High Process Count
                # Logic: Current Value > Median + 3 * MAD (Robust Z-Score > 3)
                elif spatial_median is not None:
                     # Align stats to window
                     med = spatial_median.reindex(series_window.index).fillna(0)
                     mad_val = spatial_mad.reindex(series_window.index).fillna(0).replace(0, 1e-6) # prevent div0
                     
                     # Calculate Robust Z-Score (Modified Z-Score)
                     # diff = series_window - med
                     # mod_z = 0.6745 * diff / mad (approx) -> Let's use standard deviation scale
                     # We scaled MAD by 1.4826 already, so it acts like Sigma.
                     
                     dev = series_window - med
                     z_spatial = dev / mad_val
                     
                     # Threshold: 3.5 Sigma is strong outlier
                     # Also add a "Min Difference" check to avoid noise amplification (e.g. 0.001 vs 0.002)
                     # Min diff: 1.0 for large numbers, 0.05 for small ratios like CPU?
                     # A generic way is: Ratio > 3x AND Z > 3
                     
                     # 1. High Outlier (Stress/Spike)
                     mask_high = (z_spatial > 3.0) & (series_window > med * 2) & (series_window > 0.05)
                     if mask_high.any():
                         anomaly_mask |= mask_high
                         pattern_name = "cross-component contrast"

                     # 2. Low Outlier (Drop/Dead)
                     # Used for: Network Drop, Memory Available Drop
                     # Logic: Value is near 0, while Median is healthy
                     # E.g. Net Drop: Val < 1, Median > 100
                     mask_low = (series_window < med * 0.1) & (med > 0.1) & (series_window < 1.0)
                     # Special handler for large numbers (Bytes) vs small (CPU)
                     # For network bytes (usually > 1000):
                     if 'byte' in kpi or 'packet' in kpi:
                          mask_low = (series_window < 10) & (med > 100)
                     
                     if mask_low.any():
                         anomaly_mask |= mask_low
                         pattern_name = "drop" # or cross-component contrast

                # --- Strategy D: Temporal Outlier (Self-Z-Score) ---
                # Used for: Node Memory (Singleton), Global Issues (All pods spike)
                # If Spatial failed (only 1 pod, or all pods bad)
                if not anomaly_mask.any():
                     rolling_mean = series_full.rolling(window=20, min_periods=1, center=True).mean()
                     rolling_std = series_full.rolling(window=20, min_periods=1, center=True).std()
                     mu = rolling_mean.loc[start_time:end_time]
                     sig = rolling_std.loc[start_time:end_time].replace(0, 1e-6)
                     
                     z_temporal = (series_window - mu) / sig
                     
                     # Temporal Spike (e.g. Memory Leak)
                     if (z_temporal > 3.0).any(): 
                         # Filter minor noise
                         if (series_window.mean() > 0.1): # Don't flag 0->0.01 spikes
                             anomaly_mask |= (z_temporal > 3.0)
                             pattern_name = "spike" # or surge/level_shift
                     
                     # Temporal Drop
                     if (z_temporal < -3.0).any():
                          mask_drop = (z_temporal < -3.0)
                          if mask_drop.any():
                              anomaly_mask |= mask_drop
                              pattern_name = "drop"

                # --- Reporting ---
                if anomaly_mask.any():
                    # Fallback pattern name analysis
                    if not pattern_name:
                        pattern_name = self._detect_pattern_type(series_window, anomaly_mask)
                        
                    events.append({
                        "pod": pod, "kpi": kpi, "pattern": pattern_name or "anomaly",
                        "timestamps": series_window.index[anomaly_mask].astype(str).tolist()
                    })

        return {"observation": "Dynamic rule-based analysis.", "events": events}
