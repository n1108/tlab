import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import IsolationForest

class EnsembleDetector:
    def _calculate_hbos(self, series: pd.Series, bins=10) -> np.ndarray:
        n = len(series)
        if n < 2 or series.nunique() < 2: return np.zeros(n)
        try:
            actual_bins = min(bins, max(2, n // 5))
            hist, bin_edges = np.histogram(series, bins=actual_bins, density=True)
            min_density = np.min(hist[hist > 0]) if np.any(hist > 0) else 1e-10
            hist = np.where(hist == 0, min_density * 0.1, hist)
            bin_indices = np.digitize(series, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, actual_bins - 1)
            scores = np.log(1.0 / hist[bin_indices])
            return scores
        except Exception as e: return np.zeros(n)

    def _calculate_iqr_mask(self, series: pd.Series) -> np.ndarray:
        if len(series) < 5: return np.zeros(len(series), dtype=bool)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: return np.zeros(len(series), dtype=bool)
        
        # Current state: 2.0
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        return ((series < lower_bound) | (series > upper_bound)).values

    def _detect_local_pattern(self, series: pd.Series, anomaly_indices: np.ndarray) -> str:
        if not np.any(anomaly_indices): return "normal"
        mean_val = series.mean()
        anom_indices = np.where(anomaly_indices)[0]
        anom_values = series.iloc[anom_indices]
        anom_mean = anom_values.mean()
        is_high = anom_mean > mean_val
        duration = len(anom_indices)
        last_idx = len(series) - 1
        if is_high:
            if duration == 1: return "spike"
            elif anom_indices[-1] == last_idx: return "level_shift_up"
            else: return "surge"
        else:
            if duration == 1: return "drop"
            elif anom_indices[-1] == last_idx: return "level_shift_down"
            else: return "dip"

    def detect(self, series: pd.Series):
        if len(series) < 5 or series.std() == 0: return {}
        values = series.values.reshape(-1, 1)
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=1)
        iso.fit(values)
        if_scores = iso.decision_function(values)
        hbos_scores = self._calculate_hbos(series)
        iqr_mask = self._calculate_iqr_mask(series)
        anomaly_scores = (if_scores - (0.1 * hbos_scores)) / 2
        
        score_mean = anomaly_scores.mean()
        score_std = anomaly_scores.std()
        dynamic_thresh = score_mean - 3 * score_std
        
        cv = series.std() / (abs(series.mean()) + 1e-10)
        
        # Current logic (hard cap at -0.60)
        final_thresh = min(-0.60, dynamic_thresh)
        
        is_candidate = (anomaly_scores < final_thresh) | iqr_mask
        
        print(f"Std: {series.std()}")
        print(f"Mean: {series.mean()}")
        print(f"CV: {cv}")
        print(f"Scores Mean: {score_mean}, Std: {score_std}")
        print(f"Dynamic Thresh: {dynamic_thresh}")
        print(f"Final Thresh: {final_thresh}")
        print(f"IF Scores: {if_scores}")
        print(f"HBOS Scores: {hbos_scores}")
        print(f"Fusion Scores: {anomaly_scores}")
        print(f"IQR Mask: {iqr_mask}")
        print(f"Candidates: {is_candidate}")
        
        if not np.any(is_candidate): return {}
        ratio = np.sum(is_candidate) / len(series)
        if ratio < 0.02 and not np.any(iqr_mask): return {}
        pattern = self._detect_local_pattern(series, is_candidate)
        return {
            "is_anomaly": True,
            "pattern": pattern,
            "timestamps": series.index[is_candidate].tolist(),
            "max_val": series.max(),
            "mean_val": series.mean()
        }

data = [81.0, 0.0, 7248710.3, 5399.75, 2470.9, 8859168.86, 9265089.7, 0.0, 3497.63, 6634425.27, 0.0, 10620276.51, 7504756.99, 2795.83, 0.0, 8626670.87, 8390551.59, 9454086.23, 9402607.67, 1952.08, 8029490.55]
series = pd.Series(data)
detector = EnsembleDetector()
result = detector.detect(series)
print(f"Result: {result}")
