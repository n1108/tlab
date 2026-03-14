import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import IsolationForest

def debug_scores_v4():
    series = pd.Series([2696.53, 17408.0, 3072.0, 2935.47, 3174.4, 2662.4, 1638.4, 1092.27, 1911.47, 1638.4, 546.13, 546.13, 1365.33, 1092.27, 1365.33, 1365.33, 1092.27, 1092.27, 1092.27, 1092.27, 546.13])
    values = series.values.reshape(-1, 1)

    # 1. IF
    iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=1)
    iso.fit(values)
    if_scores = iso.decision_function(values)

    # 2. HBOS
    n = len(series)
    actual_bins = min(10, max(2, n // 5))
    hist, bin_edges = np.histogram(series, bins=actual_bins, density=True)
    min_density = np.min(hist[hist > 0]) if np.any(hist > 0) else 1e-10
    hist = np.where(hist == 0, min_density * 0.1, hist)
    bin_indices = np.digitize(series, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, actual_bins - 1)
    hbos_scores = np.log(1.0 / hist[bin_indices])

    # 3. Fusion
    is_sparse = (series.abs() < 1e-9).mean() > 0.3
    if is_sparse:
        anomaly_scores = if_scores
    else:
        anomaly_scores = (if_scores - (0.1 * hbos_scores)) / 2

    idx = 1 # 17408.0
    print(f"Index {idx} Value: {series[idx]}")
    print(f"IF Score: {if_scores[idx]}")
    print(f"HBOS Score: {hbos_scores[idx]}")
    print(f"Combined Score: {anomaly_scores[idx]}")
    
    # Thresholds
    score_mean = anomaly_scores.mean()
    score_std = anomaly_scores.std()
    cv = series.std() / (abs(series.mean()) + 1e-10)
    
    if cv < 0.05:
        min_std = 0.01
        final_thresh_cap = -0.65
    else:
        min_std = 0.10
        final_thresh_cap = -0.65
        
    actual_score_std = max(score_std, min_std)
    dynamic_thresh = score_mean - 3 * actual_score_std
    final_thresh = min(dynamic_thresh, final_thresh_cap)
    
    print(f"CV: {cv:.4f}")
    print(f"MinStd: {min_std}")
    print(f"Score Mean: {score_mean:.4f}")
    print(f"Score Std (Effective): {actual_score_std:.4f}")
    print(f"Dynamic Thresh: {dynamic_thresh:.4f}")
    print(f"Final Thresh: {final_thresh:.4f}")
    
    anoms = np.where(anomaly_scores < final_thresh)[0]
    print(f"Anomalies Indices (Score < Thresh): {anoms}")

if __name__ == "__main__":
    debug_scores_v4()
