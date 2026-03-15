import pandas as pd
import numpy as np
import sys
import os

# Add path to load MetricAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp.agent.metric import EnsembleDetector

detector = EnsembleDetector()

# Test Case 1: Continuous Metric (e.g. Latency) with "Low" Local Spike but "High" Global Max
# Pattern says Max is 1000.
# Local series is [10, 10, 10]. Anomaly is 500.
# 500 < 1000.
# If filtering was applied, 500 would be ignored.
# But since zero_rate is low, filtering should NOT be applied.
series = pd.Series([10.0, 12.0, 10.0, 11.0, 500.0, 10.0, 12.0]) 
pattern_latency = {"cv": 2.0, "max": 1000.0, "zero_rate": 0.0, "unique_rate": 0.5}
print("--- Continuous Metric Spike (500 vs Max 1000) ---")
res = detector.detect(series, pattern_latency)
print(f"Result (Should be True): {bool(res)}") 

# Test Case 2: Sparse/Event Metric (e.g. Errors)
# Pattern says Max is 1000.
# Local series is [0, 0, 0]. Anomaly is 500.
# 500 < 1000.
# Since zero_rate is high, filtering SHOULD be applied.
series_error = pd.Series([0.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0])
pattern_error = {"cv": 5.0, "max": 1000.0, "zero_rate": 0.9, "unique_rate": 0.01}
print("\n--- Sparse Metric Spike (500 vs Max 1000) ---")
res2 = detector.detect(series_error, pattern_error)
print(f"Result (Should be False): {bool(res2)}") 

# Test Case 3: Drop to Zero in High Variance Metric
# CV is high. Usually drops are inhibited.
# But pattern says zero_rate is low (0.0). So drop to 0 is anomaly.
series_drop = pd.Series([100.0]*10 + [0.0]*3 + [100.0]*5)
pattern_drop = {"cv": 2.0, "max": 200.0, "zero_rate": 0.0, "unique_rate": 0.5} # Low zero rate
print("\n--- High CV Drop to Zero (Zero Rate=0) ---")
res3 = detector.detect(series_drop, pattern_drop)
print(f"Result (Should be True): {bool(res3)}")
