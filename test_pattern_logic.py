import pandas as pd
import numpy as np
import sys
import os

# Add path to load MetricAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp.agent.metric import EnsembleDetector

detector = EnsembleDetector()

# Test Case 1: High Volatility Metric (Mocking 'rrt' with CV=9.5)
# A series with CV=2.0 (High, but "Normal" for this metric) should NOT be flagged if we tell it Normal CV=9.5
# If we don't tell it, it sees CV=2.0 > 1.0, relaxes threshold, might pass.
# Let's create a SPIKE that increases local CV to 5.0. 
# If normal CV=9.5, this spike (CV=5) is actually LESS volatile than normal? 
# Wait, spike usually increases MAX.
# Let's say normal RRT is [0, 100, 0, 50, 0].
# Anomaly RRT is [0, 100, 5000, 50, 0].
series = pd.Series([10.0, 12.0, 10.0, 11.0, 500.0, 10.0, 12.0]) # Clear spike 500.
# Without pattern:
print("--- No Pattern ---")
res = detector.detect(series)
print(f"Result: {bool(res)}") # Expected: True

# With pattern: Normal Max is 1000. So 500 is normal.
pattern_robust = {"cv": 2.0, "max": 1000.0, "zero_rate": 0.0, "unique_rate": 0.5}
print("--- With High Normal Max (1000) ---")
res2 = detector.detect(series, pattern_robust)
print(f"Result: {bool(res2)}") # Expected: False (500 < 1000*1.2)

# With pattern: Normal Max is 50. So 500 is anomaly.
pattern_sensitive = {"cv": 0.1, "max": 50.0, "zero_rate": 0.0, "unique_rate": 0.5}
print("--- With Low Normal Max (50) ---")
res3 = detector.detect(series, pattern_sensitive)
print(f"Result: {bool(res3)}") # Expected: True

# Test Case 2: Sparse Metric (Mocking 'error_count')
# Series: [0, 0, 1, 0, 0]. 1 is small.
series_sparse = pd.Series([0, 0, 1, 0, 0, 0, 0])
# Without pattern: 1 might be flagged if distribution says so (IF).
# But relative deviation is inf (median=0).
print("\n--- Sparse Series (1) ---")
res4 = detector.detect(series_sparse)
print(f"No Pattern Result: {bool(res4)}")

# With pattern: Normal Max is 5. So 1 is normal.
pattern_sparse_ok = {"cv": 2.0, "max": 5.0, "zero_rate": 0.8, "unique_rate": 0.1}
res5 = detector.detect(series_sparse, pattern_sparse_ok)
print(f"With Normal Max=5 Result: {bool(res5)}") # Expected: False

# With pattern: Normal Max is 0. So 1 is anomaly.
pattern_sparse_bad = {"cv": 2.0, "max": 0.0, "zero_rate": 0.99, "unique_rate": 0.01}
res6 = detector.detect(series_sparse, pattern_sparse_bad)
print(f"With Normal Max=0 Result: {bool(res6)}") # Expected: True
