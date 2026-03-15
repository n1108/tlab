import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exp.agent.metric import EnsembleDetector

def reproduce():
    # Case provided by user
    # 345fbe93-80, currencyservice-0, pod_cpu_usage
    series = pd.Series([0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01])
    
    print(f"Stats: Max={series.max()}, Min={series.min()}, Mean={series.mean()}, Median={series.median()}, Std={series.std()}")
    
    detector = EnsembleDetector()
    result = detector.detect(series)
    
    print(f"Result: {result}")
    
    if result and result.get("is_anomaly", False):
        print("FAIL: Anomaly detected (False Positive)")
    else:
        print("PASS: No anomaly detected")

if __name__ == "__main__":
    reproduce()
