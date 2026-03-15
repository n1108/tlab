import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging

# Add path to project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp.agent.metric import MetricAgent, EnsembleDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATASET_PATH = "/home/tyt21/tlab/unit-test/metric/test_dataset.json"

def load_test_dataset():
    with open(TEST_DATASET_PATH, 'r') as f:
        data = json.load(f)
    return data

def analyze_case(agent, case):
    print(f"\nAnalyzing Case: {case['fault_type']} ({case['uuid']})")
    print(f"Time: {case['start_time']} - {case['end_time']}")
    
    start_time = pd.to_datetime(case['start_time'])
    end_time = pd.to_datetime(case['end_time'])
    
    # Load data
    df = agent.load_data(start_time, end_time)
    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} rows of data.")
    
    expected_metrics = set(case['expected_anomalies'])
    print(f"Expected Anomalies: {expected_metrics}")

    # Group by Pod and KPI
    grouped = df.groupby(['pod', 'kpi_key'])
    
    detected = []
    missed_details = []

    for (pod, kpi), group in grouped:
        try:
            series = group.set_index('time')['value'].sort_index()
            series = series.resample('1min').max().fillna(0)
        except Exception:
            continue
            
        # Run detection
        # We need to manually call detector to get raw results before filtering
        kpi_pattern = agent.patterns.get(kpi, None)
        result = agent.detector.detect(series, pattern=kpi_pattern)
        
        is_expected = kpi in expected_metrics
        # Note: 'expected_anomalies' in json lists metric names (kpi), not pod-metric pairs usually.
        # But groundtruth says "key_metrics": ["pod_cpu_usage"]...
        # The evaluation usually matches metric name.
        
        if result:
            score = result.get('score', 0)
            detected.append({'pod': pod, 'kpi': kpi, 'score': score, 'pattern': result['pattern']})
            if is_expected:
                print(f"[MATCH] Detected expected {kpi} on {pod}. Score: {score:.2f}, Pattern: {result['pattern']}")
            else:
                # Potential False Positive (or just uncited anomaly)
                pass
        else:
            if is_expected:
                # This is a MISS (False Negative)
                # But wait, expected anomalies are for the Faulty component.
                # 'root_cause_components' list the faulty components.
                rc_components = case['root_cause_components']
                
                # Check if this pod is one of the RC components
                # The pod name in data might contain the component name.
                is_rc = False
                for rc in rc_components:
                    if rc in pod:
                        is_rc = True
                        break
                
                if is_rc:
                    print(f"[MISS] Missed expected {kpi} on RC {pod}.")
                    # Analyze WHY
                    missed_details.append({
                        'pod': pod, 'kpi': kpi,
                        'series': series,
                        'pattern': kpi_pattern
                    })

    # Deep dive into misses
    if missed_details:
        print("\n--- Deep Dive into Misses ---")
        for m in missed_details[:3]: # Look at first 3 misses
            s = m['series']
            print(f"Metric: {m['kpi']} on {m['pod']}")
            print(f"Stats: Mean={s.mean():.4f}, Std={s.std():.4f}, Max={s.max():.4f}")
            if m['pattern']:
                print(f"Pattern Stats: {m['pattern']}")
            else:
                print("No pattern info found.")
                
            # Manually run detector steps to see where it fails
            detector = agent.detector
            values = s.values.reshape(-1, 1)
            
            # 1. IF
            detector.detector = EnsembleDetector() # Reset? No need.
            iso = detector.detector.detect(s) # Wait, I can't easily access internal steps of `detect` without modifying code.
            # But I can guess.
            # If std is low, maybe threshold.
            # If std is high, maybe hidden.
            print(f"Values: {s.values.tolist()}")
            print("-" * 20)

def main():
    agent = MetricAgent("/home/tyt21/tlab/dataset")
    data = load_test_dataset()
    
    # Pick a few cases
    # Case 0: CPU stress
    analyze_case(agent, data[0])
    
    # Case 1: Network corrupt
    if len(data) > 1:
        analyze_case(agent, data[1])

if __name__ == "__main__":
    main()
