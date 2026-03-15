
import json
import pandas as pd
from collections import Counter

file_path = "/home/tyt21/tlab/unit-test/metric/results/predictions_rule_based.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    patterns = Counter()
    kpis = Counter()
    services = Counter()
    
    if len(data) > 0:
        print(f"\nFirst entry structure keys: {data[0].keys()}")
        if "events" in data[0]:
            print(f"First entry events count: {len(data[0]['events'])}")
            if len(data[0]['events']) > 0:
                print(f"First event: {data[0]['events'][0]}")
    
    for entry in data:
        # Check if entry has undetected anomalies too
        anomalies = entry.get("detected_anomalies", [])
        if anomalies:
             # Look for pattern field inside detected anomalies
             for anomaly in anomalies:
                 p = anomaly.get("pattern", "unknown")
                 k = anomaly.get("metric_name", "unknown") 
                 if k == "unknown" and "kpi" in anomaly: k = anomaly["kpi"]
                 s = anomaly.get("component_name", "unknown")
                 if s == "unknown" and "service" in anomaly: s = anomaly["service"]
                 if s == "unknown" and "pod" in anomaly: s = anomaly["pod"]

                 patterns[p] += 1
                 kpis[k] += 1
                 services[s] += 1
            
    print("\nTop 5 Detected Patterns:")
    for p, c in patterns.most_common(5):
        print(f"  {p}: {c}")

    print("\nTop 10 Detected KPIs:")
    for k, c in kpis.most_common(10):
        print(f"  {k}: {c}")
        
    print("\nTop 5 Detected Services:")
    for s, c in services.most_common(5):
        print(f"  {s}: {c}")

except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print(f"Error: {e}")
