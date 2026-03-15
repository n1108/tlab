
import json
from collections import Counter

file_path = "/home/tyt21/tlab/unit-test/metric/results/predictions_rule_based.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    patterns = Counter()
    kpis = Counter()
    services = Counter()
    
    for entry in data:
        for anomaly in entry.get("detected_anomalies", []):
            # Handle pattern list
            # In debug output: [{'pattern': ['waveform_spike'], ...}]
            p_obj = anomaly.get("pattern", ["unknown"])
            if isinstance(p_obj, list):
                for p in p_obj: patterns[str(p)] += 1
            else:
                patterns[str(p_obj)] += 1
            
            # Handle metric
            k = anomaly.get("metric", "unknown")
            kpis[str(k)] += 1
            
            # Handle component list
            # In debug output: [{'component': ['adservice', 'adservice-0'], ...}]
            c_obj = anomaly.get("component")
            if isinstance(c_obj, list) and len(c_obj) > 0:
                s = str(c_obj[0]) # Service
            else:
                s = str(c_obj)
            services[s] += 1
            
    print("\nAll Detected Patterns:")
    for p, c in patterns.most_common():
        print(f"  {p}: {c}")

    print("\nTop 10 Detected KPIs:")
    for k, c in kpis.most_common(10):
        print(f"  {k}: {c}")
        
    print("\nTop 10 Detected Services:")
    for s, c in services.most_common(10):
        print(f"  {s}: {c}")

except FileNotFoundError:
    print("File not found.")
except Exception as e:
    import traceback
    traceback.print_exc()
