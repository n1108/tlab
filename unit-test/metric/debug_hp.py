import json
import logging
import pandas as pd
import sys

# Load Ground Truth
gt_file = "unit-test/metric/test_dataset.json"
pred_file = "unit-test/metric/results/predictions_hp.json"

with open(gt_file, 'r') as f:
    gts = json.load(f)

# Load Predictions
with open(pred_file, 'r') as f:
    preds = json.load(f)

# Create a map of UUID -> Predictions
pred_map = {p['uuid']: p for p in preds}

# Check the first 5 cases
gts = gts[:5]

for gt in gts:
    uuid = gt['uuid']
    fault_type = gt.get('fault_type', 'unknown')
    expected_metrics = gt.get('expected_anomalies', [])
    root_causes = gt.get('root_cause_components', [])
    expected_components = root_causes # List of strings
    
    print(f"UUID: {uuid} ({fault_type})")
    print(f"  Root Causes: {expected_components}")
    print(f"  Expected Metrics: {expected_metrics}")

    if uuid not in pred_map:
        print("  !!! NO PREDICTIONS !!!")
        continue

    p = pred_map[uuid]
    detected_list = p.get('detected_anomalies', [])
    
    found_metrics = set()
    for d in detected_list:
        metric = d['metric']
        comps = d['component'] if isinstance(d['component'], list) else [d['component']]
        
        # Check component match
        match = False
        for c in comps:
            for rc in root_causes:
                # Match strict or prefix (e.g. cartservice-0 -> cartservice)
                if c == rc or c.startswith(rc + "-"): 
                    match = True
                    break
            if match: break
        
        if match:
            if metric in expected_metrics:
                found_metrics.add(metric)

    print(f"  Found Metrics: {list(found_metrics)}")
    missed = set(expected_metrics) - found_metrics
    print(f"  Missed Metrics: {list(missed)}")
    print("-" * 30)
