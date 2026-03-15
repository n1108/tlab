
import json

file_path = "/home/tyt21/tlab/unit-test/metric/results/predictions_rule_based.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    if data:
        entry = data[0]
        print(entry.get("detected_anomalies", [])[:2])
except Exception as e:
    print(e)
