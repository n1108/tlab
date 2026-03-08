import json
import os
import sys

from datetime import datetime

# Configurations
INPUT_PATH = "/home/tyt21/tlab/dataset/input.json"
GT_PATH = "/home/tyt21/tlab/dataset/groundtruth.jsonl"
OUTPUT_DIR = "/home/tyt21/tlab/unit-test/metric"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_dataset.json")

def load_json(path):
    print(f"Loading {path}...")
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def load_jsonl(path):
    print(f"Loading {path}...")
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return data

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset files
    inputs = load_json(INPUT_PATH)
    gts = load_jsonl(GT_PATH)
    
    # Create a lookup map for groundtruth based on UUID
    gt_map = {item['uuid']: item for item in gts if 'uuid' in item}
    
    test_cases = []
    
    print(f"Processing {len(inputs)} input entries...")
    
    for inp in inputs:
        uuid = inp.get('uuid')
        if not uuid:
            continue
            
        # Find corresponding groundtruth
        gt = gt_map.get(uuid)
        
        if not gt:
            print(f"Warning: No groundtruth found for UUID {uuid}")
            continue
            
        # Extract time window (Prefer GT as it's the source of truth for the fault window)
        start_time = gt.get("start_time")
        end_time = gt.get("end_time")
        
        # Extract expected instances (The root cause components)
        # Convert to list if it's a string, or keep as list
        instances = gt.get("instance", [])
        if isinstance(instances, str):
            instances = [instances]
        
        services = gt.get("service", [])
        if isinstance(services, str):
            services = [services]
            
        # Extract expected metrics (The 'Answer')
        expected_metrics = set()
        
        # 1. From 'key_metrics' field
        if "key_metrics" in gt:
            for m in gt["key_metrics"]:
                expected_metrics.add(m)
                
        # 2. From 'key_observations' field where type is 'metric'
        if "key_observations" in gt:
            for obs in gt["key_observations"]:
                if obs.get("type") == "metric":
                    # 'keyword' is usually a list of metric names
                    keywords = obs.get("keyword", [])
                    if isinstance(keywords, list):
                        for k in keywords:
                            expected_metrics.add(k)
                    elif isinstance(keywords, str):
                        expected_metrics.add(keywords)

        # Construct the test case object
        # Deduplicate components and filter out empty strings
        all_components = set()
        if isinstance(instances, list):
            all_components.update([x for x in instances if x])
        if isinstance(services, list):
            all_components.update([x for x in services if x])
            
        test_case = {
            "uuid": uuid,
            "start_time": start_time,
            "end_time": end_time,
            "fault_type": gt.get("fault_type", "unknown"),
            "root_cause_components": sorted(list(all_components)),
            "expected_anomalies": sorted(list(expected_metrics))
        }
        
        test_cases.append(test_case)
        
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(test_cases, f, indent=4)
        
    print(f"Successfully generated {len(test_cases)} test cases in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# python3 unit-test/metric/generate_test_data.py