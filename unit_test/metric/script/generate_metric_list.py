import os
import glob
import re
import json
import pandas as pd
from pathlib import Path

# Config
DATASET_ROOT = "dataset"
OUTPUT_FILE = "unit-test/metric/metric_list.json"

def find_metric_root(root=DATASET_ROOT):
    """Find the first available date directory containing metric-parquet."""
    if not os.path.exists(root):
        print(f"Error: {root} not found.")
        return None
    
    # Sort directories to pick the earliest date or just any valid date
    # Look for YYYY-MM-DD
    dirs = [d for d in os.listdir(root) if re.match(r'\d{4}-\d{2}-\d{2}', d)]
    dirs.sort()
    
    for d in dirs:
        path = os.path.join(root, d, "metric-parquet")
        if os.path.exists(path):
            return path
            
    print("Error: No valid metric-parquet directory found in date folders.")
    return None

def extract_cols_from_parquet(file_path):
    if not file_path:
        return set()
    try:
        # Just need columns, use pyarrow to peek or pandas with nrows=0
        df = pd.read_parquet(file_path) # nrows=0 doesn't work well with pyarrow engine sometimes, just read whole file (it's small usually)
        return set(df.columns)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def extract_apm_metrics(folder_path):
    metrics = set()
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        return metrics
    
    # Check the first file as schema representative
    cols = extract_cols_from_parquet(files[0])
    
    # Filter out non-metric columns
    ignore = {'time', 'object_id', 'object_type', 'instance', 'kubernetes_node', 'pod', 'timestamp', 'kpi_key', 'value', 'service'}
    
    # Also verify if 'value' column exists - if so, it's long format, metrics are in 'kpi_key' column values
    if 'value' in cols and 'kpi_key' in cols:
        # It's long format! We need to read distinct values of kpi_key
        # But wait, looking at user context before: APM Service had ['time', 'client_error', 'request', ...]
        # So it is WIDE format.
        # But let's be safe.
        pass
        
    filtered = {c for c in cols if c not in ignore}
    metrics.update(filtered)
    
    return metrics

def extract_infra_metrics(folder_path, prefix):
    metrics = set()
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    # Helper to clean metric name from filename
    # e.g. infra_node_cpu_usage_rate_2025-06-06.parquet -> cpu_usage_rate
    # prefix is like "infra_node"
    
    for f in files:
        fname = os.path.basename(f)
        # Remove date and extension
        # pattern: {prefix}_{metric}_{date}.parquet
        # Date pattern: YYYY-MM-DD
        
        # Split by underscore, but metric name can contain underscores.
        # Strategy: Remove prefix from start, remove date+.parquet from end.
        
        name_part = fname
        if name_part.startswith(prefix + "_"):
            name_part = name_part[len(prefix)+1:]
            
        # Remove date suffix
        # Find last occurrence of _YYYY-MM-DD.parquet
        match = re.search(r'_(\d{4}-\d{2}-\d{2})\.parquet$', name_part)
        if match:
            metric_name = name_part[:match.start()]
            metrics.add(metric_name)
            
    return metrics

def main():
    root = find_metric_root()
    if not root:
        return

    print(f"Scanning metrics from: {root}")
    
    all_metrics = [] # list of dicts

    # 1. APM Service
    path = os.path.join(root, "apm", "service")
    ms = extract_apm_metrics(path)
    for m in ms:
        all_metrics.append({"metric": m, "level": "service"})
        
    # 2. APM Pod
    path = os.path.join(root, "apm", "pod")
    ms = extract_apm_metrics(path)
    for m in ms:
        all_metrics.append({"metric": m, "level": "pod"})
        
    # 3. Infra Node
    path = os.path.join(root, "infra", "infra_node")
    ms = extract_infra_metrics(path, "infra_node")
    for m in ms:
        all_metrics.append({"metric": m, "level": "node"})

    # 4. Infra Pod
    path = os.path.join(root, "infra", "infra_pod")
    ms = extract_infra_metrics(path, "infra_pod")
    for m in ms:
        all_metrics.append({"metric": m, "level": "pod"})
        
    # 5. Infra TiDB (treat as Pod)
    path = os.path.join(root, "infra", "infra_tidb")
    # Filenames: infra_tidb_<metric>...
    ms = extract_infra_metrics(path, "infra_tidb")
    for m in ms:
        all_metrics.append({"metric": m, "level": "tidb"})
        
    # 6. Other (PD/TiKV) (treat as Pod)
    path = os.path.join(root, "other")
    # These rely on infra_pd_ or infra_tikv_ prefixes
    ms_pd = extract_infra_metrics(path, "infra_pd")
    for m in ms_pd:
        all_metrics.append({"metric": m, "level": "tidb"})
        
    ms_tikv = extract_infra_metrics(path, "infra_tikv")
    for m in ms_tikv:
        all_metrics.append({"metric": m, "level": "tidb"})
        
    # Deduplicate: (metric, level)
    unique_set = set()
    final_list = []
    
    for item in all_metrics:
        key = (item['metric'], item['level'])
        if key not in unique_set:
            unique_set.add(key)
            final_list.append(item)
            
    # Sort
    final_list.sort(key=lambda x: (x['level'], x['metric']))
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_list, f, indent=4)
        
    print(f"Generated {len(final_list)} metrics in {OUTPUT_FILE}")
    print("Sample:", final_list[:5])
    
    # Validation check
    response_metrics = [x for x in final_list if x['metric'] == 'response']
    if response_metrics:
        print("Found 'response' metric:", response_metrics)
    else:
        print("WARNING: 'response' metric NOT found!")

if __name__ == "__main__":
    main()
