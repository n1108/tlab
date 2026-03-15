import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

TEST_DATASET_PATH = "/home/tyt21/tlab/unit-test/metric/test_dataset.json"
DATASET_ROOT = "/home/tyt21/tlab/dataset"
OUTPUT_FILE = "/home/tyt21/tlab/unit-test/metric/metric_patterns.csv"

def load_test_dataset():
    with open(TEST_DATASET_PATH, 'r') as f:
        data = json.load(f)
    return data

def get_parquet_files(date_str):
    search_path = os.path.join(DATASET_ROOT, date_str, "metric-parquet", "**", "*.parquet")
    files = glob.glob(search_path, recursive=True)
    return files

def process_file(file_path, time_ranges):
    """
    Process a single parquet file and extract normal data for defined time ranges.
    time_ranges: list of dicts {start, end, anomalies} for the date of this file.
    Returns: dict {metric_name: list of values (np.array)}
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    if 'time' not in df.columns:
        return {}

    # Ensure time column is datetime and UTC-aware
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'], utc=True)
    
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    else:
        df['time'] = df['time'].dt.tz_convert('UTC')
    
    # Identify metric columns (numeric, not time/dimensions)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove obvious non-metrics if any (e.g. IDs that are numeric? Usually IDs are strings)
    # Also exclude known non-metric numerics if any. 
    # Based on inspection, metrics are the numeric columns.
    
    results = {}

    for tr in time_ranges:
        start_time = pd.to_datetime(tr['start_time'])
        end_time = pd.to_datetime(tr['end_time'])
        excluded_metrics = set(tr['expected_anomalies'])
        
        # Filter time range
        # Note: The timestamps in parquet might be localized or UTC. 
        # The input JSON has 'Z', so it is UTC.
        # df['time'] should be UTC.
        
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        if not mask.any():
            continue
            
        slice_df = df.loc[mask]
        
        for col in numeric_cols:
            if col not in excluded_metrics:
                vals = slice_df[col].dropna().values
                if len(vals) > 0:
                    if col not in results:
                        results[col] = []
                    results[col].append(vals)
                    
    return results

def calculate_stats(values):
    """
    Calculate statistical features for a metric.
    values: 1D numpy array of all normal samples.
    """
    if len(values) == 0:
        return None
        
    # Basic Stats
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    q25 = np.percentile(values, 25)
    median = np.median(values)
    q75 = np.percentile(values, 75)
    
    # Distribution
    # Handle division by zero for skew/kurtosis if std is 0
    if std > 1e-9:
        # manual skew/kurt to avoid scipy dependency if not available, but numpy is fine.
        # skew = E[((x-mu)/sigma)^3]
        centered = values - mean
        skew = np.mean((centered / std) ** 3)
        kurtosis = np.mean((centered / std) ** 4) - 3 # Excess kurtosis
        cv = std / (abs(mean) + 1e-9)
    else:
        skew = 0.0
        kurtosis = 0.0
        cv = 0.0
        
    # Sparsity / Cardinality
    zero_count = np.sum(np.abs(values) < 1e-9)
    zero_rate = zero_count / len(values)
    
    # For unique rate, if array is huge, np.unique is slow. 
    # Sample if too large? Or just do it.
    if len(values) > 100000:
        # Estimate with sample
        sample = np.random.choice(values, size=100000, replace=False)
        unique_rate = len(np.unique(sample)) / 100000
    else:
        unique_rate = len(np.unique(values)) / len(values)

    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min_val,
        "25%": q25,
        "50%": median,
        "75%": q75,
        "max": max_val,
        "skew": skew,
        "kurtosis": kurtosis,
        "cv": cv,
        "zero_rate": zero_rate,
        "unique_rate": unique_rate
    }

def main():
    print("Loading test dataset map...")
    test_data = load_test_dataset()
    
    # Group by Date
    date_ranges = {}
    for entry in test_data:
        # Detect date from start_time.
        # "2025-06-05T16:10:02Z" -> "2025-06-05"
        # BUT folder structure might be different.
        # The user has folders 2025-06-06 onwards.
        # If test_dataset has 2025-06-05, and folder is missing, we skip.
        # Wait, the parquet files inside 2025-06-06 contain data for "2025-06-05 16:00:00" !
        # Remember the `head()` output?
        # 0  2025-06-05T16:00:00Z
        # So "2025-06-06" folder contains data starting from earlier? Or data FOR the fault which started on 05?
        # Actually usually dates in logs/metrics carry over.
        # If the file path is `dataset/2025-06-06/...`, it probably covers data relevant for that day's testing.
        # BUT the timestamp in the file was `2025-06-05`.
        # So I should check ALL folders? No, valid range.
        # Let's map entries to potential folders.
        # If start_time is 2025-06-05, where is the file?
        # It seems `2025-06-06/metric-parquet/` contains data for `2025-06-05`!
        # This implies `dataset/DATE` is the "Collection Date" or "Test Date", not necessarily strictly the timestamp date.
        # I should probably scan ALL folders for ALL test cases? That's too slow.
        # Heuristic: The folder date usually matches the test date OR the next day.
        # Given `unit-test/metric/test_dataset.json` has `2025-06-05` and folder has `2025-06-06`, 
        # and file content showed `2025-06-05`, it's likely mapped 1-to-1 or with offset.
        # A simpler robust way:
        # Get all folders in dataset/.
        # For each folder, load files.
        # For each file, check min/max timestamp.
        # Then map to test cases.
        # This is robust but slow-ish (reading min/max of 1000s of files).
        # Optimization: Just assume folder date corresponds to test cases in that vicinity.
        # OR: iterate through folders `2025-06-06` ... `2025-06-29`.
        # For each folder, load data.
        # Filter test cases that overlap with this data.
        pass

    # Better Strategy given small file sizes:
    # 1. Iterate folders in `dataset/`.
    # 2. For each folder (Date):
    #    a. Load all parquet files into a list of DFs (or process sequentially).
    #    b. For each DF, get time range `[min_t, max_t]`.
    #    c. Find `test_data` entries that overlap with `[min_t, max_t]`.
    #    d. Extract normal data.
    
    # Global accumulator
    metric_buffer = {} # {metric_name: [np.array, np.array, ...]}

    folders = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("2025")])
    
    print(f"Found {len(folders)} date folders.")
    
    for folder_date in folders:
        print(f"Processing {folder_date}...")
        parquet_files = get_parquet_files(folder_date)
        
        for pfile in parquet_files:
            try:
                df = pd.read_parquet(pfile)
                if 'time' not in df.columns or df.empty:
                    continue
                    
                # Ensure time column is datetime and UTC-aware
                if not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], utc=True)
                
                if df['time'].dt.tz is None:
                    df['time'] = df['time'].dt.tz_localize('UTC')
                else:
                    df['time'] = df['time'].dt.tz_convert('UTC')
                     
                t_min = df['time'].min()
                t_max = df['time'].max()
                
                # Find relevant test cases
                # A test case is relevant if its window [start, end] overlaps with [t_min, t_max]
                relevant_cases = []
                for entry in test_data:
                    estart = pd.to_datetime(entry['start_time'])
                    eend = pd.to_datetime(entry['end_time'])
                    
                    # Check overlap
                    if (estart <= t_max) and (eend >= t_min):
                        relevant_cases.append({
                            'start_time': estart,
                            'end_time': eend,
                            'expected_anomalies': entry['expected_anomalies']
                        })
                
                if not relevant_cases:
                    continue
                    
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                for tr in relevant_cases:
                    # STRICT filtering: Only data INSIDE the fault window is used for exclusion logic?
                    # Wait, we want to extract NORMAL data.
                    # The parts of DF that OVERLAP with the fault window `[tr_start, tr_end]`.
                    # In that window, we exclude `expected_anomalies`. Use others.
                    # What about data OUTSIDE the fault window? Is it normal?
                    # The prompt says "exclude abnormal time periods of abnormal metrics".
                    # It implies OUTSIDE periods are normal?
                    # If I use data outside fault windows, I can get much more normal data.
                    # BUT, the `test_dataset.json` only defines fault windows.
                    # If I assume everything else is normal, I might include undefined faults.
                    # SAFER APPROACH: Only use data *during* the known test windows, 
                    # but only the metrics that are NOT expected to be anomalous.
                    # This ensures we are evaluating "known normal behavior" (Metric A is normal while Metric B is failing).
                    
                    mask = (df['time'] >= tr['start_time']) & (df['time'] <= tr['end_time'])
                    slice_df = df.loc[mask]
                    if slice_df.empty:
                        continue
                        
                    excluded = set(tr['expected_anomalies'])
                    
                    for col in numeric_cols:
                        if col not in excluded:
                            vals = slice_df[col].dropna().values
                            if len(vals) > 0:
                                if col not in metric_buffer:
                                    metric_buffer[col] = []
                                metric_buffer[col].append(vals)

            except Exception as e:
                print(f"Failed to process {pfile}: {e}")

    print("Aggregating statistics...")
    stats_list = []
    for metric, chunks in metric_buffer.items():
        # Concatenate all chunks
        full_data = np.concatenate(chunks)
        stats = calculate_stats(full_data)
        if stats:
            stats["metric"] = metric
            stats_list.append(stats)
            
    result_df = pd.DataFrame(stats_list)
    # Reorder columns
    cols = ["metric"] + [c for c in result_df.columns if c != "metric"]
    result_df = result_df[cols]
    
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved statistics to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
