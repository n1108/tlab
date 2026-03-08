# 提取单个metric 的异常 pattern
# 输入参数：uuid, component, metric
# 输出在异常 uuid 的时间段内，component 组件的 metric 指标的时序数据，并将数据画成一条折线图。
# 时序数据保存到 metric-series.txt 文件中，每行一组时序数据，格式为 uuid, component, metric, [value1, value2, value3, ...]
# 生成的折线图保存到 pattern-analysis/img/ 文件夹下，图片文件名为 uuid_component_metric.png

# 默认输入：uuid = "345fbe93-80", component = "emailservice", metric = "pod_cpu_usage"

import sys
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add workspace root to sys.path
# Script is in unit-test/metric/pattern-analysis/
workspace_root = Path(__file__).resolve().parents[3] 
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

def extract_metric(uuid, component, metric, extend_mins=0):
    # Paths
    test_data_path = workspace_root / "unit-test/metric/test_dataset.json"
    dataset_root = workspace_root / "dataset"
    output_dir = workspace_root / "unit-test/metric/pattern-analysis"
    img_dir = output_dir / "img"
    output_file = output_dir / "metric-series.txt"
    
    # Ensure directories exist
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load test dataset to get time range
    if not test_data_path.exists():
        logger.error(f"Test dataset not found at {test_data_path}")
        return

    with open(str(test_data_path), 'r') as f:
        test_cases = json.load(f)
        
    target_case = None
    for case in test_cases:
        if case["uuid"] == uuid:
            target_case = case
            break
            
    if not target_case:
        logger.error(f"UUID {uuid} not found in dataset.")
        return

    start_time_str = target_case["start_time"]
    end_time_str = target_case["end_time"]
    
    # Parse times (ensure naive or consistent timezone)
    start_time = pd.to_datetime(start_time_str).replace(tzinfo=None) - pd.Timedelta(minutes=extend_mins)
    end_time = pd.to_datetime(end_time_str).replace(tzinfo=None) + pd.Timedelta(minutes=extend_mins)
    
    logger.info(f"Extracting {metric} for {component} (UUID: {uuid})")
    
    # 2. Initialize MetricAgent and Load Data
    agent = MetricAgent(root_path=str(dataset_root))
    
    try:
        # Load data using internal/public method
        # Note: If load_data is not available, we might need to implement similar logic
        # But based on file read, MetricAgent has a load_data method.
        # However, we must check if load_data takes 'max_workers' or other args
        # and if it returns a DF with 'pod', 'kpi_key', 'value', 'time'
        df = agent.load_data(start_time, end_time)
        
        if df is None or df.empty:
            logger.warning("No data found for this time range.")
            return

        # 3. Filter Data
        # Special Logic for APM metrics (e.g. rrt, rrt_max, error_ratio)
        # APM files: apm/pod/pod_emailservice-2_2025-06-06.parquet
        # MetricAgent loads these as: pod="emailservice-2", kpi_key="rrt", value=...
        
        # Check if the requested metric is one of the APM metrics that might need better prefix handling
        # or if the component name is a prefix of the actual pod name in the dataframe
        
        mask = (df["pod"] == component) & (df["kpi_key"] == metric)
        filtered_df = df[mask].copy()
        
        # If exact match fails, try prefix match for pod components (e.g. emailservice -> emailservice-0)
        if filtered_df.empty:
            logger.info(f"Exact match for component '{component}' failed. Trying prefix match...")
            # Use startswith for Pod names (e.g., "emailservice" matches "emailservice-1")
            mask_prefix = (df["pod"].str.startswith(component + "-")) & (df["kpi_key"] == metric)
            filtered_df = df[mask_prefix].copy()
            
            # If still empty, try to see if component is already the full name but maybe metric is different?
            # No, requirement is metric="rtt".
            
            if not filtered_df.empty:
                unique_pods = filtered_df["pod"].unique()
                # For APM metrics like 'rrt', we might have multiple pods.
                # Heuristic: Pick the one with the most data or just the first one.
                # Or better: Pick the one that actually HAS data for this metric (already filtered by kpi_key)
                
                # Sort pods to be deterministic
                sorted_pods = sorted(unique_pods)
                selected_pod = sorted_pods[0]
                
                logger.info(f"Found related pods: {sorted_pods}. Selecting: {selected_pod}")
                component = selected_pod
                filtered_df = filtered_df[filtered_df["pod"] == component]

        if filtered_df.empty:
            # Fallback debug info
            logger.warning(f"No data found for component='{component}' (or derived) and metric='{metric}'.")
            
            if "kpi_key" in df.columns:
                all_available_metrics = df["kpi_key"].unique()
                if metric not in all_available_metrics:
                    logger.warning(f"Metric '{metric}' does NOT exist in the loaded data. Did you mean one of these? {sorted(all_available_metrics)}")
                else:
                    avail_pods = df[df["kpi_key"] == metric]["pod"].unique()
                    logger.warning(f"Metric '{metric}' exists, but not for component '{component}'. Pods with this metric: {avail_pods[:10]}...") 
            return
            
        # Sort by time
        filtered_df.sort_values("time", inplace=True)
        
        # Resample to 1 minute
        resampled_df = filtered_df.set_index("time").resample("1min").max()
        is_missing = resampled_df["value"].isna().tolist()
        
        # Fill missing values with 0
        filled_df = resampled_df.fillna(0).reset_index()
        
        values = filled_df["value"].tolist()
        timestamps = filled_df["time"].tolist()
        
        # 4. Save to text file
        line_content = f"{uuid}, {component}, {metric}, {values}\n"
        
        with open(str(output_file), "a") as f:
            f.write(line_content)
        
        logger.info(f"Saved series data to {output_file}")
        
        # 5. Plot Line Chart
        plt.figure(figsize=(10, 6))
        
        # Plot continuous line (gray background)
        plt.plot(timestamps, values, linestyle='-', color='gray', alpha=0.5, label="Trend")
        
        # Plot Observed Data (Blue Dots)
        obs_x = [t for t, m in zip(timestamps, is_missing) if not m]
        obs_y = [v for v, m in zip(values, is_missing) if not m]
        plt.scatter(obs_x, obs_y, color='blue', marker='.', label="Observed")
        
        # Plot Missing Data (Red Xs)
        miss_x = [t for t, m in zip(timestamps, is_missing) if m]
        miss_y = [v for v, m in zip(values, is_missing) if m]
        if miss_x:
            plt.scatter(miss_x, miss_y, color='red', marker='x', label="Missing (Filled 0)")

        plt.title(f"{metric} on {component}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        img_filename = f"{uuid}_{component}_{metric}.png"
        img_path = img_dir / img_filename
        plt.savefig(str(img_path))
        plt.close()
        
        logger.info(f"Saved plot to {img_path}")
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metric time series for a failure case.")
    parser.add_argument("--uuid", type=str, default="345fbe93-80", help="Failure Case UUID")
    parser.add_argument("--component", type=str, default="emailservice", help="Component Name (Pod/Node)")
    parser.add_argument("--metric", type=str, default="pod_cpu_usage", help="Metric Name")
    parser.add_argument("--extend", type=int, default=0, help="Extend time window (minutes) before and after")
    
    args = parser.parse_args()
    
    extract_metric(args.uuid, args.component, args.metric, extend_mins=args.extend)

# python3 extract_metric.py --uuid=345fbe93-80 --component=emailservice --metric=pod_cpu_usage
# 特殊情况：pod 故障导致数据缺失，需要绘制前后30分钟的图像
# python3 extract_metric.py --uuid=38ee3d45-82 --component=cartservice --metric=pod_cpu_usage --extend 30