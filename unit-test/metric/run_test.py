import sys
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parents[2]
sys.path.append(str(workspace_root))

from exp.agent.metric import MetricAgent

def run_tests(limit=None):
    # Define paths
    test_data_path = workspace_root / "unit-test/metric/test_dataset.json"
    result_dir = workspace_root / "unit-test/metric/results"
    result_file = result_dir / "predictions.json"
    dataset_root = workspace_root / "dataset"
    
    # Ensure result directory exists
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    if not test_data_path.exists():
        logger.error(f"Test dataset not found at {test_data_path}")
        return
        
    with open(test_data_path, 'r') as f:
        test_cases = json.load(f)
    
    if limit is not None:
        print(f"Limiting to first {limit} test cases.")
        test_cases = test_cases[:limit]
        
    print(f"Loaded {len(test_cases)} test cases.")
    
    # Initialize MetricAgent
    agent = MetricAgent(root_path=str(dataset_root))
    
    results = []
    
    # Iterate through test cases
    for case in tqdm(test_cases, desc="Running Tests"):
        try:
            # Parse time range (ISO 8601 strings to datetime)
            # Ensure naive datetime for pyarrow compatibility if parquet has naive timestamps
            start_time = pd.to_datetime(case["start_time"]).replace(tzinfo=None)
            end_time = pd.to_datetime(case["end_time"]).replace(tzinfo=None)
            
            # Run detection
            # query_metrics returns {"observation": ..., "events": [...]}
            # Each event has: "pod", "kpi", "pattern", "timestamps"
            analysis = agent.query_metrics(start_time, end_time)
            detected_events = analysis.get("events", [])
            
            # Aggregation by metric
            aggregated_anomalies = {}
            for event in detected_events:
                pod = event.get("pod")
                kpi = event.get("kpi")
                pattern = event.get("pattern")
                timestamps = event.get("timestamps", [])
                
                key = kpi
                if key not in aggregated_anomalies:
                    aggregated_anomalies[key] = {
                        "patterns": set(),
                        "components": set(),
                        "timestamps": set()
                    }
                
                if pattern:
                    aggregated_anomalies[key]["patterns"].add(pattern)
                
                if pod:
                    aggregated_anomalies[key]["components"].add(pod)
                
                aggregated_anomalies[key]["timestamps"].update(timestamps)

            formatted_detections = []
            for kpi, data in aggregated_anomalies.items():
                formatted_detections.append({
                    "metric": kpi,
                    "pattern": sorted(list(data["patterns"])),
                    "component": sorted(list(data["components"])),
                    "timestamps": sorted(list(data["timestamps"]))
                })

            # Create result entry with only required fields
            result_entry = {
                "uuid": case.get("uuid"),
                "root_cause_components": case.get("root_cause_components"),
                "detected_anomalies": formatted_detections
            }
            
            results.append(result_entry)
            
        except Exception as e:
            
            # Create result entry with only required fields
            result_entry = {
                "uuid": case.get("uuid"),
                "root_cause_components": case.get("root_cause_components"),
                "detected_anomalies": formatted_detections
            }
            
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"Error processing case {case.get('uuid')}: {e}")
            case["error"] = str(e)
            results.append(case)
            
    # Save results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Test completed. Results saved to {result_file}")
    
    # Evaluate accuracy
    evaluate_accuracy(results, test_cases)

def evaluate_accuracy(results, test_cases):
    """
    Calculate Precision and Recall.
    Ignore anomaly pattern/morphology. Only check if the specific metric on the root cause component was detected.
    
    Recall = (Expected Metrics Detected on Root Cause) / (Total Expected Metrics on Root Cause)
    Precision = (Detected Metrics on Root Cause matching Expected) / (Total Detected Metrics)
    """
    print("\nStarting Evaluation...")
    
    # Global counts
    total_expected_count = 0 
    correctly_detected_expected_count = 0  # Recall Numerator
    
    total_detected_count = 0
    correct_detected_count = 0   # Precision Numerator
    
    test_case_map = {case["uuid"]: case for case in test_cases}
    
    for res in results:
        uuid = res.get("uuid")
        if uuid not in test_case_map:
            continue
            
        case = test_case_map[uuid]
        root_causes = set(case.get("root_cause_components", []))
        
        # Expected metrics for this case (set of strings)
        expected_metrics = set()
        for exp in case.get("expected_anomalies", []):
            if exp.get("metric"):
                expected_metrics.add(exp.get("metric"))
        
        detected_items = res.get("detected_anomalies", [])
        
        # Helper to check if a component is a root cause or a pod of it
        def is_root_cause(comp_name_or_list):
            if isinstance(comp_name_or_list, list):
                for comp_name in comp_name_or_list:
                    for rc in root_causes:
                        if comp_name == rc or comp_name.startswith(rc + "-"):
                            return True
                return False
            else:
                comp_name = comp_name_or_list
                for rc in root_causes:
                    if comp_name == rc or comp_name.startswith(rc + "-"):
                        return True
                return False

        # --- Calculate Precision (Accuracy of Detections) ---
        for d in detected_items:
            total_detected_count += 1
            
            d_comp = d.get("component", [])
            d_metric = d.get("metric")
            
            if is_root_cause(d_comp):
                # It is on a root cause component. Is it an expected metric?
                if d_metric in expected_metrics:
                    correct_detected_count += 1

        # --- Calculate Recall (Coverage of Expected Issues) ---
        total_expected_count += len(expected_metrics)
        
        for exp_metric in expected_metrics:
            # Did we find this metric on ANY valid root cause component?
            found = False
            for d in detected_items:
                d_comp = d.get("component", "")
                d_metric = d.get("metric")
                
                if d_metric == exp_metric and is_root_cause(d_comp):
                    found = True
                    break
            
            if found:
                correctly_detected_expected_count += 1

    # Final Metrics
    recall = (correctly_detected_expected_count / total_expected_count * 100) if total_expected_count > 0 else 0.0
    precision = (correct_detected_count / total_detected_count * 100) if total_detected_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"Evaluation Results:")
    print(f"Total Detected Anomalies: {total_detected_count}")
    print(f"  - Correct (Precision Numerator): {correct_detected_count}")
    print(f"  - Precision: {precision:.2f}%") 
    print(f"Total Expected Anomalies (Types per case): {total_expected_count}")
    print(f"  - Detected (Recall Numerator): {correctly_detected_expected_count}")
    print(f"  - Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    args = parser.parse_args()
    
    run_tests(limit=args.limit)
