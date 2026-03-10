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

# Add baseline directory to sys.path
sys.path.append(str(workspace_root / "unit-test/metric/baselines"))

from exp.agent.metric import MetricAgent
from rule_based import RuleBasedMetricAgent

def run_tests(limit=None, method="metric-agent"):
    # Define paths
    test_data_path = workspace_root / "unit-test/metric/test_dataset.json"
    result_dir = workspace_root / "unit-test/metric/results"
    
    # Choose output file based on method (optional, good practice)
    prediction_filename = "predictions_rule_based.json" if method == "rule-based" else "predictions.json"
    result_file = result_dir / prediction_filename
    
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
    
    # Initialize Agent
    if method == "rule-based":
        print("Using RuleBasedMetricAgent")
        agent = RuleBasedMetricAgent(root_path=str(dataset_root))
    else:
        print("Using MetricAgent (default)")
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
    Strict Mode:
     - Unit of measurement: (Metric, Component) pair.
     - Precision Denominator: Total unique (Metric, Component) pairs detected (including non-root-cause components).
     - Recall Denominator: Total predicted (Metric, RootCause) pairs (assuming Cross Product of Expected Metrics * Root Causes).
    """
    print("\nStarting Evaluation...")
    
    total_expected_count = 0 
    total_detected_count = 0
    correct_detected_count = 0   # Intersection
    
    test_case_map = {case["uuid"]: case for case in test_cases}
    
    for res in results:
        uuid = res.get("uuid")
        if uuid not in test_case_map:
            continue
            
        case = test_case_map[uuid]
        root_causes = set(case.get("root_cause_components", []))
        
        # 1. Parse Expected Metrics
        expected_metrics = set()
        for x in case.get("expected_anomalies", []):
            if isinstance(x, str):
                expected_metrics.add(x)
            elif isinstance(x, dict):
                expected_metrics.add(x.get("metric"))
        
        # Flatten Expected Set: {(metric, rc_component)}
        # We assume every expected metric applies to every root cause component
        expected_set = set()
        for rc in root_causes:
            for m in expected_metrics:
                expected_set.add((m, rc))
                
        # 2. Parse Detected Metrics
        detected_items = res.get("detected_anomalies", [])
        detected_set = set()
        
        for item in detected_items:
            metric = item.get("metric")
            components = item.get("component", []) # This is a list
            if isinstance(components, str): 
                components = [components]
                
            for comp in components:
                # Map detected component to root cause if possible
                matched_rc = None
                for rc in root_causes:
                    # Match exact or prefix (e.g. "cartservice-0" -> "cartservice")
                    if comp == rc or comp.startswith(rc + "-"):
                        matched_rc = rc
                        break
                
                if matched_rc:
                    detected_set.add((metric, matched_rc))
                else:
                    detected_set.add((metric, comp))
        
        # 3. Calculate Stats for this Case
        correct_set = expected_set.intersection(detected_set)
        
        total_expected_count += len(expected_set)
        total_detected_count += len(detected_set)
        correct_detected_count += len(correct_set)

    # Final Metrics
    recall = (correct_detected_count / total_expected_count * 100) if total_expected_count > 0 else 0.0
    precision = (correct_detected_count / total_detected_count * 100) if total_detected_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"Evaluation Results:")
    print(f"Total Detected Anomalies (Metric-Component Pairs): {total_detected_count}")
    print(f"  - Correct: {correct_detected_count}")
    print(f"  - Precision: {precision:.2f}%") 
    print(f"Total Expected Anomalies (Metric-RC Pairs): {total_expected_count}")
    print(f"  - Recalled: {correct_detected_count}")
    print(f"  - Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--method", type=str, default="metric-agent", choices=["metric-agent", "rule-based"], help="Anomaly detection method to use")
    args = parser.parse_args()
    
    run_tests(limit=args.limit, method=args.method)

# python3 unit-test/metric/run_test.py --limit 5