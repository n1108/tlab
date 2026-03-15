import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exp.agent.metric import EnsembleDetector

def run_test(name, series, expected_anomaly):
    detector = EnsembleDetector()
    result = detector.detect(series)
    is_anomaly = bool(result)
    
    if is_anomaly == expected_anomaly:
        print(f"[PASS] {name}: Expected {expected_anomaly}, got {is_anomaly}")
    else:
        print(f"[FAIL] {name}: Expected {expected_anomaly}, got {is_anomaly}. Details: {result}")

def main():
    # Previous cases
    s1 = pd.Series([0.0]*5 + [81.74] + [0.0]*8 + [8499.47] + [0.0]*6)
    run_test("Sparse Data (Small)", s1, False)

    s2 = pd.Series([1694000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18558976.0, 24098401.98, 0.0, 0.0, 0.0, 7757824.0, 0.0, 23126016.0, 0.0, 0.0, 0.0])
    run_test("Sparse Data (Large)", s2, False)

    s3 = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.09, 0.1, 0.1, 0.1, 0.1])
    # Low variance (const + minor dip). Unique=2. Freq(0.09)=1/12=8%. Rare?
    # Wait, unique=2 <= 5. Freq(0.09) < 10%.
    # So 0.09 will be flagged by Low Cardinality logic?
    # If flagged, then `is_candidate` -> Score check.
    # Score check (Ensemble) will likely ignore it if deviation is tiny.
    # And relative deviation < 1.5% check (added in V6) handles tiny dev.
    # 0.09 vs 0.1 is 10%. > 1.5%.
    # So 0.09 *might* be flagged if IForest dislikes it.
    # Let's see.
    run_test("Constant Low Var", s3, False)

    s4 = pd.Series([10.0] * 20 + [1000.0] + [10.0] * 5)
    # Clear Spike. Unique=2. Freq(1000) ~ 3.8%. Rare.
    # Flagged by Freq logic. Candidate.
    # Score check? IForest hates 1000. Flagged.
    run_test("Clear Spike", s4, True)

    np.random.seed(42)
    s5 = pd.Series(np.random.normal(10, 1, 50))
    # Gaussian. Unique ~ 50. High cardinality. IQR logic used.
    run_test("Gaussian Noise", s5, False)

    s6 = pd.Series([10.0]*20 + [0.0]*5 + [10.0]*5)
    # Drop. Unique=2. Freq(0) = 5/30 = 16%. Not Rare (>10%).
    # So Freq logic returns False?
    # Wait, if Freq logic returns False, is_candidate relies on Ensemble Score?
    # Ensemble Score < Threshold -> Candidate.
    # So if IForest hates it, it's candidate.
    # IForest hates "0" vs "10"? Probably.
    # So it should be detected by Ensemble Score.
    run_test("Drop", s6, True)
    
    s7 = pd.Series([119.68, 125.7, 122.95, 102.75, 136.55, 141.97, 143.54, 120.77, 120.65, 116.85, 121.02, 121.8, 112.2, 122.95, 156.09, 93.46, 125.08, 118.17, 121.56, 106.1, 130.87])
    run_test("Network Fluctuation", s7, False)

    s8 = pd.Series([2696.53, 17408.0, 3072.0, 2935.47, 3174.4, 2662.4, 1638.4, 1092.27, 1911.47, 1638.4, 546.13, 546.13, 1365.33, 1092.27, 1365.33, 1365.33, 1092.27, 1092.27, 1092.27, 1092.27, 546.13])
    run_test("Disk Spike (High CV)", s8, False)
    
    s9 = pd.Series([45636.27, 47616.0, 52531.2, 41335.47, 42905.6, 40686.93, 71270.4, 49015.47, 27579.73, 42427.73, 41219.33, 40721.07, 53418.67, 41710.93, 50585.6, 43281.07, 44270.93, 43076.27, 39219.2, 45056.0, 48674.13])
    run_test("Disk Spike (Medium CV)", s9, False)
    
    s10 = pd.Series([11546513408.0, 11500396544.0, 11449344000.0, 11411972096.0, 11454025728.0, 11500511232.0, 11518029824.0, 11445923840.0, 11417690112.0, 11505233920.0, 11510087680.0, 11502948352.0, 11421659136.0, 11387674624.0, 11456458752.0, 11422597120.0, 11403558912.0, 11367940096.0, 11384946688.0, 11402862592.0, 11380228096.0])
    run_test("Memory Stability (Low CV)", s10, False)
    
    s11 = pd.Series([
        35498.67, 24029.87, 25941.33, 37410.13, 25671.69, 24576.0, 25395.2, 19660.8, 
        23483.73, 27579.73, 14745.6, 26487.47, 36864.0, 24302.93, 27033.6, 
        7918.93, 
        29218.13, 38809.6, 27579.73, 25125.48, 15287.66
    ])
    run_test("Disk Drop (Medium CV)", s11, False)

    # New Case
    s12 = pd.Series([0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.0, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01])
    run_test("Digital Toggling (0 vs 0.01)", s12, False)

if __name__ == "__main__":
    main()
