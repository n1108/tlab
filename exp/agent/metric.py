import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from typing import List, Dict, Any
from sklearn.ensemble import IsolationForest
import pyarrow.dataset as ds
from exp.utils.input import load_parquet
from exp.utils.time import daterange

logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Implementation of hwlyyzc's Metric Anomaly Detection Module.
    Strategy: Multi-algorithm Fusion (IF + HBOS + IQR) + Local Pattern Verification.
    """

    def _calculate_hbos(self, series: pd.Series, bins=10) -> np.ndarray:
        """
        Histogram-based Outlier Score (HBOS).
        Higher score = Lower density = More anomalous.
        """
        n = len(series)
        if n < 2 or series.nunique() < 2:
            return np.zeros(n)
        
        try:
            # Adjust bins dynamically based on data length
            actual_bins = min(bins, max(2, n // 5))
            hist, bin_edges = np.histogram(series, bins=actual_bins, density=True)
            
            # Avoid log(0) by replacing 0 with a very small density
            min_density = np.min(hist[hist > 0]) if np.any(hist > 0) else 1e-10
            hist = np.where(hist == 0, min_density * 0.1, hist)
            
            # Map values to bins
            bin_indices = np.digitize(series, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, actual_bins - 1)
            
            # Score = log(1 / density)
            scores = np.log(1.0 / hist[bin_indices])
            return scores
        except Exception as e:
            # Fallback
            return np.zeros(n)

    def _calculate_iqr_mask(self, series: pd.Series) -> np.ndarray:
        """
        Interquartile Range (IQR) detection.
        Returns boolean mask where True indicates an outlier.
        """
        if len(series) < 5:
            return np.zeros(len(series), dtype=bool)
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
             return np.zeros(len(series), dtype=bool)
             
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return ((series < lower_bound) | (series > upper_bound)).values

    def _detect_local_pattern(self, series: pd.Series, anomaly_indices: np.ndarray) -> str:
        """
        Identify local temporal patterns (Spike, Drop, Shift) based on anomalous points.
        """
        if not np.any(anomaly_indices):
            return "normal"
            
        mean_val = series.mean()
        anom_indices = np.where(anomaly_indices)[0]
        anom_values = series.iloc[anom_indices]
        
        # Determine direction
        anom_mean = anom_values.mean()
        is_high = anom_mean > mean_val
        
        # Check duration and continuity
        duration = len(anom_indices)
        last_idx = len(series) - 1
        
        # Heuristic rules for pattern naming
        if is_high:
            if duration == 1:
                return "spike"
            elif anom_indices[-1] == last_idx:
                return "level_shift_up" # Anomaly persists until the end
            else:
                return "surge" # Anomaly lasted for a while then returned
        else:
            if duration == 1:
                return "drop"
            elif anom_indices[-1] == last_idx:
                return "level_shift_down"
            else:
                return "dip"

    def detect(self, series: pd.Series) -> Dict[str, Any]:
        # Pre-check: Skip constant or too short series
        if len(series) < 5 or series.std() == 0:
            return {}
            
        values = series.values.reshape(-1, 1)
        
        # 1. Isolation Forest (Global)
        # decision_function: lower is more anomalous (negative values are outliers)
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=1) # n_jobs=1 to avoid overhead in threadpool
        iso.fit(values)
        if_scores = iso.decision_function(values) 
        
        # 2. HBOS (Global)
        hbos_scores = self._calculate_hbos(series)
        
        # 3. IQR (Global Hard Threshold)
        iqr_mask = self._calculate_iqr_mask(series)
        
        # 4. Fusion Formula from PPT: (IF - 0.1 * HBOS) / 2
        # IF is negative for anomalies. HBOS is positive for anomalies.
        # Result: More negative = More anomalous.
        anomaly_scores = (if_scores - (0.1 * hbos_scores)) / 2
        
        # 5. Dynamic Thresholding
        # Using 3-sigma rule on the fusion score
        score_mean = anomaly_scores.mean()
        score_std = anomaly_scores.std()
        dynamic_thresh = score_mean - 3 * score_std
        
        # Hard cap to avoid false positives in noisy data
        final_thresh = min(-0.45, dynamic_thresh)
        
        is_candidate = (anomaly_scores < final_thresh) | iqr_mask
        
        if not np.any(is_candidate):
            return {}
            
        # 6. Local Pattern Verification
        ratio = np.sum(is_candidate) / len(series)
        
        # Filter out minor noise (e.g. single point in long series if not extreme)
        if ratio < 0.02 and not np.any(iqr_mask):
            return {}
            
        pattern = self._detect_local_pattern(series, is_candidate)
        
        return {
            "is_anomaly": True,
            "pattern": pattern,
            "timestamps": series.index[is_candidate].tolist(),
            "max_val": series.max(),
            "mean_val": series.mean()
        }

class MetricAgent:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.detector = EnsembleDetector()
        
        # Fields to load (Optimization: Don't load everything)
        self.apm_load_fields = ["time", "object_id", "error_ratio", "timeout", "rrt", "rrt_max"]
        self.infra_load_fields = ["time", "instance", "pod", "value", "kpi_key"]

    def load_data(self, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        """
        Loads metrics from APM, Infra, and Other sources.
        Standardizes them into [time, pod, kpi_key, value].
        """
        all_dfs = []
        tasks = []
        
        # Define filters
        time_filter = (ds.field("time") >= start) & (ds.field("time") <= end)

        # 1. APM Data Loader (Needs Melting)
        apm_glob = f"{self.root_path}/*/metric-parquet/apm/service/*.parquet"
        
        def _process_apm(f):
            try:
                df = load_parquet(Path(f), self.apm_load_fields, time_filter)
                if df.empty: return None
                # Transform wide APM table to long format
                melted = df.melt(id_vars=["time", "object_id"], 
                                 value_vars=["error_ratio", "timeout", "rrt", "rrt_max"],
                                 var_name="kpi_key", value_name="value")
                melted.rename(columns={"object_id": "pod"}, inplace=True)
                return melted
            except Exception:
                return None

        # 2. Infra & Other Data Loader (Already Long format usually)
        # Pattern covers infra_pod, infra_node, infra_tidb, and other
        infra_patterns = [
            "infra/infra_pod/*.parquet",
            "infra/infra_node/*.parquet", 
            "infra/infra_tidb/*.parquet",
            "other/*.parquet"
        ]
        
        def _process_infra(f):
            try:
                # Load minimal columns to save IO
                df = load_parquet(Path(f), filter_=time_filter)
                if df.empty: return None
                
                # Identify value column (dynamic)
                cols = set(df.columns)
                # Standard schema fields to ignore when finding 'value'
                schema_cols = {"time", "cf", "device", "instance", "kpi_key", "kpi_name", 
                               "kubernetes_node", "mountpoint", "namespace", "object_type", 
                               "pod", "sql_type", "type"}
                
                value_candidates = list(cols - schema_cols)
                if not value_candidates: return None
                
                # Normalize columns
                df["value"] = df[value_candidates[0]] # Take the metric value
                
                # Ensure 'pod' exists (Node metrics use 'instance')
                if "pod" not in df.columns:
                    df["pod"] = df["instance"] if "instance" in df.columns else "unknown"
                
                # Ensure 'kpi_key' exists
                if "kpi_key" not in df.columns:
                    # Infer kpi_key from filename if missing in file
                    df["kpi_key"] = Path(f).stem
                
                return df[["time", "pod", "kpi_key", "value"]]
            except Exception:
                return None

        # Collect file paths
        for day in daterange(start, end):
            # APM Tasks
            tasks.extend([(f, _process_apm) for f in glob.glob(apm_glob.replace("*", day, 1))])
            
            # Infra Tasks
            for pattern in infra_patterns:
                full_pattern = f"{self.root_path}/{day}/metric-parquet/{pattern}"
                tasks.extend([(f, _process_infra) for f in glob.glob(full_pattern)])

        # Execute Parallel Loading
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(func, file) for file, func in tasks]
            for future in as_completed(futures):
                res = future.result()
                if res is not None and not res.empty:
                    all_dfs.append(res)
        
        if not all_dfs:
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)

    def score(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Main entry point. Returns structured anomalies for JudgeAgent.
        """
        # Call the analysis logic
        analysis = self.query_metrics(start_time, end_time)
        scores = []
        
        # Convert internal event format to list of dicts required by main.py
        if "events" in analysis:
            for event in analysis["events"]:
                scores.append({
                    "service": event.get("pod"),
                    "kpi": event.get("kpi"),
                    "reason": f"Metric: {event.get('kpi')} {event.get('pattern')}",
                    "details": event.get("timestamps", [])[:3] # Keep it concise
                })
        return scores

    def query_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Performs the detection logic.
        """
        logger.info(f"MetricAgent: Analyzing {start_time} to {end_time}")
        
        # 1. Load Data
        df = self.load_data(start_time, end_time)
        if df.empty:
            return {"observation": "No metric data available.", "events": []}
            
        events = []
        
        # 2. Group by Pod and KPI
        # We process each time series independently
        grouped = df.groupby(['pod', 'kpi_key'])
        
        for (pod, kpi), group in grouped:
            # Resample to 1 minute to align time series and handle missing data
            # Use 'max' for downsampling to capture spikes, fillna(0) for missing
            try:
                series = group.set_index('time')['value'].sort_index()
                series = series.resample('1min').max().fillna(0)
            except Exception:
                continue
            
            # 3. Detect Anomalies
            result = self.detector.detect(series)
            
            if result:
                # Filter out low-value noise for specific KPIs if needed
                # (e.g., error_ratio < 0.01 is usually negligible)
                if "ratio" in kpi and result["max_val"] < 0.01:
                    continue
                    
                events.append({
                    "pod": pod,
                    "kpi": kpi,
                    "pattern": result["pattern"],
                    "timestamps": [str(t) for t in result["timestamps"]]
                })

        # 4. Summarize for LLM Observation
        if not events:
            observation = "No significant metric anomalies detected."
        else:
            # Consolidate by Pod to reduce token usage
            pod_anomalies = {}
            for e in events:
                p = e['pod']
                if p not in pod_anomalies:
                    pod_anomalies[p] = []
                pod_anomalies[p].append(f"{e['kpi']} ({e['pattern']})")
            
            # Construct description
            details = []
            for p, kpis in pod_anomalies.items():
                # Dedup and sort
                kpis = sorted(list(set(kpis)))
                # Limit KPIs per pod
                kpi_str = ", ".join(kpis[:4])
                if len(kpis) > 4: kpi_str += "..."
                details.append(f"{p}: [{kpi_str}]")
            
            # Limit total pods in observation
            if len(details) > 10:
                details = details[:10] + ["...others"]
                
            observation = f"Metric Anomalies Detected: {'; '.join(details)}"

        return {
            "observation": observation,
            "events": events
        }