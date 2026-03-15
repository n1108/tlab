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

    def _calculate_iqr_mask(self, series: pd.Series, pattern: Dict[str, Any] = None) -> np.ndarray:
        """
        Interquartile Range (IQR) detection.
        Returns boolean mask where True indicates an outlier.
        """
        if len(series) < 5:
            return np.zeros(len(series), dtype=bool)

        # Skip IQR if series is zero-inflated (too sparse)
        # In sparse data, IQR becomes 0 and flags any non-zero value as anomaly
        if (series.abs() < 1e-9).mean() > 0.3:
            return np.zeros(len(series), dtype=bool)
            
        # Handle Discrete/Low-Cardinality Data (e.g. 0.0 vs 0.01 flip-flops)
        # If data has very few unique values (< 5 or < 10% of length),
        # use frequency-based outlier detection instead of IQR.
        # Rule: A value is an outlier only if it appears < 10% of the time.
        pct_unique = series.nunique() / len(series)
        if series.nunique() <= 5 or pct_unique < 0.1:
            val_counts = series.value_counts(normalize=True)
            # Find the dominant mode
            mode_val = val_counts.index[0] 
            
            # Use 20% frequency threshold to catch Drop anomalies (16%)
            rare_values = val_counts[val_counts < 0.2].index
            
            # Additional Filter: Only flag rare values if they differ significantly (> 20%) from the mode
            # This handles micro-fluctuations in discrete signals (e.g. 0.1 -> 0.09)
            significant_rare = []
            for val in rare_values:
                # Calculate relative difference
                rel_diff = abs(val - mode_val) / (abs(mode_val) + 1e-9)
                if rel_diff > 0.2:
                    significant_rare.append(val)
            
            return series.isin(significant_rare).values
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            median = series.median()
            # If median is effectively zero
            if abs(median) < 1e-9:
                # If series passed sparsity check (<30% zeros), flag non-zero values
                return (series.abs() > 1e-6)
            # For non-zero median, flag values deviating by > 20%
            return (series - median).abs() > (0.2 * abs(median))
             
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        iqr_mask = (series < lower_bound) | (series > upper_bound)
        
        # Add relative deviation check for robust filtering
        # Require >30% deviation from median to be considered a true anomaly
        # Adjust threshold based on volatility (CV)
        median = series.median()
        if abs(median) > 1e-9:
            relative_deviation = (series - median).abs() / abs(median)
            cv = series.std() / (abs(series.mean()) + 1e-9)
            
            # Use stricter criteria for highly volatile metrics (CV > 1.0)
            # to filter out normal bursty behavior (e.g. disk I/O spikes)
            is_spike = series > median
            
            if cv > 1.0:
                # For spikes, require massive deviation (15x) to ignore noise
                thresh_spike = 15.0
            else:
                # Standard deviation check for stable metrics
                thresh_spike = 0.5 + (2.0 * cv)
            
            # Universal drop logic: Base 50% + volatility scaling
            # Avoids flagging normal idle periods (drops to 30-40% of mean) as anomalies
            # while still detecting complete drops in stable metrics
            
            # Determine if zero/low values are normal
            can_drop_to_zero = False
            if pattern:
                 # If zero_rate is high (>5%) or min is explicitly 0, then 0 is normal.
                 if pattern.get('zero_rate', 0) > 0.05 or pattern.get('min', 0) == 0:
                      can_drop_to_zero = True
            else:
                 # Heuristic: If currently low min, maybe normal. But rely on CV mostly.
                 can_drop_to_zero = (series.min() == 0)

            thresh_drop = 0.5 + (1.0 * cv)
            
            # If metric is strictly positive (e.g. latency, active requests), a drop to near-zero is anomaly
            if not can_drop_to_zero:
                 # Cap drop threshold at 0.9 (90% drop) to ensure near-zero drops are caught
                 thresh_drop = min(thresh_drop, 0.9)
                
            threshold_val = np.where(is_spike, thresh_spike, thresh_drop)
            
            iqr_mask = iqr_mask & (relative_deviation > threshold_val)
            
        return iqr_mask.values

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
                
    def _calculate_severity(self, series: pd.Series, anomaly_indices: np.ndarray, pattern_info: Dict[str, Any] = None) -> float:
        """
        Calculate an anomaly severity score. Prefer Robust Z-Score (using Median/IQR) 
        to handle skewed distributions (like latency) where Standard Z-Score is too weak.
        """
        if not np.any(anomaly_indices):
            return 0.0
            
        anom_values = series[anomaly_indices]
        
        # 1. Try Robust Z-Score (Median Absolute Deviation Proxy)
        # This is essential for heavy-tailed metrics (e.g. latency) where strict Std Dev is huge,
        # masking obvious anomalies.
        if pattern_info:
            q25 = pattern_info.get('25%', None)
            q50 = pattern_info.get('50%', None)
            q75 = pattern_info.get('75%', None)
            
            if q25 is not None and q75 is not None and q50 is not None:
                iqr = q75 - q25
                if iqr > 1e-9:
                    # Robust Z = |X - Median| / (IQR * 0.7413)
                    # 0.7413 is approximation for 1/1.349 (Gaussian consistent)
                    robust_z = (anom_values - q50).abs() / (iqr * 0.7413)
                    return float(robust_z.max())
                    
        # 2. Fallback to Standard Z-Score
        if pattern_info:
            ref_mean = pattern_info.get('mean', series.mean())
            ref_std = pattern_info.get('std', series.std())
            ref_max = pattern_info.get('max', series.max())
        else:
            ref_mean = series.mean()
            ref_std = series.std()
            ref_max = series.max()
            
        # Avoid division by zero
        if ref_std < 1e-9:
            ref_std = 1e-9
            
        z_scores = (anom_values - ref_mean).abs() / ref_std
        max_z = z_scores.max()
        
        # 3. Drop Logic Boost
        # If value is near zero and mean was high, boost score (Critical Failure)
        if ref_mean > 1e-6:
             min_val = anom_values.min()
             if min_val < (ref_mean * 0.1): 
                  # Determine boost based on how consistent the mean is
                  # If Ref CV is low, this is HUGE. If Ref CV is high, still bad.
                  # Force a high score to ensure visibility.
                  return max(max_z, 15.0) 
        
        # 4. Spike Logic Boost utilizing Ratio
        # For sparse metrics where Std is dominated by 0s, Z-score is good.
        # For bounded metrics, Ratio is useful.
        if ref_max > 1e-6:
             max_val = anom_values.max()
             ratio = max_val / ref_max
             if ratio > 2.0: 
                  # If we exceed historical max by 2x, boost.
                  # Add 10 to ensure it tops noise.
                  max_z = max(max_z, 10.0 + ratio)
                  
        return float(max_z)

    def detect(self, series: pd.Series, pattern: Dict[str, Any] = None) -> Dict[str, Any]:
        # Pre-check: Skip constant or too short series
        if len(series) < 5 or series.std() == 0:
            return {}
            
        values = series.values.reshape(-1, 1)
        
        # 0. Pattern-Based Tuning
        is_sparse_pattern = False
        is_discrete_pattern = False
        normal_cv = None
        
        if pattern:
            # Check for sparsity
            is_sparse_pattern = pattern.get('zero_rate', 0) > 0.3
            # Check for low cardinality (discrete)
            is_discrete_pattern = (pattern.get('unique_rate', 1.0) < 0.01) or (pattern.get('count', 0) * pattern.get('unique_rate', 0) < 50)
            # Use normal CV for sensitivity setting
            normal_cv = pattern.get('cv', 0)
        
        # 1. Isolation Forest (Global)
        # decision_function: lower is more anomalous (negative values are outliers)
        iso = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=1) # n_jobs=1 to avoid overhead in threadpool
        iso.fit(values)
        if_scores = iso.decision_function(values) 
        
        # 2. HBOS (Global)
        hbos_scores = self._calculate_hbos(series)
        
        # 3. IQR (Global Hard Threshold)
        iqr_mask = self._calculate_iqr_mask(series)
        
        # 4. Fusion
        # Handle sparse data: If data is primarily zeros (e.g. error count, memory spike),
        # HBOS scores become unreliable due to binning artifacts on small scales.
        # Use Isolation Forest score directly for sparse data.
        curr_is_sparse = (series.abs() < 1e-9).mean() > 0.3
        
        # Use pattern-based knowledge if available, otherwise fallback to current series
        use_sparse_logic = is_sparse_pattern if pattern else curr_is_sparse
        
        if use_sparse_logic:
            anomaly_scores = if_scores
        else:
            # Formula from PPT: (IF - 0.1 * HBOS) / 2
            # IF is negative for anomalies. HBOS is positive for anomalies.
            # Result: More negative = More anomalous.
            anomaly_scores = (if_scores - (0.1 * hbos_scores)) / 2
        
        # 5. Dynamic Thresholding
        # Using 3-sigma rule on the fusion score
        score_mean = anomaly_scores.mean()
        score_std = anomaly_scores.std()
        dynamic_thresh = score_mean - 3 * score_std
        
        # Hard cap to avoid false positives in noisy data
        # If low variance (stable), act more conservatively but allow subtle anomalies
        # PREFER normal CV if known, otherwise calculate from current series
        if pattern and normal_cv is not None:
             cv = normal_cv
        else:
             cv = series.std() / (abs(series.mean()) + 1e-10)
        
        if cv < 0.05:
            min_std = 0.01
            final_thresh_cap = -0.65
        elif cv > 1.0:
            min_std = 0.20 # Relax threshold significantly for high variance
            final_thresh_cap = -0.65
        else:
            min_std = 0.10
            final_thresh_cap = -0.65
            
        score_std = max(score_std, min_std)
        dynamic_thresh = score_mean - 3 * score_std
        
        final_thresh = min(final_thresh_cap, dynamic_thresh)
        
        is_candidate = (anomaly_scores < final_thresh) | iqr_mask
        
        # Post-Processing: Filter insignificant anomalies
        # Require at least 1.5% relative deviation to filter out micro-fluctuations
        # in extremely stable metrics (e.g. Memory Available 0.8% change)
        median = series.median()
        if abs(median) > 1e-9:
            relative_deviation = (series - median).abs() / abs(median)
            # Adjust noise filter based on volatility
            # For low volatility metrics, still require 1.5% deviation
            # For high volatility metrics, require 15% deviation
            filter_thresh = 0.015 if cv < 1.0 else 0.15
            is_candidate = is_candidate & (relative_deviation > filter_thresh)
        if pattern:
             # Global Max Filtering Logic
             # ONLY apply strict suppression of spikes based on global max IF the metric is
             # fundamentally sparse/event-based/discrete (e.g. error counts, restarts).
             # For continuous metrics (latency, cpu, bytes, packets), global max is dangerous
             # because different services have different scales, but share the same metric name.
             
             zero_rate = pattern.get('zero_rate', 0.0)
             # Heuristic: If zero_rate > 0.1, it's likely sparse/event-based.
             is_event_metric = (zero_rate > 0.1)
             
             if is_event_metric:
                 normal_max = pattern.get('max', 0)
                 if normal_max > 0:
                      # Enforce: Spikes must exceed (Normal Max * 1.1) to be considered Global Anomalies.
                      # Candidates that are greater than median (Spikes)
                      spikes = (series > median) & is_candidate
                      # Mask out spikes that are within normal global max
                      valid_spikes = spikes & (series > (normal_max * 1.1))
                      
                      # Drops are candidates < median
                      drops = (series < median) & is_candidate
                      
                      # Recombine
                      is_candidate = valid_spikes | drops
                  
        if pattern and abs(median) <= 1e-9:
             # Handling sparse metrics where median is 0
             # Logic is already covered by the global max check above?
             # No, if median is 0, everything > 0 is a "spike".
             # So line above handles it: valid_spikes = (series > 0) & (series > normal_max * 1.1).
             # This is correct.
             pass
        
        if not np.any(is_candidate):
            return {}
        
        if not np.any(is_candidate):
            return {}
            
        # 6. Local Pattern Verification
        ratio = np.sum(is_candidate) / len(series)
        
        # Filter out minor noise (e.g. single point in long series if not extreme)
        if ratio < 0.02 and not np.any(iqr_mask):
            return {}
            
        pattern_str = self._detect_local_pattern(series, is_candidate)
        
        # 7. Severity Scoring
        # Use pattern-based (historical) stats if available for robust scoring
        score = self._calculate_severity(series, is_candidate, pattern)
        
        return {
            "is_anomaly": True,
            "pattern": pattern_str,
            "timestamps": series.index[is_candidate].tolist(),
            "max_val": series.max(),
            "mean_val": series.mean(),
            "score": score
        }

class MetricAgent:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.detector = EnsembleDetector()
        self.patterns = self._load_patterns()
        
        # Fields to load (Optimization: Don't load everything)
        self.apm_load_fields = ["time", "object_id", "error_ratio", "client_error_ratio", "server_error_ratio", "timeout", "rrt", "rrt_max", "client_error", "server_error", "request", "response", "error"]
        self.infra_load_fields = ["time", "instance", "pod", "value", "kpi_key", "kubernetes_node"]

    def _load_patterns(self) -> Dict[str, Any]:
        """
        Load metric patterns from CSV to guide detection sensitivity.
        """
        pattern_file = Path("/home/tyt21/tlab/unit-test/metric/metric_patterns.csv")
        if not pattern_file.exists():
            logger.warning(f"Metric pattern file not found at {pattern_file}")
            return {}
            
        try:
            df = pd.read_csv(pattern_file)
            # Create dict keyed by metric name
            patterns = df.set_index('metric').to_dict(orient='index')
            logger.info(f"Loaded patterns for {len(patterns)} metrics.")
            return patterns
        except Exception as e:
            logger.error(f"Failed to load metric patterns: {e}")
            return {}

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
        # Load both service-level and pod-level APM metrics
        apm_patterns = [
            "apm/service/*.parquet",
            "apm/pod/*.parquet"
        ]
        
        def _process_apm(f):
            try:
                df = load_parquet(Path(f), self.apm_load_fields, time_filter)
                if df.empty: return None
                
                # Identify which columns are metrics (excluding ID/Time)
                metric_cols = [c for c in df.columns if c in self.apm_load_fields and c not in ["time", "object_id"]]
                
                # Transform wide APM table to long format
                melted = df.melt(id_vars=["time", "object_id"], 
                                 value_vars=metric_cols,
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

                # Pre-clean string "null" values to actual None
                for col in ["pod", "kubernetes_node", "instance", "object_type"]:
                    if col in df.columns:
                        df[col] = df[col].replace("null", None)
                
                # Correctly determine the component name (Service, Node, or Pod)
                file_str = str(f)
                if "infra_node" in file_str:
                    # Node metrics: prioritize kubernetes_node, then instance
                    df["pod"] = df["kubernetes_node"].fillna(df["instance"] if "instance" in df.columns else "unknown")
                elif "infra_pod" in file_str:
                    # Pod metrics: use pod column
                    df["pod"] = df["pod"].fillna("unknown")
                elif "infra_tidb" in file_str or "infra_pd" in file_str or "infra_tikv" in file_str:
                    # TiDB components: construct name using object_type
                    if "object_type" in df.columns:
                        # Map object_type to valid component names
                        type_map = {"tidb": "tidb-tidb", "pd": "tidb-pd", "tikv": "tidb-tikv"}
                        df["pod"] = df["object_type"].map(type_map).fillna(df["object_type"])
                        # Append -0 as per naming convention in valid components/groundtruth
                        df["pod"] = df["pod"].apply(lambda x: f"{x}-0" if x in type_map.values() else x)
                    elif "namespace" in df.columns and (df["namespace"] == "tidb").any():
                        # If namespace is tidb but no object_type, try to infer from filename
                        if "infra_pd" in file_str: df["pod"] = "tidb-pd-0"
                        elif "infra_tikv" in file_str: df["pod"] = "tidb-tikv-0"
                        elif "infra_tidb" in file_str: df["pod"] = "tidb-tidb-0"
                        else: df["pod"] = df["pod"].fillna(df["instance"] if "instance" in df.columns else "unknown")
                    else:
                        df["pod"] = df["pod"].fillna(df["instance"] if "instance" in df.columns else "unknown")
                else:
                    # Other metrics: try pod, then instance
                    df["pod"] = df["pod"].fillna(df["instance"] if "instance" in df.columns else "unknown")
                
                # Final fallback for any remaining nulls
                df["pod"] = df["pod"].fillna("unknown")
                
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
            for pattern in apm_patterns:
                full_pattern = f"{self.root_path}/{day}/metric-parquet/{pattern}"
                tasks.extend([(f, _process_apm) for f in glob.glob(full_pattern)])
            
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
                    "details": event.get("timestamps", [])
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
            kpi_pattern = self.patterns.get(kpi, None)
            result = self.detector.detect(series, pattern=kpi_pattern)
            
            if result:
                # Filter out low-value noise for specific KPIs based on normal max if known
                normal_max = kpi_pattern.get('max', 0) if kpi_pattern else 0.01
                max_tolerance = normal_max * 0.1 if normal_max > 0 else 0.01
                
                if "ratio" in kpi and result["max_val"] < max_tolerance:
                    continue

                # Severity Score Calculation
                # Use robust Z-score logic to assign high values to severe events
                # and low values to noise.
                severity = result.get("score", 0)
                
                # Removed hard filter < 3.0 because Global Z-Score can be misleadingly low 
                # for heavy-tailed metrics (e.g. latency). Rely on sorting & Top-N instead.
                # However, filter astronomically small scores (micro-noise).
                if severity < 0.1:
                    continue
                    
                events.append({
                    "pod": pod,
                    "kpi": kpi,
                    "pattern": result["pattern"],
                    "timestamps": [str(t) for t in result["timestamps"]],
                    "score": severity
                })

        # 4. Filter and Prioritize Events
        # Sort events by severity score descending
        events.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Keep only top 15 most significant anomalies to reduce noise
        if len(events) > 15:
            # Optionally log what's being dropped
            # dropped = events[15:]
            # logger.info(f"Dropping {len(dropped)} low-score events.")
            events = events[:15]

        # 5. Summarize for LLM Observation
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