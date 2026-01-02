from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import pyarrow.dataset as ds
import numpy as np
from pandas import Series
import ruptures as rpt
from exp.utils.input import load_parquet
from exp.utils.time import daterange
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import logging

# 引入增强分析器
try:
    from .enhanced_metric_analyzer import MetricAnalyzer
except ImportError:
    from enhanced_metric_analyzer import MetricAnalyzer

logger = logging.getLogger(__name__)

KPI_MODEL = {
    "cpu_usage": "rbf",
    "timeout": "poisson",
    "error": "l2"
}

KPI_METADATA = {
    "cpu_usage": {"unit": "%", "description": "CPU利用率"},
    "memory_usage": {"unit": "bytes", "description": "内存使用量"},
    "timeout": {"unit": "次", "description": "调用超时次数"},
}
def compute_kpi_correlations(pivot_df: pd.DataFrame, threshold=0.8):
    corr = pivot_df.corr(method='pearson')
    strong_relations = [
        (a, b, corr.loc[a, b])
        for a in corr.columns
        for b in corr.columns
        if a != b and corr.loc[a, b] > threshold
    ]
    return strong_relations

def cluster_time_series(df: pd.DataFrame, top_kpis: list, n_clusters=3) -> Dict:
    clusters = {}
    for kpi in top_kpis:
        pivot = df[df['kpi_key'] == kpi].pivot_table(index="time", columns="pod", values="value")
        pivot = pivot.dropna(axis=1)
        if pivot.shape[1] < 2:
            continue
        try:
            ts_data = to_time_series_dataset(pivot.T.values)
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
            labels = model.fit_predict(ts_data)
            for i, pod in enumerate(pivot.columns):
                clusters.setdefault(pod, {})[kpi] = int(labels[i])
        except Exception:
            continue
    return clusters

def multi_vote_anomaly(outliers, change_points, slope_flag) -> bool:
    vote = int(outliers.sum() > 0) + int(len(change_points) > 1) + int(slope_flag)
    return vote >= 2

def detect_change_points(series: Series, model="rbf", pen=10) -> List:
    algo = rpt.Pelt(model=model).fit(series.values)
    result = algo.predict(pen=pen)
    return result

def rolling_std_anomaly(series: Series, window=5, z_thresh=3):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score.abs() > z_thresh

def slope_anomaly(series: Series, threshold=0.75) -> Tuple[bool, float]:
    x = np.arange(len(series))
    y = series.values
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return abs(slope) > threshold and r2 > 0.85, slope

def joint_anomaly_pca(df: pd.DataFrame, top_kpis: list, threshold=3.0) -> List:
    pivot = df.pivot_table(index="time", columns="kpi_key", values="value")
    pivot = pivot[top_kpis].dropna()
    if pivot.shape[0] < 10:
        return []
    pca = PCA(n_components=1)
    scores = pca.fit_transform(pivot)
    z = (scores - scores.mean()) / scores.std()
    abnormal_times = pivot.index[(np.abs(z) > threshold).flatten()]
    if len(abnormal_times) == 0:
        return []

    time_diffs = pd.Series(abnormal_times).diff().fillna(pd.Timedelta(seconds=0))
    continuous_groups = (time_diffs <= pd.Timedelta(minutes=5)).astype(int).cumsum()
    grouped = pd.Series(abnormal_times).groupby(continuous_groups)
    filtered = [group.tolist() for name, group in grouped if len(group) >= 2]

    if len(top_kpis) >= 3 and filtered:
        return [t for group in filtered for t in group]

    return []

def extract_correlated_groups(correlations, infra_df, min_overlap_secs=180):
    pod_kpi_time = defaultdict(lambda: defaultdict(list))
    for _, row in infra_df.iterrows():
        pod_kpi_time[row["pod"]][row["kpi_key"]].append(row["time"])

    groups = []
    for kpi_a, kpi_b, corr in correlations:
        if corr < 0.9:
            continue
        pods_a = {pod for pod, kpis in pod_kpi_time.items() if kpi_a in kpis}
        pods_b = {pod for pod, kpis in pod_kpi_time.items() if kpi_b in kpis}
        shared = list(pods_a & pods_b)
        if len(shared) >= 2:
            times_a = set(pod_kpi_time[shared[0]][kpi_a])
            times_b = set(pod_kpi_time[shared[1]][kpi_b])
            common = sorted(times_a & times_b)
            if len(common) >= 2 and (common[-1] - common[0]).total_seconds() > min_overlap_secs:
                groups.append({
                    "group": shared,
                    "shared_kpi": f"{kpi_a}/{kpi_b}",
                    "time_overlap": [str(common[0]), str(common[-1])]
                })
    return groups


class MetricAgent:
    def __init__(self, root_path: str, error_threshold=0.1, timeout_threshold=10):
        self.root_path = Path(root_path)
        self.error_threshold = error_threshold
        self.timeout_threshold = timeout_threshold
        self.apm_fields = [
            "time", "request", "response", "rrt", "rrt_max", "error",
            "client_error", "server_error", "timeout",
            "error_ratio", "client_error_ratio", "server_error_ratio", "object_id", "object_type"
        ]
        self.infra_fields = [
            "time", "cf", "device", "instance", "kpi_key", "kpi_name", "kubernetes_node",
            "mountpoint", "namespace", "object_type", "pod", "value", "sql_type", "type"
        ]
        self.infra_schema_fields = [
            "time", "cf", "device", "instance", "kpi_key", "kpi_name", "kubernetes_node",
            "mountpoint", "namespace", "object_type", "pod", "sql_type", "type"
        ]
        self.DOMAIN_THRESHOLDS = {
            "cpu_usage": 80,
        }
        self.weights = {
            "error_ratio": 2.0, 
            "client_error_ratio": 1.5,
            "server_error_ratio": 2.5,
            "timeout": 1.8,
            "rrt": 1.2,
            "rrt_max": 2.0,
            }
        
        # Initialize enhanced analyzer
        self.enhanced_analyzer = MetricAnalyzer(str(self.root_path))

    def load_apm(self, start: datetime, end: datetime, max_workers=4):
        files = []
        for day in daterange(start, end):
            files.extend(glob.glob(f"{self.root_path}/{day}/metric-parquet/apm/service/*.parquet"))

        results = []
        filter = (ds.field("time") >= start) & (ds.field("time") <= end)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(load_parquet, Path(f), self.apm_fields, filter): f for f in files}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    results.append(df[self.apm_fields])

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def load_infra_or_other(self, file_pattern: str, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        files = []
        for day in daterange(start, end):
            files.extend(glob.glob(f"{self.root_path}/{day}/metric-parquet/{file_pattern}"))

        results = []
        filter = (ds.field("time") >= start) & (ds.field("time") <= end)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(load_parquet, Path(f), filter_=filter): f for f in files}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    # 识别 value 列：排除掉 Schema 中定义的列，剩下的就是 value 列
                    metric_candidates = list(set(df.columns) - set(self.infra_schema_fields))
                    if len(metric_candidates) == 1:
                        df["value"] = df[metric_candidates[0]]
                        
                        # 处理缺失 pod 列的情况（Node 或 TiDB 指标通常没有 pod 列）
                        if "pod" not in df.columns:
                            # 优先使用 instance 作为标识符（如 Node 名或 IP:Port）
                            if "instance" in df.columns:
                                df["pod"] = df["instance"]
                            else:
                                df["pod"] = "unknown"

                        df["pod"] = df["pod"].astype(str)
                        
                        # 确保所有需要的 infra_fields 都存在，不存在的填充 None
                        for field in self.infra_fields:
                            if field not in df.columns:
                                df[field] = None
                                
                        results.append(df[self.infra_fields])

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def score(self, start_time: datetime, end_time: datetime) -> List:
        scores = []
        
        # 1. 分析 APM 指标 (服务级别)
        apm = self.load_apm(start_time, end_time)
        if not apm.empty:
            for service, group in apm.groupby("object_id"):
                if group.empty:
                    continue
                
                if group['error_ratio'].mean() > self.error_threshold:
                    scores.append({
                        'service': service,
                        'kpi': 'error_ratio',
                        'reason': f"Error ratio {group['error_ratio'].mean():.2f} exceeds threshold {self.error_threshold:.2f}",\
                    })
                if group['timeout'].mean() > self.timeout_threshold:
                    scores.append({
                        'service': service,
                        'kpi': 'timeout',
                        'reason': f"Timeout {group['timeout'].mean():.2f} exceeds threshold {self.timeout_threshold:.2f}",
                    })
        
        # 2. 分析基础设施和组件指标 (Pod/Node 级别)
        # 调用 query_metrics 检测 CPU/内存/TiDB 异常
        infra_result = self.query_metrics(start_time, end_time)
        
        # 如果在基础设施中发现显著异常，将其添加到 scores 中
        # main.py 使用此处返回的列表来告知 JudgeAgent
        if infra_result and infra_result.get("observation") != "No significant metric anomalies detected.":
            scores.append({
                'service': 'Infrastructure/Components',
                'kpi': 'various',
                'reason': infra_result["observation"],
                'details': infra_result.get("events", [])
            })

        return scores

    def query_metrics(self, start_time: datetime, end_time: datetime):
        # 修改：同时加载 Pod, Node, TiDB 和 Other 指标
        infra_pod = self.load_infra_or_other('infra/infra_pod/*.parquet', start_time, end_time)
        infra_node = self.load_infra_or_other('infra/infra_node/*.parquet', start_time, end_time)
        infra_tidb = self.load_infra_or_other('infra/infra_tidb/*.parquet', start_time, end_time)
        other = self.load_infra_or_other('other/*.parquet', start_time, end_time)
        
        # 合并所有基础设施指标
        infra_and_other = pd.concat([infra_pod, infra_node, infra_tidb, other], ignore_index=True)
        
        logger.info(f"Loaded {len(infra_and_other)} infra/other records from {start_time} to {end_time}")

        anomalies = []
        details = {}
        events = []

        if infra_and_other.empty:
             return {
                "observation": "No significant metric anomalies detected.",
                "details": {},
                "events": []
            }

        for pod, pod_group in infra_and_other.groupby("pod"):
            for kpi, group in pod_group.groupby("kpi_key"):
                values = group['value'].reset_index(drop=True)
                timestamps = group['time'].reset_index(drop=True)

                if len(values) < 10:
                    continue

                try:
                    model = KPI_MODEL.get(kpi.lower(), "rbf")
                    change_points = detect_change_points(values, model=model)
                except Exception as e:
                    change_points = []

                outliers = rolling_std_anomaly(values)
                if outliers.sum() > 0:
                    outlier_times = timestamps[outliers].tolist()
                    details.setdefault(pod, {}).setdefault(kpi, {}).update({"outliers": [str(t) for t in outlier_times]})
                    events.append({"pod": pod, "kpi": kpi, "type": "rolling_outlier", "timestamps": [str(t) for t in outlier_times]})
                    anomalies.append((pod, kpi))

                if len(change_points) >= 2:
                    cp_times = [timestamps[i-1] for i in change_points if i-1 < len(timestamps)]
                    if len(cp_times) >= 2:
                        details.setdefault(kpi, {}).update({
                        "change_points": cp_times,
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99),
                        "max": values.max(),
                        "min": values.min(),
                    })
                    events.append({"pod": pod, "kpi": kpi, "type": "change_point", "timestamps": [str(t) for t in cp_times]})
                    anomalies.append((pod, kpi))

                slope_flag, slope = slope_anomaly(values)
                if slope_flag:
                    details.setdefault(kpi, {}).update({"trend_slope": slope})
                    events.append({"kpi": kpi, "type": "trend", "slope": slope})
                    anomalies.append((pod, kpi))

        top_kpis = infra_and_other['kpi_key'].value_counts().nlargest(10).index.tolist()
        
        abnormal_times = joint_anomaly_pca(infra_and_other, top_kpis)
        if abnormal_times:
            events.append({"type": "joint_anomaly", "top_kpis": top_kpis, "timestamps": [str(t) for t in abnormal_times]})
            for kpi in top_kpis:
                # 标记为系统级或特定 Pod（如果可识别），此处暂标记 KPI 为通用
                anomalies.append(("system", kpi))
                details.setdefault(kpi, {}).update({"joint_anomaly_times": abnormal_times})

        summary = {}
        for pod, kpi in anomalies:
            if pod == "null" or pod is None:
                pod = "pd"

            if pod not in summary:
                summary[pod] = set()
            
            kpi = kpi[4:] if kpi.startswith("pod_") else kpi
            summary[pod].add(kpi)

        summary = {pod: sorted(list(kpis)) for pod, kpis in summary.items()}

        if anomalies:
            observation = f"Detected anomalies in metrics: {summary}."
        else:
            observation = "No significant metric anomalies detected."

        logger.info({
            "observation": observation,
            "details": details,
            "events": events
        })

        return {
            "observation": observation,
            "details": details,
            "events": events
        }
    
    def analyze_curve_features(self, start_time: datetime, end_time: datetime) -> List[str]:
        """
        Enhanced metric analysis with curve feature extraction
        
        Returns formatted results according to specification:
        [指标粒度_指标名][实例类型 实例名称][均值变化率] ([正常均值] -> [异常均值]) ([曲线特征])
        
        Args:
            start_time: Analysis start time
            end_time: Analysis end time
            
        Returns:
            List of formatted anomaly analysis results
        """
        try:
            results = self.enhanced_analyzer.analyze_metrics(start_time, end_time)
            logger.info(f"Enhanced analysis found {len(results)} anomalous patterns")
            return results
        except Exception as e:
            logger.error(f"Enhanced curve feature analysis failed: {e}")
            return []

        # summary = {}
        # for pod, kpi in anomalies:
        #     if pod and pod != "null":
        #         summary.setdefault(pod, set()).add(kpi[4:] if kpi.startswith("pod_") else kpi)
        # summary = {pod: sorted(list(kpis)) for pod, kpis in summary.items()}

        # pivot_all = infra_and_other.pivot_table(index="time", columns="kpi_key", values="value")
        # correlations = compute_kpi_correlations(pivot_all[top_kpis]) if not pivot_all.empty else []
        # clusters = cluster_time_series(infra_and_other, top_kpis)

        # correlated_groups = extract_correlated_groups(correlations, infra_and_other)

        # anomaly_count = len(anomalies)
        # pod_count = len(summary)
        # kpi_set = sorted(set(kpi for kpis in summary.values() for kpi in kpis))
        # kpi_count = len(kpi_set)
        # kpi_list = ", ".join(kpi_set[:5]) + ("..." if kpi_count > 5 else "")

        # if anomaly_count:
        #     observation = (
        #         f"Detected {anomaly_count} metric anomalies across {pod_count} pods and {kpi_count} KPIs "
        #         f"(e.g., {kpi_list})."
        #     )
        # else:
        #     observation = "No significant metric anomalies detected."

        # return {
        #     "observation": observation,
        #     "summary": summary,
        #     "events": events,
        #     "correlated_groups": correlated_groups
        # }
