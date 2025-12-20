import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import difflib

import numpy as np
import pandas as pd
from scipy.stats import zscore

from exp.utils.input import load_parquet_by_hour

logger = logging.getLogger(__name__)

# traceID	e0b937776abecfa2d946dcd4b3f2f2cf
# spanID	c7a0a12ff9b0685e
# operationName	hipstershop.ProductCatalogService/ListProducts
# references	[{'refType': 'CHILD_OF', 'spanID': 'a97c04e2e6c86766', 'traceID': 'e0b937776abecfa2d946dcd4b3f2f2cf'}]
# startTimeMillis	1749142862303
# duration	65
# TODO: use status.code, rpc.method
# Does ip and peer need to be used?
# tags
# [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}
#  {'key': 'rpc.service', 'type': 'string', 'value': 'hipstershop.ProductCatalogService'}
#  {'key': 'rpc.method', 'type': 'string', 'value': 'ListProducts'}
#  {'key': 'net.peer.ip', 'type': 'string', 'value': '10.233.77.230'}
#  {'key': 'net.peer.port', 'type': 'string', 'value': '33572'}
#  {'key': 'instrumentation.name', 'type': 'string', 'value': 'go.opentelemetry.io/otel/sdk/tracer'}
#  {'key': 'status.code', 'type': 'int64', 'value': '0'}
#  {'key': 'status.message', 'type': 'string', 'value': ''}
#  {'key': 'span.kind', 'type': 'string', 'value': 'server'}
#  {'key': 'internal.span.format', 'type': 'string', 'value': 'jaeger'}]
# TODO: how to use logs in span message
# logs
# [{'fields': array([{'key': 'message.type', 'type': 'string', 'value': 'RECEIVED'}, {'key': 'message.id', 'type': 'int64', 'value': '1'},{'key': 'message.uncompressed_size', 'type': 'int64', 'value': '0'}, {'key': 'name', 'type': 'string', 'value': 'message'}],dtype=object), 'timestamp': 1749142862303896}
# {'fields': array([{'key': 'message.type', 'type': 'string', 'value': 'SENT'}, {'key': 'message.id', 'type': 'int64', 'value': '1'}, {'key': 'message.uncompressed_size', 'type': 'int64', 'value': '2541'}, {'key': 'name', 'type': 'string', 'value': 'message'}], dtype=object), 'timestamp': 1749142862303934}                           ]
# TODO: use tags (name -> pod, node_name -> node, namespace -> namespace)
# Does ip need to be used?
# process
# {
# 'serviceName': 'productcatalogservice',
# 'tags': array([
#     {'key': 'exporter', 'type': 'string', 'value': 'jaeger'},
#     {'key': 'float', 'type': 'float64', 'value': '312.23'},
#     {'key': 'ip', 'type': 'string', 'value': '10.233.79.154'},
#     {'key': 'name', 'type': 'string', 'value': 'productcatalogservice-1'},
#     {'key': 'node_name', 'type': 'string', 'value': 'aiops-k8s-06'},
#     {'key': 'namespace', 'type': 'string', 'value': 'hipstershop'}
#     ],
#     dtype=object)}

class TraceAgent:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.fields = [
            "traceID", "spanID", "operationName", "references", "startTimeMillis", "duration", "tags", "logs",
            "process"]
        self.analysis_fields = [
            "traceID", "spanID", "operationName", "references", "start", "end", "duration", "tags", "logs", 
            "namespace", "node", "pod", 'kind', 'status_code', 'http_code', 'is_error', 'process'
        ]

    def load_spans(self, start: datetime, end: datetime, max_workers=4):
        def callback(spans: pd.DataFrame) -> pd.DataFrame:
            # 1. 向量化处理时间（Pandas 原生向量化已经很快，保留）
            spans['start'] = pd.to_datetime(spans["startTimeMillis"], unit="ms")
            spans['end'] = spans['start'] + pd.to_timedelta(spans['duration'], unit='ms')

            # --- 优化核心：使用列表推导式代替 apply ---
            
            # 2. 优化 process 字段解析
            # 将 Series 转为原生 List，避免 Pandas 每一行创建 Series 的开销
            process_raw = spans['process'].to_list()
            nodes, nss, pods = [], [], []
            
            for p in process_raw:
                # 预设默认值
                node, ns, pod = None, None, None
                tags = p.get('tags')
                if isinstance(tags, (list, np.ndarray)):
                    for tag in tags:
                        # 假设 tag 是字典
                        k = tag.get('key')
                        if k == 'node_name': node = tag.get('value')
                        elif k == 'namespace': ns = tag.get('value')
                        elif k == 'name': pod = tag.get('value')
                nodes.append(node)
                nss.append(ns)
                pods.append(pod)
                
            spans['node'] = nodes
            spans['namespace'] = nss
            spans['pod'] = pods

            # 3. 优化 tags 字段解析
            tags_raw = spans['tags'].to_list()
            kinds, status_codes, http_codes, is_errors = [], [], [], []
            
            for tags in tags_raw:
                kind, sc, hc, err = 'internal', '0', 0, False
                if isinstance(tags, (list, np.ndarray)):
                    for tag in tags:
                        k = tag.get('key')
                        v = tag.get('value')
                        if k == "span.kind": 
                            kind = str(v).lower()
                        elif k in ("status.code", "grpc.status_code"): 
                            sc = v
                        elif k == "http.status_code": 
                            # 避免对非数字调用 int()
                            hc = int(v) if (isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())) else 0
                        elif k == "error": 
                            err = (v is True or str(v).lower() == "true")
                            
                kinds.append(kind)
                status_codes.append(sc)
                http_codes.append(hc)
                is_errors.append(err)
                
            spans['kind'] = kinds
            spans['status_code'] = status_codes
            spans['http_code'] = http_codes
            spans['is_error'] = is_errors

            return spans

        return load_parquet_by_hour(
            start, end, self.root_path,
            file_pattern="{dataset}/{day}/trace-parquet/trace_jaeger-span_{day}_{hour}-00-00.parquet",
            load_fields=self.fields,
            return_fields=self.analysis_fields,
            filter_=None,
            callback=callback,
            max_workers=max_workers
        )

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Levenshtein similarity ratio using difflib (0.0 to 1.0).
        hwlyyzc solution requires > 95% similarity for merging.
        """
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def _compress_messages(self, messages: List[str], threshold: float = 0.95) -> List[str]:
        """
        Compress redundant error messages using Levenshtein distance.
        """
        if not messages:
            return []
        
        unique_templates = []
        for msg in messages:
            if not msg: continue
            
            matched = False
            for i, existing in enumerate(unique_templates):
                if self._calculate_similarity(msg, existing) > threshold:
                    # Keep the shorter one or just the first one as representative
                    matched = True
                    break
            if not matched:
                unique_templates.append(msg)
        
        return unique_templates

    def _extract_error_message(self, logs: np.ndarray, tags: np.ndarray) -> str:
        """
        Extract meaningful error message from logs or tags.
        """
        msgs = []
        # Check logs for "message", "error", "stack", etc.
        if isinstance(logs, np.ndarray):
            for log in logs:
                fields = log.get('fields', [])
                if isinstance(fields, np.ndarray):
                    for f in fields:
                        key = f.get('key', '')
                        if key in ['message', 'error.object', 'error.kind', 'event']:
                            val = f.get('value', '')
                            if val and str(val) not in ['error', 'SENT', 'RECEIVED']:
                                msgs.append(str(val))
        
        # Check tags if no logs found
        if not msgs and isinstance(tags, np.ndarray):
             for tag in tags:
                if tag.get('key') == 'status.message':
                    msgs.append(str(tag.get('value')))

        return " | ".join(msgs[:2]) # Return first 2 distinct parts

    def _detect_pod_distribution_anomaly(self, pod_counts: Dict[str, int]) -> List[str]:
        """
        Use Z-score to detect Pod distribution anomalies.
        hwlyyzc logic: outside 3 standard deviations.
        """
        if len(pod_counts) < 2:
            return list(pod_counts.keys())
        
        pods = list(pod_counts.keys())
        counts = list(pod_counts.values())
        
        # If variance is 0 (all same), no anomaly in distribution
        if np.std(counts) == 0:
            return pods 
            
        z_scores = zscore(counts)
        # We are looking for pods with significantly HIGHER counts (positive Z-score > 3)
        anomalous_pods = []
        for i, z in enumerate(z_scores):
            # Using 3.0 as per paper, but if counts are small, Z-score might not reach 3.
            # Adding a safeguard: count must be > mean.
            if z > 2.5: # Slightly relaxed from 3 for smaller sample sizes
                anomalous_pods.append(pods[i])
        
        return anomalous_pods if anomalous_pods else pods # If no outlier, return all (systematic issue)


    def _get_stats_per_link(self, df: pd.DataFrame):
        """
        统计每个调用链路(src, dst, op)的错误和慢调用频率
        """
        stats = defaultdict(lambda: {'error_count': 0, 'slow_count': 0, 'total': 0})
        
        # 预先计算 P95 阈值
        op_thresholds = df.groupby('operationName')['duration'].quantile(0.95).to_dict()
        
        # 建立索引方便查找父节点（简化版，实际可使用 span_index）
        for _, row in df.iterrows():
            dst_svc = row['process'].get('serviceName', 'unknown')
            op = row['operationName']
            
            # 判定异常
            is_err = row['is_error'] or row['http_code'] >= 400 or row['status_code'] != '0'
            is_slow = row['duration'] > op_thresholds.get(op, float('inf'))
            
            key = (dst_svc, op) # 这里简化为目标服务和操作，也可以加上源服务
            stats[key]['total'] += 1
            if is_err: stats[key]['error_count'] += 1
            if is_slow: stats[key]['slow_count'] += 1
            
        return stats, op_thresholds

    def score(self, start_time: datetime, end_time: datetime, max_workers=4) -> List[Dict]:
        """
        按照 hwlyyzc 方案修复版：引入时间维度 Z-score 背景降噪
        """
        # 1. 加载历史基准数据 (例如异常前 30 分钟)
        baseline_start = start_time - timedelta(minutes=30)
        logger.info(f"Loading baseline from {baseline_start} to {start_time}")
        baseline_spans = self.load_spans(baseline_start, start_time, max_workers=max_workers)
        
        # 2. 加载当前异常数据
        logger.info(f"Loading anomalous data from {start_time} to {end_time}")
        current_spans = self.load_spans(start_time, end_time, max_workers=max_workers)
        
        if current_spans.empty:
            return []

        # 3. 计算基线统计量 (计算历史每分钟的错误均值和标准差)
        # 将基线按分钟切分，计算频率的波动
        baseline_spans['minute'] = baseline_spans['start'].dt.floor('min')
        baseline_min_stats = baseline_spans.groupby(['minute', 'operationName']).agg(
            error_cnt=('is_error', 'sum'),
            total_cnt=('spanID', 'count')
        ).reset_index()

        # 计算每个接口的历史均值和标准差
        history_metrics = baseline_min_stats.groupby('operationName')['error_cnt'].agg(['mean', 'std']).fillna(0).to_dict('index')

        # 4. 获取当前窗口的异常
        # 筛选条件：Error 或 Latency > P95 (这里的 P95 应该参考历史)
        global_p95 = baseline_spans.groupby('operationName')['duration'].quantile(0.95).to_dict()
        
        link_stats = defaultdict(lambda: {
            'count': 0,
            'source_pods': defaultdict(int),
            'target_pods': defaultdict(int),
            'messages': [],
            'latency_vals': [],
            'error_codes': set(),
            'burst_score': 0.0
        })

        span_index = current_spans.set_index('spanID')[['process', 'pod']].to_dict('index')

        for _, row in current_spans.iterrows():
            op = row['operationName']
            is_err = row['is_error'] or row['http_code'] >= 400 or row['status_code'] != '0'
            is_slow = row['duration'] > global_p95.get(op, float('inf'))
            
            if not (is_err or is_slow):
                continue

            # --- 时间维度 Z-score 降噪核心逻辑 ---
            if is_err and op in history_metrics:
                m = history_metrics[op]['mean']
                s = history_metrics[op]['std']
                # 计算当前错误是否为突发 (假设当前窗口是 1 分钟内的聚合)
                # Z = (Current_Error - Mean) / Std
                # 如果 Std 很小，设置一个最小 Std 保护
                z = (1 - m) / max(s, 0.1) 
                if z < 2.0 and m > 0.5: # 如果历史错误率就很高，且当前未显著突增，则过滤
                    continue 

            # --- 以下保留原有的空间维度统计和语义压缩 ---
            dst_svc = row['process'].get('serviceName', 'unknown')
            dst_pod = row.get('pod') or dst_svc
            
            # 查找父节点获取源
            src_svc, src_pod = "User", "unknown"
            refs = row.get('references')
            if isinstance(refs, np.ndarray) and len(refs) > 0:
                parent_id = refs[0].get('spanID')
                if parent_id in span_index:
                    src_svc = span_index[parent_id]['process'].get('serviceName', 'unknown')
                    src_pod = span_index[parent_id].get('pod') or src_svc

            key = (src_svc, dst_svc)
            stat = link_stats[key]
            stat['count'] += 1
            stat['target_pods'][dst_pod] += 1
            stat['source_pods'][src_pod] += 1
            
            if is_err:
                stat['error_codes'].add(f"HTTP {row['http_code']}" if row['http_code'] else f"gRPC {row['status_code']}")
                msg = self._extract_error_message(row['logs'], row['tags'])
                if msg: stat['messages'].append(msg)
            if is_slow:
                stat['latency_vals'].append(row['duration'])

        # 5. 格式化输出 (包含语义去重)
        results = []
        for (src, dst), data in link_stats.items():
            # 语义去重
            compressed_msgs = self._compress_messages(data['messages'], threshold=0.95)
            
            results.append({
                "span": {
                    "source": src,
                    "target": dst,
                    "target_pods": self._detect_pod_distribution_anomaly(data['target_pods']),
                    "count": data['count']
                },
                "message": {
                    "error_messages": compressed_msgs[:3],
                    "latency": {"avg": np.mean(data['latency_vals'])} if data['latency_vals'] else {}
                }
            })
        return results
    
# The span tag can be classified as
# [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.StackExchangeRedis'}
#  {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.10'}
#  {'key': 'db.system', 'type': 'string', 'value': 'redis'}
#  {'key': 'db.redis.flags', 'type': 'string', 'value': 'DemandMaster'}
#  {'key': 'db.statement', 'type': 'string', 'value': 'HMSET'}
#  {'key': 'net.peer.name', 'type': 'string', 'value': 'redis-cart'}
#  {'key': 'net.peer.port', 'type': 'int64', 'value': '6379'}
#  {'key': 'db.redis.database_index', 'type': 'int64', 'value': '0'}
#  {'key': 'peer.service', 'type': 'string', 'value': 'redis-cart:6379'}
#  {'key': 'span.kind', 'type': 'string', 'value': 'client'}
#  {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}]

# [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.AspNetCore'}
# {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.0'}
# {'key': 'server.address', 'type': 'string', 'value': 'cartservice'}
# {'key': 'server.port', 'type': 'int64', 'value': '7070'}
# {'key': 'http.request.method', 'type': 'string', 'value': 'POST'}
# {'key': 'url.scheme', 'type': 'string', 'value': 'http'}
# {'key': 'url.path', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'network.protocol.version', 'type': 'string', 'value': '2'}
# {'key': 'user_agent.original', 'type': 'string', 'value': 'grpc-go/1.31.0'}
# {'key': 'grpc.method', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'grpc.status_code', 'type': 'string', 'value': '0'}
# {'key': 'http.route', 'type': 'string', 'value': '/hipstershop.CartService/GetCart'}
# {'key': 'http.response.status_code', 'type': 'int64', 'value': '200'}
# {'key': 'span.kind', 'type': 'string', 'value': 'server'}
# {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}]

# [{'key': 'rpc.system', 'type': 'string', 'value': 'grpc'}
# {'key': 'rpc.service', 'type': 'string', 'value': 'hipstershop.RecommendationService'}
# {'key': 'rpc.method', 'type': 'string', 'value': 'ListRecommendations'}
# {'key': 'net.peer.ip', 'type': 'string', 'value': 'recommendationservice'}
# {'key': 'net.peer.port', 'type': 'string', 'value': '8080'}
# {'key': 'instrumentation.name', 'type': 'string', 'value': 'go.opentelemetry.io/otel/sdk/tracer'}
# {'key': 'status.code', 'type': 'int64', 'value': '0'}
# {'key': 'status.message', 'type': 'string', 'value': ''}
# {'key': 'span.kind', 'type': 'string', 'value': 'client'}
# {'key': 'internal.span.format', 'type': 'string', 'value': 'jaeger'}]
