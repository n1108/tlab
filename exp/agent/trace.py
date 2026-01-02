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
            # 1. 向量化处理时间
            spans['start'] = pd.to_datetime(spans["startTimeMillis"], unit="ms")
            
            # 精确时间过滤
            if spans.empty:
                return spans
            
            mask = (spans['start'] >= start) & (spans['start'] <= end)
            spans = spans[mask].copy()
            
            if spans.empty:
                return spans

            spans['end'] = spans['start'] + pd.to_timedelta(spans['duration'], unit='us')
            
            # 2. 优化 process 字段解析
            process_raw = spans['process'].to_list()
            nodes, nss, pods = [], [], []
            
            for p in process_raw:
                node, ns, pod = None, None, None
                tags = p.get('tags')
                if isinstance(tags, (list, np.ndarray)):
                    for tag in tags:
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
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def _compress_messages(self, messages: List[str], threshold: float = 0.95) -> List[str]:
        if not messages:
            return []
        unique_templates = []
        for msg in messages:
            if not msg: continue
            matched = False
            for i, existing in enumerate(unique_templates):
                if self._calculate_similarity(msg, existing) > threshold:
                    matched = True
                    break
            if not matched:
                unique_templates.append(msg)
        return unique_templates

    def _extract_error_message(self, logs: np.ndarray, tags: np.ndarray) -> str:
        msgs = []
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
        
        if not msgs and isinstance(tags, np.ndarray):
             for tag in tags:
                if tag.get('key') == 'status.message':
                    msgs.append(str(tag.get('value')))

        return " | ".join(msgs[:2]) 

    def _detect_pod_distribution_anomaly(self, pod_counts: Dict[str, int]) -> List[str]:
        if len(pod_counts) < 2:
            return list(pod_counts.keys())
        
        pods = list(pod_counts.keys())
        counts = list(pod_counts.values())
        
        if np.std(counts) == 0:
            return pods 
            
        z_scores = zscore(counts)
        anomalous_pods = []
        for i, z in enumerate(z_scores):
            if z > 2.5: 
                anomalous_pods.append(pods[i])
        
        return anomalous_pods if anomalous_pods else pods

    def score(self, start_time: datetime, end_time: datetime, max_workers=4) -> List[Dict]:
        # 1. 加载历史基准数据
        baseline_start = start_time - timedelta(minutes=30)
        logger.info(f"Loading baseline from {baseline_start} to {start_time}")
        baseline_spans = self.load_spans(baseline_start, start_time, max_workers=max_workers)
        
        # 2. 加载当前异常数据
        logger.info(f"Loading anomalous data from {start_time} to {end_time}")
        current_spans = self.load_spans(start_time, end_time, max_workers=max_workers)
        
        if current_spans.empty:
            return []

        # 3. 计算基线统计量
        history_metrics = {}
        if not baseline_spans.empty:
            # 计算综合错误标志
            baseline_spans['combined_error'] = (
                baseline_spans['is_error'] | 
                (baseline_spans['http_code'] >= 400) | 
                ((baseline_spans['status_code'] != '0') & (baseline_spans['status_code'] != 0))
            )
            baseline_spans['minute'] = baseline_spans['start'].dt.floor('min')
            baseline_min_stats = baseline_spans.groupby(['minute', 'operationName']).agg(
                error_cnt=('combined_error', 'sum')
            ).reset_index()
            history_metrics = baseline_min_stats.groupby('operationName')['error_cnt'].agg(['mean', 'std']).fillna(0).to_dict('index')

        # 计算 P95 延迟阈值
        if not baseline_spans.empty:
            global_p95 = baseline_spans.groupby('operationName')['duration'].quantile(0.95).to_dict()
        else:
            global_p95 = current_spans.groupby('operationName')['duration'].quantile(0.95).to_dict()

        # 4. 预计算当前窗口的统计特征
        duration_minutes = (end_time - start_time).total_seconds() / 60.0
        if duration_minutes <= 0: duration_minutes = 1.0

        # 使用综合错误条件来统计当前错误数
        is_failed_mask = (
            current_spans['is_error'] | 
            (current_spans['http_code'] >= 400) | 
            ((current_spans['status_code'] != '0') & (current_spans['status_code'] != 0))
        )
        current_error_counts = current_spans[is_failed_mask].groupby('operationName').size().to_dict()
        
        anomalous_error_ops = set()
        
        for op, count in current_error_counts.items():
            current_rate = count / duration_minutes
            
            if op not in history_metrics:
                anomalous_error_ops.add(op)
                continue
            
            stats = history_metrics[op]
            h_mean = stats['mean']
            h_std = stats['std']
            
            safe_std = max(h_std, 0.1) 
            z = (current_rate - h_mean) / safe_std
            
            # Z-Score > 3.0 或 绝对数量显著 (例如 > 5 且超过均值1.5倍)
            if z > 3.0 or (count > 5 and current_rate > h_mean * 1.5):
                anomalous_error_ops.add(op)

        # 5. 遍历 Span 进行聚合
        # 修改聚合逻辑：按 (Source, Target) 分组，内部再按 (Pod, Node) 聚合
        # link_stats structure:
        # { (src_svc, dst_svc): { (dst_pod, dst_node): { count, latency_vals, messages } } }
        link_stats = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'messages': [],
            'latency_vals': [],
            'error_codes': set()
        }))

        # 建立索引加速父子查找，加入 node 信息
        span_index = current_spans.set_index('spanID')[['process', 'pod', 'node']].to_dict('index')

        for idx, row in current_spans.iterrows():
            op = row['operationName']
            
            # 错误判定逻辑保持不变
            raw_is_err = row['is_error'] or row['http_code'] >= 400 or (row['status_code'] != '0' and row['status_code'] != 0)
            is_slow = row['duration'] > global_p95.get(op, float('inf'))
            is_valid_err = raw_is_err and (op in anomalous_error_ops)

            if not (is_valid_err or is_slow):
                continue

            dst_svc = row['process'].get('serviceName', 'unknown')
            dst_pod = row.get('pod') or dst_svc
            # 获取目标 Node
            dst_node = row.get('node') or 'unknown'
            
            src_svc = "User"
            refs = row.get('references')
            if isinstance(refs, np.ndarray) and len(refs) > 0:
                parent_id = refs[0].get('spanID')
                if parent_id in span_index:
                    src_svc = span_index[parent_id]['process'].get('serviceName', 'unknown')

            # 聚合到 Pod + Node 粒度
            stat = link_stats[(src_svc, dst_svc)][(dst_pod, dst_node)]
            stat['count'] += 1
            
            if is_valid_err:
                if row['http_code']:
                    code_str = f"HTTP {row['http_code']}"
                elif row['status_code'] != '0' and row['status_code'] != 0:
                    code_str = f"gRPC {row['status_code']}"
                else:
                    code_str = "Error Tag"
                    
                stat['error_codes'].add(code_str)
                msg = self._extract_error_message(row['logs'], row['tags'])
                if msg: stat['messages'].append(msg)
            
            # 记录所有异常或慢请求的延迟
            stat['latency_vals'].append(row['duration'])

        # 6. 格式化输出
        results = []
        for (src, dst), pod_groups in link_stats.items():
            
            pod_details = []
            
            for (pod, node), data in pod_groups.items():
                compressed_msgs = self._compress_messages(data['messages'], threshold=0.95)
                
                avg_latency_ms = 0
                if data['latency_vals']:
                    avg_latency_ms = round(float(np.mean(data['latency_vals'])) / 1000, 2)

                if not compressed_msgs and avg_latency_ms == 0:
                    continue
                
                pod_details.append({
                    "pod": pod,
                    "node": node,
                    "count": data['count'],
                    "avg_latency_ms": avg_latency_ms,
                    "error_messages": compressed_msgs[:2] # 限制每Pod的错误消息数
                })

            if not pod_details:
                continue

            # 按延迟排序，突出问题最严重的 Pod/Node
            pod_details.sort(key=lambda x: x['avg_latency_ms'], reverse=True)

            results.append({
                "span": {
                    "source": src,
                    "target": dst
                },
                "details": pod_details[:5] # 限制每个 Link 显示 Top 5 问题 Pod
            })
            
        return results