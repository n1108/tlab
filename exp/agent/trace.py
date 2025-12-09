import logging
from collections import defaultdict
from datetime import datetime
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
        # 增加解析 http code 和 error 字段
        self.analysis_fields = [
            "traceID", "spanID", "operationName", "references", "start", "end", "duration", "tags", "logs", 
            "namespace", "node", "pod", 'kind', 'status_code', 'http_code', 'is_error', 'process'
        ]

    def load_spans(self, start: datetime, end: datetime, max_workers=4):
        def callback(spans: pd.DataFrame) -> pd.DataFrame:
            def parse_process(process: Dict) -> pd.Series:
                t = {}
                tags = process.get('tags')
                if isinstance(tags, np.ndarray):
                    for tag in tags:
                        if isinstance(tag, dict) and "key" in tag and "value" in tag:
                            key = tag["key"]
                            if key in ("node_name", "namespace", "name"):
                                t[key] = tag["value"]
                return pd.Series([
                    t.get('node_name'),
                    t.get('namespace'),
                    t.get('name'),
                ])

            def parse_tags(tags: np.ndarray) -> pd.Series:
                t = {}
                is_err = False
                for tag in tags:
                    if isinstance(tag, dict) and "key" in tag and "value" in tag:
                        key = tag["key"]
                        val = tag["value"]
                        
                        if key == "span.kind":
                            t["kind"] = str(val).lower()
                        elif key in ("status.code", "grpc.status_code"):
                            t["status_code"] = val
                        elif key == "http.status_code":
                            t["http_code"] = int(val) if str(val).isdigit() else 0
                        elif key == "error":
                            if isinstance(val, bool) and val:
                                is_err = True
                            elif str(val).lower() == "true":
                                is_err = True

                return pd.Series([
                    t.get('kind', 'internal'),
                    t.get('status_code', '0'),
                    t.get('http_code', 0),
                    is_err
                ])

            spans['start'] = pd.to_datetime(spans["startTimeMillis"], unit="ms")
            spans['end'] = spans['start'] + pd.to_timedelta(spans['duration'], unit='ms')
            spans[['node', 'namespace', 'pod']] = spans['process'].apply(parse_process)
            spans[['kind', 'status_code', 'http_code', 'is_error']] = spans['tags'].apply(parse_tags)
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


    def score(self, start_time: datetime, end_time: datetime, max_workers=4) -> List[Dict]:
        """
        Score trace data using hwlyyzc's logic:
        1. P95 Latency Thresholds
        2. Error Code > 400 or error=True
        3. Pod Distribution Anomaly (Z-score)
        4. Semantic Deduplication (Levenshtein)
        """
        all_spans = self.load_spans(start_time, end_time, max_workers=max_workers)
        if all_spans.empty:
            logger.warning(f"Didn't find any spans between {start_time} and {end_time}.")
            return []

        # 1. Latency Thresholds: P95 per operation
        # "Span耗时异常：＞95分位数"
        op_durations = all_spans.groupby('operationName')['duration']
        # Compute P95 thresholds
        op_thresholds = op_durations.quantile(0.95).to_dict()

        # Data structure to aggregate anomalies
        # Key: (Source Service, Target Service, Anomaly Type)
        link_stats = defaultdict(lambda: {
            'count': 0,
            'source_pods': defaultdict(int), # For Pod distribution check
            'target_pods': defaultdict(int),
            'messages': [], # For Levenshtein compression
            'latency_vals': [],
            'error_codes': set()
        })

        # We construct a simple graph lookup for parent-child to identify Source-Target
        # Since 'process' info is on the span, we need to link child spans to parent spans to know "Source".
        # However, jaeger spans have 'references' pointing to parent.
        # Efficient approach: Index spans by SpanID, then iterate.
        
        # Filter potential anomalies first to reduce iteration size
        # Condition 1: Error
        cond_error = (all_spans['is_error'] == True) | (all_spans['http_code'] >= 400) | (all_spans['status_code'] != '0')
        
        # Condition 2: Latency > P95
        # Need to map thresholds to rows
        all_spans['threshold'] = all_spans['operationName'].map(op_thresholds).fillna(float('inf'))
        cond_latency = all_spans['duration'] > all_spans['threshold']

        anomalous_spans = all_spans[cond_error | cond_latency].copy()
        
        if anomalous_spans.empty:
            return []

        # Build Span ID -> Service/Pod map for quick lookup
        # We need this to find the "Source" (Parent) of an anomalous "Target" (Child)
        # Optimized: We only need to look up parents for the anomalous spans.
        # But parents might be healthy, so we need a global index.
        # To save memory, we only index [spanID -> {service, pod}]
        span_index = all_spans.set_index('spanID')[['process', 'pod']].to_dict('index')

        for _, row in anomalous_spans.iterrows():
            target_service = row['process'].get('serviceName', 'unknown')
            target_pod = row.get('pod') or target_service
            
            # Find Source (Parent)
            source_service = "User/Gateway"
            source_pod = "unknown"
            
            refs = row.get('references')
            if isinstance(refs, np.ndarray) and len(refs) > 0:
                for ref in refs:
                    if ref.get('refType') == 'CHILD_OF':
                        parent_id = ref.get('spanID')
                        if parent_id in span_index:
                            p_proc = span_index[parent_id]['process']
                            source_service = p_proc.get('serviceName', 'unknown')
                            source_pod = span_index[parent_id].get('pod') or source_service
                        break
            
            # Categorize Anomaly
            is_err = row['is_error'] or row['http_code'] >= 400 or row['status_code'] != '0'
            is_slow = row['duration'] > row['threshold']
            
            # We treat (Source->Target) as a link.
            link_key = (source_service, target_service)
            stat = link_stats[link_key]
            
            stat['count'] += 1
            stat['source_pods'][source_pod] += 1
            stat['target_pods'][target_pod] += 1
            
            if is_err:
                stat['error_codes'].add(f"HTTP {row['http_code']}" if row['http_code'] else f"gRPC {row['status_code']}")
                # Extract message for compression
                msg = self._extract_error_message(row['logs'], row['tags'])
                if msg:
                    stat['messages'].append(msg)
            
            if is_slow:
                stat['latency_vals'].append(row['duration'])

        # Final Result Formatting
        results = []
        for (src, dst), data in link_stats.items():
            # 3. Pod Distribution Anomaly
            # "Pod分布比例异常：... 三倍标准差外"
            anomalous_target_pods = self._detect_pod_distribution_anomaly(data['target_pods'])
            anomalous_source_pods = self._detect_pod_distribution_anomaly(data['source_pods'])
            
            # 4. Semantic Deduplication
            # "相似度大于95%被合并"
            compressed_msgs = self._compress_messages(data['messages'], threshold=0.95)
            
            span_obj = {
                "source": src,
                "source_pods": sorted(anomalous_source_pods), # Only report statistically significant pods
                "target": dst,
                "target_pods": sorted(anomalous_target_pods),
                "count": data['count'],
            }
            
            message_parts = {}
            if data['error_codes']:
                codes = sorted([c for c in data['error_codes'] if c not in ['HTTP 0', 'gRPC 0']])
                if codes:
                    message_parts["error_codes"] = codes
            
            if compressed_msgs:
                message_parts["error_messages"] = compressed_msgs[:5] # Limit to top 5 templates

            if data['latency_vals']:
                latencies = np.array(data['latency_vals'])
                message_parts["latency"] = {
                    "avg_latency_ms": round(np.mean(latencies), 2),
                    "max_latency_ms": round(np.max(latencies), 2),
                    # Since threshold is P95 per op, we average the thresholds encountered? 
                    # Simpler to just show what the typical P95 was. 
                    # For simplicity, we omit exact threshold here or assume it's context dependent.
                }

            results.append({
                "span": span_obj,
                "message": message_parts
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
