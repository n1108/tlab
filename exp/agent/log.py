import logging
import re
import json
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from datetime import datetime
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from exp.utils.input import load_parquet_by_hour

logger = logging.getLogger(__name__)

# 定义错误关键词 (识别错误模式)
ERROR_KEYWORDS = {
    'error', 'exception', 'fail', 'warning', 'critical', 'timeout', 'panic', 'refused', 'reset'
}

class LogAgent:
    """
    LogAgent implementing hwlyyzc's strategy:
    1. Log Cleaning (Masking numbers/IPs).
    2. Error Pattern Matching (Keyword filtering).
    3. Drain3 Template Parsing.
    4. Anomaly Detection (Frequency burst/New templates).
    """

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.fields = [
            "k8_namespace", "@timestamp",
            "agent_name",
            "k8_pod", "message", "k8_node_name"
        ]
        
        # 初始化 Drain3
        config = TemplateMinerConfig()
        config.load("exp/template/drain3_log.ini")
        # 不使用持久化，保证每次分析是针对当前窗口的独立统计，或者根据需求开启
        # persistence = FilePersistence("drain3_state.bin") 
        self.template_miner = TemplateMiner(None, config)
        
        # 预编译正则，用于代码层面的深度清洗 (将数字、哈希值替换为 <*>)
        self.mask_patterns = [
            (re.compile(r'(\d{1,3}\.){3}\d{1,3}(:\d+)?'), '<IP>'), # IP
            (re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'), '<UUID>'), # UUID
            (re.compile(r'\b\d+\b'), '<NUM>'), # Numbers
            (re.compile(r'0x[0-9a-fA-F]+'), '<HEX>') # Hex
        ]

    def _preprocess_message(self, raw_message: str) -> str | None:
        """
        解析 JSON 并进行清洗
        """
        log_content = raw_message
        
        # 1. JSON 解析
        if raw_message.strip().startswith('{'):
            try:
                log_json = json.loads(raw_message)
                # 优先获取 error 字段，其次是 message 字段
                if 'error' in log_json and log_json['error']:
                    log_content = log_json['error']
                elif 'message' in log_json:
                    log_content = log_json['message']
            except json.JSONDecodeError:
                pass # 解析失败则当做纯文本处理

        if not isinstance(log_content, str):
            log_content = str(log_content)

        # 2. 错误关键词过滤 (错误模式匹配)
        # 只有包含错误关键词的日志才进入后续模板分析，减少噪声
        if not any(kw in log_content.lower() for kw in ERROR_KEYWORDS):
            return None

        # 3. 强清洗 (移除变量信息，替换为 <*>)
        # Drain3 内部有 masking，但这里做显式替换能提高聚类效果
        # for pattern, mask in self.mask_patterns:
        #     log_content = pattern.sub(mask, log_content)
            
        return log_content

    def load_logs(self, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        """
        加载并预处理日志
        """
        def callback(df: pd.DataFrame) -> pd.DataFrame:
            # 应用预处理
            df['cleaned_message'] = df['message'].apply(self._preprocess_message)
            # 过滤掉非错误日志 (None)
            df = df.dropna(subset=['cleaned_message'])
            return df

        # 显式构造 PyArrow Scalar，强制指定精度为 ms 且带 UTC 时区
        # 这样与 ds.field("@timestamp").cast(...) 后的类型完全一致
        pa_start = pa.scalar(start, type=pa.timestamp('ms', tz='UTC'))
        pa_end = pa.scalar(end, type=pa.timestamp('ms', tz='UTC'))
        
        # 构建过滤器
        # 注意：这里假设 Parquet 中的 @timestamp 可能是字符串或不一致的格式，所以保留左边的 cast
        # 如果 Parquet 原生就是 timestamp[ms, UTC]，左边的 cast 也是安全的
        filter_expression = (
            (ds.field("@timestamp").cast(pa.timestamp('ms', tz='UTC')) >= pa_start) & 
            (ds.field("@timestamp").cast(pa.timestamp('ms', tz='UTC')) <= pa_end)
        )

        return load_parquet_by_hour(
            start, end, self.root_path,
            file_pattern="{dataset}/{day}/log-parquet/log_filebeat-server_{day}_{hour}-00-00.parquet",
            load_fields=self.fields,
            return_fields=self.fields + ['cleaned_message'],
            filter_=filter_expression, # 使用新的 filter
            callback=callback,
            max_workers=max_workers
        )

    def score(self, start_time: datetime, end_time: datetime, max_workers=4):
        """
        主分析函数：模板聚类 + 频率统计
        """
        # 1. 加载过滤后的错误日志
        logs_df = self.load_logs(start_time, end_time, max_workers=max_workers)
        
        if logs_df.empty:
            logger.info("No error logs found in the specified time range.")
            return []

        # 2. Drain3 模板聚类 (PPT: 模板解析)
        # 统计结构: pod -> template_id -> {count, sample, template_str}
        pod_stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'sample': '', 'template': ''}))
        
        # 重新初始化 miner 以保证分析的是当前窗口的分布
        # 注意：真实场景下检测“新出现”需要持久化状态，但在此离线分析逻辑中，
        # 我们假设窗口内的错误日志本身就是异常的候选。
        self.template_miner = TemplateMiner(None, self.template_miner.config)

        for _, row in logs_df.iterrows():
            pod_name = row['k8_pod']
            # 去掉 Pod 后面的随机字符，聚合到 workload 级别 (例如 cartservice-2 -> cartservice)
            # 这一步对于聚合统计很重要
            service_name = "-".join(pod_name.split("-")[:-1]) if "-" in pod_name else pod_name
            
            content = row['cleaned_message']
            
            # Drain3 添加日志并获取 Cluster
            cluster = self.template_miner.add_log_message(content)
            template_id = cluster["cluster_id"]
            template_str = cluster["template_mined"]

            stats = pod_stats[service_name][template_id]
            stats['count'] += 1
            stats['template'] = template_str
            # 保留第一条原始日志作为样本（未清洗前的message中提取的内容）
            if stats['count'] == 1:
                stats['sample'] = content

        # 3. 结果格式化与筛选 (PPT: 异常日志压缩)
        results = []
        
        for service, templates in pod_stats.items():
            error_count = sum(t['count'] for t in templates.values())
            
            # 提取 Top N 频率的错误模板
            # PPT 提到关注频率突增，这里输出频率最高的错误模板
            sorted_templates = sorted(templates.items(), key=lambda x: x[1]['count'], reverse=True)
            
            top_templates = []
            for tid, stat in sorted_templates[:5]: # 取前5个主要错误
                top_templates.append({
                    "template": stat['template'],
                    "count": stat['count'],
                    "sample": stat['sample']
                })

            if error_count > 0:
                results.append({
                    "component": service, # 使用 Service 名而不是 Pod 名，便于后续归因
                    "error_count": error_count,
                    "top_patterns": top_templates,
                    "observation": f"Service {service} has {error_count} error logs. Top pattern: {top_templates[0]['template']}"
                })

        logger.info(f"Log analysis found anomalies in {len(results)} components.")
        return results