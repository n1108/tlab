import logging
import re
import json
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from datetime import datetime, timedelta
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from exp.utils.input import load_parquet_by_hour

logger = logging.getLogger(__name__)

# 定义错误关键词
ERROR_KEYWORDS = {
    'error', 'exception', 'fail', 'warning', 'critical', 'timeout', 'panic', 'refused', 'reset', 'unavailable'
}

class LogAgent:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.fields = [
            "k8_namespace", "@timestamp",
            "agent_name",
            "k8_pod", "message", "k8_node_name"
        ]
        
        config = TemplateMinerConfig()
        config.load("exp/template/drain3_log.ini")
        self.config = config
        
        # 预编译正则
        self.mask_patterns = [
            (re.compile(r'(\d{1,3}\.){3}\d{1,3}(:\d+)?'), '<IP>'),
            (re.compile(r'\b\d+\b'), '<NUM>'),
        ]

    def _preprocess_message(self, raw_message: str) -> str | None:
        if raw_message is None:
            return None
            
        log_content = raw_message
        if raw_message.strip().startswith('{'):
            try:
                log_json = json.loads(raw_message)
                if 'error' in log_json and log_json['error']:
                    log_content = log_json['error']
                elif 'message' in log_json:
                    log_content = log_json['message']
            except json.JSONDecodeError:
                pass 

        if not isinstance(log_content, str):
            log_content = str(log_content)

        if not any(kw in log_content.lower() for kw in ERROR_KEYWORDS):
            return None
            
        return log_content

    def load_logs(self, start: datetime, end: datetime, max_workers=4) -> pd.DataFrame:
        def callback(df: pd.DataFrame) -> pd.DataFrame:
            df['cleaned_message'] = df['message'].apply(self._preprocess_message)
            df = df.dropna(subset=['cleaned_message'])
            return df

        pa_start = pa.scalar(start, type=pa.timestamp('ms', tz='UTC'))
        pa_end = pa.scalar(end, type=pa.timestamp('ms', tz='UTC'))
        
        filter_expression = (
            (ds.field("@timestamp").cast(pa.timestamp('ms', tz='UTC')) >= pa_start) & 
            (ds.field("@timestamp").cast(pa.timestamp('ms', tz='UTC')) <= pa_end)
        )

        return load_parquet_by_hour(
            start, end, self.root_path,
            file_pattern="{dataset}/{day}/log-parquet/log_filebeat-server_{day}_{hour}-00-00.parquet",
            load_fields=self.fields,
            return_fields=self.fields + ['cleaned_message'],
            filter_=filter_expression,
            callback=callback,
            max_workers=max_workers
        )

    def score(self, start_time: datetime, end_time: datetime, max_workers=4):
        # 1. 加载基线和当前数据
        baseline_duration = timedelta(minutes=30)
        baseline_start = start_time - baseline_duration
        
        baseline_df = self.load_logs(baseline_start, start_time, max_workers)
        current_df = self.load_logs(start_time, end_time, max_workers)
        
        if current_df.empty and baseline_df.empty:
            return []

        miner = TemplateMiner(None, self.config)

        # 2. 统计基线
        baseline_stats = defaultdict(lambda: defaultdict(int))
        if not baseline_df.empty:
            for _, row in baseline_df.iterrows():
                svc = self._get_service_name(row['k8_pod'])
                res = miner.add_log_message(row['cleaned_message'])
                baseline_stats[svc][res["cluster_id"]] += 1

        # 3. 统计当前
        current_stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'template': '', 'sample': ''}))
        if not current_df.empty:
            for _, row in current_df.iterrows():
                svc = self._get_service_name(row['k8_pod'])
                content = row['cleaned_message']
                res = miner.add_log_message(content)
                t_id = res["cluster_id"]
                entry = current_stats[svc][t_id]
                entry['count'] += 1
                entry['template'] = res["template_mined"]
                if entry['count'] == 1:
                    entry['sample'] = content

        anomalies = []
        
        # 4. 检测异常：新增(New)、突增(Surge)、持续(Persistent)
        for svc, templates in current_stats.items():
            svc_anomalies = []
            total_logs = 0
            
            for t_id, info in templates.items():
                curr_cnt = info['count']
                base_cnt = baseline_stats[svc].get(t_id, 0)
                total_logs += curr_cnt
                
                anomaly_type = None
                
                if base_cnt == 0:
                    anomaly_type = "New Pattern"
                elif curr_cnt > base_cnt * 3 and curr_cnt > 5:
                    anomaly_type = "Frequency Surge"
                elif curr_cnt > 10: 
                    # 新增：虽然没有突增，但绝对数量较高，属于持续报错
                    # 避免漏掉已经开始一段时间的故障
                    anomaly_type = "Persistent Error"
                
                if anomaly_type:
                    svc_anomalies.append({
                        "type": anomaly_type,
                        "template": info['template'],
                        "current_count": curr_cnt,
                        "baseline_count": base_cnt,
                        "sample": info['sample']
                    })
            
            # 排序优先级：New > Surge > Persistent
            type_priority = {"New Pattern": 3, "Frequency Surge": 2, "Persistent Error": 1}
            svc_anomalies.sort(key=lambda x: (type_priority[x['type']], x['current_count']), reverse=True)
            
            if svc_anomalies:
                top_pattern = svc_anomalies[0]
                anomalies.append({
                    "component": svc,
                    "total_error_count": total_logs,
                    "anomalous_patterns": svc_anomalies[:5],
                    "observation": f"{svc}: Detected {len(svc_anomalies)} anomalies. Top: [{top_pattern['type']}] {top_pattern['template']} (Count: {top_pattern['current_count']})"
                })

        # 5. 检测突降 (Service Drop)
        for svc, base_templates in baseline_stats.items():
            if svc not in current_stats:
                total_base = sum(base_templates.values())
                if total_base > 10:
                    anomalies.append({
                        "component": svc,
                        "observation": f"{svc}: Error logs disappeared completely (Frequency Drop). Service might be down."
                    })

        return anomalies

    def _get_service_name(self, pod_name: str) -> str:
        if not pod_name:
            return "unknown"
        parts = pod_name.split("-")
        if len(parts) > 2 and (parts[-1].isdigit() or len(parts[-1]) > 4):
             return "-".join(parts[:-2])
        if len(parts) > 1 and (parts[-1].isdigit() or len(parts[-1]) > 4):
            return "-".join(parts[:-1])
        return pod_name