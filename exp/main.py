import logging
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List, Optional
from datetime import datetime

from exp.agent.metric import MetricAgent
from exp.agent.trace import TraceAgent
from exp.agent.log import LogAgent
from exp.agent.judge import JudgeAgent
from exp.utils.log import setup_logger
from exp.utils.time import parse_time_range

logger = logging.getLogger(__name__)


def load_precomputed_metrics(uuid: str, dataset: str) -> List[Dict]:
    """从 ranked_anomaly_with_pattern.csv 文件加载指定 uuid 的预计算指标"""
    import csv
    
    csv_path = f"unit_test/metric/results/ranked_anomaly_with_pattern.csv"
    metrics = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['uuid'] == uuid:
                    component = row.get('component', 'unknown')
                    metric = row.get('metric', 'unknown')
                    pattern = row.get('pattern', 'unknown')
                    metrics.append({
                        # 与 JudgeAgent._format_observation(metric) 兼容
                        'service': component,
                        'kpi': metric,
                        'reason': f"precomputed_ranked_metric({pattern})",
                        # details 原本是时间戳列表；预计算文件里没有时间，因此附带 pattern 供展示
                        'details': [f"pattern:{pattern}"],
                        # 保留原字段，兼容其他下游逻辑
                        'component': component,
                        'metric': metric,
                        'pattern': pattern,
                    })
        
        logger.info(f"Loaded {len(metrics)} precomputed metrics for uuid: {uuid}")
        
    except Exception as e:
        logger.error(f"Failed to load precomputed metrics from {csv_path}: {e}")
        return []
    
    return metrics


def process_anomaly(item: Dict, metric_agent: MetricAgent, trace_agent: TraceAgent, log_agent: LogAgent,
                    judge_agent: JudgeAgent, use_precomputed: bool = False):
    uuid = str(item.get("uuid", ""))
    description = str(item.get("Anomaly Description", ""))
    start_time, end_time = parse_time_range(description)
    
    if not start_time or not end_time:
        logger.warning(f"Warning: Could not parse time range from description: {description}")
        return {
            "uuid": uuid,
            "component": "Unknown",
            "reason": "Time range parsing failed.",
            "reasoning_trace": []
        }

    logger.info(f"Processing {uuid} | Time: {start_time} - {end_time}")

    # 1. 获取各 Agent 的原始结果 (List or Dict)
    # 根据参数决定是使用预计算结果还是调用 MetricAgent
    metric_info = ""
    if use_precomputed:
        # 从 CSV 文件加载预计算的指标结果（已按根因可能性排序）
        metric_result = load_precomputed_metrics(uuid, metric_agent.root_path)
        if not metric_result:
            logger.warning(f"No precomputed metrics found for uuid: {uuid}, falling back to MetricAgent")
            metric_result = metric_agent.score(start_time, end_time)
        else:
            # 添加给大模型的提示信息（英文）
            metric_info = "\n[NOTE] The above metrics are ranked by root cause likelihood. Higher-ranked metrics are more likely to be the root cause of the failure.\n"
    else:
        # MetricAgent.score 返回 List[Dict]
        metric_result = metric_agent.score(start_time, end_time)
    
    # TraceAgent.score 返回 List[Dict] (aggregated links)
    trace_result = trace_agent.score(start_time, end_time)
    
    # LogAgent.score 返回 List[Dict] (anomalies)
    log_result = log_agent.score(start_time, end_time)

    # 2. 将原始结果传给 JudgeAgent 进行融合推理
    # 如果有预计算指标的提示信息，添加到 description 中一起传给大模型
    full_description = description + metric_info if metric_info else description
    analysis = judge_agent.analyze(uuid, full_description, metric_result, trace_result, log_result)
    
    return analysis


def main(args: argparse.Namespace, uuid: str):
    dataset = str(args.dataset)
    log_file = f"results/{dataset}/logs/{uuid}.log"
    log_level = str(args.log_level).upper()
    input_file = f"{dataset}/input.json"
    output = f"results/{dataset}/answer/{uuid}-output.jsonl"
    max_workers = int(args.max_workers)
    use_precomputed = getattr(args, 'use_precomputed', False)

    setup_logger(log_file, log_level)
    logger.info(f"Logger initialized. Dataset: {dataset}, UUID: {uuid}, Use Precomputed: {use_precomputed}")

    metric_agent = MetricAgent(dataset)
    trace_agent = TraceAgent(dataset)
    log_agent = LogAgent(dataset)
    # JudgeAgent 需要 API Key
    judge_agent = JudgeAgent(None, None) 

    try:
        with open(input_file, 'r') as f:
            anomalies = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {input_file}: {e}")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    with open(output, 'a', encoding='utf-8') as o:
        for anomaly in anomalies:
             res = process_anomaly(anomaly, metric_agent, trace_agent, log_agent, judge_agent, use_precomputed)
             if res:
                 o.write(json.dumps(res, ensure_ascii=False) + "\n")
                 o.flush()

if __name__ == "__main__":
    uuid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dataset")
    parser.add_argument('--max_workers', type=int, default=2)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--use_precomputed', action='store_true', 
                        help='使用预计算的 ranked_anomaly_with_pattern.csv 中的指标结果，而非调用 MetricAgent')
    args = parser.parse_args()
    main(args, uuid)

# python3 -m exp.main --use_precomputed