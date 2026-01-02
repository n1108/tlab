import logging
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict
from datetime import datetime

from exp.agent.metric import MetricAgent
from exp.agent.trace import TraceAgent
from exp.agent.log import LogAgent
from exp.agent.judge import JudgeAgent
from exp.utils.log import setup_logger
from exp.utils.time import parse_time_range

logger = logging.getLogger(__name__)

def process_anomaly(item: Dict, metric_agent: MetricAgent, trace_agent: TraceAgent, log_agent: LogAgent,
                    judge_agent: JudgeAgent):
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
    # MetricAgent.score 返回 List[Dict]
    metric_result = metric_agent.score(start_time, end_time)
    
    # TraceAgent.score 返回 List[Dict] (aggregated links)
    trace_result = trace_agent.score(start_time, end_time)
    
    # LogAgent.score 返回 List[Dict] (anomalies)
    log_result = log_agent.score(start_time, end_time)

    # 2. 将原始结果传给 JudgeAgent 进行融合推理
    analysis = judge_agent.analyze(uuid, description, metric_result, trace_result, log_result)
    
    return analysis

def main(args: argparse.Namespace, uuid: str):
    dataset = str(args.dataset)
    log_file = f"results/{dataset}/logs/{uuid}.log"
    log_level = str(args.log_level).upper()
    input_file = f"{dataset}/input.json" # rename input variable to avoid conflict
    output = f"results/{dataset}/answer/{uuid}-output.jsonl"
    max_workers = int(args.max_workers)

    setup_logger(log_file, log_level)
    logger.info(f"Logger initialized. Dataset: {dataset}, UUID: {uuid}")

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
             res = process_anomaly(anomaly, metric_agent, trace_agent, log_agent, judge_agent)
             if res:
                 o.write(json.dumps(res, ensure_ascii=False) + "\n")
                 o.flush()

if __name__ == "__main__":
    uuid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="phasetwo")
    parser.add_argument('--max_workers', type=int, default=2)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    main(args, uuid)