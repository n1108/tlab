import logging
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import re
from typing import Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from exp.agent.metric import MetricAgent
from exp.agent.trace import TraceAgent
from exp.agent.log import LogAgent
from exp.agent.judge import JudgeAgent
from exp.utils.log import setup_logger
from exp.utils.time import parse_time_range

logger = logging.getLogger(__name__)
_PRECOMPUTED_METRIC_BLOCKS: Optional[Dict[str, str]] = None


def _load_precomputed_metric_blocks(summary_path: str = "results/metric_summary.txt") -> Dict[str, str]:
    global _PRECOMPUTED_METRIC_BLOCKS
    if _PRECOMPUTED_METRIC_BLOCKS is not None:
        return _PRECOMPUTED_METRIC_BLOCKS

    blocks: Dict[str, str] = {}
    current_uuid: Optional[str] = None
    current_lines: List[str] = []

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                m = re.match(r"^# \[\d+/\d+\] UUID: (.+)$", line)
                if m:
                    if current_uuid is not None:
                        blocks[current_uuid] = "\n".join(current_lines).strip()
                    current_uuid = m.group(1).strip()
                    current_lines = []
                    continue
                if current_uuid is not None:
                    current_lines.append(line)

        if current_uuid is not None:
            blocks[current_uuid] = "\n".join(current_lines).strip()

        logger.info("Loaded %s metric-summary blocks from %s", len(blocks), summary_path)
    except Exception as e:
        logger.error("Failed to load metric summary blocks from %s: %s", summary_path, e)
        blocks = {}

    _PRECOMPUTED_METRIC_BLOCKS = blocks
    return _PRECOMPUTED_METRIC_BLOCKS


def load_precomputed_metrics(uuid: str, dataset: str, top_k: int = 30) -> str:
    """从 metric_summary.txt 中加载指定 uuid 的预计算指标摘要文本"""
    _ = dataset, top_k
    blocks = _load_precomputed_metric_blocks("results/metric_summary.txt")
    text = blocks.get(str(uuid), "").strip()
    if text:
        logger.info("Loaded metric summary text for uuid: %s", uuid)
    else:
        logger.warning("No metric summary text found for uuid: %s", uuid)
    return text


def process_anomaly(item: Dict, metric_agent: MetricAgent, trace_agent: TraceAgent, log_agent: LogAgent,
                    judge_agent: JudgeAgent, use_precomputed: bool = False,
                    precomputed_top_k: int = 30):
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
        # 从 metric_summary.txt 加载预计算的指标摘要文本
        metric_result = load_precomputed_metrics(uuid, metric_agent.root_path, top_k=precomputed_top_k)
        if not metric_result:
            logger.warning(f"No precomputed metric summary found for uuid: {uuid}, falling back to MetricAgent")
            metric_result = metric_agent.score(start_time, end_time)
        else:
            metric_info = (
                "\n[NOTE] The [METRICS] section above is a precomputed metric evidence summary for this UUID. "
                "Treat it as structured prior evidence, not as a hard conclusion. "
                "Some listed anomalies may be co-occurring side effects rather than the root cause. "
                "In ordinary microservice cases, node or TiDB/TiKV anomalies may be collateral unless traces/logs support infrastructure or database-local failure. "
                "Please combine them with traces, logs, topology, and fault semantics before the final judgment.\n"
            )
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
    precomputed_top_k = int(getattr(args, 'precomputed_top_k', 30))
    llm_provider = str(getattr(args, 'llm_provider', 'deepseek'))
    llm_model = str(getattr(args, 'llm_model', '')).strip() or None
    llm_api_key = str(getattr(args, 'llm_api_key', '')).strip() or None
    llm_api_url = str(getattr(args, 'llm_api_url', '')).strip() or None

    setup_logger(log_file, log_level)
    logger.info(f"Logger initialized. Dataset: {dataset}, UUID: {uuid}, Use Precomputed: {use_precomputed}")

    metric_agent = MetricAgent(dataset)
    trace_agent = TraceAgent(dataset)
    log_agent = LogAgent(dataset)
    # JudgeAgent 需要 API Key
    judge_agent = JudgeAgent(
        llm_api_key,
        llm_api_url,
        provider=llm_provider,
        model=llm_model,
    )
    if not judge_agent.api_key:
        logger.error(
            "No LLM API key configured. Set YUZO_API_KEY, DEEPSHIELDS_API_KEY, "
            "DEEPSEEK_API_KEY, OPENAI_API_KEY, or use --llm_api_key."
        )
        return

    try:
        with open(input_file, 'r') as f:
            anomalies = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {input_file}: {e}")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)

    workers = max(1, int(max_workers))
    # 边跑边写入：避免 list(executor.map(...)) 卡到全部完成才创建/写入 jsonl（大批量时会误以为「没生成文件」）。
    with open(output, "w", encoding="utf-8") as o:
        if workers == 1:
            for anomaly in anomalies:
                res = process_anomaly(
                    anomaly,
                    metric_agent,
                    trace_agent,
                    log_agent,
                    judge_agent,
                    use_precomputed,
                    precomputed_top_k,
                )
                if res:
                    o.write(json.dumps(res, ensure_ascii=False) + "\n")
                    o.flush()
        else:
            logger.info("Running with %s worker threads (parallel per anomaly)", workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for res in executor.map(
                    lambda a: process_anomaly(
                        a,
                        metric_agent,
                        trace_agent,
                        log_agent,
                        judge_agent,
                        use_precomputed,
                        precomputed_top_k,
                    ),
                    anomalies,
                ):
                    if res:
                        o.write(json.dumps(res, ensure_ascii=False) + "\n")
                        o.flush()

    logger.info("Finished; results written to %s", output)

if __name__ == "__main__":
    uuid = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dataset")
    parser.add_argument(
        '--max_workers',
        type=int,
        default=2,
        help='并行处理 input.json 中每条故障的线程数（每条含 trace/log/metric 与一次 Judge LLM）。'
        ' 过大可能触发 Yuzo/DeepSeek 限流（429）；调试可设为 1。',
    )
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--use_precomputed', action='store_true', 
                        help='使用 results/metric_summary.txt 中对应 UUID 的指标摘要，而非调用 MetricAgent')
    parser.add_argument('--precomputed_top_k', type=int, default=30,
                        help='兼容旧参数；当前使用 metric_summary.txt 时该参数不生效')
    parser.add_argument('--llm_provider', type=str, default='deepseek', choices=['deepseek', 'yuzo'],
                        help='Judge 使用的大模型提供方：deepseek 或 yuzo')
    parser.add_argument('--llm_model', type=str, default='',
                        help='覆盖默认模型名。yuzo 默认 reasoner，deepseek 默认 deepseek-chat')
    parser.add_argument('--llm_api_key', type=str, default='',
                        help='覆盖环境变量中的 API Key')
    parser.add_argument('--llm_api_url', type=str, default='',
                        help='覆盖默认 Base URL（yuzo 默认 https://api.deepshields.com/v1）')
    args = parser.parse_args()
    main(args, uuid)

# python3 -m exp.main --use_precomputed