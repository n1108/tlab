import json
import logging
import os
import re
from openai import OpenAI
from typing import Dict, Any, List

# 导入 hwlyyzc 方案的 Prompt 模板和静态知识
from exp.prompt.agent import HWLYYZC_SYSTEM_PROMPT, CALL_TOPOLOGY, VALID_COMPONENTS

logger = logging.getLogger(__name__)

class JudgeAgent:
    """
    Implements the 'Large Model Root Cause Reasoning Layer' from hwlyyzc team.
    Fuses system architecture, valid component list, and multi-source anomaly data.
    """

    def __init__(self, api_key: str | None, api_url: str | None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.getenv("DEEPSEEK_API_URL")
        if not self.api_key:
            logger.warning("JudgeAgent: API key not found. Reasoning will fail.")

    def _format_observation(self, obs_data: Any, source_type: str) -> str:
        """Helper to format complex observation objects into concise text for LLM."""
        if not obs_data:
            return "No significant anomalies detected."
        
        # 如果已经是字符串，直接返回
        if isinstance(obs_data, str):
            return obs_data

        summary = []
        try:
            if source_type == "metric":
                # MetricAgent 返回的是 list of dicts
                if isinstance(obs_data, list):
                    for item in obs_data[:15]: # 增加上下文长度
                        svc = item.get('service', '')
                        kpi = item.get('kpi', '')
                        reason = item.get('reason', '')
                        details = item.get('details', [])
                        detail_str = f", Details: {json.dumps(details, ensure_ascii=False)}" if details else ""
                        summary.append(f"- Component: {svc}, KPI: {kpi}, Info: {reason}{detail_str}")
                elif isinstance(obs_data, dict):
                    # 处理 MetricAgent.query_metrics 的返回格式
                    events = obs_data.get('events', [])
                    for e in events[:15]:
                        summary.append(f"- {e.get('pod', 'unknown')} ({e.get('kpi', 'unknown')}): {e.get('type')}")
                    if not events:
                        summary.append(str(obs_data.get('observation', '')))

            elif source_type == "trace":
                # TraceAgent 返回 list of dicts (links)
                if isinstance(obs_data, list):
                    for link in obs_data[:10]: # Top 10 problematic links
                        span = link.get('span', {})
                        details = link.get('details', [])
                        src, tgt = span.get('source'), span.get('target')
                        for d in details[:3]: # Top 3 pods per link
                            pod = d.get('pod')
                            node = d.get('node', 'unknown')
                            lat = d.get('avg_latency_ms')
                            errs = d.get('error_messages', [])
                            err_str = f", Errors: {errs}" if errs else ""
                            summary.append(f"- Link {src}->{tgt}: Pod {pod} (Node: {node}, Latency: {lat}ms{err_str})")

            elif source_type == "log":
                # LogAgent 返回 list of dicts
                if isinstance(obs_data, list):
                    for item in obs_data[:10]:
                        comp = item.get('component')
                        svc = item.get('service', 'unknown')
                        node = item.get('node', 'unknown')
                        obs = item.get('observation')
                        summary.append(f"- Component: {comp} (Service: {svc}, Node: {node}), Log Analysis: {obs}")
        
        except Exception as e:
            logger.error(f"Error formatting {source_type} observation: {e}")
            return str(obs_data)

        return "\n".join(summary) if summary else "No significant details found."

    def analyze(self, uuid: str, description: str, metric_result: Any, trace_result: Any, log_result: Any) -> Dict:
        """
        Execute the reasoning process.
        """
        print(f"\n{'='*20} JudgeAgent Analysis: {uuid} {'='*20}")
        logger.info(f"JudgeAgent: Analyzing anomaly {uuid}")

        # 1. Format inputs
        metric_obs_str = self._format_observation(metric_result, "metric")
        trace_obs_str = self._format_observation(trace_result, "trace")
        log_obs_str = self._format_observation(log_result, "log")

        # 2. Construct User Prompt
        user_prompt = f"""
Anomaly Description: {description}

### MULTI-SOURCE OBSERVATIONS

[1. METRIC ANOMALIES]
{metric_obs_str}

[2. TRACE ANOMALIES]
{trace_obs_str}

[3. LOG ANOMALIES]
{log_obs_str}

### INSTRUCTIONS
Based on the observations above and the SCORING RULES in the system prompt:
1. Identify all suspect components.
2. Apply the **Downstream Priority** rule: If 'frontend' and 'checkoutservice' are both anomalous, and frontend calls checkoutservice, prioritize 'checkoutservice'.
3. Check for **Restart Signals**: If metrics show gaps or logs show startup messages, prioritize 'Pod Kill/Restart'.
4. Check for **Node Level Issues**: If multiple pods on the same node (e.g., aiops-k8s-04) are anomalous, the root cause is likely the Node.

Diagnose the single root cause component and the specific reason.
"""
        # --- LOGGING PROMPT ---
        print(f"--- [PROMPT CONSTRUCTED] ---\n{user_prompt}\n----------------------------")

        # 3. Call LLM
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": HWLYYZC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1, 
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            # --- LOGGING RESPONSE ---
            print(f"--- [LLM RAW RESPONSE] ---\n{content}\n--------------------------")
            
            # 解析 JSON
            content = re.sub(r'^```json\s*|\s*```$', '', content.strip())
            parsed = json.loads(content)

            # 4. 结果后处理与校验
            component = parsed.get("component", "unknown")
            reason = parsed.get("reason", "unknown")
            
            # 简单的防幻觉校验
            if component not in VALID_COMPONENTS:
                base_svc = component.rsplit('-', 1)[0]
                if base_svc in VALID_COMPONENTS:
                    pass # 合法 Pod
                elif component.startswith("aiops-k8s") or component.startswith("k8s-master"):
                     pass # 合法 Node
                else:
                    logger.warning(f"JudgeAgent: Potential hallucinated component '{component}'.")
            
            print(f"--- [FINAL RESULT] Component: {component}, Reason: {reason}")
            
            return {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": parsed.get("reasoning_trace", [])
            }

        except Exception as e:
            logger.error(f"JudgeAgent Analysis Failed: {e}", exc_info=True)
            print(f"ERROR in JudgeAgent: {e}")
            return {
                "uuid": uuid,
                "component": "unknown",
                "reason": "Analysis failed due to internal error.",
                "reasoning_trace": []
            }