import json
import logging
import os
import re
from openai import OpenAI
from typing import Dict, Any, List

# 导入更新后的 Prompt 模板
from exp.prompt.agent import HWLYYZC_SYSTEM_PROMPT, VALID_COMPONENTS

logger = logging.getLogger(__name__)

class JudgeAgent:
    """
    Implements the 'Large Model Root Cause Reasoning Layer' aligned with PPT Page 14.
    """

    def __init__(self, api_key: str | None, api_url: str | None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.getenv("DEEPSEEK_API_URL")
        if not self.api_key:
            logger.warning("JudgeAgent: API key not found.")

    def _format_observation(self, obs_data: Any, source_type: str) -> str:
        """
        Formats raw observations into text, preserving TIMESTAMPS for 'Time Priority' logic.
        """
        if not obs_data:
            return "No anomalies detected."
        
        if isinstance(obs_data, str):
            return obs_data

        summary = []
        try:
            if source_type == "metric":
                # MetricAgent returns list of dicts: 
                # [{'service': 'adservice-0', 'kpi': 'cpu', 'reason': '...', 'details': ['2025-06-06 10:00:00']}]
                if isinstance(obs_data, list):
                    comp_metrics = {}
                    for item in obs_data:
                        svc = item.get('service', 'unknown')
                        kpi = item.get('kpi', 'unknown')
                        # Extract timestamps to help LLM with Time Priority
                        timestamps = item.get('details', [])
                        first_time = timestamps[0] if timestamps else "unknown time"
                        
                        if svc not in comp_metrics:
                            comp_metrics[svc] = []
                        # 格式: kpi (time)
                        comp_metrics[svc].append(f"{kpi} at {first_time}")
                    
                    for svc, details in list(comp_metrics.items())[:100]: 
                        detail_str = "; ".join(details[:4]) # Limit per component
                        if len(details) > 4: detail_str += "..."
                        summary.append(f"- {svc}: [{detail_str}]")
                
                # Handle direct dict output (fallback)
                elif isinstance(obs_data, dict):
                    events = obs_data.get('events', [])
                    for e in events[:50]:
                        ts = e.get('timestamps', [])
                        t_str = ts[0] if ts else ""
                        summary.append(f"- {e.get('pod')} {e.get('kpi')}: {e.get('pattern')} at {t_str}")

            elif source_type == "trace":
                # TraceAgent returns aggregated links
                if isinstance(obs_data, list):
                    for link in obs_data[:30]:
                        span = link.get('span', {})
                        details = link.get('details', [])
                        src, tgt = span.get('source'), span.get('target')
                        for d in details[:3]: 
                            pod = d.get('pod')
                            node = d.get('node', 'unknown')
                            lat = d.get('avg_latency_ms')
                            errs = d.get('error_messages', [])
                            # Highlight errors for 'Trace Severity (+2)' rule
                            err_str = f", Errs: {errs[:1]}" if errs else ""
                            summary.append(f"- {src}->{tgt}: {pod} (Node:{node}, {lat}ms{err_str})")

            elif source_type == "log":
                # LogAgent returns anomalies list
                if isinstance(obs_data, list):
                    for item in obs_data[:30]:
                        comp = item.get('component')
                        # Log keywords like 'connection refused' are crucial for 'Restart (+10)'
                        obs = item.get('observation', '')
                        # Try to keep it concise for the 20-word limit context
                        summary.append(f"- {comp}: {obs}")
        
        except Exception as e:
            logger.error(f"Error formatting {source_type}: {e}")
            return "Format error."

        return "\n".join(summary) if summary else "No significant details."

    def analyze(self, uuid: str, description: str, metric_result: Any, trace_result: Any, log_result: Any) -> Dict:
        """
        Constructs the prompt and calls the LLM.
        """
        logger.info(f"JudgeAgent: Analyzing {uuid}")

        # 1. Format inputs (Injecting Data)
        metric_obs = self._format_observation(metric_result, "metric")
        trace_obs = self._format_observation(trace_result, "trace")
        log_obs = self._format_observation(log_result, "log")

        # 2. Construct User Prompt (The Dynamic Part)
        user_prompt = f"""
Anomaly Time/Desc: {description}

[METRICS]
{metric_obs}

[TRACES]
{trace_obs}

[LOGS]
{log_obs}

DIAGNOSE based on System Context & Scoring Criteria provided in System Prompt.
"""
        
        print(f"--- [PROMPT] ---\n{user_prompt}\n----------------")

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
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            print(f"--- [RESPONSE] ---\n{content}\n------------------")
            
            # 4. Parse & Validate
            content = re.sub(r'^```json\s*|\s*```$', '', content.strip())
            parsed = json.loads(content)
            
            component = parsed.get("component", "unknown")
            # Fallback validation
            if component not in VALID_COMPONENTS:
                # Simple heuristic correction if LLM outputs specific pod instead of service
                base = component.rsplit('-', 1)[0]
                if base in VALID_COMPONENTS:
                    pass # Allow valid pods
                elif component.startswith("aiops-k8s") or component.startswith("k8s-master"):
                    pass # Allow nodes
                else:
                    logger.warning(f"Invalid component: {component}")

            # Enforce 20-word limit on reason/observation (Post-processing)
            reason = " ".join(parsed.get("reason", "").split()[:20])
            trace = parsed.get("reasoning_trace", [])
            for step in trace:
                if "observation" in step:
                    step["observation"] = " ".join(str(step["observation"]).split()[:20])

            return {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": trace
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "uuid": uuid,
                "component": "unknown",
                "reason": "Analysis failed",
                "reasoning_trace": []
            }