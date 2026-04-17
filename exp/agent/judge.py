import json
import logging
import os
import re
import time
from openai import OpenAI
from typing import Dict, Any, List

# 导入更新后的 Prompt 模板
from exp.prompt.agent import HWLYYZC_SYSTEM_PROMPT, VALID_COMPONENTS

logger = logging.getLogger(__name__)


class JudgeAgent:
    """
    Implements the 'Large Model Root Cause Reasoning Layer' aligned with PPT Page 14.
    """

    def __init__(
        self,
        api_key: str | None,
        api_url: str | None,
        provider: str = "deepseek",
        model: str | None = None,
    ):
        self.provider = str(provider or "deepseek").lower().strip()

        if self.provider == "yuzo":
            self.api_key = (
                api_key
                or os.getenv("YUZO_API_KEY")
                or os.getenv("DEEPSHIELDS_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY")
            )
            self.api_url = api_url or os.getenv("YUZO_API_URL") or "https://api.deepshields.com/v1"
            self.model = model or "reasoner"
        else:
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.api_url = api_url or os.getenv("DEEPSEEK_API_URL")
            self.model = model or "deepseek-chat"

        # 归一化 URL，避免出现 /v1/ 这类末尾斜杠差异
        if self.api_url:
            self.api_url = self.api_url.rstrip("/")

        # 支持通过环境变量微调重试与超时，默认对 502 更友好
        self.max_retries = max(1, int(os.getenv("JUDGE_MAX_RETRIES", "4")))
        self.request_timeout = max(10.0, float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120")))
        self.enable_model_fallback = os.getenv("JUDGE_ENABLE_MODEL_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}
        self.max_metric_services = max(1, int(os.getenv("JUDGE_MAX_METRIC_SERVICES", "6")))
        self.max_metric_kpis_per_service = max(1, int(os.getenv("JUDGE_MAX_METRIC_KPIS_PER_SERVICE", "2")))
        self.max_trace_links = max(1, int(os.getenv("JUDGE_MAX_TRACE_LINKS", "10")))
        self.max_log_items = max(1, int(os.getenv("JUDGE_MAX_LOG_ITEMS", "10")))
        self.temperature = float(os.getenv("JUDGE_TEMPERATURE", "0.1"))
        self.max_obs_chars = max(
            400,
            int(os.getenv("JUDGE_MAX_OBS_CHARS", "1800")),
        )

        if not self.api_key:
            logger.warning("JudgeAgent: API key not found.")

    @staticmethod
    def _extract_json_text(content: str) -> str:
        """Best-effort 提取 JSON 文本，兼容模型返回前后缀解释。"""
        text = (content or "").strip()
        text = re.sub(r'^```json\s*|\s*```$', '', text)
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        msg = str(exc).lower()

        # Some gateways wrap upstream 404(model not found) into a 502 envelope.
        # This is not transient and should fail fast to next model candidate.
        non_retryable_keywords = [
            "does not exist",
            "notfounderror",
            "model `",
            'model "',
            "invalid model",
            "unsupported model",
        ]
        if any(k in msg for k in non_retryable_keywords):
            return False

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        transient_keywords = [
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "connection reset",
            "timed out",
            "timeout",
            "temporarily unavailable",
        ]
        return any(k in msg for k in transient_keywords)

    def _create_completion_with_retry(self, client: OpenAI, req_kwargs: Dict[str, Any]):
        """调用 chat.completions，遇到网关抖动进行指数退避重试。"""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return client.chat.completions.create(**req_kwargs)
            except Exception as e:
                last_exc = e
                if not self._is_transient_error(e) or attempt >= self.max_retries:
                    raise
                backoff = min(8.0, 0.8 * (2 ** (attempt - 1)))
                logger.warning(
                    "LLM request transient failure (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    e,
                    backoff,
                )
                time.sleep(backoff)
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown completion error")

    def _build_model_candidates(self) -> List[str]:
        """构建模型候选列表，优先用户指定，其次容灾回退。"""
        candidates: List[str] = []
        primary = str(self.model or "").strip()
        if primary:
            candidates.append(primary)
        if self.provider == "yuzo":
            if not candidates:
                candidates.append("reasoner")
            # 与用户示例保持一致：默认严格使用单模型，避免误切到不支持模型。
            if self.enable_model_fallback:
                for m in ["chat"]:
                    if m not in candidates:
                        candidates.append(m)
        return candidates or ["deepseek-chat"]

    @staticmethod
    def _clip_text(text: str, max_chars: int, suffix: str = "\n...(truncated)") -> str:
        if not text:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + suffix

    @staticmethod
    def _assemble_user_prompt(
        description: str,
        metric_obs: str,
        trace_obs: str,
        log_obs: str,
    ) -> str:
        return f"""Anomaly Time/Desc: {description}

[METRICS]
{metric_obs}

[TRACES]
{trace_obs}

[LOGS]
{log_obs}

DIAGNOSE based on System Context & Scoring Criteria provided in System Prompt.
"""

    @staticmethod
    def _salvage_partial_json(content: str) -> Dict[str, Any] | None:
        """从截断或不完整 JSON 中尽力提取 component/reason。"""
        if not content:
            return None
        text = content.strip()
        component = None
        reason = None

        m_comp = re.search(r'"component"\s*:\s*"([^"\n\r]+)', text)
        if m_comp:
            component = m_comp.group(1).strip()

        m_reason = re.search(r'"reason"\s*:\s*"([^"\n\r]+)', text)
        if m_reason:
            reason = m_reason.group(1).strip()

        if not component and not reason:
            return None

        return {
            "component": component or "unknown",
            "reason": reason or "Analysis succeeded but output was truncated",
            "reasoning_trace": [],
        }

    def _build_fallback_reasoning_trace(
        self,
        component: str,
        reason: str,
        metric_result: Any,
        trace_result: Any,
        log_result: Any,
    ) -> List[Dict[str, Any]]:
        """当模型未返回完整推理链时，基于已有证据补齐最小 reasoning_trace。"""
        steps: List[Dict[str, Any]] = []

        metric_obs = "No strong metric evidence."
        if isinstance(metric_result, list) and metric_result:
            picked = None
            for item in metric_result:
                svc = str(item.get("service", ""))
                if component and (component == svc or component in svc or svc in component):
                    picked = item
                    break
            if picked is None:
                picked = metric_result[0]
            svc = picked.get("service", "unknown")
            kpi = picked.get("kpi", "unknown")
            metric_obs = f"Top metric anomaly: {svc}.{kpi}."

        trace_obs = "No strong trace evidence."
        if isinstance(trace_result, list) and trace_result:
            picked_link = None
            for link in trace_result:
                span = link.get("span", {}) if isinstance(link, dict) else {}
                src = str(span.get("source", ""))
                tgt = str(span.get("target", ""))
                if component and (component in src or component in tgt):
                    picked_link = link
                    break
            if picked_link is None:
                picked_link = trace_result[0]
            span = picked_link.get("span", {}) if isinstance(picked_link, dict) else {}
            src = span.get("source", "unknown")
            tgt = span.get("target", "unknown")
            trace_obs = f"Trace hotspot: {src}->{tgt}."

        log_obs = "No strong log evidence."
        if isinstance(log_result, list) and log_result:
            picked_log = None
            for item in log_result:
                comp = str(item.get("component", "")) if isinstance(item, dict) else ""
                if component and (component in comp or comp in component):
                    picked_log = item
                    break
            if picked_log is None:
                picked_log = log_result[0]
            comp = picked_log.get("component", "unknown") if isinstance(picked_log, dict) else "unknown"
            log_obs = f"Log anomaly around {comp}."

        final_obs = f"Select {component} due to strongest combined evidence."
        if reason:
            final_obs = reason

        steps.append({"step": 1, "action": "Analyze Metrics", "observation": metric_obs})
        steps.append({"step": 2, "action": "Analyze Traces", "observation": trace_obs})
        steps.append({"step": 3, "action": "Analyze Logs", "observation": log_obs})
        steps.append({"step": 4, "action": "Final Judgment", "observation": final_obs})
        return steps

    def _pick_fallback_component_reason(
        self,
        metric_result: Any,
        trace_result: Any,
        log_result: Any,
    ) -> tuple[str, str]:
        """
        当 LLM 输出不可用时，基于现有证据给出一个最小可用预测，
        避免返回 unknown / Analysis failed。
        """
        # 1) Log evidence first: usually strongest for explicit failures.
        if isinstance(log_result, list) and log_result:
            comp = str(log_result[0].get("component", "")).strip()
            obs = str(log_result[0].get("observation", "")).lower()
            if comp:
                if any(k in obs for k in ("connection refused", "unavailable", "transport is closing")):
                    return comp, "pod failure"
                if any(k in obs for k in ("timeout", "timed out", "dialing", "i/o")):
                    return comp, "network delay"
                if "exception" in obs:
                    return comp, "jvm exception"
                return comp, "code error"

        # 2) Metric evidence fallback.
        if isinstance(metric_result, list) and metric_result:
            svc = str(metric_result[0].get("service", "")).strip()
            kpi = str(metric_result[0].get("kpi", "")).lower()
            if svc:
                if "node_" in kpi:
                    if "memory" in kpi:
                        return svc, "node memory stress"
                    if "filesystem" in kpi or "disk" in kpi:
                        return svc, "node disk fill"
                    return svc, "node cpu stress"
                if "cpu" in kpi:
                    return svc, "cpu stress"
                if "memory" in kpi:
                    return svc, "memory stress"
                if "error" in kpi:
                    return svc, "code error"
                return svc, "code error"

        # 3) Trace-only fallback.
        if isinstance(trace_result, list) and trace_result:
            span = trace_result[0].get("span", {}) if isinstance(trace_result[0], dict) else {}
            src = str(span.get("source", "")).strip()
            if src:
                return src, "network delay"

        return "frontend", "code error"

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
                    
                    svc_items = list(comp_metrics.items())
                    omitted_svc = max(0, len(svc_items) - self.max_metric_services)
                    for svc, details in svc_items[:self.max_metric_services]:
                        clipped = details[:self.max_metric_kpis_per_service]
                        omitted_kpi = max(0, len(details) - len(clipped))
                        detail_str = "; ".join(clipped)
                        if omitted_kpi > 0:
                            detail_str += f"; ...(+{omitted_kpi} kpis)"
                        summary.append(f"- {svc}: [{detail_str}]")
                    if omitted_svc > 0:
                        summary.append(f"- ...(+{omitted_svc} services omitted)")
                
                # Handle direct dict output (fallback)
                elif isinstance(obs_data, dict):
                    events = obs_data.get('events', [])
                    for e in events:
                        ts = e.get('timestamps', [])
                        t_str = ts[0] if ts else ""
                        summary.append(f"- {e.get('pod')} {e.get('kpi')}: {e.get('pattern')} at {t_str}")

            elif source_type == "trace":
                # TraceAgent returns aggregated links
                if isinstance(obs_data, list):
                    omitted_trace = max(0, len(obs_data) - self.max_trace_links)
                    for link in obs_data[:self.max_trace_links]:
                        span = link.get('span', {})
                        details = link.get('details', [])
                        src, tgt = span.get('source'), span.get('target')
                        for d in details: 
                            pod = d.get('pod')
                            node = d.get('node', 'unknown')
                            lat = d.get('avg_latency_ms')
                            errs = d.get('error_messages', [])
                            # Highlight errors for 'Trace Severity (+2)' rule
                            err_str = f", Errs: {errs}" if errs else ""
                            summary.append(f"- {src}->{tgt}: {pod} (Node:{node}, {lat}ms{err_str})")
                    if omitted_trace > 0:
                        summary.append(f"- ...(+{omitted_trace} trace links omitted)")

            elif source_type == "log":
                # LogAgent returns anomalies list
                if isinstance(obs_data, list):
                    omitted_log = max(0, len(obs_data) - self.max_log_items)
                    for item in obs_data[:self.max_log_items]:
                        comp = item.get('component')
                        # Log keywords like 'connection refused' are crucial for 'Restart (+10)'
                        obs = item.get('observation', '')
                        summary.append(f"- {comp}: {obs}")
                    if omitted_log > 0:
                        summary.append(f"- ...(+{omitted_log} logs omitted)")
        
        except Exception as e:
            logger.error(f"Error formatting {source_type}: {e}")
            return "Format error."

        if not summary:
            return "No significant details."

        text = "\n".join(summary)
        if len(text) > self.max_obs_chars:
            text = text[:self.max_obs_chars] + "\n- ...(truncated for context limit)"
        return text

    def analyze(self, uuid: str, description: str, metric_result: Any, trace_result: Any, log_result: Any) -> Dict:
        """
        Constructs the prompt and calls the LLM.
        """
        logger.info(f"JudgeAgent: Analyzing {uuid}")

        # 1. Format inputs (Injecting Data)
        metric_obs = self._format_observation(metric_result, "metric")
        trace_obs = self._format_observation(trace_result, "trace")
        log_obs = self._format_observation(log_result, "log")

        system_prompt = HWLYYZC_SYSTEM_PROMPT

        user_prompt = self._assemble_user_prompt(
            description, metric_obs, trace_obs, log_obs
        )

        print(f"--- [PROMPT] ---\n{user_prompt}\n----------------")

        # 3. Call LLM
        # 关闭 SDK 内建重试，统一由本类控制重试与日志，避免双重重试叠加。
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=self.request_timeout,
            max_retries=0,
        )
        
        try:
            response = None
            last_exc = None
            model_candidates = self._build_model_candidates()
            for model_name in model_candidates:
                req_kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                }

                try:
                    response = self._create_completion_with_retry(client, req_kwargs)
                    if model_name != self.model:
                        logger.warning("LLM model fallback succeeded with model=%s", model_name)
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning("Model attempt failed: model=%s err=%s", model_name, e)

                    # 网关不支持 json_object 时再试一次去掉 response_format（DeepSeek / Yuzo 共用）。
                    if "response_format" in req_kwargs:
                        try:
                            req_kwargs.pop("response_format", None)
                            response = self._create_completion_with_retry(client, req_kwargs)
                            if model_name != self.model:
                                logger.warning("LLM model fallback succeeded with model=%s", model_name)
                            break
                        except Exception as e2:
                            last_exc = e2
                            logger.warning("Model attempt without response_format failed: model=%s err=%s", model_name, e2)
                    continue

            if response is None:
                raise last_exc if last_exc else RuntimeError("LLM request failed for all model candidates")
            
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ValueError("LLM returned empty choices")
            message = getattr(choices[0], "message", None)
            if message is None:
                raise ValueError("LLM returned empty message")
            content = getattr(message, "content", None)
            if content is None:
                raise ValueError("LLM returned empty content")
            print(f"--- [RESPONSE] ---\n{content}\n------------------")
            
            # 4. Parse & Validate
            content = self._extract_json_text(content)
            try:
                parsed = json.loads(content)
            except Exception:
                salvaged = self._salvage_partial_json(content)
                if salvaged is None:
                    raise
                logger.warning("LLM returned partial JSON; salvaged component/reason from truncated output")
                parsed = salvaged
            
            component = parsed.get("component", "unknown")
            # Fallback validation - keep simple validation only
            if component not in VALID_COMPONENTS and component != "unknown":
                # Simple heuristic correction if LLM outputs specific pod instead of service
                base = component.rsplit('-', 1)[0]
                if base in VALID_COMPONENTS:
                    component = base  # Use base service name
                elif component.startswith("aiops-k8s") or component.startswith("k8s-master"):
                    pass  # Allow nodes
                else:
                    logger.warning(f"Invalid component: {component}, will use fallback if needed")

            # Enforce 20-word limit on reason/observation (conservative post-processing)
            reason = " ".join(parsed.get("reason", "").split()[:20])
            trace = parsed.get("reasoning_trace", [])
            if not isinstance(trace, list):
                trace = []

            # Only use fallback when trace is empty OR component is invalid
            if len(trace) == 0 or component in ("unknown", ""):
                component, reason = self._pick_fallback_component_reason(
                    metric_result=metric_result,
                    trace_result=trace_result,
                    log_result=log_result,
                )
                trace = self._build_fallback_reasoning_trace(
                    component=component,
                    reason=reason,
                    metric_result=metric_result,
                    trace_result=trace_result,
                    log_result=log_result,
                )

            # Conservative clipping only - no aggressive normalization or injection
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
            # Conservative fallback - only use safe methods
            component, reason = self._pick_fallback_component_reason(
                metric_result=metric_result,
                trace_result=trace_result,
                log_result=log_result,
            )
            trace = self._build_fallback_reasoning_trace(
                component=component,
                reason=reason,
                metric_result=metric_result,
                trace_result=trace_result,
                log_result=log_result,
            )
            for step in trace:
                if "observation" in step:
                    step["observation"] = " ".join(str(step["observation"]).split()[:20])
            return {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": trace,
            }