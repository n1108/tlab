"""Wraps bundled Metric/Trace/Log agents as tool implementations."""

from __future__ import annotations

import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from rag_agent.bundled.agent.judge import JudgeAgent
from rag_agent.bundled.agent.log import LogAgent
from rag_agent.bundled.agent.metric import MetricAgent
from rag_agent.bundled.agent.trace import TraceAgent

logger = logging.getLogger(__name__)

# Cap tool payload size for LLM context
_MAX_TOOL_CHARS = 14_000


class DetectionToolkit:
    """Holds detectors and optional caches for one case."""

    def __init__(self, dataset_root: str) -> None:
        self._root = dataset_root
        self.metric_agent = MetricAgent(dataset_root)
        self.trace_agent = TraceAgent(dataset_root)
        self.log_agent = LogAgent(dataset_root)
        self._formatter: Optional[JudgeAgent] = None
        self._cache: Dict[str, Any] = {}
        self._formatted_cache: Dict[str, str] = {}

    def _format(self, obs: Any, source_type: str) -> str:
        if self._formatter is None:
            # API not used for formatting-only path
            self._formatter = JudgeAgent(api_key="unused-formatting-only", api_url=None)
        return self._formatter._format_observation(obs, source_type)

    def _clip(self, text: str) -> str:
        if len(text) <= _MAX_TOOL_CHARS:
            return text
        return text[: _MAX_TOOL_CHARS] + "\n...(truncated for tool output limit)"

    def get_metric_anomalies(self, start: datetime, end: datetime) -> str:
        key = "metric"
        if key not in self._cache:
            logger.info("Tool get_metric_anomalies: running detector")
            raw = self.metric_agent.score(start, end)
            self._cache[key] = raw
        if key not in self._formatted_cache:
            raw = self._cache[key]
            self._formatted_cache[key] = self._clip(self._format(raw, "metric"))
        return self._formatted_cache[key]

    def get_trace_anomalies(self, start: datetime, end: datetime) -> str:
        key = "trace"
        if key not in self._cache:
            logger.info("Tool get_trace_anomalies: running detector")
            raw = self.trace_agent.score(start, end)
            self._cache[key] = raw
        if key not in self._formatted_cache:
            raw = self._cache[key]
            self._formatted_cache[key] = self._clip(self._format(raw, "trace"))
        return self._formatted_cache[key]

    def get_log_anomalies(self, start: datetime, end: datetime) -> str:
        key = "log"
        if key not in self._cache:
            logger.info("Tool get_log_anomalies: running detector")
            raw = self.log_agent.score(start, end)
            self._cache[key] = raw
        if key not in self._formatted_cache:
            raw = self._cache[key]
            self._formatted_cache[key] = self._clip(self._format(raw, "log"))
        return self._formatted_cache[key]

    def prefetch_all(self, start: datetime, end: datetime) -> None:
        """
        Warm all detector caches in parallel for faster tool rounds.
        Safe to call multiple times.
        """
        missing = [k for k in ("metric", "trace", "log") if k not in self._cache]
        if not missing:
            return

        def _run(name: str):
            if name == "metric":
                self._cache["metric"] = self.metric_agent.score(start, end)
            elif name == "trace":
                self._cache["trace"] = self.trace_agent.score(start, end)
            elif name == "log":
                self._cache["log"] = self.log_agent.score(start, end)

        workers = min(3, len(missing))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_run, missing))

    def dispatch(
        self, name: str, start: datetime, end: datetime
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Returns (tool_result_string, parsed_submit_dict_or_None).
        """
        if name == "get_metric_anomalies":
            return self.get_metric_anomalies(start, end), None
        if name == "get_trace_anomalies":
            return self.get_trace_anomalies(start, end), None
        if name == "get_log_anomalies":
            return self.get_log_anomalies(start, end), None
        return f"Unknown tool: {name}", None


def openai_tool_schemas() -> List[Dict[str, Any]]:
    """Schemas for Chat Completions `tools` parameter (OpenAI-compatible)."""
    empty_params = {"type": "object", "properties": {}, "additionalProperties": False}
    submit_params = {
        "type": "object",
        "properties": {
            "component": {"type": "string", "description": "Root-cause component name"},
            "reason": {"type": "string", "description": "Single standard fault-type phrase"},
            "reasoning_trace": {
                "type": "array",
                "description": "Ordered analysis steps",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "action": {"type": "string"},
                        "observation": {"type": "string"},
                    },
                    "required": ["step", "action", "observation"],
                },
            },
        },
        "required": ["component", "reason", "reasoning_trace"],
        "additionalProperties": False,
    }

    def fn(name: str, desc: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": parameters}}

    return [
        fn(
            "get_metric_anomalies",
            "Run metric anomaly detection for the case time window; returns a text summary.",
            empty_params,
        ),
        fn(
            "get_trace_anomalies",
            "Run trace-based anomaly detection (latency/errors on call edges); returns a text summary.",
            empty_params,
        ),
        fn(
            "get_log_anomalies",
            "Run log anomaly detection (template surges / new patterns); returns a text summary.",
            empty_params,
        ),
        fn(
            "submit_root_cause_analysis",
            "Submit the final root-cause JSON fields when ready to conclude.",
            submit_params,
        ),
    ]
