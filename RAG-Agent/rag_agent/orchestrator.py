"""Multi-turn LLM loop with tool calls; falls back to one-shot JudgeAgent.analyze."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from rag_agent.bundled.agent.judge import JudgeAgent
from rag_agent.bundled.utils.time import parse_time_range
from rag_agent.prompts import RAG_AGENT_SYSTEM_PROMPT
from rag_agent.tool_runner import DetectionToolkit, openai_tool_schemas

logger = logging.getLogger(__name__)


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return any(
        k in msg
        for k in (
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "timed out",
            "timeout",
        )
    )


def _completion_with_retry(client: OpenAI, req_kwargs: Dict[str, Any], max_retries: int = 4):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**req_kwargs)
        except Exception as e:
            last_exc = e
            if not _is_transient_error(e) or attempt >= max_retries:
                raise
            backoff = min(8.0, 0.8 * (2 ** (attempt - 1)))
            logger.warning("LLM transient error (attempt %s/%s): %s; sleep %.1fs", attempt, max_retries, e, backoff)
            time.sleep(backoff)
    if last_exc:
        raise last_exc
    raise RuntimeError("completion failed")


def _build_user_message(uuid: str, description: str, start: datetime, end: datetime) -> str:
    return (
        f"Case UUID: {uuid}\n"
        f"Anomaly description (includes UTC window): {description}\n"
        f"Parsed window (naive local as in detectors): start={start}, end={end}\n\n"
        "Call the detector tools as needed, then submit_root_cause_analysis."
    )


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    """Convert SDK message to API-shaped dict for next request."""
    d: Dict[str, Any] = {"role": "assistant", "content": msg.content or None}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in tcs
        ]
    return d


def run_rag_case(
    uuid: str,
    description: str,
    dataset_root: str,
    judge: JudgeAgent,
    max_turns: int = 8,
) -> Dict[str, Any]:
    """
    Run tool-augmented RCA. On failure or missing submit tool, falls back to JudgeAgent.analyze.
    """
    start, end = parse_time_range(description)
    if not start or not end:
        return {
            "uuid": uuid,
            "component": "Unknown",
            "reason": "Time range parsing failed.",
            "reasoning_trace": [],
            "rag_meta": {"mode": "error", "detail": "parse_time_range"},
        }

    toolkit = DetectionToolkit(dataset_root)
    tools = openai_tool_schemas()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": RAG_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(uuid, description, start, end)},
    ]

    client = OpenAI(
        api_key=judge.api_key,
        base_url=judge.api_url,
        timeout=judge.request_timeout,
        max_retries=0,
    )
    model = judge.model or "deepseek-chat"
    temperature = float(os.getenv("RAG_AGENT_TEMPERATURE", str(judge.temperature)))

    submit_result: Optional[Dict[str, Any]] = None
    tool_rounds = 0

    for turn in range(max_turns):
        req_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            resp = _completion_with_retry(client, req_kwargs)
        except Exception as e:
            logger.error("RAG LLM call failed: %s", e)
            break

        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []

        if not tcs:
            # Model answered in plain text — nudge or exit
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
            messages.append(
                {
                    "role": "user",
                    "content": "You must call the detector tools (metric/trace/log) and then "
                    "submit_root_cause_analysis with structured fields. If already done, call submit_root_cause_analysis now.",
                }
            )
            continue

        messages.append(_assistant_message_dict(msg))

        for tc in tcs:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}

            if name == "submit_root_cause_analysis":
                submit_result = {
                    "component": str(args.get("component", "unknown")).strip(),
                    "reason": " ".join(str(args.get("reason", "")).split()[:20]),
                    "reasoning_trace": args.get("reasoning_trace") or [],
                }
                if not isinstance(submit_result["reasoning_trace"], list):
                    submit_result["reasoning_trace"] = []
                out = json.dumps({"status": "accepted", "message": "Final answer recorded."}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                continue

            try:
                payload, _ = toolkit.dispatch(name, start, end)
            except Exception as e:
                payload = f"Tool error ({name}): {e}"
                logger.exception("Tool %s failed", name)
            tool_rounds += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload})

        if submit_result is not None:
            # Normalize reasoning_trace steps (20-word clip like JudgeAgent)
            trace = []
            for i, step in enumerate(submit_result["reasoning_trace"]):
                if not isinstance(step, dict):
                    continue
                trace.append(
                    {
                        "step": int(step.get("step", i + 1)),
                        "action": str(step.get("action", ""))[:200],
                        "observation": " ".join(str(step.get("observation", "")).split()[:20]),
                    }
                )
            if len(trace) < 4:
                m = toolkit._cache.get("metric")
                t = toolkit._cache.get("trace")
                lg = toolkit._cache.get("log")
                if m is None:
                    m = toolkit.metric_agent.score(start, end)
                if t is None:
                    t = toolkit.trace_agent.score(start, end)
                if lg is None:
                    lg = toolkit.log_agent.score(start, end)
                trace = judge._build_fallback_reasoning_trace(
                    submit_result["component"],
                    submit_result["reason"],
                    m,
                    t,
                    lg,
                )
            return {
                "uuid": uuid,
                "component": submit_result["component"],
                "reason": submit_result["reason"],
                "reasoning_trace": trace,
                "rag_meta": {
                    "mode": "rag_tools",
                    "turns_used": turn + 1,
                    "detector_tool_invocations": tool_rounds,
                },
            }

    # Fallback: one-shot judge with full detector outputs (same as exp main)
    logger.warning("RAG tool loop did not submit; falling back to JudgeAgent.analyze")
    metric_result = toolkit.metric_agent.score(start, end)
    trace_result = toolkit.trace_agent.score(start, end)
    log_result = toolkit.log_agent.score(start, end)
    analysis = judge.analyze(uuid, description, metric_result, trace_result, log_result)
    analysis["rag_meta"] = {
        "mode": "fallback_judge_analyze",
        "turns_used": max_turns,
        "detector_tool_invocations": tool_rounds,
    }
    return analysis
