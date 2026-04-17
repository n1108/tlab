"""
LogAgent baseline wrapper for orchestrator.
"""
from __future__ import annotations

from exp.agent.log import LogAgent


class LogAgentBaseline:
    """Wrapper for LogAgent to fit orchestrator interface"""

    def __init__(self):
        self.name = "log_agent"
        self.agent = None

    def score(self, fault_texts: list[str], normal_texts: list[str] | None = None) -> dict:
        # In practice, LogAgent uses time windows, not text lists.
        # This is a simplified wrapper - real implementation uses time-based scoring.
        if not fault_texts:
            return {"text": "- no anomaly", "count": 0}

        # For now return placeholder. Real LogAgent is called via time windows in orchestrator.
        return {
            "text": f"LogAgent detected {len(fault_texts)} potential anomalies (simplified)",
            "count": len(fault_texts),
        }
