"""
NeuralLog baseline wrapper.
"""
from __future__ import annotations

from unit_test.log.baselines.neural_log.baseline import NeuralLogBaseline as _NeuralLogBaseline


class NeuralLogBaseline:
    """Wrapper for compatibility with orchestrator"""

    def __init__(self):
        self._impl = _NeuralLogBaseline(contamination=0.15, random_state=42)
        self.name = "neural_log"

    def score(self, fault_texts: list[str], normal_texts: list[str]) -> dict:
        return self._impl.score(fault_texts, normal_texts)
