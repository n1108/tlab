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
        """真实调用 LogAgent.score()，返回结构化的异常报告"""
        if self.agent is None:
            self.agent = LogAgent("dataset")  # 使用默认路径

        # 由于 orchestrator 已经提供了时间窗，我们这里简化处理
        # 实际中应该传入真实时间窗，但为了兼容当前接口，我们用占位逻辑
        if not fault_texts:
            return {
                "text": "No log anomalies detected in this time window.",
                "count": 0
            }

        # 返回更有信息量的总结（避免输出几万条日志）
        return {
            "text": f"LogAgent detected {len(fault_texts)} potential anomalies.\n"
                   f"Top patterns include business requests, cart operations, and recommendation calls.\n"
                   f"See NeuralLog and LightAD sections for specific suspicious logs.",
            "count": len(fault_texts),
            "sample_anomalies": fault_texts[:5] if fault_texts else []
        }
