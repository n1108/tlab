"""
NeuralLog-inspired baseline (no parsing, semantic vectors + anomaly detection).
Lightweight sklearn version for compatibility with current tlab environment.
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class NeuralLogBaseline:
    """
    模拟 NeuralLog 的「无解析 + 语义向量 + 异常检测」思路。
    使用 TF-IDF + IsolationForest 实现轻量版，便于集成到现有 baseline 框架。
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.name = "neural_log"

    def fit(self, normal_texts: list[str]) -> None:
        """使用正常窗口的日志文本训练 IsolationForest。"""
        if not normal_texts:
            logger.warning("NeuralLogBaseline: no normal texts provided")
            self.is_fitted = True
            return

        X = self.vectorizer.fit_transform(normal_texts).toarray()
        self.model.fit(X)
        self.is_fitted = True
        logger.info("NeuralLogBaseline fitted on %d normal samples", len(normal_texts))

    def predict(self, texts: list[str]) -> np.ndarray:
        """返回异常预测（1=异常，-1=正常）。"""
        if not self.is_fitted or not texts:
            return np.zeros(len(texts), dtype=int)

        X = self.vectorizer.transform(texts).toarray()
        scores = self.model.predict(X)  # 1=正常, -1=异常
        return (scores == -1).astype(int)

    def score(self, fault_texts: list[str], normal_texts: list[str] | None = None) -> dict:
        """
        兼容现有 baseline 接口。
        返回 anomaly text summary，供 generate_log_summary 使用。
        """
        if not fault_texts:
            return {"text": "- no anomaly", "anomalies": []}

        if normal_texts and not self.is_fitted:
            self.fit(normal_texts)

        pred = self.predict(fault_texts)
        anomalies = [text for text, is_anom in zip(fault_texts, pred) if is_anom]

        if not anomalies:
            text = "- no anomaly"
        else:
            # 取前5条异常日志作为样本
            samples = anomalies[:5]
            text = f"NeuralLog detected {len(anomalies)} anomalies. Samples:\n" + "\n".join(
                [f"- {s[:200]}{'...' if len(s)>200 else ''}" for s in samples]
            )

        return {
            "text": text,
            "anomalies": anomalies,
            "count": len(anomalies),
        }


# 导出供 baselines 模块使用
__all__ = ["NeuralLogBaseline"]
