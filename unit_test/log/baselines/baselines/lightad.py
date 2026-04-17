"""
LightAD baselines (KNN, DT, SLFN) - 封装在单个文件中。
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


# === Inline functions previously from log_summary_common.py ===
def build_vectorizer(texts: list[str]):
    """Build CountVectorizer (same as old common)."""
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(
        min_df=2,
        max_df=0.95,
        stop_words="english",
        binary=True,
        ngram_range=(1, 2),
    ).fit(texts)


def make_synthetic_anomalies(x_norm: np.ndarray, rng: np.random.Generator, noise_scale: float = 0.3):
    """Create synthetic anomalies by adding noise (same logic as old common)."""
    x_syn = x_norm.copy()
    noise = rng.normal(0, noise_scale, x_syn.shape).astype(np.float32)
    x_syn = np.clip(x_syn + noise, 0, 1)
    return x_syn


def normal_windows(start: datetime, end: datetime, minutes: int = 30):
    """Return (normal_start, normal_end) windows."""
    from datetime import timedelta
    delta = timedelta(minutes=minutes)
    return (start - delta, start), (end, end + delta)


def summarize_baseline_predictions(
    fault_df: pd.DataFrame,
    pred_positive: np.ndarray,
    components: set[str] | None = None,
    max_samples: int = 8,
) -> tuple[str, list[dict]]:
    """Simple summarization for baseline predictions (error-like first)."""
    if not isinstance(pred_positive, np.ndarray) or len(pred_positive) == 0:
        return "- no anomaly detected", []

    positive_mask = pred_positive.astype(bool)
    positive_df = fault_df[positive_mask].copy() if isinstance(fault_df, pd.DataFrame) else fault_df

    if len(positive_df) == 0:
        return "- no anomaly detected", []

    # Take first few error-like samples
    samples = positive_df.head(max_samples).to_dict("records")
    lines = []
    for row in samples:
        comp = str(row.get("k8_pod") or row.get("component", "unknown"))
        obs = str(row.get("text_line", row.get("observation", "")))[:120]
        lines.append(f"{comp}: {obs}")

    summary = "\n".join(lines)
    return summary, samples


# === End inline functions ===


class LightADBaseline:
    """封装 LightAD 风格的 KNN/DT/SLFN baseline"""

    def __init__(self, method: Literal["knn", "dt", "slfn"] = "knn", seed: int = 42):
        self.method = method
        self.seed = seed
        self.name = f"lightad_{method}"
        self.np_rng = np.random.default_rng(seed)

    def score(self, fault_texts: list[str], normal_texts: list[str]) -> dict:
        """Run LightAD-style detection and return summary text."""
        if len(normal_texts) < 20 or not fault_texts:
            return {"text": f"- no anomaly ({self.method})", "count": 0}

        vec = build_vectorizer(normal_texts)
        x_norm = vec.transform(normal_texts).toarray().astype(np.float32)
        x_syn = make_synthetic_anomalies(x_norm, self.np_rng if hasattr(self, 'np_rng') else np.random.default_rng(self.seed))
        x_train = np.vstack([x_norm, x_syn])
        y_train = np.concatenate([np.zeros(len(x_norm)), np.ones(len(x_syn))])

        x_fault = vec.transform(fault_texts).toarray().astype(np.float32)
        fault_df = pd.DataFrame({"text_line": fault_texts, "k8_pod": ["unknown"] * len(fault_texts)})

        if self.method == "knn":
            clf = KNeighborsClassifier(n_neighbors=1, metric="minkowski", n_jobs=-1)
        elif self.method == "dt":
            clf = DecisionTreeClassifier(random_state=self.seed, class_weight="balanced")
        else:  # slfn
            clf = MLPClassifier(hidden_layer_sizes=(25,), max_iter=300, random_state=self.seed)

        clf.fit(x_train, y_train)
        pred = clf.predict(x_fault)
        text, _ = summarize_baseline_predictions(fault_df, pred == 1, set())

        return {"text": text, "count": int((pred == 1).sum())}
