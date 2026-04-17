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

from unit_test.log.baselines.log_summary_common import (
    summarize_baseline_predictions,
    build_vectorizer,
    make_synthetic_anomalies,
    normal_windows,
)

logger = logging.getLogger(__name__)


class LightADBaseline:
    """封装 LightAD 风格的 KNN/DT/SLFN baseline"""

    def __init__(self, method: Literal["knn", "dt", "slfn"] = "knn", seed: int = 42):
        self.method = method
        self.seed = seed
        self.name = f"lightad_{method}"

    def score(self, fault_texts: list[str], normal_texts: list[str]) -> dict:
        if len(normal_texts) < 20 or not fault_texts:
            return {"text": f"- no anomaly ({self.method})", "count": 0}

        vec = build_vectorizer(normal_texts)
        x_norm = vec.transform(normal_texts).toarray().astype(np.float32)
        x_syn = make_synthetic_anomalies(x_norm, np.random.default_rng(self.seed))
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
