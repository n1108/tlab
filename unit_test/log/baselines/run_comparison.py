"""
逐 uuid 的实时检测对比：LogAgent vs LightAD 风格 KNN / DT / SLFN。

对每个故障 uuid：
- 正常模式窗口：故障前 30 分钟 + 故障后 30 分钟（若存在）
- 检测窗口：故障时间段本身
- LogAgent：保持原实现（前 30 分钟 baseline 对比）
- KNN/DT/SLFN：仅用“正常窗口”训练（实时/近实时设定）
  - 为了继续使用 LightAD 的二分类器，构造合成异常样本（打乱正常向量列）作为负类
  - 在故障窗日志上预测为异常(label=1)的文本参与关键词召回评分
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from unit_test.log.baselines.loader import load_window_lines

logger = logging.getLogger(__name__)

PROJECT_ROOT = _ROOT
DATASET_JSON = PROJECT_ROOT / "unit_test/log/log_unit_test_dataset.json"
REPORT_PATH = PROJECT_ROOT / "unit_test/log/results/baseline_comparison_report.md"


def _load_eval_module():
    path = PROJECT_ROOT / "unit_test/log/evaluate_log_agent.py"
    spec = importlib.util.spec_from_file_location("evaluate_log_agent", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


eva = _load_eval_module()


def _normal_windows(start: datetime, end: datetime, minutes: int = 30):
    delta = timedelta(minutes=minutes)
    return (start - delta, start), (end, end + delta)


def _score_keyword_recall(
    fault_df: pd.DataFrame,
    pred_positive: np.ndarray,
    components: set[str],
    patterns: list[list[str]],
) -> tuple[int, int]:
    total = len(patterns)
    if total == 0:
        return 0, 0

    sub = fault_df.loc[pred_positive.astype(bool)]
    blobs: dict[str, str] = {}
    for _, row in sub.iterrows():
        pod = str(row.get("k8_pod") or "unknown")
        t = str(row.get("text_line") or "")
        blobs[pod] = blobs.get(pod, "") + " " + t

    hits = 0
    for seq in patterns:
        matched = False
        for pod, blob in blobs.items():
            if not components:
                continue
            if not eva._component_matches(pod, components):
                continue
            if eva._keywords_in_order(blob, seq):
                matched = True
                break
        if matched:
            hits += 1
    return hits, total


def _match_patterns_with_blobs(
    blobs: dict[str, str],
    components: set[str],
    patterns: list[list[str]],
) -> list[bool]:
    matched_flags: list[bool] = []
    for seq in patterns:
        matched = False
        for pod, blob in blobs.items():
            if not components:
                continue
            if not eva._component_matches(pod, components):
                continue
            if eva._keywords_in_order(blob, seq):
                matched = True
                break
        matched_flags.append(matched)
    return matched_flags


def _log_agent_pattern_matches(agent, start: datetime, end: datetime, components: set[str], patterns: list[list[str]]):
    anomalies = agent.score(start, end)
    blobs: dict[str, str] = {}
    for block in anomalies:
        comp = str(block.get("component", ""))
        if not comp:
            continue
        blobs[comp] = blobs.get(comp, "") + " " + eva._anomaly_text(block)
    return _match_patterns_with_blobs(blobs, components, patterns)


def _build_vectorizer(texts: list[str]) -> CountVectorizer:
    vec = CountVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1, max_df=0.98)
    vec.fit(texts)
    return vec


def _make_synthetic_anomalies(x_norm_dense: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """列内打乱，破坏 token 共现关系，作为伪异常。"""
    x_syn = x_norm_dense.copy()
    n = x_syn.shape[0]
    for j in range(x_syn.shape[1]):
        idx = rng.permutation(n)
        x_syn[:, j] = x_syn[idx, j]
    return x_syn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--limit-uuids", type=int, default=0, help="0=全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-window-minutes", type=int, default=30)
    parser.add_argument("--max-normal-lines", type=int, default=800)
    parser.add_argument("--output-md", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("exp", "exp.utils.input", "exp.agent.log", "drain3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    with DATASET_JSON.open(encoding="utf-8") as f:
        cases: list[dict] = json.load(f)
    if args.limit_uuids > 0:
        cases = cases[: args.limit_uuids]

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    from exp.agent.log import LogAgent

    root = str(args.dataset_root)
    agent = LogAgent(root)

    totals = {"log_agent": (0, 0, 0.0), "knn": (0, 0, 0.0), "dt": (0, 0, 0.0), "slfn": (0, 0, 0.0), "union": (0, 0, 0.0)}
    used_uuids = 0
    skipped_uuids = 0

    for i, item in enumerate(cases, start=1):
        uuid = str(item.get("uuid", ""))
        patterns = item.get("expected_log_patterns") or []
        components = set(str(x) for x in item.get("root_cause_components", []) if x)

        if not patterns or not components:
            continue

        try:
            start = eva._parse_iso_utc(str(item["start_time"]))
            end = eva._parse_iso_utc(str(item["end_time"]))
        except (ValueError, TypeError, KeyError):
            skipped_uuids += 1
            continue

        t0 = time.perf_counter()
        logagent_flags = _log_agent_pattern_matches(agent, start, end, components, patterns)
        h = sum(1 for x in logagent_flags if x)
        tot = len(patterns)
        ph, pt, ps = totals["log_agent"]
        totals["log_agent"] = (ph + h, pt + tot, ps + (time.perf_counter() - t0))
        union_flags = list(logagent_flags)

        (n1s, n1e), (n2s, n2e) = _normal_windows(start, end, args.normal_window_minutes)
        try:
            n1 = load_window_lines(root, n1s, n1e)
            n2 = load_window_lines(root, n2s, n2e)
            fault_df = load_window_lines(root, start, end)
        except Exception as e:
            logger.warning("[%s/%s] uuid=%s load error: %s", i, len(cases), uuid, e)
            n1, n2, fault_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        normal_df = pd.concat([n1, n2], ignore_index=True) if (not n1.empty or not n2.empty) else pd.DataFrame()

        if fault_df.empty:
            for m in ("knn", "dt", "slfn"):
                mh, mt, ms = totals[m]
                totals[m] = (mh, mt + len(patterns), ms)
            uh, ut, us = totals["union"]
            totals["union"] = (uh + sum(1 for x in union_flags if x), ut + len(patterns), us)
            skipped_uuids += 1
            continue

        normal_lines = normal_df["text_line"].astype(str).tolist() if not normal_df.empty else []
        if len(normal_lines) > args.max_normal_lines:
            normal_lines = rng_py.sample(normal_lines, args.max_normal_lines)

        if len(normal_lines) < 20:
            for m in ("knn", "dt", "slfn"):
                mh, mt, ms = totals[m]
                totals[m] = (mh, mt + len(patterns), ms)
            uh, ut, us = totals["union"]
            totals["union"] = (uh + sum(1 for x in union_flags if x), ut + len(patterns), us)
            skipped_uuids += 1
            continue

        vec = _build_vectorizer(normal_lines)
        x_norm_dense = vec.transform(normal_lines).toarray().astype(np.float32)
        x_syn = _make_synthetic_anomalies(x_norm_dense, rng_np)
        x_train_dense = np.vstack([x_norm_dense, x_syn])
        y_train = np.concatenate(
            [np.zeros(len(x_norm_dense), dtype=np.int32), np.ones(len(x_syn), dtype=np.int32)]
        )

        models = {
            "knn": KNeighborsClassifier(n_neighbors=1, metric="minkowski", n_jobs=-1),
            "dt": DecisionTreeClassifier(random_state=args.seed, class_weight="balanced"),
            "slfn": MLPClassifier(hidden_layer_sizes=(25,), max_iter=300, random_state=args.seed),
        }

        x_fault_dense = vec.transform(fault_df["text_line"].astype(str)).toarray().astype(np.float32)

        for name, clf in models.items():
            t0 = time.perf_counter()
            clf.fit(x_train_dense, y_train)
            pred = clf.predict(x_fault_dense)
            infer_sec = time.perf_counter() - t0
            pos = pred == 1
            h, tot = _score_keyword_recall(fault_df, pos, components, patterns)
            mh, mt, ms = totals[name]
            totals[name] = (mh + h, mt + tot, ms + infer_sec)
            sub = fault_df.loc[pos.astype(bool)]
            blobs: dict[str, str] = {}
            for _, row in sub.iterrows():
                pod = str(row.get("k8_pod") or "unknown")
                text_line = str(row.get("text_line") or "")
                blobs[pod] = blobs.get(pod, "") + " " + text_line
            model_flags = _match_patterns_with_blobs(blobs, components, patterns)
            union_flags = [a or b for a, b in zip(union_flags, model_flags)]

        used_uuids += 1
        uh, ut, us = totals["union"]
        totals["union"] = (uh + sum(1 for x in union_flags if x), ut + len(patterns), us)
        if i % 20 == 0:
            logger.info("processed %s/%s", i, len(cases))

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Log 异常检测对比报告（实时窗对齐版）\n\n")
    lines.append(f"生成时间: {gen_time}\n\n")
    lines.append("## 设置\n\n")
    ds_desc = f"前 {len(cases)} 条" if args.limit_uuids > 0 else f"全量 {len(cases)} 条"
    lines.append(f"- **数据集**: `log_unit_test_dataset.json`，{ds_desc}\n")
    lines.append(f"- **正常窗口**: 每个 uuid 的故障前后各 {args.normal_window_minutes} 分钟（排除故障窗）\n")
    lines.append("- **检测目标**: 故障窗内实时识别异常日志，再按根因组件关键词序列计召回\n")
    lines.append(f"- **有效 uuid 数**: {used_uuids}（样本不足/空窗跳过 {skipped_uuids}）\n")
    lines.append(f"- **Parquet 根目录**: `{args.dataset_root.resolve()}`\n\n")

    lines.append("## 方法对齐说明\n\n")
    lines.append("- **LogAgent**: 保持原仓库规则（前 30 分钟基线 vs 故障窗）。\n")
    lines.append("- **KNN/DT/SLFN**: 每个 uuid 独立建模，仅用该 uuid 正常窗训练；为复用 LightAD 二分类器，使用列打乱构造伪异常类。\n")
    lines.append("- **评分**: 与 `evaluate_log_agent.py` 一致，统计关键词序列召回。\n\n")

    lines.append("## 结果\n\n")
    lines.append("| 方法 | 命中/期望模式 | 召回率 | 累计耗时(秒) |\n")
    lines.append("|------|----------------|--------|-------------|\n")
    for key, label in (
        ("log_agent", "LogAgent"),
        ("knn", "KNN (LightAD)"),
        ("dt", "Decision Tree (LightAD)"),
        ("slfn", "SLFN / MLP (LightAD)"),
        ("union", "Union (all methods)"),
    ):
        h, t, sec = totals[key]
        r = (h / t) if t else 0.0
        lines.append(f"| {label} | {h}/{t} | {r:.4f} | {sec:.2f} |\n")

    lines.append("\n## 说明\n\n")
    lines.append("- 该对比强调**每个故障独立、邻近时间窗建模**，避免跨 uuid 离线训练信息泄漏。\n")
    lines.append("- 部分小时 parquet 缺失会导致相应用例计入分母但难以命中。\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {args.output_md}")
    for key in ("log_agent", "knn", "dt", "slfn"):
        h, t, sec = totals[key]
        r = (h / t) if t else 0.0
        print(f"  {key}: {h}/{t} = {r:.4f} (sec={sec:.2f})")
    h, t, _ = totals["union"]
    r = (h / t) if t else 0.0
    print(f"  union: {h}/{t} = {r:.4f}")


if __name__ == "__main__":
    main()
