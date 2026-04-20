"""
**Deprecated / Optional** - Baseline comparison report.

**Recommendation**: Use `orchestrator.py` (the single entrypoint) to run all baselines and generate `results/log_summary.txt` for JudgeAgent.

检测仅用 parquet 时间窗与日志文本；评测标注来自 `log_unit_test_dataset.json`（期望模式与组件），不传入各 baseline 模型。
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
from unit_test.log.baselines.neural_log.baseline import NeuralLogBaseline
from unit_test.log.input_cases import load_input_json_cases

logger = logging.getLogger(__name__)

PROJECT_ROOT = _ROOT
REPORT_PATH = PROJECT_ROOT / "unit_test/log/results/baseline_comparison_report.md"
DEFAULT_LOG_UNIT_TEST_GT = PROJECT_ROOT / "unit_test" / "log" / "log_unit_test_dataset.json"
DEFAULT_PRECOMPUTE_DIR = PROJECT_ROOT / "results" / "log_precompute"


def load_precompute_jsonl(path: Path | None) -> dict[str, dict]:
    """uuid -> 预计算行（orchestrator 写出的 jsonl）。"""
    if path is None or not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(r.get("uuid", "") or "").strip()
            if uid:
                out[uid] = r
    return out


def pick_precompute_file(directory: Path, *filenames: str) -> Path | None:
    for name in filenames:
        p = directory / name
        if p.is_file():
            return p
    return None


def _score_keywords_on_text(text: str, patterns: list[list[str]]) -> tuple[int, int]:
    tot = len(patterns)
    if not tot:
        return 0, 0
    t = text or ""
    hits = sum(1 for seq in patterns if eva._keywords_in_order(t, seq))
    return hits, tot


def _match_patterns_on_text(text: str, patterns: list[list[str]]) -> list[bool]:
    if not patterns:
        return []
    t = text or ""
    return [eva._keywords_in_order(t, seq) for seq in patterns]


def load_log_unit_test_ground_truth(path: Path) -> dict[str, dict[str, object]]:
    """
    仅评测用：expected_log_patterns、root_cause_components。
    不参与各 baseline 的特征提取或训练（检测阶段不使用该文件）。
    """
    if not path.is_file():
        raise FileNotFoundError(f"log unit-test groundtruth not found: {path}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array in {path}")
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("uuid", "") or "").strip()
        if not uid:
            continue
        comps_raw = row.get("root_cause_components") or []
        components: set[str] = {str(c).strip() for c in comps_raw if str(c).strip()}
        pats_raw = row.get("expected_log_patterns") or []
        patterns: list[list[str]] = []
        for seq in pats_raw:
            if isinstance(seq, list):
                patterns.append([str(x) for x in seq])
            elif isinstance(seq, str) and seq.strip():
                patterns.append([seq.strip()])
        out[uid] = {"patterns": patterns, "components": components}
    return out


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


def _merged_text_fault_gt_pods(fault_df: pd.DataFrame, components: set[str]) -> str:
    """故障窗内、根因组件对应 pod 上的全部日志文本（不限制为 IF 阳性）。"""
    if not components:
        return ""
    parts: list[str] = []
    for _, row in fault_df.iterrows():
        pod = str(row.get("k8_pod") or "unknown")
        if not eva._component_matches(pod, components):
            continue
        parts.append(str(row.get("text_line") or ""))
    return " ".join(parts)


def _neural_log_has_positive_on_gt_pods(
    fault_df: pd.DataFrame, pred_positive: np.ndarray, components: set[str]
) -> bool:
    """IF 至少在一条属于根因组件的 pod 上判为异常。"""
    if not components:
        return False
    flags = pred_positive.astype(bool)
    for j, (_, row) in enumerate(fault_df.iterrows()):
        if j >= len(flags) or not flags[j]:
            continue
        pod = str(row.get("k8_pod") or "unknown")
        if eva._component_matches(pod, components):
            return True
    return False


def _score_neural_log_recall(
    fault_df: pd.DataFrame,
    pred_positive: np.ndarray,
    components: set[str],
    patterns: list[list[str]],
) -> tuple[int, int]:
    """
    NeuralLog 专用：关键词必须在根因组件 pod 的故障窗全文上出现（组件语义）；
    且 IF 须在至少一条该组件 pod 上有阳性（避免未触发模型也给分）。
    """
    total = len(patterns)
    if total == 0:
        return 0, 0
    if not components:
        return 0, total
    blob_gt = _merged_text_fault_gt_pods(fault_df, components)
    if not blob_gt.strip():
        return 0, total
    if not _neural_log_has_positive_on_gt_pods(fault_df, pred_positive, components):
        return 0, total
    hits = sum(1 for seq in patterns if eva._keywords_in_order(blob_gt, seq))
    return hits, total


def _match_patterns_neural_log(
    fault_df: pd.DataFrame,
    pred_positive: np.ndarray,
    components: set[str],
    patterns: list[list[str]],
) -> list[bool]:
    if not patterns:
        return []
    if not components:
        return [False] * len(patterns)
    blob_gt = _merged_text_fault_gt_pods(fault_df, components)
    if not blob_gt.strip():
        return [False] * len(patterns)
    if not _neural_log_has_positive_on_gt_pods(fault_df, pred_positive, components):
        return [False] * len(patterns)
    return [eva._keywords_in_order(blob_gt, seq) for seq in patterns]


def _score_neural_log_recall_cached(
    fault_df: pd.DataFrame,
    components: set[str],
    patterns: list[list[str]],
    has_if_positive_on_gt_pod: bool,
) -> tuple[int, int]:
    """使用 orchestrator 写入的 has_if_positive_on_gt_pod，不重跑 IF。"""
    tot = len(patterns)
    if tot == 0 or not components:
        return 0, tot
    blob_gt = _merged_text_fault_gt_pods(fault_df, components)
    if not blob_gt.strip() or not has_if_positive_on_gt_pod:
        return 0, tot
    hits = sum(1 for seq in patterns if eva._keywords_in_order(blob_gt, seq))
    return hits, tot


def _match_patterns_neural_log_cached(
    fault_df: pd.DataFrame,
    components: set[str],
    patterns: list[list[str]],
    has_if_positive_on_gt_pod: bool,
) -> list[bool]:
    if not patterns:
        return []
    if not components or not has_if_positive_on_gt_pod:
        return [False] * len(patterns)
    blob_gt = _merged_text_fault_gt_pods(fault_df, components)
    if not blob_gt.strip():
        return [False] * len(patterns)
    return [eva._keywords_in_order(blob_gt, seq) for seq in patterns]


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
    parser.add_argument(
        "--log-groundtruth-json",
        type=Path,
        default=DEFAULT_LOG_UNIT_TEST_GT,
        help="仅评测：expected_log_patterns 与 root_cause_components（NeuralLog 计分规则见脚本内说明）；不喂给检测模型",
    )
    parser.add_argument(
        "--include-input-without-gt",
        action="store_true",
        help="保留 input.json 中不在上述 JSON 的 uuid（无标注时期望模式分母为 0）",
    )
    parser.add_argument(
        "--from-precompute",
        action="store_true",
        help="优先读预计算 jsonl，不重跑已缓存的检测；仍加载故障窗 parquet（NeuralLog 计分需要）。"
        "NeuralLog 需 jsonl 含 has_if_positive_on_gt_pod（请用新版 orchestrator 重跑 neural_log）。",
    )
    parser.add_argument(
        "--precompute-dir",
        type=Path,
        default=DEFAULT_PRECOMPUTE_DIR,
        help="orchestrator 输出目录，默认 results/log_precompute",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("exp", "exp.utils.input", "exp.agent.log", "drain3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    gt_by_uuid = load_log_unit_test_ground_truth(args.log_groundtruth_json)
    cases_all = load_input_json_cases(PROJECT_ROOT)
    n_input = len(cases_all)
    if not args.include_input_without_gt:
        cases = [c for c in cases_all if str(c.get("uuid", "") or "").strip() in gt_by_uuid]
        logger.info(
            "含 log_unit_test 标注且存在于 input 的 uuid: %s（input.json 共 %s）",
            len(cases),
            n_input,
        )
    else:
        cases = cases_all
    if args.limit_uuids > 0:
        cases = cases[: args.limit_uuids]
    logger.info("本次实际评测用例数: %s", len(cases))

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    prec_dir = args.precompute_dir
    prec_log_agent: dict[str, dict] = {}
    prec_knn: dict[str, dict] = {}
    prec_dt: dict[str, dict] = {}
    prec_slfn: dict[str, dict] = {}
    prec_nl: dict[str, dict] = {}
    if args.from_precompute:
        prec_log_agent = load_precompute_jsonl(pick_precompute_file(prec_dir, "log_agent.jsonl"))
        prec_knn = load_precompute_jsonl(
            pick_precompute_file(prec_dir, "lightad_knn.jsonl", "knn.jsonl")
        )
        prec_dt = load_precompute_jsonl(pick_precompute_file(prec_dir, "lightad_dt.jsonl", "dt.jsonl"))
        prec_slfn = load_precompute_jsonl(
            pick_precompute_file(prec_dir, "lightad_slfn.jsonl", "slfn.jsonl")
        )
        prec_nl = load_precompute_jsonl(pick_precompute_file(prec_dir, "neural_log.jsonl"))
        logger.info(
            "from-precompute: log_agent=%s knn=%s dt=%s slfn=%s neural_log=%s（目录 %s）",
            len(prec_log_agent),
            len(prec_knn),
            len(prec_dt),
            len(prec_slfn),
            len(prec_nl),
            prec_dir,
        )

    from exp.agent.log import LogAgent

    root = str(args.dataset_root)
    agent = LogAgent(root)

    totals = {
        "log_agent": (0, 0, 0.0),
        "knn": (0, 0, 0.0),
        "dt": (0, 0, 0.0),
        "slfn": (0, 0, 0.0),
        "neural_log": (0, 0, 0.0),
        "union": (0, 0, 0.0),
    }
    used_uuids = 0
    skipped_uuids = 0

    for i, item in enumerate(cases, start=1):
        uuid = str(item.get("uuid", "") or "").strip()
        gt = gt_by_uuid.get(uuid, {})
        patterns: list[list[str]] = list(gt.get("patterns") or [])
        components: set[str] = set(gt.get("components") or [])

        if not item.get("parse_ok", True):
            skipped_uuids += 1
            continue

        try:
            start = eva._parse_iso_utc(str(item["start_time"]))
            end = eva._parse_iso_utc(str(item["end_time"]))
        except (ValueError, TypeError, KeyError):
            skipped_uuids += 1
            continue

        use_la_prec = args.from_precompute and uuid in prec_log_agent
        t0_la = time.perf_counter()
        if use_la_prec:
            la_text = str(prec_log_agent[uuid].get("text") or "")
            h_la, tot_la = _score_keywords_on_text(la_text, patterns)
            logagent_flags = _match_patterns_on_text(la_text, patterns)
        else:
            logagent_flags = _log_agent_pattern_matches(agent, start, end, components, patterns)
            h_la = sum(1 for x in logagent_flags if x)
            tot_la = len(patterns)
        ph, pt, ps = totals["log_agent"]
        totals["log_agent"] = (ph + h_la, pt + tot_la, ps + (time.perf_counter() - t0_la))
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
            for m in ("knn", "dt", "slfn", "neural_log"):
                mh, mt, ms = totals[m]
                totals[m] = (mh, mt + len(patterns), ms)
            uh, ut, us = totals["union"]
            totals["union"] = (uh + sum(1 for x in union_flags if x), ut + len(patterns), us)
            skipped_uuids += 1
            continue

        normal_lines = normal_df["text_line"].astype(str).tolist() if not normal_df.empty else []
        if len(normal_lines) > args.max_normal_lines:
            normal_lines = rng_py.sample(normal_lines, args.max_normal_lines)

        use_knn_prec = args.from_precompute and uuid in prec_knn
        use_dt_prec = args.from_precompute and uuid in prec_dt
        use_slfn_prec = args.from_precompute and uuid in prec_slfn
        nl_prec_ready = (
            args.from_precompute
            and uuid in prec_nl
            and isinstance(prec_nl[uuid], dict)
            and "has_if_positive_on_gt_pod" in prec_nl[uuid]
        )
        needs_sklearn = not (use_knn_prec and use_dt_prec and use_slfn_prec)
        needs_nl_if = not nl_prec_ready

        if len(normal_lines) < 20 and (needs_sklearn or needs_nl_if):
            for m in ("knn", "dt", "slfn", "neural_log"):
                mh, mt, ms = totals[m]
                totals[m] = (mh, mt + len(patterns), ms)
            uh, ut, us = totals["union"]
            totals["union"] = (uh + sum(1 for x in union_flags if x), ut + len(patterns), us)
            skipped_uuids += 1
            continue

        vec = None
        x_train_dense = None
        y_train = None
        x_fault_dense = None
        if needs_sklearn:
            vec = _build_vectorizer(normal_lines)
            x_norm_dense = vec.transform(normal_lines).toarray().astype(np.float32)
            x_syn = _make_synthetic_anomalies(x_norm_dense, rng_np)
            x_train_dense = np.vstack([x_norm_dense, x_syn])
            y_train = np.concatenate(
                [np.zeros(len(x_norm_dense), dtype=np.int32), np.ones(len(x_syn), dtype=np.int32)]
            )
            x_fault_dense = vec.transform(fault_df["text_line"].astype(str)).toarray().astype(np.float32)

        models = {
            "knn": (KNeighborsClassifier(n_neighbors=1, metric="minkowski", n_jobs=-1), prec_knn, use_knn_prec),
            "dt": (DecisionTreeClassifier(random_state=args.seed, class_weight="balanced"), prec_dt, use_dt_prec),
            "slfn": (
                MLPClassifier(hidden_layer_sizes=(25,), max_iter=300, random_state=args.seed),
                prec_slfn,
                use_slfn_prec,
            ),
        }

        for name, (clf, prec_map, use_prec) in models.items():
            if use_prec:
                t0 = time.perf_counter()
                txt = str(prec_map[uuid].get("text") or "")
                h, tot = _score_keywords_on_text(txt, patterns)
                infer_sec = time.perf_counter() - t0
                mh, mt, ms = totals[name]
                totals[name] = (mh + h, mt + tot, ms + infer_sec)
                model_flags = _match_patterns_on_text(txt, patterns)
                union_flags = [a or b for a, b in zip(union_flags, model_flags)]
            else:
                assert vec is not None and x_train_dense is not None and y_train is not None
                assert x_fault_dense is not None
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

        if nl_prec_ready:
            has_if = bool(prec_nl[uuid].get("has_if_positive_on_gt_pod"))
            t0_nl = time.perf_counter()
            h_nl, tot_nl = _score_neural_log_recall_cached(
                fault_df, components, patterns, has_if
            )
            infer_sec_nl = time.perf_counter() - t0_nl
            mh, mt, ms = totals["neural_log"]
            totals["neural_log"] = (mh + h_nl, mt + tot_nl, ms + infer_sec_nl)
            nl_flags = _match_patterns_neural_log_cached(
                fault_df, components, patterns, has_if
            )
            union_flags = [a or b for a, b in zip(union_flags, nl_flags)]
        else:
            nl_model = NeuralLogBaseline(contamination=0.15, random_state=args.seed)
            t0_nl = time.perf_counter()
            try:
                nl_model.fit(normal_lines)
                fault_texts_nl = fault_df["text_line"].astype(str).tolist()
                pos_nl = nl_model.predict(fault_texts_nl).astype(bool)
            except Exception as e:
                logger.warning("[%s/%s] uuid=%s NeuralLog failed: %s", i, len(cases), uuid, e)
                pos_nl = np.zeros(len(fault_df), dtype=bool)
            infer_sec_nl = time.perf_counter() - t0_nl
            h_nl, tot_nl = _score_neural_log_recall(fault_df, pos_nl, components, patterns)
            mh, mt, ms = totals["neural_log"]
            totals["neural_log"] = (mh + h_nl, mt + tot_nl, ms + infer_sec_nl)
            nl_flags = _match_patterns_neural_log(fault_df, pos_nl, components, patterns)
            union_flags = [a or b for a, b in zip(union_flags, nl_flags)]

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
    lines.append(f"- **故障时间窗**: `dataset/input.json`（描述解析），{ds_desc}\n")
    try:
        gt_show = str(args.log_groundtruth_json.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        gt_show = str(args.log_groundtruth_json.resolve())
    lines.append(
        f"- **评测标注（仅计分，不输入模型）**: `{gt_show}` — "
        "`expected_log_patterns` + `root_cause_components`。\n"
    )
    lines.append(f"- **正常窗口**: 每个 uuid 的故障前后各 {args.normal_window_minutes} 分钟（排除故障窗）\n")
    if args.include_input_without_gt:
        lines.append("- **uuid 范围**: 含 input 中无 log_unit_test 标注的 uuid（该部分期望模式数为 0）。\n")
    else:
        lines.append("- **uuid 范围**: 仅统计在 log_unit_test_dataset 中有标注且 input 可解析时间窗的交集。\n")
    lines.append(f"- **有效 uuid 数**: {used_uuids}（样本不足/空窗跳过 {skipped_uuids}）\n")
    lines.append(f"- **Parquet 根目录**: `{args.dataset_root.resolve()}`\n")
    if args.from_precompute:
        try:
            pd_rel = str(args.precompute_dir.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            pd_rel = str(args.precompute_dir.resolve())
        lines.append(
            f"- **预计算模式 (`--from-precompute`)**: 读 `{pd_rel}/*.jsonl`，"
            "LogAgent / KNN / DT / SLFN 在**预计算摘要 text**上做顺序关键词命中（与逐行+组件在线计分不同）；"
            "NeuralLog 使用 jsonl 中的 `has_if_positive_on_gt_pod` + 本脚本故障窗 parquet 做组件内关键词计分。\n"
        )
    lines.append("\n")

    lines.append("## 方法对齐说明\n\n")
    lines.append("- **LogAgent**: 保持原仓库规则（前 30 分钟基线 vs 故障窗）。\n")
    lines.append("- **KNN/DT/SLFN**: 每个 uuid 独立建模，仅用该 uuid 正常窗训练；为复用 LightAD 二分类器，使用列打乱构造伪异常类。\n")
    lines.append(
        "- **NeuralLog**: 与 `orchestrator` 一致，为 TF-IDF + IsolationForest 轻量实现；"
        "在正常窗上拟合，再对故障窗逐行打分。\n"
    )
    lines.append(
        "- **评分（LogAgent / KNN / DT / SLFN）**: 在**判为异常且 pod 匹配根因组件**的合并文本上，"
        "按顺序匹配每条期望模式。\n"
    )
    lines.append(
        "- **评分（NeuralLog 专用）**: 关键词在**故障窗内、根因组件 pod 上的全部日志**中按序匹配；"
        "且要求 IF 在至少一条该组件 pod 上判阳（避免未触发检测也给分）。\n\n"
    )

    lines.append("## 结果\n\n")
    lines.append("| 方法 | 命中/期望模式 | 召回率 | 累计耗时(秒) |\n")
    lines.append("|------|----------------|--------|-------------|\n")
    for key, label in (
        ("log_agent", "LogAgent"),
        ("knn", "KNN (LightAD)"),
        ("dt", "Decision Tree (LightAD)"),
        ("slfn", "SLFN / MLP (LightAD)"),
        ("neural_log", "NeuralLog (TF-IDF + IF)"),
        ("union", "Union (all methods)"),
    ):
        h, t, sec = totals[key]
        r = (h / t) if t else 0.0
        lines.append(f"| {label} | {h}/{t} | {r:.4f} | {sec:.2f} |\n")

    lines.append("\n## 说明\n\n")
    lines.append("- 该对比强调**每个故障独立、邻近时间窗建模**，避免跨 uuid 离线训练信息泄漏。\n")
    lines.append("- 部分小时 parquet 缺失会导致跳过或空窗。\n")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {args.output_md}")
    for key in ("log_agent", "knn", "dt", "slfn", "neural_log"):
        h, t, sec = totals[key]
        r = (h / t) if t else 0.0
        print(f"  {key}: {h}/{t} = {r:.4f} (sec={sec:.2f})")
    h, t, _ = totals["union"]
    r = (h / t) if t else 0.0
    print(f"  union: {h}/{t} = {r:.4f}")


if __name__ == "__main__":
    main()
