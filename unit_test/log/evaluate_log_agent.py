"""
调用 exp.agent.log.LogAgent.score，对 log_unit_test_dataset.json 中的期望 log 模式进行评分。

评分规则（与 metric 单测思路一致：根因组件 + 信号命中）：
- 对每个 uuid 的每条 expected_log_patterns 中的关键词序列，若存在某个检测到的异常项，
  其 component 与 groundtruth 中的 root_cause_components 匹配（含 pod 名变体），且
  该异常相关的文本（observation、各 anomalous_patterns 的 template 与 sample）拼接后，
  按顺序包含全部关键词（大小写不敏感），则计为命中。
- 总体分数 = 命中模式数 / 期望模式总数（仅统计含期望模式的用例）。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.agent.log import LogAgent

DATASET_FILE = PROJECT_ROOT / "unit_test/log/log_unit_test_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "unit_test/log/results"


def _parse_iso_utc(time_str: str) -> datetime:
    if not time_str:
        raise ValueError("empty time string")
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt


def _component_aliases(component: str) -> set[str]:
    aliases = {component}
    if component.startswith("aiops-k8s-"):
        return aliases
    if re.match(r".+-\d+$", component):
        aliases.add(component.rsplit("-", 1)[0])
    return aliases


def _component_matches(pred_component: str, expected_components: set[str]) -> bool:
    pred_aliases = _component_aliases(pred_component)
    expected_aliases: set[str] = set()
    for expected in expected_components:
        expected_aliases.update(_component_aliases(expected))
    return len(pred_aliases & expected_aliases) > 0


def _anomaly_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    obs = block.get("observation")
    if obs:
        parts.append(str(obs))
    for ap in block.get("anomalous_patterns") or []:
        if isinstance(ap, dict):
            for k in ("template", "sample", "type"):
                v = ap.get(k)
                if v:
                    parts.append(str(v))
    return " ".join(parts)


def _keywords_in_order(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    pos = 0
    for kw in keywords:
        k = kw.lower()
        i = lower.find(k, pos)
        if i == -1:
            return False
        pos = i + len(k)
    return True


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"log test dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON array")
    return data


def evaluate(
    dataset_root: Path,
    limit: int | None = None,
    *,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[float, int, int, list[dict]]:
    cases = _load_cases(DATASET_FILE)
    if limit is not None and limit > 0:
        cases = cases[:limit]

    n_cases = len(cases)
    logger.info("评测用例数: %s（dataset-root=%s）", n_cases, dataset_root)

    agent = LogAgent(str(dataset_root))
    total_patterns = 0
    hit_patterns = 0
    per_uuid: list[dict] = []

    use_bar = show_progress and tqdm is not None
    case_iter = enumerate(cases, start=1)
    if tqdm is not None:
        iterator = tqdm(
            case_iter,
            total=n_cases,
            desc="LogAgent eval",
            unit="case",
            disable=not use_bar,
        )
    else:
        iterator = case_iter

    for idx, item in iterator:
        uuid = str(item.get("uuid", ""))
        patterns: list[list[str]] = item.get("expected_log_patterns") or []
        components = set(str(c) for c in item.get("root_cause_components", []) if c)
        if not patterns:
            continue

        start_s = item.get("start_time")
        end_s = item.get("end_time")

        if use_bar and isinstance(iterator, tqdm):
            iterator.set_postfix_str(uuid[:20], refresh=False)

        if verbose:
            logger.info(
                "[%s/%s] uuid=%s | %s .. %s | patterns=%s | components=%s",
                idx,
                n_cases,
                uuid,
                start_s,
                end_s,
                len(patterns),
                sorted(components),
            )
        elif not use_bar:
            # 无 tqdm 时每条用例打一行，避免长时间无输出
            print(f"[{idx}/{n_cases}] {uuid}", flush=True)

        try:
            start = _parse_iso_utc(str(start_s))
            end = _parse_iso_utc(str(end_s))
        except (ValueError, TypeError) as e:
            logger.warning("[%s/%s] uuid=%s 时间解析失败: %s", idx, n_cases, uuid, e)
            per_uuid.append(
                {
                    "uuid": uuid,
                    "error": f"time parse: {e}",
                    "expected_hits": 0,
                    "expected_total": len(patterns),
                }
            )
            total_patterns += len(patterns)
            continue

        try:
            anomalies = agent.score(start, end)
        except Exception as e:
            logger.exception("[%s/%s] uuid=%s LogAgent.score 异常", idx, n_cases, uuid)
            per_uuid.append(
                {
                    "uuid": uuid,
                    "error": str(e),
                    "expected_hits": 0,
                    "expected_total": len(patterns),
                }
            )
            total_patterns += len(patterns)
            continue

        n_anom = len(anomalies) if isinstance(anomalies, list) else 0
        if verbose:
            logger.info(
                "[%s/%s] uuid=%s -> LogAgent 返回 %s 条异常组件",
                idx,
                n_cases,
                uuid,
                n_anom,
            )

        uuid_hits = 0
        pattern_results: list[dict] = []
        for seq in patterns:
            total_patterns += 1
            matched = False
            for block in anomalies:
                comp = str(block.get("component", ""))
                if not comp or not components:
                    continue
                if not _component_matches(comp, components):
                    continue
                blob = _anomaly_text(block)
                if _keywords_in_order(blob, seq):
                    matched = True
                    break
            if matched:
                hit_patterns += 1
                uuid_hits += 1
            pattern_results.append({"keywords": seq, "matched": matched})

        if verbose:
            logger.info(
                "[%s/%s] uuid=%s -> 模式命中 %s/%s",
                idx,
                n_cases,
                uuid,
                uuid_hits,
                len(patterns),
            )

        per_uuid.append(
            {
                "uuid": uuid,
                "expected_hits": uuid_hits,
                "expected_total": len(patterns),
                "pattern_results": pattern_results,
            }
        )

    score = (hit_patterns / total_patterns) if total_patterns else 0.0
    return score, hit_patterns, total_patterns, per_uuid


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LogAgent against log_unit_test_dataset.json")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Parquet 根目录（与 exp/main.py 中 LogAgent 一致）",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 n 条用例")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "log_agent_evaluation.json",
        help="写入逐 uuid 明细",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="打印每条用例的窗口、组件、LogAgent 返回条数与模式命中数（需配合日志级别）",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条；未装 tqdm 时默认逐行打印 [i/n] uuid",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（--verbose 时建议 INFO 或 DEBUG）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # 进度条与 exp 里逐小时 INFO 混在一起难以阅读；非 verbose 时压低子模块日志
    if not args.verbose and not args.no_progress:
        for name in ("exp", "exp.utils.input", "exp.agent.log", "drain3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    score, hits, total, per_uuid = evaluate(
        args.dataset_root,
        limit=args.limit,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "score": score,
        "hit_patterns": hits,
        "total_expected_patterns": total,
        "dataset_root": str(args.dataset_root.resolve()),
        "per_uuid": per_uuid,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"LogAgent evaluation: score={score:.4f} ({hits}/{total} patterns hit), "
        f"written {args.output_json}"
    )


if __name__ == "__main__":
    main()
