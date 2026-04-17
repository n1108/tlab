"""
LogAgent / Baseline 评分脚本（类似 unit_test/metric/score.py）。

- 仅使用 dataset/input.json（时间窗 + 描述）
- 对每个 uuid 调用 LogAgent.score，返回 anomaly list
- 输出 per-uuid 统计（检测到的 component 数、anomaly 数），无 groundtruth 时不计算 pattern_score
- 供后续 judge 或手动分析使用
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
from unit_test.log.input_cases import load_input_json_cases

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


def evaluate(
    dataset_root: Path,
    limit: int | None = None,
    *,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[None, int, int, list[dict]]:
    cases = load_input_json_cases(PROJECT_ROOT)
    if limit is not None and limit > 0:
        cases = cases[:limit]

    n_cases = len(cases)
    logger.info(
        "用例数: %s（仅 dataset/input.json；无标注模式分，dataset-root=%s）",
        n_cases,
        dataset_root,
    )

    agent = LogAgent(str(dataset_root))
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
        if use_bar and isinstance(iterator, tqdm):
            iterator.set_postfix_str(uuid[:20], refresh=False)

        if not item.get("parse_ok", True):
            logger.warning("[%s/%s] uuid=%s 无法从描述解析时间窗", idx, n_cases, uuid)
            per_uuid.append(
                {
                    "uuid": uuid,
                    "error": "time range not parsed from Anomaly Description",
                    "anomaly_count": 0,
                    "components_detected": [],
                }
            )
            continue

        start_s = item.get("start_time")
        end_s = item.get("end_time")

        if verbose:
            logger.info("[%s/%s] uuid=%s | %s .. %s", idx, n_cases, uuid, start_s, end_s)
        elif not use_bar:
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
                    "anomaly_count": 0,
                    "components_detected": [],
                }
            )
            continue

        try:
            anomalies = agent.score(start, end)
        except Exception as e:
            logger.exception("[%s/%s] uuid=%s LogAgent.score 异常", idx, n_cases, uuid)
            per_uuid.append(
                {
                    "uuid": uuid,
                    "error": str(e),
                    "anomaly_count": 0,
                    "components_detected": [],
                }
            )
            continue

        n_anom = len(anomalies) if isinstance(anomalies, list) else 0
        comps: set[str] = set()
        if isinstance(anomalies, list):
            for block in anomalies:
                c = str(block.get("component", "") or "").strip()
                if c:
                    comps.add(c)

        if verbose:
            logger.info(
                "[%s/%s] uuid=%s -> %s 条异常, components=%s",
                idx,
                n_cases,
                uuid,
                n_anom,
                sorted(comps),
            )

        per_uuid.append(
            {
                "uuid": uuid,
                "anomaly_count": n_anom,
                "components_detected": sorted(comps),
            }
        )

    # 无标注：返回结构化评分结果（类似 metric score.py）
    return {
        "mode": "input-only-score",
        "total_cases": n_cases,
        "successful_cases": len([p for p in per_uuid if p.get("anomaly_count", 0) > 0]),
        "total_anomalies_detected": sum(p.get("anomaly_count", 0) for p in per_uuid),
        "per_uuid": per_uuid,
        "note": "No ground-truth patterns. Use orchestrator.py for full baseline comparison + summary for JudgeAgent.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score LogAgent/baselines on dataset/input.json (no labels) - similar to metric/score.py"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Parquet root (consistent with exp/main.py)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "log_score.json",
        help="Output JSON with per-uuid stats",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-case details",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not args.verbose and not args.no_progress:
        for name in ("exp", "exp.utils.input", "exp.agent.log", "drain3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    result = score(
        args.dataset_root,
        limit=args.limit,
        verbose=args.verbose,
        show_progress=not args.no_progress,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Score completed: {result.get('successful_cases', 0)}/{result.get('total_cases', 0)} cases")
    print(f"Total anomalies detected: {result.get('total_anomalies_detected', 0)}")
    print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
