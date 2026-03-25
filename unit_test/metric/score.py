# 评分脚本，有两种评分模式
# 1. 获取单个 baseline 的输出 results/result_baseline_<method>.csv 进行评分
# 2. 对多个 baseline 的结果取并集进行评分
# 评分机制：
# 设 data/metric_dataset.json 中的某个故障（uuid）的故障组件集合为 C，故障指标集合为 M
# 该故障（uuid）检测出的异常指标列表为 List[(c0, m0), (c1, m1), ...]
# 如果 mi ∈ M 且 ci ∈ C，则认为指标 mi 检测正确，所有检测正确的指标的集合为 M_c
# 正确率为 M_c / M，这里 M_c 和 M 表示所有故障的正确检测指标数量和总的故障指标数量之和
# 评分结果输出到 results/score.csv 中作为新的一行，字段为：time, method, score
# time 为脚本运行时间（北京时间），格式为 YYYY-MM-DD-HH:MM，method 格式类似 1 或 1+2，score 为正确率，保留两位小数

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "unit_test/metric/data/metric_dataset.json"
RESULT_DIR = PROJECT_ROOT / "unit_test/metric/results"
SCORE_FILE = RESULT_DIR / "score.csv"


BEIJING_TZ = timezone(timedelta(hours=8))


def _load_metric_dataset() -> List[dict]:
	if not DATASET_FILE.exists():
		raise FileNotFoundError(f"metric dataset not found: {DATASET_FILE}")

	with DATASET_FILE.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise ValueError(f"Invalid metric dataset format: {DATASET_FILE}")

	return data


def _parse_method_ids(method: str) -> List[str]:
	parts = [m.strip() for m in method.split("+") if m.strip()]
	if not parts:
		raise ValueError("method is empty")
	return parts


def _load_predictions(method: str) -> pd.DataFrame:
	method_ids = _parse_method_ids(method)
	all_frames: List[pd.DataFrame] = []

	for method_id in method_ids:
		result_file = RESULT_DIR / f"result_baseline_{method_id}.csv"
		if not result_file.exists():
			raise FileNotFoundError(f"baseline result not found: {result_file}")

		frame = pd.read_csv(result_file)
		required_cols = {"uuid", "component", "metric"}
		if not required_cols.issubset(set(frame.columns)):
			raise ValueError(f"Invalid columns in {result_file}, required: {required_cols}")

		all_frames.append(frame[["uuid", "component", "metric"]])

	pred_df = pd.concat(all_frames, ignore_index=True)
	pred_df = pred_df.dropna(subset=["uuid", "component", "metric"])
	pred_df["uuid"] = pred_df["uuid"].astype(str)
	pred_df["component"] = pred_df["component"].astype(str)
	pred_df["metric"] = pred_df["metric"].astype(str)

	return pred_df.drop_duplicates().reset_index(drop=True)


def _group_predictions(pred_df: pd.DataFrame) -> Dict[str, Set[Tuple[str, str]]]:
	pred_map: Dict[str, Set[Tuple[str, str]]] = {}
	for row in pred_df.itertuples(index=False):
		uuid = row.uuid
		component = row.component
		metric = row.metric
		pred_map.setdefault(uuid, set()).add((component, metric))
	return pred_map


def _component_aliases(component: str) -> Set[str]:
	aliases = {component}

	if component.startswith("aiops-k8s-"):
		return aliases

	if re.match(r".+-\d+$", component):
		aliases.add(component.rsplit("-", 1)[0])

	return aliases


def _component_matches(pred_component: str, expected_components: Set[str]) -> bool:
	pred_aliases = _component_aliases(pred_component)
	expected_aliases: Set[str] = set()
	for expected in expected_components:
		expected_aliases.update(_component_aliases(expected))

	return len(pred_aliases & expected_aliases) > 0


def evaluate(method: str, limit: int | None = None) -> float:
	dataset = _load_metric_dataset()
	if limit is not None:
		if limit <= 0:
			raise ValueError("limit must be a positive integer")
		dataset = dataset[:limit]

	pred_df = _load_predictions(method)
	pred_map = _group_predictions(pred_df)

	correct_metric_count = 0
	total_metric_count = 0

	for item in dataset:
		uuid = str(item.get("uuid", "")).strip()
		components = set(str(c) for c in item.get("root_cause_components", []) if c)
		metrics = set(str(m) for m in item.get("expected_anomalies", []) if m)

		if not uuid:
			continue

		total_metric_count += len(metrics)
		if not metrics or not components:
			continue

		detected_pairs = pred_map.get(uuid, set())
		correct_metrics_for_uuid: Set[str] = set()
		for component, metric in detected_pairs:
			if _component_matches(component, components) and metric in metrics:
				correct_metrics_for_uuid.add(metric)

		correct_metric_count += len(correct_metrics_for_uuid)

	if total_metric_count == 0:
		return 0.0

	return correct_metric_count / total_metric_count


def append_score(method: str, score: float, limit: int | None) -> None:
	RESULT_DIR.mkdir(parents=True, exist_ok=True)

	now = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d-%H:%M")
	limit_text = str(limit) if limit is not None else "all"
	row = pd.DataFrame(
		[{
			"time": now,
			"method": method,
			"top_n": limit_text,
			"score": f"{score:.4f}",
		}]
	)

	if SCORE_FILE.exists():
		try:
			existing = pd.read_csv(SCORE_FILE)
		except EmptyDataError:
			existing = pd.DataFrame(columns=["time", "method", "top_n", "score"])
		for col in ["time", "method", "top_n", "score"]:
			if col not in existing.columns:
				existing[col] = ""
		existing = existing[["time", "method", "top_n", "score"]]
		merged = pd.concat([existing, row], ignore_index=True)
		merged.to_csv(SCORE_FILE, index=False)
	else:
		row.to_csv(SCORE_FILE, index=False)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--method",
		type=str,
		required=True,
		help="Scoring method id(s), e.g. 1 or 1+2",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Only evaluate the first n test cases",
	)
	args = parser.parse_args()

	score = evaluate(args.method, args.limit)
	append_score(args.method, score, args.limit)
	print(f"method={args.method}, top_n={args.limit if args.limit is not None else 'all'}, score={score:.4f}")


if __name__ == "__main__":
	main()
