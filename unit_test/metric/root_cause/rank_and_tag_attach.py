# 读取 results/result_baseline<2+4+5>.csv
# 获取 Baseline 2,4,5 检测出的所有异常指标
# Step 1:
# 调用 root_cause/baro/ 目录下的根因定位算法
# 对每个故障（uuid）的所有异常指标进行排序，排序越靠前越可能是根因指标
# Step 2:
# 给异常指标添加一个局部异常模式，输出标记后的序列
# 采用 exp/agent/metric.py 中的 _detect_local_pattern 算法
# 输出一个 csv 文件，记录每个故障（uuid）排序后的异常指标列表
# 格式为 uuid, component, metric, pattern
# 改进：增加 score 列，过滤 normal，按 uuid 截断 top_k，降低噪声输入

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT_CAUSE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "unit_test/metric/results"
INPUT_FILE = PROJECT_ROOT / "dataset/input.json"
OUTPUT_FILE = RESULT_DIR / "ranked_anomaly_with_pattern.csv"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT_CAUSE_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_CAUSE_DIR))

from exp.agent.metric import MetricAgent
from exp.utils.time import parse_time_range
from baro.root_cause_analysis import robust_scorer


def _parse_iso_utc(time_str: str) -> datetime:
	dt = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
	if dt.tzinfo:
		dt = dt.replace(tzinfo=None)
	return dt


def _load_input_cases() -> list[dict]:
	if not INPUT_FILE.exists():
		raise FileNotFoundError(f"input file not found: {INPUT_FILE}")
	with INPUT_FILE.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("dataset/input.json must be a list")

	cases: list[dict] = []
	for item in data:
		uuid = str(item.get("uuid", "")).strip()
		desc = str(item.get("Anomaly Description", "")).strip()
		if not uuid or not desc:
			continue
		start_time, end_time = parse_time_range(desc)
		if not start_time or not end_time:
			continue
		cases.append({
			"uuid": uuid,
			"start_time": start_time,
			"end_time": end_time,
		})

	return cases


def _load_union_anomalies() -> pd.DataFrame:
	files = [
		("b2", RESULT_DIR / "result_baseline2.csv"),
		("b4", RESULT_DIR / "result_baseline4.csv"),
		("b5", RESULT_DIR / "result_baseline5.csv"),
	]
	frames = []
	for baseline, file in files:
		if not file.exists():
			raise FileNotFoundError(f"baseline result not found: {file}")
		frame = pd.read_csv(file)
		frame["baseline"] = baseline
		frames.append(frame)
	df = pd.concat(frames, ignore_index=True)

	required_cols = {"uuid", "component", "metric"}
	if not required_cols.issubset(df.columns):
		raise ValueError(f"input columns must include {required_cols}")

	df = df[["uuid", "component", "metric", "baseline"]].dropna()
	df["uuid"] = df["uuid"].astype(str)
	df["component"] = df["component"].astype(str)
	df["metric"] = df["metric"].astype(str)

	# 投票：同一 (uuid, component, metric) 被几个 baseline 命中
	votes = (
		df.drop_duplicates(["uuid", "component", "metric", "baseline"])
		.groupby(["uuid", "component", "metric"], as_index=False)["baseline"]
		.nunique()
		.rename(columns={"baseline": "votes"})
	)
	return votes


def _normalize_component(component: str) -> str:
	if not isinstance(component, str):
		return "unknown"
	if component.startswith("aiops-k8s-") or component.startswith("k8s-master"):
		return component
	parts = component.rsplit("-", 1)
	if len(parts) == 2 and parts[1].isdigit():
		return parts[0]
	return component


def _pattern_weight(pattern: str) -> float:
	weights = {
		"spike": 1.0,
		"surge": 0.95,
		"level_shift_up": 1.0,
		"level_shift_down": 0.9,
		"drop": 0.8,
		"dip": 0.75,
		"normal": 0.1,
	}
	return weights.get(str(pattern), 0.8)


def _detect_local_pattern(series: pd.Series, anomaly_indices: np.ndarray) -> str:
	if not np.any(anomaly_indices):
		return "normal"

	mean_val = series.mean()
	anom_indices = np.where(anomaly_indices)[0]
	anom_values = series.iloc[anom_indices]
	anom_mean = anom_values.mean()
	is_high = anom_mean > mean_val

	duration = len(anom_indices)
	last_idx = len(series) - 1

	if is_high:
		if duration == 1:
			return "spike"
		if anom_indices[-1] == last_idx:
			return "level_shift_up"
		return "surge"

	if duration == 1:
		return "drop"
	if anom_indices[-1] == last_idx:
		return "level_shift_down"
	return "dip"


def _longest_true_run(mask: np.ndarray) -> int:
	max_run = 0
	run = 0
	for v in mask:
		if bool(v):
			run += 1
			if run > max_run:
				max_run = run
		else:
			run = 0
	return max_run


def _pattern_from_series(series: pd.Series, fault_start: datetime) -> tuple[str, float, float, int]:
	if series.empty or len(series) < 5:
		return "normal", 0.0, 0.0, 0

	pre = series[series.index < fault_start]
	post = series[series.index >= fault_start]
	if pre.empty or post.empty:
		return "normal", 0.0, 0.0, 0

	base_mean = float(pre.mean())
	base_std = float(pre.std())

	if base_std > 1e-8:
		post_mask = ((post - base_mean).abs() > 3.0 * base_std).to_numpy()
	else:
		post_mask = ((post - base_mean).abs() > 0.05).to_numpy()

	if not np.any(post_mask):
		return "normal", 0.0, 0.0, 0

	duration_ratio = float(np.mean(post_mask))
	run_length = int(_longest_true_run(post_mask))
	first_idx = int(np.where(post_mask)[0][0])
	first_time = post.index[first_idx]
	if isinstance(first_time, pd.Timestamp):
		first_time = first_time.to_pydatetime()
	delta_minutes = max(0.0, float((first_time - fault_start).total_seconds() / 60.0))
	onset_score = float(np.exp(-delta_minutes / 5.0))

	full_mask = np.zeros(len(series), dtype=bool)
	start_idx = len(pre)
	full_mask[start_idx:start_idx + len(post_mask)] = post_mask
	pattern = _detect_local_pattern(series, full_mask)

	# 对于持续性很弱的非瞬时模式做降级，减少伪 shift/surge
	if pattern in {"surge", "level_shift_up", "level_shift_down", "dip"} and run_length < 2 and duration_ratio < 0.08:
		pattern = "spike" if pattern in {"surge", "level_shift_up"} else "drop"

	return pattern, onset_score, duration_ratio, run_length


def _build_metric_dict_for_candidates(
	df: pd.DataFrame,
	candidates: list[tuple[str, str]],
) -> tuple[dict[str, list[list[float]]], dict[str, pd.Series]]:
	metric_dict: dict[str, list[list[float]]] = {}
	series_dict: dict[str, pd.Series] = {}

	for component, metric in candidates:
		sub = df[(df["pod"] == component) & (df["kpi_key"] == metric)][["time", "value"]].copy()
		if sub.empty:
			continue

		sub = sub.sort_values("time")
		key = f"{component}::{metric}"
		metric_dict[key] = [[float(t.timestamp()), float(v)] for t, v in zip(sub["time"], sub["value"])]
		series_dict[key] = pd.Series(sub["value"].values, index=pd.to_datetime(sub["time"]))

	return metric_dict, series_dict


def run(
	limit: int | None = None,
	uuid: str | None = None,
	top_k: int = 30,
	drop_normal: bool = True,
	min_votes: int = 2,
	max_per_component: int = 4,
) -> None:
	dataset = _load_input_cases()
	if uuid:
		dataset = [item for item in dataset if str(item.get("uuid", "")).strip() == uuid]
	if limit is not None:
		dataset = dataset[:limit]

	uuid_order = {
		str(item.get("uuid", "")).strip(): idx
		for idx, item in enumerate(dataset)
		if str(item.get("uuid", "")).strip()
	}

	anomalies = _load_union_anomalies()
	anomalies_by_uuid: dict[str, list[tuple[str, str, int]]] = {}
	for row in anomalies.itertuples(index=False):
		if int(row.votes) < max(1, int(min_votes)):
			continue
		anomalies_by_uuid.setdefault(row.uuid, []).append((row.component, row.metric, int(row.votes)))

	metric_agent = MetricAgent(str(PROJECT_ROOT / "dataset"))
	output_rows: list[dict] = []

	for item in dataset:
		case_uuid = str(item.get("uuid", "")).strip()
		if not case_uuid:
			continue

		candidates_raw = anomalies_by_uuid.get(case_uuid, [])
		candidates = list(dict.fromkeys(candidates_raw))
		if not candidates:
			continue

		candidate_votes = {
			(component, metric): votes
			for component, metric, votes in candidates
		}

		start_time = item.get("start_time")
		end_time = item.get("end_time")
		if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
			continue
		load_start = start_time - timedelta(minutes=30)

		try:
			metric_df = metric_agent.load_data(load_start, end_time)
		except Exception as exc:
			logger.warning("load_data failed for %s: %s", case_uuid, exc)
			continue

		if metric_df.empty:
			continue

		pair_candidates = [(component, metric) for component, metric, _ in candidates]
		metric_dict, series_dict = _build_metric_dict_for_candidates(metric_df, pair_candidates)
		if not metric_dict:
			continue

		try:
			ranking = robust_scorer(metric_dict, inject_time=int(start_time.timestamp()))
			ranks = ranking.get("ranks", [])
			score_map = {k: float(v) for k, v in ranking.get("scores", [])}
		except Exception as exc:
			logger.warning("robust_scorer failed for %s: %s", case_uuid, exc)
			ranks = []
			score_map = {}

		ranked_keys = [k for k in ranks if k in series_dict]
		existed = set(ranked_keys)
		ranked_keys.extend([k for k in series_dict.keys() if k not in existed])

		for key in ranked_keys:
			component, metric = key.split("::", 1)
			component_group = _normalize_component(component)

			pattern, onset_score, duration_ratio, run_length = _pattern_from_series(series_dict[key], start_time)
			raw_score = float(score_map.get(key, 0.0))
			votes = int(candidate_votes.get((component, metric), 1))
			vote_weight = 1.0 + 0.35 * max(0, votes - 1)
			# 使用 log1p 压缩量级，避免个别超大值淹没全部先验
			score_base = float(np.log1p(max(0.0, raw_score)))
			weighted_score = score_base * _pattern_weight(pattern) * vote_weight

			# 时间先发分 + 持续性分：越早出现、持续越稳定，分越高
			temporal_weight = (0.75 + 0.5 * onset_score) * (0.75 + 0.5 * min(1.0, duration_ratio / 0.2))
			if pattern in {"surge", "level_shift_up", "level_shift_down", "dip", "drop"}:
				temporal_weight *= (0.8 + 0.2 * min(1.0, run_length / 3.0))

			final_score = weighted_score * temporal_weight
			if drop_normal and pattern == "normal":
				continue
			# 极晚出现且非常短促的单点峰值，视为低价值噪声
			if pattern == "spike" and onset_score < 0.2 and duration_ratio < 0.02:
				continue
			output_rows.append(
				{
					"uuid": case_uuid,
					"component": component,
					"component_group": component_group,
					"metric": metric,
					"pattern": pattern,
					"votes": votes,
					"raw_score": raw_score,
					"score": weighted_score,
					"onset_score": onset_score,
					"duration_ratio": duration_ratio,
					"run_length": run_length,
					"temporal_weight": temporal_weight,
					"final_score": final_score,
				}
			)

	result_df = pd.DataFrame(
		output_rows,
		columns=["uuid", "component", "component_group", "metric", "pattern", "votes", "raw_score", "score", "onset_score", "duration_ratio", "run_length", "temporal_weight", "final_score"],
	)
	if not result_df.empty:
		result_df = result_df.drop_duplicates()
		result_df["score"] = pd.to_numeric(result_df["score"], errors="coerce").fillna(0.0)
		result_df["raw_score"] = pd.to_numeric(result_df["raw_score"], errors="coerce").fillna(0.0)
		result_df["onset_score"] = pd.to_numeric(result_df["onset_score"], errors="coerce").fillna(0.0)
		result_df["duration_ratio"] = pd.to_numeric(result_df["duration_ratio"], errors="coerce").fillna(0.0)
		result_df["run_length"] = pd.to_numeric(result_df["run_length"], errors="coerce").fillna(0).astype(int)
		result_df["temporal_weight"] = pd.to_numeric(result_df["temporal_weight"], errors="coerce").fillna(1.0)
		result_df["final_score"] = pd.to_numeric(result_df["final_score"], errors="coerce").fillna(0.0)
		result_df["votes"] = pd.to_numeric(result_df["votes"], errors="coerce").fillna(1).astype(int)

		# 组件聚合分：同组件组取前3个指标分数求和，优先高一致性的组件
		group_score = (
			result_df.sort_values(["uuid", "component_group", "final_score"], ascending=[True, True, False])
			.groupby(["uuid", "component_group"], as_index=False)
			.head(3)
			.groupby(["uuid", "component_group"], as_index=False)["final_score"]
			.sum()
			.rename(columns={"final_score": "component_score"})
		)
		result_df = result_df.merge(group_score, on=["uuid", "component_group"], how="left")

		result_df["_order"] = result_df["uuid"].map(uuid_order).fillna(len(uuid_order))
		result_df = result_df.sort_values(
			["_order", "component_score", "final_score", "score", "raw_score"],
			ascending=[True, False, False, False, False],
		)

		# 限制每个组件最多保留若干条，避免单组件淹没整体线索
		result_df["_comp_rank"] = result_df.groupby(["uuid", "component_group"]).cumcount()
		result_df = result_df[result_df["_comp_rank"] < max(1, int(max_per_component))]

		result_df = result_df.groupby("uuid", as_index=False).head(max(1, int(top_k)))
		result_df = result_df.drop(columns=["_order", "_comp_rank"]).reset_index(drop=True)

	RESULT_DIR.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(OUTPUT_FILE, index=False)
	logger.info("saved %d rows to %s", len(result_df), OUTPUT_FILE)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--limit", type=int, default=None, help="Only process first n cases")
	parser.add_argument("--uuid", type=str, default=None, help="Only process one uuid")
	parser.add_argument("--top_k", type=int, default=30, help="Keep top-k metrics per uuid")
	parser.add_argument("--keep_normal", action="store_true", help="Keep rows with normal pattern")
	parser.add_argument("--min_votes", type=int, default=2, help="Require at least N baseline votes")
	parser.add_argument("--max_per_component", type=int, default=4, help="Max rows per component group")
	args = parser.parse_args()
	run(
		limit=args.limit,
		uuid=args.uuid,
		top_k=args.top_k,
		drop_normal=not args.keep_normal,
		min_votes=args.min_votes,
		max_per_component=args.max_per_component,
	)


if __name__ == "__main__":
	main()

# python3 unit_test/metric/root_cause/_k=30 --min_votes=2 --max_per_component=4rank_and_tag_attach.py --top_k=30 --min_votes=2 --max_per_component=4