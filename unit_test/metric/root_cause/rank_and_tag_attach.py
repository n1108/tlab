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
DATASET_FILE = PROJECT_ROOT / "unit_test/metric/data/metric_dataset.json"
OUTPUT_FILE = RESULT_DIR / "ranked_anomaly_with_pattern.csv"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT_CAUSE_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_CAUSE_DIR))

from exp.agent.metric import MetricAgent
from baro.root_cause_analysis import robust_scorer


def _parse_iso_utc(time_str: str) -> datetime:
	dt = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
	if dt.tzinfo:
		dt = dt.replace(tzinfo=None)
	return dt


def _load_metric_dataset() -> list[dict]:
	if not DATASET_FILE.exists():
		raise FileNotFoundError(f"dataset file not found: {DATASET_FILE}")
	with DATASET_FILE.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("metric_dataset.json must be a list")
	return data


def _load_union_anomalies() -> pd.DataFrame:
	combined_file = RESULT_DIR / "result_baseline2+4+5.csv"
	if combined_file.exists():
		df = pd.read_csv(combined_file)
	else:
		files = [
			RESULT_DIR / "result_baseline2.csv",
			RESULT_DIR / "result_baseline4.csv",
			RESULT_DIR / "result_baseline5.csv",
		]
		frames = []
		for file in files:
			if not file.exists():
				raise FileNotFoundError(f"baseline result not found: {file}")
			frames.append(pd.read_csv(file))
		df = pd.concat(frames, ignore_index=True)

	required_cols = {"uuid", "component", "metric"}
	if not required_cols.issubset(df.columns):
		raise ValueError(f"input columns must include {required_cols}")

	df = df[["uuid", "component", "metric"]].dropna()
	df["uuid"] = df["uuid"].astype(str)
	df["component"] = df["component"].astype(str)
	df["metric"] = df["metric"].astype(str)
	return df.drop_duplicates().reset_index(drop=True)


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


def _pattern_from_series(series: pd.Series, fault_start: datetime) -> str:
	if series.empty or len(series) < 5:
		return "normal"

	pre = series[series.index < fault_start]
	post = series[series.index >= fault_start]
	if pre.empty or post.empty:
		return "normal"

	base_mean = float(pre.mean())
	base_std = float(pre.std())

	if base_std > 1e-8:
		post_mask = ((post - base_mean).abs() > 3.0 * base_std).to_numpy()
	else:
		post_mask = ((post - base_mean).abs() > 0.05).to_numpy()

	if not np.any(post_mask):
		return "normal"

	full_mask = np.zeros(len(series), dtype=bool)
	start_idx = len(pre)
	full_mask[start_idx:start_idx + len(post_mask)] = post_mask
	return _detect_local_pattern(series, full_mask)


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


def run(limit: int | None = None, uuid: str | None = None) -> None:
	dataset = _load_metric_dataset()
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
	anomalies_by_uuid: dict[str, list[tuple[str, str]]] = {}
	for row in anomalies.itertuples(index=False):
		anomalies_by_uuid.setdefault(row.uuid, []).append((row.component, row.metric))

	metric_agent = MetricAgent(str(PROJECT_ROOT / "dataset"))
	output_rows: list[dict] = []

	for item in dataset:
		case_uuid = str(item.get("uuid", "")).strip()
		if not case_uuid:
			continue

		candidates = list(dict.fromkeys(anomalies_by_uuid.get(case_uuid, [])))
		if not candidates:
			continue

		start_time = _parse_iso_utc(str(item.get("start_time")))
		end_time = _parse_iso_utc(str(item.get("end_time")))
		load_start = start_time - timedelta(minutes=30)

		try:
			metric_df = metric_agent.load_data(load_start, end_time)
		except Exception as exc:
			logger.warning("load_data failed for %s: %s", case_uuid, exc)
			continue

		if metric_df.empty:
			continue

		metric_dict, series_dict = _build_metric_dict_for_candidates(metric_df, candidates)
		if not metric_dict:
			continue

		try:
			ranking = robust_scorer(metric_dict, inject_time=int(start_time.timestamp()))
			ranks = ranking.get("ranks", [])
		except Exception as exc:
			logger.warning("robust_scorer failed for %s: %s", case_uuid, exc)
			ranks = []

		ranked_keys = [k for k in ranks if k in series_dict]
		existed = set(ranked_keys)
		ranked_keys.extend([k for k in series_dict.keys() if k not in existed])

		for key in ranked_keys:
			component, metric = key.split("::", 1)
			pattern = _pattern_from_series(series_dict[key], start_time)
			output_rows.append(
				{
					"uuid": case_uuid,
					"component": component,
					"metric": metric,
					"pattern": pattern,
				}
			)

	result_df = pd.DataFrame(output_rows, columns=["uuid", "component", "metric", "pattern"])
	if not result_df.empty:
		result_df = result_df.drop_duplicates()
		result_df["_order"] = result_df["uuid"].map(uuid_order).fillna(len(uuid_order))
		result_df = result_df.sort_values(["_order"]).drop(columns=["_order"]).reset_index(drop=True)

	RESULT_DIR.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(OUTPUT_FILE, index=False)
	logger.info("saved %d rows to %s", len(result_df), OUTPUT_FILE)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--limit", type=int, default=None, help="Only process first n cases")
	parser.add_argument("--uuid", type=str, default=None, help="Only process one uuid")
	args = parser.parse_args()
	run(limit=args.limit, uuid=args.uuid)


if __name__ == "__main__":
	main()