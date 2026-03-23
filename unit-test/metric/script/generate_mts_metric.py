# 读取 dataset 文件夹下的 metric 数据
# 在 unit-test/metric/time-series-data 文件夹下生成 csv 文件
# dataset 文件夹下每个日期的 metric 对应一个文件
# 文件格式为：第一列为 time，其他列为 service_metric 格式的指标列
# unit-test/metric/time-series-data/simple_data.csv 文件是一个例子

import re
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = WORKSPACE_ROOT / "dataset"
OUTPUT_DIR = WORKSPACE_ROOT / "unit-test/metric/time-series-data"
LEVELS = ("service", "pod", "node")


def _is_date_dir(path: Path) -> bool:
	return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name))


def _normalize_service_name(value: str) -> str:
	if value is None:
		return ""

	service = str(value).strip().replace(" (deleted)", "")
	return service


def _normalize_entity_name(value: str, level: str) -> str:
	name = _normalize_service_name(value)
	if level == "service":
		name = re.sub(r"-\d+$", "", name)
	return name


def _fallback_entity_from_file(file_path: Path) -> str:
	name = file_path.name
	if name.startswith("infra_pd_"):
		return "pd"
	if name.startswith("infra_tikv_"):
		return "tikv"
	if name.startswith("infra_tidb_"):
		return "tidb"
	if name.startswith("infra_node_"):
		return "node"
	return "global"


def _build_entity_series(df: pd.DataFrame, entity_col: str | None, file_path: Path, level: str) -> pd.Series:
	fallback_entity = _fallback_entity_from_file(file_path)

	if entity_col is None:
		return pd.Series([fallback_entity] * len(df), index=df.index)

	entity = df[entity_col].astype(str).map(str.strip)
	entity = entity.replace({"": pd.NA, "null": pd.NA, "None": pd.NA, "nan": pd.NA})

	if entity.notna().sum() == 0:
		return pd.Series([fallback_entity] * len(df), index=df.index)

	entity = entity.fillna(fallback_entity)
	return entity.map(lambda x: _normalize_entity_name(x, level))


def _to_epoch_seconds(series: pd.Series) -> pd.Series:
	ts = pd.to_datetime(series, errors="coerce", utc=True)
	return (ts.astype("int64") // 10**9).astype("int64")


def _extract_entity_col(df: pd.DataFrame, level: str) -> str | None:
	if level == "service":
		entity_candidates = ["service", "object_id"]
	elif level == "pod":
		entity_candidates = ["pod", "object_id"]
	else:
		entity_candidates = ["kubernetes_node", "instance", "node", "host", "object_id"]

	for col in entity_candidates:
		if col in df.columns:
			return col
	return None


def _extract_long_metrics_from_file(file_path: Path, level: str) -> pd.DataFrame:
	try:
		df = pd.read_parquet(file_path)
	except Exception:
		return pd.DataFrame(columns=["time", "service", "metric", "value"])

	if "time" not in df.columns:
		return pd.DataFrame(columns=["time", "service", "metric", "value"])

	if "object_type" in df.columns:
		target = level
		df = df[df["object_type"].astype(str).str.lower() == target]
		if df.empty:
			return pd.DataFrame(columns=["time", "service", "metric", "value"])

	entity_col = _extract_entity_col(df, level)
	entity_series = _build_entity_series(df, entity_col, file_path, level)

	if "kpi_key" in df.columns and "value" in df.columns:
		tmp = pd.DataFrame(
			{
				"time": df["time"],
				"service": entity_series,
				"metric": df["kpi_key"],
				"value": pd.to_numeric(df["value"], errors="coerce"),
			}
		)
		return tmp.dropna(subset=["time", "service", "metric", "value"])

	non_metric_cols = {
		"time",
		"timestamp",
		"service",
		"object_id",
		"object_type",
		"pod",
		"instance",
		"kubernetes_node",
		"node",
		"host",
		"namespace",
		"kpi_key",
		"kpi_name",
		"cf",
		"device",
		"mountpoint",
		"sql_type",
		"type",
	}

	metric_cols = []
	for col in df.columns:
		if col in non_metric_cols:
			continue
		if pd.api.types.is_numeric_dtype(df[col]):
			metric_cols.append(col)

	if not metric_cols:
		return pd.DataFrame(columns=["time", "service", "metric", "value"])

	tmp = pd.DataFrame({
		"time": df["time"],
		"service": entity_series,
	})
	tmp = pd.concat([tmp, df[metric_cols]], axis=1)

	melted = tmp.melt(
		id_vars=["time", "service"],
		value_vars=metric_cols,
		var_name="metric",
		value_name="value",
	)
	melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
	return melted.dropna(subset=["time", "service", "metric", "value"])


def _get_level_parquet_files(metric_root: Path, level: str) -> list[Path]:
	if level == "service":
		service_dir = metric_root / "apm" / "service"
		return sorted(service_dir.glob("*.parquet")) if service_dir.exists() else []

	if level == "pod":
		apm_pod_dir = metric_root / "apm" / "pod"
		infra_pod_dir = metric_root / "infra" / "infra_pod"
		files = []
		if apm_pod_dir.exists():
			files.extend(sorted(apm_pod_dir.glob("*.parquet")))
		if infra_pod_dir.exists():
			files.extend(sorted(infra_pod_dir.glob("*.parquet")))
		return files

	node_dir = metric_root / "infra" / "infra_node"
	return sorted(node_dir.glob("*.parquet")) if node_dir.exists() else []


def _read_all_metrics(metric_root: Path, level: str) -> pd.DataFrame:
	parquet_files = _get_level_parquet_files(metric_root, level)
	if not parquet_files:
		return pd.DataFrame(columns=["time", "service", "metric", "value"])

	frames = []
	for file_path in parquet_files:
		long_df = _extract_long_metrics_from_file(file_path, level)
		if long_df.empty:
			continue
		frames.append(long_df)

	if not frames:
		return pd.DataFrame(columns=["time", "service", "metric", "value"])

	return pd.concat(frames, ignore_index=True)


def build_date_csv(date_dir: Path, level: str) -> pd.DataFrame:
	metric_root = date_dir / "metric-parquet"
	if not metric_root.exists():
		return pd.DataFrame()

	all_metrics = _read_all_metrics(metric_root, level)
	if all_metrics.empty:
		return pd.DataFrame()

	all_metrics = all_metrics.dropna(subset=["time", "service", "metric", "value"])
	all_metrics = all_metrics[all_metrics["service"].astype(str).str.len() > 0]
	all_metrics["time"] = _to_epoch_seconds(all_metrics["time"])

	all_metrics = all_metrics.groupby(["time", "service", "metric"], as_index=False)["value"].mean()
	all_metrics["column_name"] = all_metrics["service"] + "_" + all_metrics["metric"]

	wide_df = all_metrics.pivot(index="time", columns="column_name", values="value").reset_index()
	metric_cols = sorted([c for c in wide_df.columns if c != "time"])
	wide_df = wide_df[["time"] + metric_cols].sort_values("time").reset_index(drop=True)

	return wide_df


def _cleanup_old_outputs() -> None:
	if not OUTPUT_DIR.exists():
		return

	for file_path in OUTPUT_DIR.glob("*.csv"):
		if file_path.name == "simple_data.csv":
			continue
		file_path.unlink(missing_ok=True)


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	_cleanup_old_outputs()

	date_dirs = sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir() and _is_date_dir(p)])
	if not date_dirs:
		print(f"未在 {DATASET_ROOT} 下找到日期目录")
		return

	for date_dir in date_dirs:
		for level in LEVELS:
			result_df = build_date_csv(date_dir, level)
			if result_df.empty:
				print(f"{date_dir.name} [{level}]: 无可用 metric 数据，跳过")
				continue

			output_file = OUTPUT_DIR / f"{date_dir.name}_{level}.csv"
			result_df.to_csv(output_file, index=False)
			print(f"{date_dir.name} [{level}]: 已生成 {output_file}")


if __name__ == "__main__":
	main()