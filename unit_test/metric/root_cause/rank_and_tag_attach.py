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
# 排名：同一故障内多路特征的倒数秩融合 RRF（Cormack et al.；平滑常数 k=RRF_K），final_score 即 RRF。
# 算法步骤（与标注无关）：① baseline 并集候选 → BARO raw_score + 局部形态量；
# ② 各特征在 uuid 内分别排序 → RRF 合并；③ 可选 Trace；④ HipsterShop 多 Pod 同指标同形态可聚合；
# ⑤ 组件聚合与 top_k。默认每个 uuid 处理完即追加写入 OUTPUT_FILE；--batch_csv 则攒全表再一次性写入。
# Trace 默认关（扫 parquet 较慢）。

import argparse
import json
import logging
import re
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
from exp.agent.trace import TraceAgent
from exp.prompt.agent import CALL_TOPOLOGY
from exp.utils.time import parse_time_range
from baro.root_cause_analysis import robust_scorer

# Reciprocal rank fusion smoothing constant (Cormack et al.; widely cited default).
RRF_K = 60
# Metric 序列加载：故障开始时刻前推的窗口长度（分钟）。
LOAD_LOOKBACK_MINUTES = 30

# ---------------------------------------------------------------------------
# 部署与架构（基于 Google HipsterShop / Online Boutique 类微服务演示系统）
# - 核心微服务多语言实现；常见为每 Service 3 个 Pod，命名为 {service}-{0,1,2}
# - TiDB：tidb-tidb、tidb-pd、tidb-tikv 等通常各 1 Pod，不按三副本聚合
# - 节点：aiops-k8s-01..08、k8s-master* 等无 {service}-N 形态，不参与本聚合
# 下列集合用于识别「可对齐副本行」的 component_group；聚合在 run() 中可选开启。
# ---------------------------------------------------------------------------
HIPSTER_MICROSERVICES_THREE_PODS = frozenset(
	{
		"adservice",
		"cartservice",
		"currencyservice",
		"productcatalogservice",
		"checkoutservice",
		"recommendationservice",
		"shippingservice",
		"emailservice",
		"paymentservice",
		"redis-cart",
		"frontend",
	}
)


def _replica_index_for_hipster(component: str, component_group: str) -> int | None:
	"""若 component 为 {component_group}-{0..2} 且 group 为 Hipster 微服务，返回副本下标，否则 None。"""
	cg = str(component_group)
	if cg not in HIPSTER_MICROSERVICES_THREE_PODS:
		return None
	m = re.match(rf"^{re.escape(cg)}-(\d+)$", str(component))
	if not m:
		return None
	return int(m.group(1))


def _aggregate_hipster_replica_rows(df: pd.DataFrame) -> pd.DataFrame:
	"""
	将同一 uuid 下、同一微服务、同一 (metric, pattern) 的多条 Pod 行合并为一行，
	节省 component_group 内 max_per_component 名额；合并规则见函数内注释。
	"""
	if df.empty:
		return df
	df = df.copy()
	df["_rep_idx"] = df.apply(
		lambda r: _replica_index_for_hipster(str(r["component"]), str(r["component_group"])),
		axis=1,
	)
	eligible = df["component_group"].isin(HIPSTER_MICROSERVICES_THREE_PODS) & df["_rep_idx"].notna()
	sub = df.loc[eligible].drop(columns=["_rep_idx"])
	rest = df.loc[~eligible].drop(columns=["_rep_idx"])

	if sub.empty:
		if not rest.empty:
			rest = rest.copy()
			rest["replica_count"] = 1
			rest["pod_members"] = rest["component"].astype(str)
		return rest

	out_chunks: list[pd.DataFrame] = []
	for _, g in sub.groupby(["uuid", "component_group", "metric", "pattern"], sort=False):
		if len(g) == 1:
			row = g.iloc[0].to_dict()
			row["replica_count"] = 1
			row["pod_members"] = str(row.get("component", ""))
			out_chunks.append(pd.DataFrame([row]))
			continue
		# 取 RRF 最高的一条为模板；标量取「最严重/最早」的合理聚合
		best_i = int(g["final_score"].idxmax())
		base = g.loc[best_i].to_dict()
		pods = sorted(
			g["component"].astype(str).unique().tolist(),
			key=lambda x: int(x.rsplit("-", 1)[-1]) if x.rsplit("-", 1)[-1].isdigit() else 0,
		)
		base["component"] = str(base["component_group"])
		pods = _normalize_pod_members_list(str(base["component_group"]), pods)
		base["replica_count"] = len(pods)
		base["pod_members"] = ",".join(pods)
		base["delta_minutes"] = float(g["delta_minutes"].min())
		base["onset_score"] = float(1.0 / (1.0 + max(0.0, base["delta_minutes"])))
		base["duration_ratio"] = float(g["duration_ratio"].max())
		base["run_length"] = int(g["run_length"].max())
		base["raw_score"] = float(g["raw_score"].max())
		base["votes"] = int(g["votes"].max())
		base["score"] = float(np.log1p(max(0.0, base["raw_score"])))
		base["final_score"] = float(g["final_score"].max())
		for col in ("trace_boost_hit", "trace_callee_in", "trace_rca_ordinal", "trace_hot"):
			if col in base:
				base[col] = int(g[col].fillna(0).max())
		out_chunks.append(pd.DataFrame([base]))

	merged_sub = pd.concat(out_chunks, ignore_index=True)
	if not rest.empty:
		rest = rest.copy()
		rest["replica_count"] = 1
		rest["pod_members"] = rest["component"].astype(str)
		out = pd.concat([merged_sub, rest], ignore_index=True)
	else:
		out = merged_sub
	return _merge_duplicate_metric_pattern_rows(out)


def _pod_member_sort_key(name: str) -> tuple:
	s = str(name).strip()
	parts = s.rsplit("-", 1)
	if len(parts) == 2 and parts[1].isdigit():
		return (0, parts[0], int(parts[1]))
	return (1, s, 0)


def _normalize_pod_members_list(component_group: str, names: list[str]) -> list[str]:
	"""去重；若已有 {service}-N 副本，则去掉裸的 service 名（baseline 常同时给二者）。"""
	cg = str(component_group)
	seen: list[str] = []
	for n in sorted(names, key=_pod_member_sort_key):
		t = n.strip()
		if not t or t in seen:
			continue
		seen.append(t)
	has_replica = any(re.match(rf"^{re.escape(cg)}-\d+$", x) for x in seen)
	if has_replica:
		seen = [x for x in seen if x != cg]
	return seen


def _merge_duplicate_metric_pattern_rows(df: pd.DataFrame) -> pd.DataFrame:
	"""
	合并同一 (uuid, component_group, metric, pattern) 的多行（例如 baseline 同时给出
	service 名与 pod 名），避免 emailservice 与 emailservice-0..2 各占一行。
	"""
	if df.empty:
		return df
	chunks: list[pd.DataFrame] = []
	for _, g in df.groupby(["uuid", "component_group", "metric", "pattern"], sort=False):
		if len(g) == 1:
			chunks.append(g)
			continue
		best_i = int(g["final_score"].idxmax())
		base = g.loc[best_i].to_dict()
		names: list[str] = []
		for _, r in g.iterrows():
			for part in str(r.get("pod_members", "")).split(","):
				p = part.strip()
				if p and p not in names:
					names.append(p)
			c = str(r["component"]).strip()
			if c and c not in names:
				names.append(c)
		names = _normalize_pod_members_list(str(base["component_group"]), names)
		base["pod_members"] = ",".join(names)
		base["replica_count"] = len(names)
		base["component"] = str(base["component_group"])
		base["delta_minutes"] = float(g["delta_minutes"].min())
		base["onset_score"] = float(1.0 / (1.0 + max(0.0, base["delta_minutes"])))
		base["duration_ratio"] = float(g["duration_ratio"].max())
		base["run_length"] = int(g["run_length"].max())
		base["raw_score"] = float(g["raw_score"].max())
		base["votes"] = int(g["votes"].max())
		base["score"] = float(np.log1p(max(0.0, base["raw_score"])))
		base["final_score"] = float(g["final_score"].max())
		for col in ("trace_boost_hit", "trace_callee_in", "trace_rca_ordinal", "trace_hot"):
			if col in base:
				base[col] = int(g[col].fillna(0).max())
		chunks.append(pd.DataFrame([base]))
	return pd.concat(chunks, ignore_index=True)


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


def _trace_hot_component_groups(
	trace_result: list,
	latency_ms: float = 200.0,
) -> set[str]:
	"""
	从 TraceAgent.score 结果中提取「边上确有异常表现」涉及的服务（归一化后的 component_group 小写集合）。
	条件：该 link 下任一 pod 明细带 error_messages，或 avg_latency_ms 超过阈值。
	用于与 metric 排名结合：调用链上被点名的服务更可能是根因或一级受害者，可对其 KPI 加权。
	"""
	out: set[str] = set()
	if not trace_result:
		return out
	for link in trace_result:
		span = link.get("span") or {}
		src_raw = str(span.get("source") or "")
		tgt_raw = str(span.get("target") or "")
		hot_edge = False
		for d in link.get("details") or []:
			errs = d.get("error_messages") or []
			if errs:
				hot_edge = True
				break
			try:
				lat = float(d.get("avg_latency_ms") or 0)
			except (TypeError, ValueError):
				lat = 0.0
			if lat >= latency_ms:
				hot_edge = True
				break
		if not hot_edge:
			continue
		for raw in (src_raw, tgt_raw):
			s = raw.strip()
			if not s or s.lower() == "user":
				continue
			out.add(_normalize_component(s).lower())
	return out


def _metric_kind_for_rca(metric: str) -> str:
	"""将 KPI 粗分为三类，用于 Trace RCA 序数（internal / network / sli）。"""
	m = str(metric).lower()
	if any(
		k in m
		for k in (
			"pod_cpu",
			"pod_memory",
			"pod_process",
			"node_",
			"jvm",
			"memory_usage",
			"cpu_usage",
			"working_set",
			"disk",
			"_fs",
			"io_util",
			"load_average",
			"load_",
		)
	):
		return "internal"
	if any(k in m for k in ("network_receive", "network_transmit", "network_", "packet", "byte")):
		return "network"
	return "sli"


def _compute_rrf_scores(
	pdf: pd.DataFrame,
	trace_mode: str,
) -> pd.Series:
	"""
	倒数秩融合：对同一 uuid 内每条候选，在若干「只依赖本故障内相对强弱」的排序上取秩，
	再按 sum 1/(k+rank) 合并。k=RRF_K 为文献常用常数，不是对评测集调参。
	trace_mode: "" | "rca" | "boost"
	"""
	if pdf.empty:
		return pd.Series(dtype=float)
	s = pd.Series(0.0, index=pdf.index)
	r = pdf["raw_score"].rank(ascending=False, method="min")
	s = s + 1.0 / (RRF_K + r)
	r = pdf["delta_minutes"].rank(ascending=True, method="min")
	s = s + 1.0 / (RRF_K + r)
	r = pdf["votes"].rank(ascending=False, method="min")
	s = s + 1.0 / (RRF_K + r)
	r = pdf["duration_ratio"].rank(ascending=False, method="min")
	s = s + 1.0 / (RRF_K + r)
	r = pdf["run_length"].rank(ascending=False, method="min")
	s = s + 1.0 / (RRF_K + r)
	if trace_mode == "rca":
		r = pdf["trace_callee_in"].rank(ascending=False, method="min")
		s = s + 1.0 / (RRF_K + r)
		r = pdf["trace_rca_ordinal"].rank(ascending=False, method="min")
		s = s + 1.0 / (RRF_K + r)
	elif trace_mode == "boost":
		r = pdf["trace_hot"].rank(ascending=False, method="min")
		s = s + 1.0 / (RRF_K + r)
	return s


def _trace_hot_in_out_degree(
	trace_result: list,
	latency_ms: float,
) -> tuple[dict[str, int], dict[str, int]]:
	"""
	对「热点边」统计每个服务作为 callee(target) / caller(source) 的出现次数。
	热点边定义与 _trace_hot_component_groups 一致。
	"""
	in_hot: dict[str, int] = {}
	out_hot: dict[str, int] = {}
	if not trace_result:
		return in_hot, out_hot
	for link in trace_result:
		span = link.get("span") or {}
		src_raw = str(span.get("source") or "")
		tgt_raw = str(span.get("target") or "")
		hot_edge = False
		for d in link.get("details") or []:
			errs = d.get("error_messages") or []
			if errs:
				hot_edge = True
				break
			try:
				lat = float(d.get("avg_latency_ms") or 0)
			except (TypeError, ValueError):
				lat = 0.0
			if lat >= latency_ms:
				hot_edge = True
				break
		if not hot_edge:
			continue
		src = _normalize_component(src_raw).lower()
		tgt = _normalize_component(tgt_raw).lower()
		if src and src != "user":
			out_hot[src] = out_hot.get(src, 0) + 1
		if tgt:
			in_hot[tgt] = in_hot.get(tgt, 0) + 1
	return in_hot, out_hot


def _reachable_hot_callee_from_caller(
	start: str,
	topology: dict[str, list[str]],
	callee_hot: set[str],
) -> bool:
	"""
	从 start 沿静态拓扑「向下游 callee」BFS，是否可达任一 trace 上作为热点 target 的服务。
	用于压低：仅表现为 SLI 异常、但下游 callee 已在 trace 上承压的「连带指标」。
	"""
	if not callee_hot or not start:
		return False
	s = str(start).lower()
	if s in callee_hot:
		return False
	from collections import deque

	q = deque(topology.get(s, []))
	seen = {s}
	while q:
		n = q.popleft()
		if n in seen:
			continue
		seen.add(n)
		if n in callee_hot:
			return True
		for ch in topology.get(n, []):
			if ch not in seen:
				q.append(ch)
	return False


def _trace_rca_ordinal(
	component_group: str,
	metric: str,
	in_hot: dict[str, int],
	out_hot: dict[str, int],
	topology: dict[str, list[str]],
	callee_hot: set[str],
) -> int:
	"""
	Trace 侧离散序数（整数），只表达相对次序，参与 RRF 的一路排序；不用浮点乘子。
	语义：热点 callee 上 internal/network 高于 SLI；纯 caller 上 SLI 偏低；拓扑下游可达热点 callee 的 SLI 偏低。
	"""
	c = str(component_group).lower()
	kind = _metric_kind_for_rca(metric)
	inn = int(in_hot.get(c, 0))
	outn = int(out_hot.get(c, 0))
	callee_on_hot_edge = inn >= 1
	caller_only = outn >= 1 and inn == 0

	ordv = 0
	if callee_on_hot_edge:
		if kind == "internal":
			ordv = 3
		elif kind == "network":
			ordv = 2
		else:
			ordv = 1
	if caller_only and kind == "sli":
		ordv -= 2
	elif caller_only and kind == "network":
		ordv -= 1
	if kind == "sli" and not callee_on_hot_edge and _reachable_hot_callee_from_caller(c, topology, callee_hot):
		ordv -= 1
	return int(ordv)


def _pattern_from_series(series: pd.Series, fault_start: datetime) -> tuple[str, float, float, int]:
	if series.empty or len(series) < 5:
		return "normal", 0.0, 0.0, 0

	pre = series[series.index < fault_start]
	post = series[series.index >= fault_start]
	if pre.empty or post.empty:
		return "normal", 0.0, 0.0, 0

	base_mean = float(pre.mean())
	base_std = float(pre.std())

	# 故障后相对故障前：3σ 规则（近似正态时）；近常数序列用固定绝对阈值区分噪声。
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

	full_mask = np.zeros(len(series), dtype=bool)
	start_idx = len(pre)
	full_mask[start_idx:start_idx + len(post_mask)] = post_mask
	pattern = _detect_local_pattern(series, full_mask)

	# 形态过弱时降级标签，避免把单点抖动标成 shift（启发式，仅影响 pattern 名称）
	if pattern in {"surge", "level_shift_up", "level_shift_down", "dip"} and run_length < 2 and duration_ratio < 0.08:
		pattern = "spike" if pattern in {"surge", "level_shift_up"} else "drop"

	return pattern, delta_minutes, duration_ratio, run_length


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


def _rank_one_uuid(
	case_uuid: str,
	start_time: datetime,
	end_time: datetime,
	candidates_raw: list[tuple[str, str, int]],
	dataset_root: str,
	drop_normal: bool,
	use_trace_rca: bool,
	use_trace_boost: bool,
	trace_latency_ms: float,
) -> list[dict]:
	"""单条故障：BARO + 局部形态特征，同一 uuid 内 RRF 融合；可选 Trace 以整数序数/计数参与 RRF。"""
	candidates = list(dict.fromkeys(candidates_raw))
	if not candidates:
		return []

	candidate_votes = {(c, m): v for c, m, v in candidates}
	load_start = start_time - timedelta(minutes=LOAD_LOOKBACK_MINUTES)
	metric_agent = MetricAgent(dataset_root)
	trace_agent: TraceAgent | None = None
	if use_trace_rca or use_trace_boost:
		trace_agent = TraceAgent(dataset_root)

	try:
		metric_df = metric_agent.load_data(load_start, end_time)
	except Exception as exc:
		logger.warning("load_data failed for %s: %s", case_uuid, exc)
		return []

	if metric_df.empty:
		return []

	pair_candidates = [(c, m) for c, m, _ in candidates]
	metric_dict, series_dict = _build_metric_dict_for_candidates(metric_df, pair_candidates)
	if not metric_dict:
		return []

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

	case_rows: list[dict] = []
	for key in ranked_keys:
		component, metric = key.split("::", 1)
		component_group = _normalize_component(component)

		pattern, delta_minutes, duration_ratio, run_length = _pattern_from_series(series_dict[key], start_time)
		raw_score = float(score_map.get(key, 0.0))
		votes = int(candidate_votes.get((component, metric), 1))
		score_log = float(np.log1p(max(0.0, raw_score)))
		# 可解释 onset：距注入时刻越短越高（分钟为量纲，+1 避免除零）
		onset_score = float(1.0 / (1.0 + max(0.0, delta_minutes)))

		if drop_normal and pattern == "normal":
			continue

		case_rows.append(
			{
				"uuid": case_uuid,
				"component": component,
				"component_group": component_group,
				"metric": metric,
				"pattern": pattern,
				"votes": votes,
				"raw_score": raw_score,
				"score": score_log,
				"delta_minutes": float(delta_minutes),
				"onset_score": onset_score,
				"duration_ratio": duration_ratio,
				"run_length": run_length,
				"trace_callee_in": 0,
				"trace_rca_ordinal": 0,
				"trace_hot": 0,
				"trace_boost_hit": 0,
			}
		)

	if not case_rows:
		return []

	trace_mode = ""
	if trace_agent is not None and (use_trace_rca or use_trace_boost):
		try:
			trace_result = trace_agent.score(start_time, end_time)
			if use_trace_rca:
				trace_mode = "rca"
				in_hot, out_hot = _trace_hot_in_out_degree(trace_result, trace_latency_ms)
				callee_hot = {s for s, v in in_hot.items() if v >= 1}
				for r in case_rows:
					c = str(r["component_group"]).lower()
					r["trace_callee_in"] = int(in_hot.get(c, 0))
					r["trace_rca_ordinal"] = _trace_rca_ordinal(
						str(r["component_group"]),
						str(r["metric"]),
						in_hot,
						out_hot,
						CALL_TOPOLOGY,
						callee_hot,
					)
			elif use_trace_boost:
				trace_mode = "boost"
				hot_groups = _trace_hot_component_groups(trace_result, latency_ms=trace_latency_ms)
				for r in case_rows:
					g = str(r["component_group"]).lower()
					hit = 1 if hot_groups and g in hot_groups else 0
					r["trace_hot"] = hit
					r["trace_boost_hit"] = hit
		except Exception as exc:
			logger.warning("trace post-process failed for %s: %s", case_uuid, exc)
			trace_mode = ""

	pdf = pd.DataFrame(case_rows)
	pdf["final_score"] = _compute_rrf_scores(pdf, trace_mode)
	return pdf.to_dict("records")


def _finalize_result_df(
	result_df: pd.DataFrame,
	uuid_order: dict[str, int],
	aggregate_replicas: bool,
	max_per_component: int,
	top_k: int,
	use_trace_boost: bool,
	use_trace_rca: bool,
) -> pd.DataFrame:
	"""对已有 ranking 行做数值化、副本聚合、组件分与截断、列顺序。"""
	if result_df.empty:
		return result_df
	result_df = result_df.drop_duplicates()
	result_df["score"] = pd.to_numeric(result_df["score"], errors="coerce").fillna(0.0)
	result_df["raw_score"] = pd.to_numeric(result_df["raw_score"], errors="coerce").fillna(0.0)
	result_df["delta_minutes"] = pd.to_numeric(result_df["delta_minutes"], errors="coerce").fillna(0.0)
	result_df["onset_score"] = pd.to_numeric(result_df["onset_score"], errors="coerce").fillna(0.0)
	result_df["duration_ratio"] = pd.to_numeric(result_df["duration_ratio"], errors="coerce").fillna(0.0)
	result_df["run_length"] = pd.to_numeric(result_df["run_length"], errors="coerce").fillna(0).astype(int)
	result_df["final_score"] = pd.to_numeric(result_df["final_score"], errors="coerce").fillna(0.0)
	result_df["votes"] = pd.to_numeric(result_df["votes"], errors="coerce").fillna(1).astype(int)
	if "trace_boost_hit" in result_df.columns:
		result_df["trace_boost_hit"] = pd.to_numeric(
			result_df["trace_boost_hit"], errors="coerce"
		).fillna(0).astype(int)
	if "trace_callee_in" in result_df.columns:
		result_df["trace_callee_in"] = pd.to_numeric(
			result_df["trace_callee_in"], errors="coerce"
		).fillna(0).astype(int)
	if "trace_rca_ordinal" in result_df.columns:
		result_df["trace_rca_ordinal"] = pd.to_numeric(
			result_df["trace_rca_ordinal"], errors="coerce"
		).fillna(0).astype(int)
	if "trace_hot" in result_df.columns:
		result_df["trace_hot"] = pd.to_numeric(result_df["trace_hot"], errors="coerce").fillna(0).astype(int)

	if aggregate_replicas:
		result_df = _aggregate_hipster_replica_rows(result_df)

	group_score = (
		result_df.sort_values(["uuid", "component_group", "final_score"], ascending=[True, True, False])
		.groupby(["uuid", "component_group"], as_index=False)
		.head(3)
		.groupby(["uuid", "component_group"], as_index=False)["final_score"]
		.sum()
		.rename(columns={"final_score": "component_score"})
	)
	group_peak = (
		result_df.groupby(["uuid", "component_group"], as_index=False)["final_score"]
		.max()
		.rename(columns={"final_score": "component_peak_score"})
	)
	result_df = result_df.merge(group_score, on=["uuid", "component_group"], how="left")
	result_df = result_df.merge(group_peak, on=["uuid", "component_group"], how="left")

	result_df["_order"] = result_df["uuid"].map(uuid_order).fillna(len(uuid_order))
	result_df = result_df.sort_values(
		[
			"_order",
			"component_peak_score",
			"component_score",
			"final_score",
			"score",
			"raw_score",
		],
		ascending=[True, False, False, False, False, False],
	)

	result_df["_comp_rank"] = result_df.groupby(["uuid", "component_group"]).cumcount()
	result_df = result_df[result_df["_comp_rank"] < max(1, int(max_per_component))]

	result_df = result_df.groupby("uuid", as_index=False).head(max(1, int(top_k)))
	drop_cols = ["_order"]
	if "_comp_rank" in result_df.columns:
		drop_cols.append("_comp_rank")
	result_df = result_df.drop(columns=drop_cols).reset_index(drop=True)
	if not use_trace_boost and "trace_boost_hit" in result_df.columns:
		result_df = result_df.drop(columns=["trace_boost_hit"])
	if not use_trace_boost and "trace_hot" in result_df.columns:
		result_df = result_df.drop(columns=["trace_hot"])
	if not use_trace_rca and "trace_rca_ordinal" in result_df.columns:
		result_df = result_df.drop(columns=["trace_rca_ordinal"])
	if not use_trace_rca and "trace_callee_in" in result_df.columns:
		result_df = result_df.drop(columns=["trace_callee_in"])

	_preferred_cols = [
		"uuid",
		"component",
		"component_group",
		"replica_count",
		"pod_members",
		"metric",
		"pattern",
		"votes",
		"raw_score",
		"score",
		"delta_minutes",
		"onset_score",
		"duration_ratio",
		"run_length",
		"final_score",
		"component_score",
		"component_peak_score",
	]
	_extra = [c for c in result_df.columns if c not in _preferred_cols]
	result_df = result_df[[c for c in _preferred_cols if c in result_df.columns] + _extra]
	return result_df


def run(
	limit: int | None = None,
	uuid: str | None = None,
	top_k: int = 30,
	drop_normal: bool = True,
	min_votes: int = 1,
	max_per_component: int = 4,
	aggregate_replicas: bool = True,
	stream_csv: bool = True,
	use_trace_boost: bool = False,
	trace_latency_ms: float = 200.0,
	use_trace_rca: bool = False,
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

	dataset_root = str(PROJECT_ROOT / "dataset")
	if use_trace_rca:
		logger.info(
			"trace RCA enabled (slow): latency_threshold_ms=%s",
			trace_latency_ms,
		)
	elif use_trace_boost:
		logger.info(
			"trace boost enabled (RRF list): latency_threshold_ms=%s",
			trace_latency_ms,
		)

	jobs: list[tuple] = []
	for item in dataset:
		case_uuid = str(item.get("uuid", "")).strip()
		if not case_uuid:
			continue
		candidates_raw = anomalies_by_uuid.get(case_uuid, [])
		if not candidates_raw:
			continue
		start_time = item.get("start_time")
		end_time = item.get("end_time")
		if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
			continue
		jobs.append(
			(
				case_uuid,
				start_time,
				end_time,
				candidates_raw,
				dataset_root,
				drop_normal,
				use_trace_rca,
				use_trace_boost,
				trace_latency_ms,
			)
		)

	RESULT_DIR.mkdir(parents=True, exist_ok=True)

	if stream_csv:
		header_written = False
		total_rows = 0
		with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as outf:
			for job in jobs:
				chunk = _rank_one_uuid(*job)
				if not chunk:
					continue
				result_df = pd.DataFrame(chunk)
				result_df = _finalize_result_df(
					result_df,
					uuid_order,
					aggregate_replicas,
					max_per_component,
					top_k,
					use_trace_boost,
					use_trace_rca,
				)
				if result_df.empty:
					continue
				result_df.to_csv(outf, index=False, header=not header_written)
				header_written = True
				total_rows += len(result_df)
		logger.info("saved %d rows to %s (stream_csv)", total_rows, OUTPUT_FILE)
	else:
		output_rows: list[dict] = []
		for job in jobs:
			output_rows.extend(_rank_one_uuid(*job))

		result_df = pd.DataFrame(output_rows)
		if not result_df.empty:
			result_df = _finalize_result_df(
				result_df,
				uuid_order,
				aggregate_replicas,
				max_per_component,
				top_k,
				use_trace_boost,
				use_trace_rca,
			)
		result_df.to_csv(OUTPUT_FILE, index=False)
		logger.info("saved %d rows to %s (batch_csv)", len(result_df), OUTPUT_FILE)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--limit", type=int, default=None, help="Only process first n cases")
	parser.add_argument("--uuid", type=str, default=None, help="Only process one uuid")
	parser.add_argument("--top_k", type=int, default=30, help="Keep top-k metrics per uuid")
	parser.add_argument("--keep_normal", action="store_true", help="Keep rows with normal pattern")
	parser.add_argument(
		"--min_votes",
		type=int,
		default=1,
		help="至少 N 个 baseline 同时命中 (uuid,component,metric)；默认 1 以召回单 baseline 真根因，设为 2 可降噪",
	)
	parser.add_argument("--max_per_component", type=int, default=4, help="Max rows per component group")
	parser.add_argument(
		"--no_aggregate_replicas",
		action="store_true",
		help="关闭 HipsterShop 微服务多 Pod 同指标同形态合并（默认开启，节省每组件行数）",
	)
	parser.add_argument(
		"--batch_csv",
		action="store_true",
		help="整表算完后一次性写入；默认每个 uuid 算完即追加写入 OUTPUT_FILE",
	)
	parser.add_argument(
		"--trace_boost",
		action="store_true",
		help="结合 Trace：热点边上的服务在 RRF 中增加一路「是否命中」排序",
	)
	parser.add_argument(
		"--trace_latency_ms",
		type=float,
		default=200.0,
		help="Trace 明细 avg_latency_ms 超过该阈值视为热点边（与 error_messages 二选一即可）",
	)
	parser.add_argument(
		"--trace_rca",
		action="store_true",
		help="启用 Trace 根因/连带区分（每个 uuid 多扫两遍 trace，明显变慢）",
	)
	args = parser.parse_args()
	if args.trace_boost and args.trace_rca:
		logger.warning("同时指定 --trace_rca 与 --trace_boost 时仅使用 --trace_rca（忽略 trace_boost）")
	run(
		limit=args.limit,
		uuid=args.uuid,
		top_k=args.top_k,
		drop_normal=not args.keep_normal,
		min_votes=args.min_votes,
		max_per_component=args.max_per_component,
		aggregate_replicas=not args.no_aggregate_replicas,
		stream_csv=not args.batch_csv,
		use_trace_boost=args.trace_boost and not args.trace_rca,
		trace_latency_ms=args.trace_latency_ms,
		use_trace_rca=args.trace_rca,
	)


if __name__ == "__main__":
	main()

# 示例: python3 unit_test/metric/root_cause/rank_and_tag_attach.py --top_k=30 --min_votes=1 --max_per_component=4