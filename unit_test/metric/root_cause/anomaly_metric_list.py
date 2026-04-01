import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "unit_test/metric/results"
DEFAULT_OUTPUT = RESULT_DIR / "anomaly_metric_list.csv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unit_test.metric.root_cause import rank_and_tag_attach as R


def _load_baseline_hits() -> dict[tuple[str, str, str], str]:
    """(uuid,component,metric) -> 'b2,b4,...'"""
    files = [
        ("b2", RESULT_DIR / "result_baseline2.csv"),
        ("b4", RESULT_DIR / "result_baseline4.csv"),
        ("b5", RESULT_DIR / "result_baseline5.csv"),
    ]
    frames: list[pd.DataFrame] = []
    for name, path in files:
        if not path.exists():
            raise FileNotFoundError(f"baseline result not found: {path}")
        df = pd.read_csv(path)
        req = {"uuid", "component", "metric"}
        if not req.issubset(df.columns):
            raise ValueError(f"{path} must include columns: {req}")
        df = df[["uuid", "component", "metric"]].dropna().copy()
        df["uuid"] = df["uuid"].astype(str)
        df["component"] = df["component"].astype(str)
        df["metric"] = df["metric"].astype(str)
        df["baseline"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["uuid", "component", "metric", "baseline"])
    out = (
        merged.groupby(["uuid", "component", "metric"], as_index=False)["baseline"]
        .agg(lambda s: ",".join(sorted(set(map(str, s)))))
        .rename(columns={"baseline": "baseline_hits"})
    )
    return {
        (str(r.uuid), str(r.component), str(r.metric)): str(r.baseline_hits)
        for r in out.itertuples(index=False)
    }


def build_anomaly_metric_list(
    limit: int | None = None,
    uuid: str | None = None,
    min_votes: int = 1,
    keep_normal: bool = False,
    use_trace_rca: bool = False,
    use_trace_boost: bool = False,
    trace_latency_ms: float = 200.0,
) -> pd.DataFrame:
    cases = R._load_input_cases()
    if uuid:
        cases = [x for x in cases if x["uuid"] == str(uuid).strip()]
    if limit is not None:
        cases = cases[: max(1, int(limit))]
    if not cases:
        return pd.DataFrame()

    uuid_order = {x["uuid"]: i for i, x in enumerate(cases)}
    votes_df = R._load_union_anomalies()
    anomalies_by_uuid: dict[str, list[tuple[str, str, int]]] = {}
    for row in votes_df.itertuples(index=False):
        if int(row.votes) < max(1, int(min_votes)):
            continue
        anomalies_by_uuid.setdefault(str(row.uuid), []).append(
            (str(row.component), str(row.metric), int(row.votes))
        )
    baseline_hits = _load_baseline_hits()

    rows: list[dict] = []
    dataset_root = str(PROJECT_ROOT / "dataset")
    for i, case in enumerate(cases, start=1):
        case_uuid = case["uuid"]
        candidates = anomalies_by_uuid.get(case_uuid, [])
        if not candidates:
            print(f"[{i}/{len(cases)}] uuid={case_uuid} no candidate from baseline245")
            continue
        print(f"[{i}/{len(cases)}] uuid={case_uuid} candidates={len(candidates)}")
        case_rows = R._rank_one_uuid(
            case_uuid=case_uuid,
            start_time=case["start_time"],
            end_time=case["end_time"],
            candidates_raw=candidates,
            dataset_root=dataset_root,
            drop_normal=not keep_normal,
            use_trace_rca=use_trace_rca,
            use_trace_boost=use_trace_boost,
            trace_latency_ms=trace_latency_ms,
        )
        for r in case_rows:
            key = (str(r.get("uuid", "")), str(r.get("component", "")), str(r.get("metric", "")))
            r["baseline_hits"] = baseline_hits.get(key, "")
            r["metric_kind"] = R._metric_kind_for_rca(str(r.get("metric", "")))
            rows.append(r)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["votes"] = pd.to_numeric(out["votes"], errors="coerce").fillna(1).astype(int)
    out["raw_score"] = pd.to_numeric(out["raw_score"], errors="coerce").fillna(0.0)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    out["delta_minutes"] = pd.to_numeric(out["delta_minutes"], errors="coerce").fillna(0.0)
    out["onset_score"] = pd.to_numeric(out["onset_score"], errors="coerce").fillna(0.0)
    out["duration_ratio"] = pd.to_numeric(out["duration_ratio"], errors="coerce").fillna(0.0)
    out["run_length"] = pd.to_numeric(out["run_length"], errors="coerce").fillna(0).astype(int)
    out["final_score"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)

    out["_order"] = out["uuid"].map(uuid_order).fillna(len(uuid_order))
    out = out.sort_values(
        ["_order", "uuid", "final_score", "votes", "raw_score"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    out["rank_in_uuid"] = out.groupby("uuid").cumcount() + 1
    out = out.drop(columns=["_order"])

    preferred = [
        "uuid",
        "rank_in_uuid",
        "component",
        "component_group",
        "metric",
        "metric_kind",
        "pattern",
        "votes",
        "baseline_hits",
        "raw_score",
        "score",
        "delta_minutes",
        "onset_score",
        "duration_ratio",
        "run_length",
        "final_score",
        "trace_hot",
        "trace_callee_in",
        "trace_rca_ordinal",
        "trace_boost_hit",
    ]
    extra = [c for c in out.columns if c not in preferred]
    out = out[[c for c in preferred if c in out.columns] + extra]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build anomaly_metric_list.csv with BARO ranking + temporal features, "
            "using baseline2/4/5 union candidates."
        )
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first n cases")
    parser.add_argument("--uuid", type=str, default=None, help="Only process one uuid")
    parser.add_argument("--min_votes", type=int, default=1, help="Min baseline vote threshold")
    parser.add_argument("--keep_normal", action="store_true", help="Keep rows with pattern=normal")
    parser.add_argument("--trace_boost", action="store_true", help="Enable trace boost channel")
    parser.add_argument("--trace_rca", action="store_true", help="Enable trace RCA ordinal channel")
    parser.add_argument("--trace_latency_ms", type=float, default=200.0, help="Trace latency threshold")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output csv path")
    args = parser.parse_args()

    out_df = build_anomaly_metric_list(
        limit=args.limit,
        uuid=args.uuid,
        min_votes=args.min_votes,
        keep_normal=args.keep_normal,
        use_trace_rca=bool(args.trace_rca),
        use_trace_boost=bool(args.trace_boost) and not bool(args.trace_rca),
        trace_latency_ms=float(args.trace_latency_ms),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"saved {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()

# python3 unit_test/metric/root_cause/anomaly_metric_list.py