# 基于大模型总结异常指标列表，输出到 results/metric_summary.txt 中
# metric_utils.py 是参考代码，参考这个代码实现一个两阶段的 LLM 总结，调用 Yuzo API
# 读取 anomaly_metric_list.csv 文件中检测出的所有异常指标，分类为 Service/Pod/Node/TiDB 四类
# 总结完成后生成异常 metric 总结，作为提供给大模型进行多模态关联分析的 prompt
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "unit_test/metric/results/anomaly_metric_list.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/metric_summary.txt"
DEFAULT_CASES_FILE = PROJECT_ROOT / "dataset/input.json"


def _normalize_component(component: str) -> str:
    c = str(component)
    if c.startswith("aiops-k8s-") or c.startswith("k8s-master"):
        return c
    return c


def _is_tidb_related(component: str, component_group: str, metric: str) -> bool:
    text = f"{component} {component_group} {metric}".lower()
    keys = (
        "tidb",
        "tikv",
        "pd",
        "tiflash",
        "ticdc",
        "drainer",
        "pump",
    )
    return any(k in text for k in keys)


def _metric_category(component: str, component_group: str, metric: str) -> str:
    c = str(component)
    cg = str(component_group)
    m = str(metric).lower()
    if _is_tidb_related(c, cg, m):
        return "TiDB"
    if cg.startswith("aiops-k8s-") or cg.startswith("k8s-master") or m.startswith("node_"):
        return "Node"
    if m.startswith("pod_"):
        return "Pod"
    return "Service"


def _build_category_payload(
    df: pd.DataFrame,
    category: str,
    top_n: int = 24,
    max_chars: int = 2600,
) -> str:
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return f"[{category}] no anomalies found."

    total_rows = len(sub)
    unique_uuid = sub["uuid"].nunique()
    unique_components = sub["component_group"].nunique()
    unique_metrics = sub["metric"].nunique()

    # 核心单元：component_group + metric，避免“同 metric/同组件”分开描述造成重复噪声。
    unit_stats = (
        sub.groupby(["component_group", "metric"], as_index=False)
        .agg(
            hit_count=("metric", "size"),
            uuid_count=("uuid", "nunique"),
            votes_avg=("votes", "mean"),
            votes_max=("votes", "max"),
            final_score_max=("final_score", "max") if "final_score" in sub.columns else ("votes", "max"),
            final_score_avg=("final_score", "mean") if "final_score" in sub.columns else ("votes", "mean"),
            pattern_top=("pattern", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
        )
        .sort_values(
            ["final_score_max", "votes_max", "uuid_count", "hit_count", "component_group", "metric"],
            ascending=[False, False, False, False, True, True],
        )
        .head(max(1, int(top_n)))
    )

    comp_stats = (
        sub.groupby("component_group", as_index=False)
        .agg(
            hit_count=("metric", "size"),
            uuid_count=("uuid", "nunique"),
            metric_count=("metric", "nunique"),
            votes_avg=("votes", "mean"),
            votes_max=("votes", "max"),
            final_score_max=("final_score", "max") if "final_score" in sub.columns else ("votes", "max"),
        )
        .sort_values(
            ["final_score_max", "votes_max", "uuid_count", "hit_count", "metric_count", "component_group"],
            ascending=[False, False, False, False, False, True],
        )
        .head(10)
    )

    unit_lines = [
        (
            f"- ({r.component_group}, {r.metric}) | pattern={r.pattern_top} "
            f"| score_max={float(r.final_score_max):.4f} | score_avg={float(r.final_score_avg):.4f} "
            f"| votes_max={int(r.votes_max)} | uuids={int(r.uuid_count)} | hits={int(r.hit_count)}"
        )
        for r in unit_stats.itertuples(index=False)
    ]
    comp_lines = [
        (
            f"- {r.component_group} | score_max={float(r.final_score_max):.4f} "
            f"| votes_max={int(r.votes_max)} | uuids={int(r.uuid_count)} "
            f"| metrics={int(r.metric_count)} | hits={int(r.hit_count)}"
        )
        for r in comp_stats.itertuples(index=False)
    ]

    # 组件打包视图：每个组件保留多个高优先级指标（而不是只留一个）
    bundle_lines: list[str] = []
    for r in comp_stats.itertuples(index=False):
        cg = str(r.component_group)
        g = sub[sub["component_group"] == cg].copy()
        metric_pick = (
            g.groupby("metric", as_index=False)
            .agg(
                score_best=("final_score", "max"),
                votes_max=("votes", "max"),
                pattern_top=("pattern", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
            )
            .sort_values(["score_best", "votes_max", "metric"], ascending=[False, False, True])
            .head(4)
        )
        metric_tokens = [
            f"{x.metric}({x.pattern_top},s={float(x.score_best):.4f},v={int(x.votes_max)})"
            for x in metric_pick.itertuples(index=False)
        ]
        bundle_lines.append(f"- ({cg}, " + ", ".join(metric_tokens) + ")")

    payload = (
        f"[{category}] total_rows={total_rows}, unique_uuid={unique_uuid}, "
        f"unique_components={unique_components}, unique_metrics={unique_metrics}\n"
        "Top component-metric units:\n"
        + "\n".join(unit_lines)
        + "\nComponent bundles (preferred for reasoning):\n"
        + "\n".join(bundle_lines)
        + "\nTop components:\n"
        + "\n".join(comp_lines)
    )
    if len(payload) > max_chars:
        payload = payload[: max_chars - 40] + "\n...(truncated)"
    return payload


def _call_yuzo(client: Any, model: str, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    return str(content or "").strip()


def _clean_llm_output(text: str) -> str:
    """Strip code fences and trailing chatter; keep concise body."""
    s = str(text or "").strip()
    if s.startswith("```"):
        s = s.replace("```markdown", "").replace("```md", "").replace("```", "").strip()
    # Remove common trailing meta-commentary lines.
    bad_prefix = (
        "这个总结",
        "以上总结",
        "总结如下",
        "注：",
        "说明：",
    )
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if any(t.startswith(p) for p in bad_prefix):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def _build_client(api_key: str | None, api_url: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Python package 'openai' is required for Yuzo API calls. Install with: pip install openai"
        ) from exc

    key = (
        api_key
        or os.getenv("YUZO_API_KEY")
        or os.getenv("DEEPSHIELDS_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError(
            "API key not found. Set YUZO_API_KEY/DEEPSHIELDS_API_KEY/DEEPSEEK_API_KEY/OPENAI_API_KEY or pass --api_key."
        )
    url = api_url or os.getenv("YUZO_API_URL") or "https://api.deepshields.com/v1"
    return OpenAI(api_key=key, base_url=str(url).rstrip("/"))


def _load_uuid_order(cases_file: Path, first_k: int | None) -> List[str]:
    if not cases_file.exists():
        raise FileNotFoundError(f"cases file not found: {cases_file}")
    with cases_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{cases_file} must be a JSON array")

    uuids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        u = str(item.get("uuid", "")).strip()
        if not u:
            continue
        uuids.append(u)
    if first_k is not None:
        uuids = uuids[: max(1, int(first_k))]
    return uuids


def _metric_list_semantics_note() -> str:
    return (
        "【异常指标列表语义说明】\n"
        "1) 列表的基本证据单元是 (component_group, metric)。\n"
        "2) 每条记录来自 baseline2/4/5 候选，经 BARO + 时序特征 + RRF 融合后排序。\n"
        "3) 关键字段含义：\n"
        "   - rank_in_uuid: 该 uuid 内排序名次（1 为最高优先级）。\n"
        "   - final_score: 多路特征融合后的最终分（越大越可能是根因相关）。\n"
        "   - raw_score: BARO 原始异常强度分（越大异常越强）。\n"
        "   - votes: baseline245 命中票数（3>2>1，越大鲁棒性越高）。\n"
        "   - pattern: 局部时序形态（如 level_shift_up/spike/surge/drop/dip/normal/missing_data）。\n"
        "     其中 missing_data 表示故障窗口内存在明显采样缺口/数据缺失，应视为一种观测异常，而不是普通涨跌形态。\n"
        "   - delta_minutes/onset_score: 异常出现相对故障起点的早晚信息。\n"
        "   - duration_ratio/run_length: 异常持续性强度信息。\n"
        "4) 解读原则：优先关注 rank 靠前 + final_score 高 + votes 高 + 非 normal pattern 的单元。\n"
        "5) 你的结论必须引用具体 (component_group, metric) 单元，不要泛泛描述。\n"
    )


def _metric_meaning_note() -> str:
    return (
        "【常见指标含义说明】\n"
        "1) Service / 应用性能类指标：\n"
        "   - `request` / `response`: 请求量与响应量，反映业务吞吐变化。\n"
        "   - `rrt` / `rrt_max`: 服务响应时间，反映延迟变化。\n"
        "   - `client_error_ratio` / `error_ratio` / `server_error_ratio`: 错误比例，反映请求失败或服务异常。\n"
        "   - `timeout`: 超时相关指标，反映调用超时现象。\n"
        "2) Pod / 容器资源类指标：\n"
        "   - `pod_cpu_usage`: Pod CPU 使用率。\n"
        "   - `pod_memory_working_set_bytes`: Pod 工作集内存使用量。\n"
        "   - `pod_processes`: Pod 内运行进程数量。\n"
        "   - `pod_fs_reads_bytes` / `pod_fs_writes_bytes`: Pod 文件系统读写字节数。\n"
        "   - `pod_network_receive_bytes` / `pod_network_transmit_bytes`: Pod 网络接收/发送字节数。\n"
        "   - `pod_network_receive_packets` / `pod_network_transmit_packets`: Pod 网络收发包数。\n"
        "3) Node / 基础设施资源类指标：\n"
        "   - `node_cpu_usage_rate`: 节点 CPU 使用率。\n"
        "   - `node_memory_usage_rate` / `node_memory_MemAvailable_bytes`: 节点内存使用率 / 可用内存。\n"
        "   - `node_disk_read_bytes_total` / `node_disk_written_bytes_total`: 节点磁盘读写强度。\n"
        "   - `node_filesystem_usage_rate` / `node_filesystem_free_bytes`: 节点文件系统使用率 / 空闲空间。\n"
        "   - `node_network_receive_bytes_total` / `node_network_transmit_bytes_total`: 节点网络收发字节数。\n"
        "   - `node_network_receive_packets_total` / `node_network_transmit_packets_total`: 节点网络收发包数。\n"
        "   - `node_sockstat_TCP_inuse`: 节点 TCP 连接占用情况。\n"
        "4) TiDB / 数据库相关指标：\n"
        "   - `memory_usage` / `cpu_usage`: 数据库组件资源使用情况。\n"
        "   - `qps` / `grpc_qps`: 查询或 RPC 吞吐变化。\n"
        "   - `duration_99th`: 99 分位延迟，反映尾延迟。\n"
        "   - `io_util` / `read_mbps` / `write_wal_mbps`: 存储 I/O 压力与吞吐。\n"
        "   - `connection_count`: 连接数变化。\n"
        "5) 解读提醒：\n"
        "   - 资源类指标更偏向解释负载、资源竞争、容量压力。\n"
        "   - 请求/响应/延迟/错误类指标更偏向解释业务层受影响现象。\n"
        "   - `normal` 可作为对照；`missing_data` 表示采样缺口，不等于普通涨跌。\n"
    )


def _build_must_keep_units(df: pd.DataFrame, max_items: int = 6) -> list[str]:
    """Pick top ranked (component_group, metric) units that must be covered."""
    if df.empty:
        return []
    work = df.copy()
    if "rank_in_uuid" in work.columns:
        work["rank_in_uuid"] = pd.to_numeric(work["rank_in_uuid"], errors="coerce").fillna(10_000).astype(int)
    else:
        work["rank_in_uuid"] = 10_000
    if "final_score" in work.columns:
        work["final_score"] = pd.to_numeric(work["final_score"], errors="coerce").fillna(0.0)
    else:
        work["final_score"] = 0.0
    if "pattern" not in work.columns:
        work["pattern"] = "unknown"

    # Prefer non-normal top-ranked units; fallback to top-ranked regardless of pattern.
    non_normal = work[work["pattern"].astype(str).ne("normal")].copy()
    pool = non_normal if not non_normal.empty else work
    comp_col = "evidence_component" if "evidence_component" in pool.columns else "component_group"
    unit_rank = (
        pool.groupby([comp_col, "metric"], as_index=False)
        .agg(
            rank_best=("rank_in_uuid", "min"),
            score_best=("final_score", "max"),
            pattern_top=("pattern", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
        )
        .sort_values(["rank_best", "score_best"], ascending=[True, False])
        .head(max(1, int(max_items)))
    )
    out: list[str] = []
    for r in unit_rank.itertuples(index=False):
        comp = getattr(r, comp_col)
        out.append(
            f"({comp}, {r.metric}) | rank_best={int(r.rank_best)} "
            f"| pattern={r.pattern_top} | score_best={float(r.score_best):.4f}"
        )
    return out


def _build_component_bundle_lines(
    df: pd.DataFrame,
    max_components: int = 5,
    max_metrics_per_component: int = 3,
) -> list[str]:
    """Build enforced '(component, metric1, metric2, ...)' evidence lines."""
    if df.empty:
        return []
    work = df.copy()
    if "final_score" in work.columns:
        work["final_score"] = pd.to_numeric(work["final_score"], errors="coerce").fillna(0.0)
    else:
        work["final_score"] = 0.0
    if "votes" in work.columns:
        work["votes"] = pd.to_numeric(work["votes"], errors="coerce").fillna(1).astype(int)
    else:
        work["votes"] = 1

    comp_col = "evidence_component" if "evidence_component" in work.columns else "component_group"
    comp_rank = (
        work.groupby(comp_col, as_index=False)
        .agg(score_best=("final_score", "max"), votes_max=("votes", "max"), metric_cnt=("metric", "nunique"))
        .sort_values(["score_best", "votes_max", "metric_cnt"], ascending=[False, False, False])
        .head(max(1, int(max_components)))
    )
    lines: list[str] = []
    for r in comp_rank.itertuples(index=False):
        cg = str(getattr(r, comp_col))
        g = work[work[comp_col] == cg].copy()
        top_metrics = (
            g.groupby("metric", as_index=False)
            .agg(
                score_best=("final_score", "max"),
                votes_max=("votes", "max"),
                pattern_top=("pattern", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
            )
            .sort_values(["score_best", "votes_max", "metric"], ascending=[False, False, True])
            .head(max(1, int(max_metrics_per_component)))
        )
        def _shape_desc(p: str) -> str:
            p = str(p)
            if p in {"level_shift_up", "surge", "spike"}:
                return f"{p}|上升"
            if p in {"level_shift_down", "drop", "dip"}:
                return f"{p}|下降"
            if p == "normal":
                return "normal|平稳"
            return f"{p}|未知"

        metrics = [
            f"{x.metric}({_shape_desc(x.pattern_top)})"
            for x in top_metrics.itertuples(index=False)
        ]
        lines.append(
            f"- ({cg}, {', '.join(metrics)}) | score_max={float(r.score_best):.4f} | votes_max={int(r.votes_max)}"
        )
    return lines


def _build_component_blocks(
    df: pd.DataFrame,
    max_components: int = 8,
    max_metrics_per_component: int = 5,
) -> list[str]:
    """Build semi-structured per-component blocks to keep component-metric binding stable."""
    if df.empty:
        return []
    work = df.copy()
    work["final_score"] = pd.to_numeric(work.get("final_score", 0.0), errors="coerce").fillna(0.0)
    work["votes"] = pd.to_numeric(work.get("votes", 1), errors="coerce").fillna(1).astype(int)
    if "pattern" not in work.columns:
        work["pattern"] = "unknown"

    def _shape_desc(p: str) -> str:
        p = str(p)
        if p in {"level_shift_up", "surge", "spike"}:
            return "up"
        if p in {"level_shift_down", "drop", "dip"}:
            return "down"
        if p == "normal":
            return "flat"
        return "unknown"

    comp_col = "evidence_component" if "evidence_component" in work.columns else "component_group"
    comp_rank = (
        work.groupby(comp_col, as_index=False)
        .agg(score_best=("final_score", "max"), votes_max=("votes", "max"), metric_cnt=("metric", "nunique"))
        .sort_values(["score_best", "votes_max", "metric_cnt"], ascending=[False, False, False])
        .head(max(1, int(max_components)))
    )
    blocks: list[str] = []
    for r in comp_rank.itertuples(index=False):
        cg = str(getattr(r, comp_col))
        g = work[work[comp_col] == cg].copy()
        top_metrics = (
            g.groupby("metric", as_index=False)
            .agg(
                score_best=("final_score", "max"),
                votes_max=("votes", "max"),
                pattern_top=("pattern", lambda s: s.value_counts().index[0] if len(s) else "unknown"),
            )
            .sort_values(["score_best", "votes_max", "metric"], ascending=[False, False, True])
            .head(max(1, int(max_metrics_per_component)))
        )
        lines = [
            f"- metric={x.metric} | pattern={x.pattern_top} | direction={_shape_desc(x.pattern_top)} | score={float(x.score_best):.4f} | votes={int(x.votes_max)}"
            for x in top_metrics.itertuples(index=False)
        ]
        block = (
            f"[COMPONENT] {cg}\n"
            f"score_max={float(r.score_best):.4f} | votes_max={int(r.votes_max)} | metric_count={int(r.metric_cnt)}\n"
            + "\n".join(lines)
        )
        blocks.append(block)
    return blocks


def _build_layer_decision_hints(df: pd.DataFrame, max_items: int = 8) -> str:
    """Build replica-consistency and layer-selection hints inspired by reference prompts."""
    if df.empty:
        return "- (none)"

    work = df.copy()
    work["component"] = work["component"].astype(str)
    work["component_group"] = work["component_group"].astype(str)
    work["final_score"] = pd.to_numeric(work.get("final_score", 0.0), errors="coerce").fillna(0.0)
    work["votes"] = pd.to_numeric(work.get("votes", 1), errors="coerce").fillna(1).astype(int)
    if "pattern" not in work.columns:
        work["pattern"] = "unknown"

    lines: list[str] = []

    pod_sub = work[work["category"] == "Pod"].copy()
    if not pod_sub.empty:
        replica_stats = (
            pod_sub.groupby("component_group", as_index=False)
            .agg(
                replica_cnt=("component", "nunique"),
                metric_cnt=("metric", "nunique"),
                score_best=("final_score", "max"),
                votes_max=("votes", "max"),
            )
            .sort_values(["replica_cnt", "score_best", "votes_max", "metric_cnt"], ascending=[False, False, False, False])
        )
        for r in replica_stats.head(max_items).itertuples(index=False):
            service = str(r.component_group)
            if int(r.replica_cnt) >= 2:
                lines.append(
                    f"- service_hint: {service} has replica-consistent pod anomalies across {int(r.replica_cnt)} replicas; "
                    "this is strong service-level evidence."
                )
            elif int(r.replica_cnt) == 1:
                lines.append(
                    f"- pod_hint: {service} currently shows pod anomalies on only 1 replica; "
                    "if peer replicas stay relatively stable, prefer pod-local / service-local interpretation over node-wide inference."
                )

    svc_sub = work[work["category"] == "Service"].copy()
    if not svc_sub.empty:
        service_stats = (
            svc_sub.groupby("component_group", as_index=False)
            .agg(metric_cnt=("metric", "nunique"), score_best=("final_score", "max"), votes_max=("votes", "max"))
            .sort_values(["score_best", "votes_max", "metric_cnt"], ascending=[False, False, False])
        )
        for r in service_stats.head(max_items).itertuples(index=False):
            if int(r.metric_cnt) >= 2:
                lines.append(
                    f"- service_hint: {str(r.component_group)} has multiple service-level metrics changing together; "
                    "this strengthens service-level explanation."
                )

    node_sub = work[work["category"] == "Node"].copy()
    if not node_sub.empty:
        node_stats = (
            node_sub.groupby("component_group", as_index=False)
            .agg(metric_cnt=("metric", "nunique"), score_best=("final_score", "max"), votes_max=("votes", "max"))
            .sort_values(["score_best", "votes_max", "metric_cnt"], ascending=[False, False, False])
        )
        for r in node_stats.head(max_items).itertuples(index=False):
            lines.append(
                f"- node_hint: {str(r.component_group)} has node-level resource anomalies, but node conclusion requires corroboration that multiple different services/pods are broadly impacted; "
                "node metrics alone are insufficient."
            )

    seen: set[str] = set()
    uniq_lines: list[str] = []
    for line in lines:
        if line not in seen:
            uniq_lines.append(line)
            seen.add(line)
    return "\n".join(uniq_lines[: max(1, int(max_items))]) if uniq_lines else "- (none)"


def _build_single_call_user_prompt(
    uuid: str,
    payloads: dict[str, str],
    must_keep_text: str,
    bundle_text: str,
    component_block_text: str,
    layer_hint_text: str,
) -> str:
    sections = "\n\n".join([f"## {c}\n{payloads[c]}" for c in ["Service", "Pod", "Node", "TiDB"]])
    return (
        f"UUID: {uuid}\n"
        f"{_metric_list_semantics_note()}\n"
        f"{_metric_meaning_note()}\n"
        "下面是当前 UUID 的指标观测结果。你的任务是做“现象总结”，不是做根因裁决。\n"
        "请客观描述观察到的变化，不要判断唯一根因，不要下结论说哪个组件一定有问题。\n"
        "缺失或为空的数据、以及 pattern=normal 的项，默认可视为波动小或相对稳定，只在需要作为对照/伴生信号时提及。\n"
        "若 pattern=missing_data，请明确指出这是数据缺失/采样缺口型异常，不要把它误写成上升或下降。\n\n"
        "必须覆盖清单（高优先级观测对象，优先描述）：\n"
        f"{must_keep_text}\n\n"
        "组件固定证据块（组件-指标绑定以这里为准，禁止跨组件串用指标）：\n"
        f"{component_block_text}\n\n"
        "层级判别辅助线索（用于区分 service / pod / node 叙事，不是最终裁决）：\n"
        f"{layer_hint_text}\n\n"
        "下面是按 Service / Pod / Node / TiDB 分层整理后的指标输入。\n"
        f"{sections}\n\n"
        "写作要求：\n"
        "1. 输入顺序只表示统计分析给出的初始优先级/先验，不是最终结论；描述时可优先覆盖前面的对象，但不要把顺序当成绝对正确。\n"
        "2. 每次提及指标时，必须绑定原始组件，不能串组件。\n"
        "3. 必须保留 pattern 和方向信息（上升/下降/平稳）；若为 missing_data，则方向写为 missing/缺失。\n"
        "4. 只描述现象、对比、共振、先后和可能受影响面，不做根因裁决。\n"
        "5. 若多个副本出现相似 pod/internal 异常，应明确写成 service-level consistency；若仅单副本异常，也要明确写出 single-replica / pod-local 特征。\n"
        "6. 写到 node 时，必须强调这是基础设施候选线索；若没有跨多个不同服务/Pod 的广泛共振线索，不要把 node 描述写得比 service/pod 更确定。\n"
        "7. 需要为后续多模态推理保留足够线索：后续会结合 Trace、Log 和领域知识来修正或确认这些统计候选。\n\n"
        "请严格按模板输出：\n"
        "## 高优先级指标现象\n"
        "- 列出不超过20条，格式建议为：(组件, 指标1(pattern|方向), 指标2(pattern|方向), ...)\n"
        "## Service级别现象\n"
        "- 描述不超过10条 service 层面的显著变化；若多个副本表现一致，要明确写出这是同一服务多个副本的一致异常\n"
        "## Pod级别现象\n"
        "- 描述不超过10条 pod/internal 指标变化；若只是单副本异常，要明确写出 peer replicas 相对稳定/未见一致异常\n"
        "## Node级别现象\n"
        "- 描述不超过10条 node 指标变化；只有在基础设施范围线索明显时才写强，避免把孤立 node 指标误写成确定性根因\n"
        "## TiDB级别现象\n"
        "- 描述不超过10条 TiDB 相关变化，保留组件和指标名\n"
        "## 可能受影响/伴生现象\n"
        "- 描述不超过10条，可包含 normal 项作为对照，但不要当成根因结论\n"
        "## 待结合Trace/Log验证的观察点\n"
        "- 描述不超过10条需要后续多模态确认的现象\n"
        "## Judge可用摘要\n"
        "- 输出一段不超过 500 字的 METRIC_EVIDENCE_SUMMARY，只总结现象，不裁决根因。\n"
        "禁止输出代码块、禁止“这个总结/以上”等解释性尾巴。"
    )


def _summarize_one_uuid(
    uuid: str,
    sub: pd.DataFrame,
    top_n_per_category: int,
    payload_max_chars: int,
    dry_run: bool,
    client: Any,
    model: str,
) -> str:
    categories = ["Service", "Pod", "Node", "TiDB"]
    must_keep_units = _build_must_keep_units(sub, max_items=6)
    bundle_lines = _build_component_bundle_lines(
        sub,
        max_components=6,
        max_metrics_per_component=4,
    )
    component_blocks = _build_component_blocks(
        sub,
        max_components=8,
        max_metrics_per_component=5,
    )
    layer_hint_text = _build_layer_decision_hints(sub, max_items=8)
    must_keep_text = "\n".join([f"- {x}" for x in must_keep_units]) if must_keep_units else "- (none)"
    bundle_text = "\n".join(bundle_lines) if bundle_lines else "- (none)"
    component_block_text = "\n\n".join(component_blocks) if component_blocks else "[COMPONENT] none"
    payloads = {
        c: _build_category_payload(
            sub,
            c,
            top_n=top_n_per_category,
            max_chars=payload_max_chars,
        )
        for c in categories
    }

    if dry_run:
        return (
            "## 高优先级指标现象\n"
            + component_block_text
            + "\n\n## Service级别现象\n"
            "- 观察 service 指标在故障窗口内的上升/下降/平稳变化。\n\n"
            "## Pod级别现象\n"
            "- 观察 pod 内部资源指标是否出现共振变化。\n\n"
            "## Node级别现象\n"
            "- 观察 node 资源或网络指标是否存在同步变化。\n\n"
            "## TiDB级别现象\n"
            "- 观察 TiDB 组件是否存在资源或时延变化。\n\n"
            "## 可能受影响/伴生现象\n"
            "- 将 normal 或较弱变化作为伴生/对照现象记录。\n\n"
            "## 待结合Trace/Log验证的观察点\n"
            "- 结合 Trace/Log 验证时序先后、调用链扩散和错误传播。\n\n"
            "## Judge可用摘要\n"
            "METRIC_EVIDENCE_SUMMARY: 以下内容仅总结该 UUID 的指标现象与变化方向，供后续结合 Trace/Log 做综合判断。"
        )

    system_prompt = (
        "你是SRE指标现象总结助手，不是最终裁决器。"
        "你的职责是把当前 UUID 的指标变化现象整理清楚，供后续多模态根因推理使用。"
        "只做现象总结、对比和观察，不做唯一根因判断。"
        "只输出模板内容，不要代码块，不要多余解释。"
    )
    user_prompt = _build_single_call_user_prompt(
        uuid,
        payloads,
        must_keep_text,
        bundle_text,
        component_block_text,
        layer_hint_text,
    )
    return _clean_llm_output(_call_yuzo(client, model, system_prompt, user_prompt))


def two_stage_metric_summary(
    input_csv: Path,
    cases_file: Path,
    output_txt: Path,
    model: str,
    api_key: str | None,
    api_url: str | None,
    top_n_per_category: int,
    payload_max_chars: int,
    first_k: int | None,
    max_workers: int,
    dry_run: bool = False,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    case_uuids = _load_uuid_order(cases_file, first_k=first_k)
    if not case_uuids:
        raise ValueError(f"no valid uuid found in {cases_file}")

    print(
        f"[init] input_csv={input_csv} cases_file={cases_file} "
        f"target_cases={len(case_uuids)} dry_run={dry_run} model={model}"
    )
    df = pd.read_csv(input_csv)
    required = {"uuid", "component", "component_group", "metric", "votes"}
    if not required.issubset(df.columns):
        raise ValueError(f"{input_csv} must include columns: {required}")

    work = df.copy()
    work["uuid"] = work["uuid"].astype(str)
    work["component"] = work["component"].astype(str)
    work["component_group"] = work["component_group"].astype(str).map(_normalize_component)
    work["metric"] = work["metric"].astype(str)
    work["votes"] = pd.to_numeric(work["votes"], errors="coerce").fillna(1).astype(int)
    if "final_score" in work.columns:
        work["final_score"] = pd.to_numeric(work["final_score"], errors="coerce").fillna(0.0)
    else:
        work["final_score"] = work["votes"].astype(float)
    if "pattern" not in work.columns:
        work["pattern"] = "unknown"
    work["category"] = work.apply(
        lambda r: _metric_category(r["component"], r["component_group"], r["metric"]), axis=1
    )
    # For pod-level evidence, keep concrete pod instance names to avoid losing pod-local signals.
    work["evidence_component"] = work.apply(
        lambda r: r["component"] if str(r["category"]) == "Pod" else r["component_group"],
        axis=1,
    )
    work = work[work["uuid"].isin(case_uuids)].copy()
    present_uuids = set(work["uuid"].unique().tolist())
    print(
        f"[init] anomaly_metric_list rows(after filter)={len(work)} "
        f"uuids_with_rows={len(present_uuids)} uuids_without_rows={len(case_uuids) - len(present_uuids)}"
    )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("", encoding="utf-8")

    client = None if dry_run else _build_client(api_key=api_key, api_url=api_url)

    all_uuids = case_uuids
    max_workers = max(1, int(max_workers))
    with output_txt.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {}
        t_starts: dict[str, float] = {}
        for idx, uuid in enumerate(all_uuids, start=1):
            sub = work[work["uuid"] == uuid].copy()
            print(
                f"[{idx}/{len(all_uuids)}] submit uuid={uuid} "
                f"rows={len(sub)} has_anomaly_rows={'yes' if len(sub) > 0 else 'no'}"
            )
            t_starts[uuid] = time.time()
            fut = ex.submit(
                _summarize_one_uuid,
                uuid,
                sub,
                top_n_per_category,
                payload_max_chars,
                dry_run,
                client,
                model,
            )
            future_map[fut] = (idx, uuid)

        done_count = 0
        next_to_write = 1
        pending: dict[int, tuple[str, str, float]] = {}
        for fut in as_completed(future_map):
            idx, uuid = future_map[fut]
            try:
                summary = fut.result()
            except Exception as exc:
                summary = (
                    "## 指标证据池（保留候选，不定责）\n"
                    f"- 生成失败：{exc}\n\n"
                    "## 待多模态验证点（Trace/Log）\n"
                    "- 需要重试该 uuid 的 summary 生成。"
                )
            elapsed = time.time() - t_starts.get(uuid, time.time())
            done_count += 1
            pending[idx] = (uuid, str(summary).strip(), elapsed)
            print(f"[{done_count}/{len(all_uuids)} done] uuid={uuid} elapsed={elapsed:.2f}s (ready)")

            # 保序写入：只按 idx 从小到大写，避免并行完成顺序打乱文件顺序
            while next_to_write in pending:
                w_uuid, w_summary, w_elapsed = pending.pop(next_to_write)
                f.write(f"# [{next_to_write}/{len(all_uuids)}] UUID: {w_uuid}\n")
                f.write(w_summary + "\n\n")
                f.flush()
                print(
                    f"[write {next_to_write}/{len(all_uuids)}] uuid={w_uuid} "
                    f"elapsed={w_elapsed:.2f}s (stream-written-ordered)"
                )
                next_to_write += 1

    print(f"saved metric summaries to {output_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage LLM summary for anomaly_metric_list.csv (Service/Pod/Node/TiDB)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input anomaly csv path")
    parser.add_argument(
        "--cases_file",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help="Case order file, default dataset/input.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output summary txt path (default: results/metric_summary.txt)",
    )
    parser.add_argument("--model", type=str, default="reasoner", help="Yuzo model name")
    parser.add_argument("--api_key", type=str, default=None, help="Yuzo API key (optional)")
    parser.add_argument("--api_url", type=str, default=None, help="Yuzo base URL (optional)")
    parser.add_argument(
        "--top_n_per_category",
        type=int,
        default=40,
        help="Top (component,metric) units kept for each category payload",
    )
    parser.add_argument(
        "--payload_max_chars",
        type=int,
        default=4200,
        help="Max payload chars per category sent to LLM",
    )
    parser.add_argument(
        "--first_k",
        type=int,
        default=400,
        help="Only process first K uuids from cases_file order (default: 400)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip LLM API calls and output deterministic local summary",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel worker count for per-uuid single-call summaries",
    )
    args = parser.parse_args()

    two_stage_metric_summary(
        input_csv=args.input,
        cases_file=args.cases_file,
        output_txt=args.output,
        model=args.model,
        api_key=args.api_key,
        api_url=args.api_url,
        top_n_per_category=max(1, int(args.top_n_per_category)),
        payload_max_chars=max(600, int(args.payload_max_chars)),
        first_k=args.first_k,
        max_workers=max(1, int(args.max_workers)),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()

# python3 unit_test/metric/root_cause/metric_summary_llm.py --dry_run