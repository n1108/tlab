"""
从 dataset parquet 加载时间窗内日志行，供 baseline 向量化（不过滤 ERROR_KEYWORDS，与 LogAgent 公平对比信息量）。
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

from exp.utils.input import load_parquet_by_hour

logger = logging.getLogger(__name__)


def _extract_message_text(raw_message: str | None) -> str | None:
    if raw_message is None:
        return None
    log_content = raw_message
    if raw_message.strip().startswith("{"):
        try:
            log_json = json.loads(raw_message)
            if "error" in log_json and log_json["error"]:
                log_content = log_json["error"]
            elif "message" in log_json:
                log_content = log_json["message"]
        except json.JSONDecodeError:
            pass
    if not isinstance(log_content, str):
        log_content = str(log_content)
    s = log_content.strip()
    return s if s else None


def load_window_lines(
    root_path: str,
    start: datetime,
    end: datetime,
    *,
    max_workers: int = 4,
) -> pd.DataFrame:
    """加载 [start, end] 内所有日志行，列含 k8_pod, text_line, @timestamp。"""
    fields = [
        "k8_namespace",
        "@timestamp",
        "agent_name",
        "k8_pod",
        "message",
        "k8_node_name",
    ]

    def callback(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["k8_pod", "k8_node_name"]:
            if col in df.columns:
                df[col] = df[col].replace("null", None)
        df["text_line"] = df["message"].apply(_extract_message_text)
        df = df.dropna(subset=["text_line"])
        return df

    pa_start = pa.scalar(start, type=pa.timestamp("ms", tz="UTC"))
    pa_end = pa.scalar(end, type=pa.timestamp("ms", tz="UTC"))
    filter_expression = (
        ds.field("@timestamp").cast(pa.timestamp("ms", tz="UTC")) >= pa_start
    ) & (ds.field("@timestamp").cast(pa.timestamp("ms", tz="UTC")) <= pa_end)

    return load_parquet_by_hour(
        start,
        end,
        root_path,
        file_pattern="{dataset}/{day}/log-parquet/log_filebeat-server_{day}_{hour}-00-00.parquet",
        load_fields=fields,
        return_fields=fields + ["text_line"],
        filter_=filter_expression,
        callback=callback,
        max_workers=max_workers,
    )


def baseline_fault_windows(start: datetime, end: datetime) -> tuple[datetime, datetime, datetime, datetime]:
    """(baseline_start, baseline_end, fault_start, fault_end)；与 LogAgent.score 一致。"""
    baseline_duration = timedelta(minutes=30)
    baseline_start = start - baseline_duration
    return baseline_start, start, start, end
