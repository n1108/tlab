"""
从 dataset/input.json 加载每条用例的故障时间窗（parse_time_range），不读取任何标注文件。

时间语义（与 exp/main 一致）：
- 描述里的 `...T..:..:..Z` 经 parse_time_range 得到「去掉 tz 的 naive datetime」，数值上等于 UTC。
- 序列化时用带 Z 的 ISO 字符串，供 LogAgent / loader 再解析。

与 parquet 路径的关系（不是漏转时区）：
- exp.utils.input.load_parquet_by_hour 会对 start/end 先 utc_to_cst，再按「北京时间」拼
  dataset/{日期}/log-parquet/log_filebeat-server_{日期}_{HH}-00-00.parquet（见 dataset/README：文件名时间为 CST）。
- 因此日志里出现 `..._2025-06-05_23-00-00.parquet` 是正常现象（例如故障前 30 分钟基线落在 UTC 15:40 附近 → CST 23:40 档）。
- 若本机缺少该小时文件，会 WARNING，与 input.json 是否「再转一次时区」无关。

与旧版 log_unit_test_dataset 的差异：
- 旧数据只含 groundtruth 里带 log 观测的 uuid；当前按 input.json 全量遍历，首条等用例
  可能以前未进入 log 单测集，从而首次触发对某些小时 parquet 的查找与缺文件告警。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp.utils.time import parse_time_range


def load_input_json_cases(project_root: Path) -> list[dict[str, Any]]:
    """
    返回与 input.json 同序的列表；每项含 uuid、start_time、end_time（ISO 字符串，UTC Z），
    解析失败时对应字段为 None 且 parse_ok=False。
    """
    path = project_root / "dataset" / "input.json"
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    out: list[dict[str, Any]] = []
    for item in rows:
        uuid = str(item.get("uuid", "") or "").strip()
        if not uuid:
            continue
        desc = str(item.get("Anomaly Description", "") or "")
        start, end = parse_time_range(desc)
        if start is None or end is None:
            out.append(
                {
                    "uuid": uuid,
                    "start_time": None,
                    "end_time": None,
                    "parse_ok": False,
                }
            )
            continue
        out.append(
            {
                "uuid": uuid,
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "parse_ok": True,
            }
        )
    return out
