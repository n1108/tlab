"""
从 dataset/groundtruth.jsonl 生成 log 单元测试数据集，写入 log_unit_test_dataset.json。
仅包含在 key_observations 中至少有一条 type 为 "log" 的故障（与 metric 脚本类似，但过滤 log 相关项）。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_PATH = PROJECT_ROOT / "dataset/groundtruth.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "unit_test/log"
OUTPUT_FILE = OUTPUT_DIR / "log_unit_test_dataset.json"


def load_jsonl(path: Path) -> list:
    print(f"Loading {path}...")
    data = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except OSError as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)
    return data


def _root_cause_components(gt: dict) -> list[str]:
    instances = gt.get("instance", [])
    if isinstance(instances, str):
        instances = [instances]
    services = gt.get("service", [])
    if isinstance(services, str):
        services = [services]
    all_c = set()
    if isinstance(instances, list):
        all_c.update(x for x in instances if x)
    if isinstance(services, list):
        all_c.update(x for x in services if x)
    return sorted(all_c)


def _extract_log_patterns(gt: dict) -> list[list[str]]:
    patterns: list[list[str]] = []
    for obs in gt.get("key_observations") or []:
        if obs.get("type") != "log":
            continue
        kw = obs.get("keyword") or []
        if isinstance(kw, str):
            kw = [kw]
        seq = [str(k).strip() for k in kw if str(k).strip()]
        if seq:
            patterns.append(seq)
    # 去重，保持顺序
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for p in patterns:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique.append(p)
    return unique


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(GT_PATH)
    test_cases: list[dict] = []

    for gt in rows:
        uuid = gt.get("uuid")
        if not uuid:
            continue
        patterns = _extract_log_patterns(gt)
        if not patterns:
            continue

        start_time = gt.get("start_time")
        end_time = gt.get("end_time")
        test_cases.append(
            {
                "uuid": uuid,
                "start_time": start_time,
                "end_time": end_time,
                "fault_type": gt.get("fault_type", "unknown"),
                "fault_category": gt.get("fault_category", ""),
                "root_cause_components": _root_cause_components(gt),
                "expected_log_patterns": patterns,
            }
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(test_cases)} log test cases to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
