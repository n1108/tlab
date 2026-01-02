import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ROOT_DIR = Path("phaseone")
# START_DATE = datetime.strptime("2025-06-06", "%Y-%m-%d")
# END_DATE = datetime.strptime("2025-06-14", "%Y-%m-%d")

# ROOT_DIR = Path("phasetwo")
# START_DATE = datetime.strptime("2025-06-17", "%Y-%m-%d")
# END_DATE = datetime.strptime("2025-06-29", "%Y-%m-%d")

ROOT_DIR = Path("dataset")
START_DATE = datetime.strptime("2025-06-06", "%Y-%m-%d")
END_DATE = datetime.strptime("2025-06-29", "%Y-%m-%d")


def daterange(start_date: datetime, end_date: datetime):
    for n in range((end_date - start_date).days + 1):
        yield start_date + timedelta(n)


def preprocess_parquet(file_path: Path) -> None:
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logging.error(f"Read file error: {file_path}: {e}")
        return
    column = "@timestamp" if file_path.name.startswith("log_") else "time"
    if column not in df.columns:
        logging.warning(f"Skip {file_path.name}, less {column} column")
        return

    # 统一转换为 UTC 时间戳
    ts = pd.to_datetime(df[column], errors="coerce", utc=True)
    
    # 如果全是空，可能是 Unix 时间戳格式
    if ts.isnull().all():
        ts = pd.to_datetime(df[column], errors="coerce", unit="s", utc=True)

    if ts.isnull().any():
        logging.warning(f"{file_path.name} contains invalid timestamps, dropping rows")

    # 去除时区信息（变为 naive timestamp），以兼容某些 PyArrow 版本写入
    df[column] = ts.dt.tz_localize(None)

    try:
        # 使用 PyArrow 引擎写回，确保类型被正确保存
        df.to_parquet(file_path, engine="pyarrow", allow_truncated_timestamps=True)
        logging.info(f"✅ processed {file_path.name} successfully")
    except Exception as e:
        logging.error(f"Failed to write {file_path.name}: {e}")


def process_directory_for_date(date: datetime, subdirs: list[str]) -> None:
    day_str = date.strftime("%Y-%m-%d")
    for subdir in subdirs:
        dir_path = ROOT_DIR / day_str / subdir
        # 兼容路径不存在的情况
        if not dir_path.exists():
            # logging.debug(f"Skip: {dir_path} (not found)")
            continue

        parquet_files = list(dir_path.glob("*.parquet"))
        if not parquet_files:
            continue

        logging.info(f"Processing directory: {dir_path}")
        for file_path in parquet_files:
            preprocess_parquet(file_path)


def main():
    service_dirs = ["metric-parquet/apm/service"]
    
    # 添加 infra_node 和 infra_tidb
    infra_dirs = [
        "metric-parquet/infra/infra_pod",
        "metric-parquet/infra/infra_node", 
        "metric-parquet/infra/infra_tidb"
    ]
    
    other_dirs = ["metric-parquet/other"]
    log_dirs = ["log-parquet"]

    # 遍历日期范围进行处理
    for date in daterange(START_DATE, END_DATE):
        process_directory_for_date(date, service_dirs + infra_dirs + other_dirs + log_dirs)


if __name__ == "__main__":
    main()