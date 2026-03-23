# metric 异常检测 baseline 运行脚本
# 运行测试后，在 unit-test/metric/results 目录下生成结果文件 result_baseline_[method].csv
# 结果文件为所有故障时间段（uuid）内检测到的所有指标异常（组件+指标）
# 文件格式为 uuid, component(node/service/pod), metric

# baseline-1: IF+HBOS+IQR, exp/agent/metric.py
# baseline-2: BOCPD（待写）


import sys
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def run_tests(limit=None, method="1", uuid=None):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--method", type=str, default="1", choices=["1", "2"], help="Anomaly detection method to use")
    parser.add_argument("--uuid", type=str, default=None, help="Run a specific test case by UUID")
    args = parser.parse_args()
    
    run_tests(limit=args.limit, method=args.method, uuid=args.uuid)

# python3 unit-test/metric/run_test.py --limit=5 --method=1