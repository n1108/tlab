# baseline6（PATE-inspired）

该 baseline 从 `../PATE` 中抽取了可直接复用的轻量工具函数（事件区间转换、buffer 点生成），并实现了一个最小可跑的 metric 异常检测流程。

## 引入的 PATE 代码

- 来源：`/home/tyt21/PATE/pate/PATE_utils.py`
- 复用函数：
  - `convert_vector_to_events_PATE`
  - `generate_buffer_points`
- 许可证：MIT（PATE 上游为 MIT）

实现文件：

- `pate_utils.py`（复制并精简后的工具函数）
- `run_baseline6.py`（检测主流程）

## 检测思路

1. 按 `(pod, kpi_key)` 构建 1 分钟重采样序列；
2. 计算 robust z-score（MAD）得到逐点异常分数；
3. 根据分位数阈值把分数二值化；
4. 使用 PATE 的 buffer 思路做邻域扩张；
5. 使用 `convert_vector_to_events_PATE` 转为区间事件；
6. 存在有效事件则输出该 `(uuid, component, metric)`。

## 接口对齐

与其他 baseline 保持一致：

- 输入：`unit_test/metric/data/metric_dataset.json`
- 输出字段：`uuid, component, metric`
- 输出文件：`unit_test/metric/results/result_baseline6.csv`
- 命令行参数：
  - `--limit`
  - `--uuid`

## 运行

```bash
python3 unit_test/metric/baselines/baseline6/run_baseline6.py --limit=5
```

或指定单个 case：

```bash
python3 unit_test/metric/baselines/baseline6/run_baseline6.py --uuid 345fbe93-80
```
