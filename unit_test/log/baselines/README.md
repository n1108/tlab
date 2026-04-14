# Log baselines（LightAD 移植）

本目录包含对 [LightAD](https://github.com/BoxiYu/LightAD)（ICSE'24）中 **KNN / 决策树 / SLFN（`sklearn` MLPClassifier）** 分类器的代码移植与适配脚本。

| 文件 | 说明 |
|------|------|
| `lightad_classifiers.py` | 与上游 `models/classifiers.py` 一致的 `KNN` / `decision_tree` / `MLP` 封装。 |
| `loader.py` | 从 tlab `dataset/.../log-parquet` 按时间窗加载日志行（不过滤 ERROR 关键词，供向量化）。 |
| `run_comparison.py` | 在 `log_unit_test_dataset.json` 上训练弱监督分类器，与 **LogAgent** 用同一关键词召回指标对比，并生成 `results/baseline_comparison_report.md`。 |

上游论文引用：

```bibtex
@inproceedings{10.1145/3597503.3623308,
  author = {Yu, Boxi and Yao, Jiayi and Fu, Qiuai and others},
  title = {Deep Learning or Classical Machine Learning? An Empirical Study on Log-Based Anomaly Detection},
  year = {2024},
  booktitle = {ICSE '24}
}
```

运行示例（在仓库根目录）：

```bash
# 全量 225 条用例（默认 --limit-uuids 0）
python -m unit_test.log.baselines.run_comparison --dataset-root dataset

# 子集快速试跑
python -m unit_test.log.baselines.run_comparison --limit-uuids 40
```
