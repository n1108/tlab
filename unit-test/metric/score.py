# 评分脚本，有两种评分模式
# 1. 获取单个 baseline 的输出 results/result_baseline_<method>.csv 进行评分
# 2. 对多个 baseline 的结果取并集进行评分
# 评分机制：
# 设 data/metric_dataset.json 中的某个故障（uuid）的故障组件集合为 C，故障指标集合为 M
# 该故障（uuid）检测出的异常指标列表为 List[(c0, m0), (c1, m1), ...]
# 如果 mi ∈ M 且 ci ∈ C，则认为指标 mi 检测正确，所有检测正确的指标的集合为 M_c
# 正确率为 M_c / M，这里 M_c 和 M 表示所有故障的正确检测指标数量和总的故障指标数量之和
# 评分结果输出到 results/score.csv 中作为新的一行，字段为：time, method, score
# time 为脚本运行时间，格式为 YYMMDD_HHMMSS，method 格式类似 1 或 1+2，score 为正确率，保留两位小数
