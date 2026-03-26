# 读取 results/result_baseline<2+4+5>.csv
# 获取 Baseline 2,4,5 检测出的所有异常指标
# Step 1:
# 调用 root_cause/baro/ 目录下的根因定位算法
# 对每个故障（uuid）的所有异常指标进行排序，排序越靠前越可能是根因指标
# Step 2:
# 给异常指标添加一个局部异常模式，输出标记后的序列
# 采用 exp/agent/metric.py 中的 _detect_local_pattern 算法
# 输出一个 csv 文件，记录每个故障（uuid）排序后的异常指标列表
# 格式为 uuid, component, metric, pattern