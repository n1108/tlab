import pandas as pd
from pathlib import Path
from collections import Counter

# 读取漏报数据
missed_file = Path("/home/tyt21/tlab/unit_test/metric/results/missed_anomalies.csv")
df_missed = pd.read_csv(missed_file)

print("=" * 80)
print("漏报异常特征分析报告")
print("=" * 80)

# 1. 按 metric 类型统计
print("\n【1. 按 Metric 类型分布】")
metric_counts = df_missed['component.metric'].apply(lambda x: x.split('.')[-1]).value_counts()
print(f"总漏报数：{len(df_missed)}")
print(f"不同 metric 数量：{len(metric_counts)}")
print("\nTop 15 漏报 metric:")
for metric, count in metric_counts.head(15).items():
    print(f"  {metric:40s} {count:4d} ({count/len(df_missed)*100:.1f}%)")

# 2. 按组件类型统计
print("\n【2. 按 Component 类型分布】")
component_counts = df_missed['component.metric'].apply(lambda x: x.split('.')[0]).value_counts()
print(f"受影响组件数：{len(component_counts)}")
print("\nTop 10 漏报组件:")
for comp, count in component_counts.head(10).items():
    print(f"  {comp:40s} {count:4d} ({count/len(df_missed)*100:.1f}%)")

# 3. 按 UUID 统计（哪些故障案例漏报最严重）
print("\n【3. 按 UUID 分布（漏报最严重的案例）】")
uuid_counts = df_missed['uuid'].value_counts()
print(f"涉及 UUID 数量：{len(uuid_counts)}")
print("\nTop 10 漏报 UUID:")
for uuid, count in uuid_counts.head(10).items():
    print(f"  {uuid:40s} {count:4d} 个指标漏报")

# 4. Metric 类型分类统计
print("\n【4. Metric 类型分类统计】")
def categorize_metric(metric_name):
    if 'error' in metric_name.lower():
        return '错误类 (Error)'
    elif 'network' in metric_name.lower():
        return '网络类 (Network)'
    elif 'memory' in metric_name.lower() or 'storage' in metric_name.lower() or 'store_size' in metric_name.lower():
        return '内存/存储类 (Memory/Storage)'
    elif 'cpu' in metric_name.lower():
        return 'CPU 类'
    elif 'process' in metric_name.lower():
        return '进程类 (Process)'
    elif 'qps' in metric_name.lower() or 'request' in metric_name.lower() or 'response' in metric_name.lower():
        return '流量类 (QPS/Request)'
    elif 'region' in metric_name.lower() or 'leader' in metric_name.lower():
        return 'TiDB 集群类 (Region/Leader)'
    else:
        return '其他 (Other)'

df_missed['metric_category'] = df_missed['component.metric'].apply(lambda x: categorize_metric(x.split('.')[-1]))
category_counts = df_missed['metric_category'].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat:30s} {count:4d} ({count/len(df_missed)*100:.1f}%)")

# 5. 组件类型分类
print("\n【5. 组件类型分类】")
def categorize_component(comp_name):
    if 'tidb' in comp_name.lower():
        return 'TiDB 数据库组件'
    elif 'service' in comp_name.lower():
        return '微服务组件'
    elif 'redis' in comp_name.lower():
        return '缓存组件'
    elif 'node' in comp_name.lower():
        return '节点级指标'
    else:
        return '其他'

df_missed['comp_category'] = df_missed['component.metric'].apply(lambda x: categorize_component(x.split('.')[0]))
comp_cat_counts = df_missed['comp_category'].value_counts()
for cat, count in comp_cat_counts.items():
    print(f"  {cat:30s} {count:4d} ({count/len(df_missed)*100:.1f}%)")

print("\n" + "=" * 80)
print("关键发现:")
print("1. 错误类指标 (error, client_error, error_ratio) 是漏报重灾区")
print("2. TiDB 相关指标 (store_size, memory_usage, region_pending) 大量漏报")
print("3. 网络类指标 (pod_network_*) 普遍漏报")
print("4. 微服务组件比基础设施组件漏报更多")
print("=" * 80)
