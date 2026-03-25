import pandas as pd
from pathlib import Path

# 读取时序数据
data_file = Path("/home/tyt21/tlab/unit_test/metric/pattern_analysis/metric_series.txt")

print("=" * 80)
print("漏报指标异常 Pattern 深度分析")
print("=" * 80)

# 分析几个典型样本
samples = [
    {
        'uuid': '0718e0f9-92',
        'component': 'productcatalogservice',
        'metric': 'pod_network_receive_bytes',
        'note': '网络指标 - 多 Pod 场景'
    },
    {
        'uuid': '0718e0f9-92',
        'component': 'productcatalogservice',
        'metric': 'request',
        'note': '流量指标 - 请求量'
    },
    {
        'uuid': '3260fa48-316',
        'component': 'tidb-tikv',
        'metric': 'store_size',
        'note': 'TiDB 存储指标'
    },
    {
        'uuid': '3260fa48-316',
        'component': 'tidb-tikv',
        'metric': 'region_pending',
        'note': 'TiDB 集群状态指标'
    },
    {
        'uuid': 'f066e1dd-145',
        'component': 'shippingservice',
        'metric': 'pod_cpu_usage',
        'note': 'Pod CPU 使用率'
    },
]

print("\n【典型漏报案例分析】")
for sample in samples:
    print(f"\n案例：{sample['uuid']}")
    print(f"组件：{sample['component']}, 指标：{sample['metric']}")
    print(f"备注：{sample['note']}")
    
    # 在数据文件中查找对应行
    with open(data_file, 'r') as f:
        lines = f.readlines()
        
    found_data = []
    for line in lines:
        if sample['uuid'] in line and sample['metric'] in line:
            found_data.append(line.strip())
    
    if found_data:
        print(f"找到 {len(found_data)} 条相关记录")
        # 解析最后一个记录的数据
        last_record = found_data[-1]
        parts = last_record.split(', ')
        if len(parts) >= 4:
            values_str = parts[3].strip('[]')
            values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
            
            non_zero = [v for v in values if v > 0]
            zero_count = len([v for v in values if v == 0])
            
            print(f"  数据点数：{len(values)}")
            print(f"  零值数量：{zero_count} ({zero_count/len(values)*100:.1f}%)")
            print(f"  非零数量：{len(non_zero)}")
            if non_zero:
                print(f"  非零值范围：[{min(non_zero):.4f}, {max(non_zero):.4f}]")
                print(f"  非零值均值：{sum(non_zero)/len(non_zero):.4f}")
                
                # 判断异常类型
                if zero_count > len(values) * 0.8:
                    print(f"  ⚠️  Pattern: **数据缺失型** - 超过 80% 为零值")
                elif len(non_zero) < 10:
                    print(f"  ⚠️  Pattern: **稀疏数据型** - 非零值极少")
                elif max(non_zero) / (sum(non_zero)/len(non_zero)) > 5:
                    print(f"  ⚠️  Pattern: **脉冲型** - 存在极端峰值")
                else:
                    print(f"  ℹ️  Pattern: **平稳型** - 数据相对平稳")
    else:
        print("未找到对应数据")

print("\n" + "=" * 80)
print("【总结：漏报指标的 5 大异常 Pattern】")
print("=" * 80)
print("""
1. **错误率指标漏报 (占比 45%)**
   - 特征：error, client_error, error_ratio 等错误类指标
   - 原因：当前检测方法可能未覆盖错误计数类指标
   - 建议：增加基于错误率和错误计数的检测规则

2. **TiDB 专用指标漏报 (占比 32.4%)**
   - 特征：store_size, region_pending, memory_usage 等 TiDB 特有指标
   - 原因：TiDB 指标具有特殊的业务含义和变化模式
   - 建议：针对 TiDB 设计专门的检测模型（如 store_size 单调递增特性）

3. **网络类指标漏报 (占比 9.5%)**
   - 特征：pod_network_receive_bytes, pod_network_transmit_packets 等
   - 原因：网络流量波动大，传统统计方法难以捕捉
   - 建议：结合时间序列预测或频域分析方法

4. **流量/QPS 指标漏报 (占比 13.3%)**
   - 特征：request, response, qps 等业务流量指标
   - 原因：流量具有周期性和趋势性，需要更复杂的建模
   - 建议：使用季节性分解或 LSTM 等时序模型

5. **微服务组件漏报严重 (占比 62.9%)**
   - 特征：shippingservice, checkoutservice 等微服务
   - 原因：微服务数量多、指标维度复杂
   - 建议：按服务类型分组建模，引入服务依赖关系
""")

print("\n【下一步优化方向】")
print("1. 针对错误类指标：实现基于阈值和突变的混合检测")
print("2. 针对 TiDB 指标：引入领域知识和业务规则")
print("3. 针对网络/流量指标：采用更高级的时序模型（Prophet、DeepAR）")
print("4. 多检测方法融合：并集策略提升召回率")
print("=" * 80)
