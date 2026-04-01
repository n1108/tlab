# 多模态 RCA 微调模板

## 目标
本文档用于整理适合监督微调的大模型训练样本，面向当前数据集中的多模态根因分析任务。

适用输入通常包括：
- 故障描述，
- 指标证据摘要，
- trace 证据，
- log 证据，
- 可选的拓扑 / 部署上下文。

期望模型学会：
- 识别正确的根因组件，
- 给出正确的故障类型和推理路径，
- 区分根因与下游受害者，
- 正确选择层级：service / pod / node。

## 推荐训练样本结构

每个监督样本建议包含：

```json
{
  "uuid": "case uuid",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "自然语言故障描述",
    "metrics": "预计算好的指标摘要文本",
    "traces": "trace 观测",
    "logs": "log 观测",
    "topology": "可选的拓扑 / 部署说明"
  },
  "output": {
    "component": "最终根因组件",
    "fault_type": "标准化故障类型",
    "layer": "service | pod | node",
    "reason": "简短最终理由",
    "reasoning_steps": [
      "step 1 ...",
      "step 2 ...",
      "step 3 ...",
      "step 4 ..."
    ]
  }
}
```

## 输出风格建议

- `component` 应与评测使用的最终答案组件一致。
- `fault_type` 尽量贴近数据集语义，例如：
  - `cpu stress`
  - `memory stress`
  - `pod failure`
  - `pod kill`
  - `node cpu stress`
  - `node memory stress`
  - `node disk fill`
  - `network corrupt`
  - `network loss`
  - `network delay`
  - `jvm cpu`
  - `jvm gc`
  - `jvm latency`
  - `jvm exception`
  - `code error`
  - `dns error`
  - `target port misconfig`
  - `io fault`
- `layer` 要和 groundtruth 中的组件层级一致。
- `reasoning_steps` 强调证据对齐，不要写成很长的故事。

## 样本构造原则

### 1. 指标要保留结构化信息
- 对 service / pod 压力类故障，保留直接内部指标，例如：
  - `pod_cpu_usage`
  - `pod_processes`
  - `pod_memory_working_set_bytes`
- 对 node 故障，保留直接 node 指标，例如：
  - `node_cpu_usage_rate`
  - `node_memory_usage_rate`
  - `node_memory_MemAvailable_bytes`
  - `node_filesystem_usage_rate`

### 2. 网络故障必须保留方向
- 对 `network corrupt / loss / delay`，必须显式保留：
  - `source`
  - `destination`
  - 关键调用侧 traces/logs
- 不要把模型训练成默认总选 destination service。

### 3. 日志语义要保留原味
- 对 `pod failure / pod kill / code error / dns error / jvm*`，日志往往比指标更直接定义故障类型。
- 应保留有判别力的关键词，例如：
  - `Connection refused`
  - `Error while dialing`
  - `timeout`
  - `unavailable`
  - `deadlineexceeded`
  - `FailedPrecondition`
  - `InvocationTargetException`
  - `TransformListener`
  - `adservice--gc`
  - `adservice--stress`

### 4. 层级语义要清楚
- service：多个副本呈现相似证据
- pod：只有一个具体实例局部异常
- node：node 指标是最直接的强证据

## 可选的细粒度指标抽取

如果某些训练样本需要更细的指标序列，可以用：

```bash
conda run -n agent python3 "unit_test/metric/pattern_analysis/extract_metric.py" \
  --uuid <UUID> \
  --component <COMPONENT> \
  --metric <METRIC> \
  --extend <MINUTES>
```

说明：
- 需要在 `agent` conda 环境里运行。
- 时序值会写到 `metric-series.txt`。
- 图像会输出到 `pattern-analysis/img/`。
- 适合给训练样本补充类似“持续上升”“尖峰”“故障窗口内消失”这类更细的现象描述。

## 高质量训练样例

## 样例 1：Service 级 CPU Stress

Groundtruth：
- `uuid`: `345fbe93-80`
- `fault_type`: `cpu stress`
- `component`: `emailservice`

训练样本：

```json
{
  "uuid": "345fbe93-80",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "系统在故障窗口内出现异常。",
    "metrics": "emailservice 在多个副本上都出现 pod_processes 和 pod_cpu_usage 上升，rrt/rrt_max 也相关；同时可能存在泛化的 node CPU 波动，但特异性较弱。",
    "traces": "没有更强的调用链证据表明其他组件才是源头。",
    "logs": "没有决定性的应用失败日志。",
    "topology": "emailservice 是结账链路中的下游服务。"
  },
  "output": {
    "component": "emailservice",
    "fault_type": "cpu stress",
    "layer": "service",
    "reason": "跨副本一致的 pod_cpu_usage 和 pod_processes 上升直接指向 emailservice 的 service 级 CPU 压力。",
    "reasoning_steps": [
      "service 内部指标是最直接证据。",
      "异常在多个副本上同时出现，因此 service 级解释强于 pod 局部解释。",
      "node CPU 波动范围更广，但不如 emailservice 内部指标具体。",
      "最终选择 emailservice。"
    ]
  }
}
```

## 样例 2：带方向的 Network Corrupt

Groundtruth：
- `uuid`: `74a44ae7-81`
- `fault_type`: `network corrupt`
- `source`: `checkoutservice`
- `destination`: `shippingservice`

训练样本：

```json
{
  "uuid": "74a44ae7-81",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "checkoutservice 与 shippingservice 之间出现异常。",
    "metrics": "shippingservice 一侧可见 memory 增长、client_error_ratio 上升，以及 request/response、rrt 类服务症状。",
    "traces": "checkoutservice/frontend 到 shippingservice 的路径上出现 timeout 类异常。",
    "logs": "日志包含 dialing failure、timeout、unavailable、canceled 等语义。",
    "topology": "checkoutservice 调用 shippingservice。"
  },
  "output": {
    "component": "checkoutservice",
    "fault_type": "network corrupt",
    "layer": "service",
    "reason": "调用侧到被调侧的 timeout / dialing failure 语义更符合 source 侧网络损坏故障。",
    "reasoning_steps": [
      "这是网络故障场景，因此 source/destination 方向很关键。",
      "trace 和 log 的方向性证据强于 destination 一侧的资源症状。",
      "shippingservice 的报错更可能是下游受害现象，而不是根因。",
      "最终选择 checkoutservice。"
    ]
  }
}
```

## 样例 3：Service 级 Pod Failure

Groundtruth：
- `uuid`: `38ee3d45-82`
- `fault_type`: `pod failure`
- `component`: `cartservice`

训练样本：

```json
{
  "uuid": "38ee3d45-82",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "购物车相关请求出现失败。",
    "metrics": "cartservice 的 request/response 相关指标在故障窗口内消失或明显不完整。",
    "traces": "frontend 与 checkoutservice 到 cartservice 的调用出现 connection-refused 风格异常。",
    "logs": "日志反复出现 connection refused / unavailable 语义。",
    "topology": "frontend 和 checkoutservice 都依赖 cartservice。"
  },
  "output": {
    "component": "cartservice",
    "fault_type": "pod failure",
    "layer": "service",
    "reason": "连接被拒绝加上服务可用性消失，说明 cartservice 发生 pod / service 不可用故障。",
    "reasoning_steps": [
      "这是典型可用性故障场景。",
      "缺失指标本身较弱，但和 connection refused 结合后就成为强不可用证据。",
      "上游调用侧只是观察到 cartservice 不可达。",
      "最终选择 cartservice，并归为 pod failure。"
    ]
  }
}
```

## 样例 4：Node Memory Stress

Groundtruth：
- `uuid`: `b1ab098d-83`
- `fault_type`: `node memory stress`
- `component`: `aiops-k8s-06`

训练样本：

```json
{
  "uuid": "b1ab098d-83",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "故障窗口内出现 node 级异常。",
    "metrics": "aiops-k8s-06 的 node_memory_usage_rate / node_memory_MemAvailable_bytes 异常是最直接证据。",
    "traces": "没有更强的 service 专属故障路径。",
    "logs": "没有更强的应用级失败语义。",
    "topology": "可能有多个业务服务受到次生影响。"
  },
  "output": {
    "component": "aiops-k8s-06",
    "fault_type": "node memory stress",
    "layer": "node",
    "reason": "直接 node 内存指标最强、最具体，能够直接定义故障。",
    "reasoning_steps": [
      "node 内存指标直接匹配标签语义。",
      "service 级异常更像次生现象，特异性更弱。",
      "trace/log 没有提供更强的竞争性 service 根因。",
      "最终选择 aiops-k8s-06。"
    ]
  }
}
```

## 样例 5：Pod 级 Memory Stress

Groundtruth：
- `uuid`: `31392fda-93`
- `fault_type`: `memory stress`
- `component`: `shippingservice-0`

训练样本：

```json
{
  "uuid": "31392fda-93",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "shippingservice 出现局部内存异常。",
    "metrics": "只有一个 shippingservice 副本出现 pod_memory_working_set_bytes 上升及相关局部信号。",
    "traces": "可能有下游影响，但没有证据表明所有副本都同时异常。",
    "logs": "没有更强的跨副本失败语义。",
    "topology": "shippingservice 是 checkoutservice 的下游。"
  },
  "output": {
    "component": "shippingservice-0",
    "fault_type": "memory stress",
    "layer": "pod",
    "reason": "单副本内存上升说明这是 pod 局部 memory stress，而不是 service 级问题。",
    "reasoning_steps": [
      "异常只集中在一个副本。",
      "没有足够证据支持 service 级 memory stress。",
      "node 级解释范围更大，但不够具体。",
      "最终选择 shippingservice-0。"
    ]
  }
}
```

## 样例 6：JVM CPU Fault

Groundtruth：
- `uuid`: `20622318-94`
- `fault_type`: `jvm cpu`
- `component`: `adservice`

训练样本：

```json
{
  "uuid": "20622318-94",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "adservice 出现 JVM 相关异常。",
    "metrics": "adservice 有 pod_cpu_usage 上升，并伴随延迟类服务症状。",
    "traces": "frontend 路径在 adservice 下游出现延迟上升。",
    "logs": "日志包含 adservice--stress、InvocationTargetException、TransformListener 等 JVM 特征。",
    "topology": "frontend 会调用 adservice。"
  },
  "output": {
    "component": "adservice",
    "fault_type": "jvm cpu",
    "layer": "service",
    "reason": "JVM stress 日志与 pod_cpu_usage 一起直接指向 adservice 的 JVM CPU 故障。",
    "reasoning_steps": [
      "JVM 子类型主要由日志语义定义。",
      "adservice 自身有直接 pod_cpu_usage 支撑。",
      "trace 延迟与 adservice 向下游传播的影响一致。",
      "最终选择 adservice。"
    ]
  }
}
```

## 样例 7：DNS Error

Groundtruth：
- `uuid`: `18b8cf77-318`
- `fault_type`: `dns error`
- `component`: `checkoutservice`

训练样本：

```json
{
  "uuid": "18b8cf77-318",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "checkoutservice 无法正确访问 productcatalogservice。",
    "metrics": "checkoutservice 的 error/server_error 类指标上升。",
    "traces": "frontend 侧可能有次生影响，但不是根因位置。",
    "logs": "出现解析失败风格的请求错误语义。",
    "topology": "checkoutservice 依赖 productcatalogservice。"
  },
  "output": {
    "component": "checkoutservice",
    "fault_type": "dns error",
    "layer": "service",
    "reason": "调用侧解析失败语义说明是 checkoutservice 的 DNS 错误。",
    "reasoning_steps": [
      "DNS 故障通常应判给发起查找失败的调用方。",
      "被调用服务本身不是主要根因对象。",
      "error/server_error 指标支持调用侧解析失败。",
      "最终选择 checkoutservice。"
    ]
  }
}
```

## 样例 8：Target Port Misconfiguration

Groundtruth：
- `uuid`: `a42d2eb4-319`
- `fault_type`: `target port misconfig`
- `component`: `emailservice`

训练样本：

```json
{
  "uuid": "a42d2eb4-319",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "服务端口配置错误导致通信失败。",
    "metrics": "emailservice 的 request/response 行为异常。",
    "traces": "调用路径失败，但不像普通的 packet loss / delay。",
    "logs": "语义更符合 target port 不可达或端口映射错误。",
    "topology": "emailservice 在结账链路中提供服务。"
  },
  "output": {
    "component": "emailservice",
    "fault_type": "target port misconfig",
    "layer": "service",
    "reason": "端口级 service 配置错误可以直接解释 request/response 异常。",
    "reasoning_steps": [
      "这是配置错误场景，不应被归成普通网络抖动。",
      "发生错误的对象本身就是 service 配置。",
      "request/response 异常与错误 target port 完全一致。",
      "最终选择 emailservice。"
    ]
  }
}
```

## 样例 9：TiKV IO Fault

Groundtruth：
- `uuid`: `332adc3a-317`
- `fault_type`: `io fault`
- `component`: `tidb-tikv`

训练样本：

```json
{
  "uuid": "332adc3a-317",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "存储层 IO 故障影响 TiKV。",
    "metrics": "tidb-tikv 出现 io_util、region_pending、raft_apply_wait、snapshot/store 等存储侧指标异常，并伴随 memory 相关变化。",
    "traces": "productcatalogservice 路径出现延迟上升，但更像下游影响。",
    "logs": "没有更强的普通业务服务级应用错误。",
    "topology": "业务服务依赖 TiDB 存储链路。"
  },
  "output": {
    "component": "tidb-tikv",
    "fault_type": "io fault",
    "layer": "pod",
    "reason": "TiKV 专属 IO 指标直接定义了存储 IO 故障位置。",
    "reasoning_steps": [
      "存储侧指标是最直接的模态。",
      "trace 延迟只是下游受影响现象，不是根因位置。",
      "不能把这种故障回推到普通微服务。",
      "最终选择 tidb-tikv。"
    ]
  }
}
```

## 样例 10：Code Error

Groundtruth：
- `uuid`: `50bce1c4-311`
- `fault_type`: `code error`
- `component`: `cartservice`

训练样本：

```json
{
  "uuid": "50bce1c4-311",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "cartservice 的代码变更导致运行时失败。",
    "metrics": "cartservice 周围出现 request/response 与 process/cpu 相关症状。",
    "traces": "frontend 出现下游受影响现象。",
    "logs": "日志包含 FailedPrecondition、request error、rpc error 以及数据库失败相关语义。",
    "topology": "frontend 依赖 cartservice。"
  },
  "output": {
    "component": "cartservice",
    "fault_type": "code error",
    "layer": "service",
    "reason": "应用错误日志直接表明 cartservice 存在代码级故障。",
    "reasoning_steps": [
      "code error 主要由日志语义定义。",
      "错误关键词是 cartservice 逻辑本身产生的。",
      "metrics 与 traces 主要体现的是下游影响。",
      "最终选择 cartservice。"
    ]
  }
}
```

## 建议的微调样本覆盖面

建议均衡覆盖：
- service 级 stress
- pod 级 stress
- service 级 pod failure
- pod 级 pod failure
- node CPU / memory / disk 故障
- network corrupt / loss / delay
- JVM cpu / gc / latency / exception
- DNS error
- target port misconfiguration
- code error
- TiDB pod failure / TiKV IO fault

## 最后建议

不要只训练模型学“答案格式”。
更重要的是让它学会：
- 方向性：`source` 与 `destination`
- 层级特异性：`service` / `pod` / `node`
- 模态分工：metrics / traces / logs 各自负责什么
- 故障语义：资源压力、可用性故障、网络、配置、JVM、IO 如何区分
