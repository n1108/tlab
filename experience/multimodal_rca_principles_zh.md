# 本数据集的多模态 RCA 推理原则

## 文档目的
本文档基于 `dataset/groundtruth.jsonl` 的完整标注，总结本数据集中的多模态根因分析经验原则。
目标是为后续的 prompt 设计、证据组织、层级判定和最终 RCA 裁决提供一个稳定的参考基线。

核心思想是：
- 指标负责提供结构化异常候选，
- Trace 负责描述传播路径、延迟变化和请求比例变化，
- Log 负责提供更强的故障语义，
- 专家知识负责把这些现象映射到正确的故障类型和故障层级。

## 推荐推理流程
建议按如下顺序进行分析：

1. 先判断当前更像哪一类故障场景。
2. 再判断最可能的层级：`service -> pod -> node`。
3. 区分根因和下游受害者。
4. 确认最终选择的组件是否对应“最直接”的证据，而不是“最宽泛”的解释。

这里的 `service -> pod -> node` 是推理顺序，不是死板的优先级。
最终答案应由整体多模态一致性决定。

## 数据集覆盖校验
为了避免只根据前部样本形成偏见，这里补充整份 `groundtruth` 的覆盖分布摘要：

- 总样本数：`400`
- `fault_category` 分布：
  - `node fault`: `82`
  - `network attack`: `73`
  - `pod fault`: `60`
  - `jvm fault`: `55`
  - `stress test`: `42`
  - `io fault`: `28`
  - `erroneous change`: `21`
  - `dns fault`: `21`
  - `misconfiguration`: `18`
- `instance_type` 分布：
  - `service`: `195`
  - `pod`: `123`
  - `node`: `82`
- `fault_type` 的高频项包括：
  - `pod failure`: `45`
  - `node memory stress`: `38`
  - `io fault`: `28`
  - `network corrupt`: `27`
  - `network delay`: `25`
  - `node cpu stress`: `23`
  - `cpu stress`: `22`
  - `network loss`: `21`
  - `node disk fill`: `21`
  - `code error`: `21`
  - `dns error`: `21`
  - `memory stress`: `20`
  - `target port misconfig`: `18`
  - `pod kill`: `15`
  - `jvm latency`: `15`
  - `jvm gc`: `14`
  - `jvm cpu`: `13`
  - `jvm exception`: `13`

这说明本文档不能只覆盖前几个 HipsterShop 常见样例，还必须覆盖：
- TiDB/TiKV 专项故障，
- `pod` 级与 `service` 级混合标注，
- `dns error` / `target port misconfig` 这类更偏语义型而非资源型故障，
- JVM 四个细分子类型。

## 跨场景通用原则

### 1. 优先使用直接证据，而不是宽泛证据
- 像 `pod_cpu_usage`、`pod_processes`、`pod_memory_working_set_bytes` 这样的 service/pod 内部指标，通常比泛化的 node CPU/内存波动更具体。
- 单独的 node 指标不应轻易覆盖已经存在的 service/pod 直接内部证据，除非 traces/logs 也同时支持该 node。
- 如果有多个解释候选，应优先选择与故障语义更直接匹配的那个。

### 2. 明确区分根因与受害者
- 出现 timeout/error 的下游服务，经常只是受害者，尤其在网络故障场景下。
- 真正的根因组件，通常是最先表现出直接内部异常或明确失败语义的组件。
- caller->callee 的错误路径，并不自动意味着 callee 就是根因。

### 3. `missing_data` 单独很弱，但与不可用信号结合时很强
- `missing_data` 或采样缺口，单独不足以作为根因依据。
- 但如果 `missing_data` 和 `connection refused`、重启、请求/响应消失等现象同时出现，那就很可能是 service/pod 不可用。

### 4. 层级选择要匹配影响范围
- `service`：同一服务的多个副本出现相似异常。
- `pod`：单个具体副本/实例异常，而 peer replicas 相对正常。
- `node`：故障明显是基础设施层面的，或 node 指标本身就是最直接证据。
- 补充一点：本数据集里并不是所有 `pod` 级 case 都会给出完整的“副本对照证据”；有些 `pod` 级标注本身就依赖实例名、TiDB 专有指标或故障注入对象来确定。

### 5. 推理必须贴合数据集里的故障语义
- 这个数据集不是在问“泛化意义下最合理的系统诊断”。
- 它更接近在问：当前样本究竟属于哪一种标注好的 fault type / layer。
- 因此推理要尽量贴近 `groundtruth` 中反复出现的模式，如：
  - `cpu stress`
  - `memory stress`
  - `pod failure`
  - `network corrupt`
  - `dns error`
  - `node disk fill`
  - `jvm gc`
  - `code error`
  - `io fault`

### 6. 要接受“某些 case 信息很稀疏”这一事实
- 在完整 `groundtruth` 中，部分 case 的 `key_observations` 很少，甚至只有单类 metric。
- 这类 case 不能强行要求“metrics + traces + logs 三模态齐全”。
- 如果某个标签在数据集中反复以单一强证据出现，例如 `node_filesystem_usage_rate -> node disk fill`，就应承认这已经足够。

### 7. 数据集里有明显的“组件-故障类型绑定”
- `adservice` 高度集中承载 `jvm cpu/gc/latency/exception`。
- `checkoutservice` 覆盖了全部 `dns error`。
- `tidb-tikv` 覆盖全部 `io fault`，并承担大量 `pod failure`。
- `target port misconfig` 只出现在 `service` 级。
- 这些绑定不是绝对规则，但在证据不足时，是很重要的数据集先验。

## 各类场景的专家规则

## 压力测试：CPU Stress

### 常见标签模式
- `fault_category = stress test`
- `fault_type = cpu stress`
- 常见于 `instance_type = service` 或 `instance_type = pod`

### 强证据
- 指标：`pod_cpu_usage`、`pod_processes`
- 辅助指标：`rrt`、`rrt_max`
- 有时会伴随下游 trace 延迟上升

### 专家规则
- 如果同一 service 的多个副本都出现 `pod_cpu_usage` 和 `pod_processes` 上升，优先考虑 service 级 `cpu stress`。
- 如果只有一个具体副本/pod 出现 CPU/process 异常，优先考虑 pod 级。
- 不能因为 node CPU 也在波动，就轻易把 service/pod CPU stress 升级为 node cpu stress，除非 node 证据本身更直接。

## 压力测试：Memory Stress

### 常见标签模式
- `fault_category = stress test`
- `fault_type = memory stress`
- 可能是 `service` 级，也可能是 `pod` 级

### 强证据
- 指标：`pod_memory_working_set_bytes`
- 有时伴随 `pod_processes`
- 有时会有类似 `could not load` 的辅助日志

### 专家规则
- 某个 service 多个副本都出现 `pod_memory_working_set_bytes` 上升时，通常是 service 级 memory stress。
- 单个副本内存上升更像 pod 级 memory stress。
- 除非 node 内存证据本身最直接，否则不要把 service/pod memory stress 重新解释为 node memory stress。

## Pod Fault：Pod Failure / Pod Kill

### 常见标签模式
- `fault_category = pod fault`
- `fault_type = pod failure` 或 `pod kill`
- `instance_type` 可能是 `service`，也可能是 `pod`

### 强证据
- 日志：`Connection refused`、`Error while dialing`、`failure`、`unavailable`
- Trace：request proportion 异常、下游调用失败
- 指标：request/response 消失，error ratio 上升，某些 pod 网络/资源指标变化

### 专家规则
- `connection refused` 是 service/pod 不可用的重要信号。
- 如果 groundtruth 是 pod 级，且有实例隔离特征，应优先选具体 pod。
- 如果 groundtruth 是 service 级 pod failure，则可以选 service，尤其当调用方只感知到“整个 service 不可用”时。
- 这类故障应优先解释为 “unavailability / pod crash / pod failure”，而不是泛化成资源压力。

### 关于 `pod kill` 的补充
- `pod kill` 往往故障窗口很短，日志不一定充分，更多体现为：
  - request proportion 突变，
  - request/response 短时消失，
  - error ratio 抬升，
  - 局部资源/网络指标同步异常。
- 因此 `pod kill` 不应要求必须看到非常完整的崩溃日志；短窗口内的可用性突变本身就很关键。

## Node Fault：Node CPU Stress

### 常见标签模式
- `fault_category = node fault`
- `fault_type = node cpu stress`
- `instance_type = node`

### 强证据
- 指标：`node_cpu_usage_rate`
- 有时伴随 node 网络包指标
- 有时会在 colocated service 上看到 trace latency 增大

### 专家规则
- 这类 case 的核心证据就是 node 指标本身。
- 如果最强直接异常就是某个 node 的 `node_cpu_usage_rate`，不要被 service 的次生症状抢走解释权。
- 这类故障的原因表达应更偏基础设施层面，而不是应用层。

## Node Fault：Node Memory Stress

### 常见标签模式
- `fault_category = node fault`
- `fault_type = node memory stress`
- `instance_type = node`

### 强证据
- 指标：`node_memory_usage_rate`、`node_memory_MemAvailable_bytes`
- 常伴随磁盘读类指标

### 专家规则
- 这类 case 往往主要依赖 node 指标判定，而不是 service 指标。
- 如果某个 node 的内存异常最直接，而 service 证据只是泛化症状，应优先选 node。
- service 症状更适合作为影响描述，而不是主解释。

## Node Fault：Node Disk Fill

### 常见标签模式
- `fault_category = node fault`
- `fault_type = node disk fill`
- `instance_type = node`

### 强证据
- 指标：`node_filesystem_usage_rate`

### 专家规则
- `node_filesystem_usage_rate` 的语义非常强，通常应直接作为 node 根因证据。
- 不要因为同节点上的某个 service 也退化，就把这类 case 解释成 service 故障。

## Network Attack：Network Corrupt

### 常见标签模式
- `fault_category = network attack`
- `fault_type = network corrupt`
- `instance_type = service`
- `groundtruth` 里通常明确给了 `source` 和 `destination`

### 强证据
- 日志：`Error while dialing`、`Rpc error`、`context canceled`、`timeout`、`unavailable`
- Trace：caller 侧延迟异常、上游传播
- 指标：`rrt`、`rrt_max`、`request`、`response`、网络包/字节指标

### 专家规则
- source/destination 的方向性非常重要。
- 不要因为 destination service 出现错误或资源异常，就直接判 destination 是根因。
- 如果 logs/traces 主要体现的是 caller 侧 dialing/timeout/canceled 语义，应优先考虑 `source` 一侧或其所在路径。
- destination 的资源异常可能只是次生影响。
- 对于 `cartservice -> redis-cart` 这类依赖链，也应保持同样原则：不要因为 `redis-cart` 或 `cartservice` 某一侧更“显眼”，就忽略 `source/destination` 已经给出的方向信息。

## Network Attack：Network Loss

### 常见标签模式
- `fault_category = network attack`
- `fault_type = network loss`
- `instance_type = service`

### 强证据
- 日志：`deadlineexceeded`、`timeout`、`Rpc error`、`context canceled`、`unavailable`
- Trace：latency 或 request proportion 异常
- 指标：`rrt`、`rrt_max`、`request`、`response`、error ratio、网络指标

### 专家规则
- loss 类故障通常表现为 timeout-heavy 的传播链。
- caller 侧证据非常重要；出现错误的 destination 不一定是根因。
- 如果 `groundtruth` 给了 `source` 和 `destination`，除非 destination 有更直接的内部失败证据，否则更应从 source 侧解释。

## Network Attack：Network Delay

### 常见标签模式
- `fault_category = network attack`
- `fault_type = network delay`
- `instance_type = service`

### 强证据
- Trace：latency anomalies 往往是最强模态
- 指标：`rrt`、`rrt_max`、`request`、`response`、网络指标

### 专家规则
- delay 场景通常更像“链路问题”，而不是“destination 资源问题”。
- 如果 trace 是最强模态，就不要过度拟合到某个 service 的 incidental 资源波动。
- 要显式利用 `source/destination` 关系。

## JVM Fault

### 常见类型
- `jvm cpu`
- `jvm gc`
- `jvm latency`
- `jvm exception`

### 常见服务
- 常见于 `adservice`

### 强证据
- 日志关键字：
  - `adservice--stress`
  - `adservice--gc`
  - `adservice-getRandomAds-latency`
  - `adservice-getrandomads-exception`
  - `GCHelper`
  - `InvocationTargetException`
  - `TransformListener`
- 辅助指标：
  - `pod_cpu_usage`
  - `rrt`
  - `rrt_max`

### 专家规则
- JVM fault 往往是 log 主导的场景。
- metrics 和 traces 主要用来做支撑，真正决定 fault subtype 的通常是日志语义。
- 如果 `groundtruth` 是 pod 级 JVM 故障，不要轻易上升成整个 service，除非多个副本明确共享同类 JVM 症状。

### 四种 JVM 子类型的区分提示
- `jvm cpu`：更常见 `adservice--stress`，并伴随 `pod_cpu_usage` 升高。
- `jvm gc`：`adservice--gc`、`GCHelper` 最关键。
- `jvm latency`：`adservice-getRandomAds-latency` 更关键，延迟语义强于资源语义。
- `jvm exception`：`adservice-getrandomads-exception`、`Caught throwexception`、`InvocationTargetException` 更关键。

## Erroneous Change：Code Error

### 常见服务
- `cartservice`
- `checkoutservice`
- `currencyservice`
- `frontend`
- `productcatalogservice`

### 强证据
- 日志：`FailedPrecondition`、`request error`、`rpc error`、`http.resp.status 302`、`context canceled`
- Trace：request proportion 或 latency 异常
- 指标：request/response/timeout 变化，有时伴随 `pod_processes` 或 `pod_cpu_usage`

### 专家规则
- code error 主要是日志语义主导的故障。
- 优先选择发出应用错误信号的 service。
- metrics 和 traces 主要告诉你影响范围和时序，真正的 fault type 主要靠 logs。

### 数据集里的常见 code error 语义
- `cartservice` 常见数据库错误，如 `FailedPrecondition`。
- `checkoutservice` / `currencyservice` / `productcatalogservice` 更常见 `request error`、`rpc error`、`context canceled`、`Internal` 一类语义。
- `frontend` 更常见配置或 URL 错误，容易传播成多个下游失败，但根因仍是 `frontend` 自身。

## DNS Fault

### 常见模式
- `fault_category = dns fault`
- `fault_type = dns error`
- 常见在 `checkoutservice` 或某个 checkoutservice pod
- `patterns` 字段会指出受影响的下游 service
- 完整数据集里，`dns error` 全部发生在 `checkoutservice` 或其具体 pod 上

### 强证据
- 日志：`http.resp.status 302`、`Internal`、`request error`、`rpc error`、`failure`、`unavailable`
- 指标：`error`、`error_ratio`、`server_error`、`server_error_ratio`

### 专家规则
- DNS fault 通常是调用侧 service/pod 的解析失败，而不是 destination service 本身坏掉。
- 在本数据集中，更应选择“发生 lookup failure 的 service/pod”，而不是被解析的目标服务。
- 最终 reason 里最好显式保留 DNS 语义。

## Misconfiguration：Target Port Misconfig

### 常见模式
- `fault_category = misconfiguration`
- `fault_type = target port misconfig`
- `service` 一般就是被配置错的那个 service
- 完整数据集里这类 case 都是 `service` 级，而不是 `pod` 级

### 强证据
- 指标：`request`、`response`
- 有时伴随 trace/request proportion 异常

### 专家规则
- 这是配置错误，不是泛化的网络不稳定。
- 更应直接选择配置错误的 service。
- 即使 traces/logs 体现为 timeout、unavailable 或 request error，也不要优先往 `network loss/corrupt` 去套，除非存在明确链路攻击方向证据。

## IO Fault

### 常见模式
- `fault_category = io fault`
- `fault_type = io fault`
- 常见于 `service = tidb-tikv`
- `instance_type = pod`
- 完整数据集里，`io fault` 全部落在 `tidb-tikv`

### 强证据
- 指标：`io_util`、`region_pending`、`raft_apply_wait`、`store_size`、`qps`、`grpc_qps`、`memory_usage`、`cpu_usage`
- Trace：依赖 TiDB 的上游服务延迟升高

### 专家规则
- TiKV 的 IO fault 主要是存储层指标主导。
- 应直接选择对应的 TiKV pod/service，而不是泛化成前台 service 或更宽泛的 TiDB。
- 当同时看到 `region_pending`、`io_util`、`raft_apply_wait`、`store_size`、`grpc_qps/qps` 中的多项时，应显式认识到这是 TiKV/存储层现象，而不是普通业务容器资源波动。

## TiDB Pod Failure

### 常见模式
- `service = tidb-tidb` 或 `tidb-pd`
- `fault_type = pod failure`

### 强证据
- TiDB 专有指标：
  - `connection_count`
  - `uptime`
  - `qps`
  - `block_cache_size`
  - `abnormal_region_count`
  - `region_health`
  - `leader_count`

### 专家规则
- TiDB 控制面/数据面指标语义很强，应直接作为 TiDB 组件证据。
- 不要把这类故障映射回普通业务微服务。
- 其中：
  - `tidb-tidb` 更常见 `connection_count`、`uptime`、`qps`、`block_cache_size`
  - `tidb-pd` 更常见 `abnormal_region_count`、`region_health`、`leader_count`
  - `tidb-tikv` 的 `pod failure` 常伴随 `grpc_qps/qps`、`region_pending`、`io_util`

## 模态权重经验

### 指标最强的情况
- 故障本质是资源压力或资源饱和；
- 指标本身和 fault type 高度对应；
- 指标所在层级和 groundtruth 标注层级一致。

### 日志最强的情况
- `pod failure`、`pod kill`、`code error`、`dns error`、`jvm exception/gc/cpu/latency`；
- 日志关键字明确描述了失败语义。

### Trace 最强的情况
- `network delay`、`network loss`、`network corrupt`；
- 需要区分 caller/source 与 callee/destination；
- 需要分析传播路径和请求比例变化。

## 常见误判模式

- 因为 node CPU/内存有波动，就把 service/pod 的直接内部异常覆盖成 node。
- 在 network fault 中，把 destination service 的症状直接当成根因。
- 把 `missing_data` 单独当作主要证据。
- 在没有副本一致性证据时，把 pod fault 提升成 service fault。
- 在没有明显基础设施直接证据时，把 service 级重复副本压力误判成 node fault。
- 忽略 TiDB 专有指标，错误映射到普通业务服务。

## Prompt 级检查清单
在最终选择组件前，建议依次确认：

1. 当前更像哪一类故障场景：stress、pod failure、node fault、network、JVM、code error、DNS、IO、misconfiguration？
2. 哪个模态提供了最直接的故障证据？
3. 当前选中的组件是真正的根因，还是只是下游受害者？
4. 当前选中的层级，是否和证据反映的影响范围一致？
5. 是否有另一个组件拥有更具体、更直接的 metric/log 关键词？

## 一句话总结
优先选择“证据最直接、故障语义最匹配、层级最一致”的组件，不要让更宽泛的次生症状覆盖更具体的根因解释。
