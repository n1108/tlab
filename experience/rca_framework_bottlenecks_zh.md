# 根因分析框架瓶颈与缺陷总结（对照评测输出与 Ground Truth）

**分析对象**

- 模型输出：`results/dataset/answer/2026-04-09_16-44-23-output.jsonl`（400 条，字段含 `uuid`、`component`、`reason`、`reasoning_trace`）
- 参考答案：`dataset/groundtruth.jsonl`（400 条，含 `fault_type`、`instance`/`service`、`source`/`destination` 等）

**说明**：下文中的定量对比用于**归纳框架行为与误差结构**，不是官方评分脚本；组件匹配规则为：对预测 `component` 与 GT 中 `service`/`instance`（列表展开）及 `source`/`destination` 做去副本后缀后的集合包含判断。

---

## 1. 整体结论摘要

| 维度 | 观察 |
|------|------|
| **组件定位** | 在「去 `-0/-1` 等服务级归一」规则下，约 **69.8%**（279/400）用例的预测组件落在 GT 给出的组件集合内；仍有约 **30%** 存在根因实体选错（含多跳链路边界选错、节点 vs 工作负载混淆等）。 |
| **故障类型** | 预测字段 `reason` 与 GT `fault_type` **字符串完全一致**约占 **31.3%**（125/400）；多数误差来自**网络类子类型混淆**（corrupt / delay / loss）、**节点子类型混淆**（CPU / 内存 / 磁盘）、**JVM 子类型**（gc / cpu / latency / exception）及 **pod kill** 等短窗场景被归为「无证据」或其它类型。 |
| **推理链** | 大量 `reasoning_trace` 中出现 **「No direct trace anomalies」**、**「No log anomalies」** 类表述（抽样统计量级约百余条量级），说明 Trace/Log 在不少 case 上**未形成可区分根因的强信号**，模型主要依赖 Metric 摘要与先验叙事。 |

---

## 2. Metric 侧：瓶颈与缺陷

1. **单模态主导、与其它模态脱节**  
   当输出声称「各层指标无显著异常」或仅依赖 `METRIC_EVIDENCE_SUMMARY` 的文本形态时，容易在 **pod kill、极短故障窗** 等场景下与 GT 不一致（例如 GT 中 `pod kill` 与预测中 `no direct evidence` 的组合在统计上多次出现）。

2. **节点 vs 数据库 vs 服务边界的歧义**  
   典型混淆：GT 为 **node memory stress**（实例为某 `aiops-k8s-*`），预测却指向 **TiDB 组件 + io fault** 等。说明在「基础设施资源异常」与「TiDB 存储/IO 指标冲高」之间，**缺少统一的归因优先级与互斥校验**（例如应先验证节点级内存/磁盘是否与 TiDB pod 同节点共现）。

3. **网络类指标语义重叠**  
   `network corrupt` / `network delay` / `network loss` 在现象上都常表现为 **RRT 上升、超时、错误率上升**；若 Metric 阶段未区分「时延型」「丢包型」「链路/握手失败型」特征，后续推理会坍缩为笼统的 **network delay**（统计上 `network corrupt`→`network delay` 等错配较多）。

4. **多组件链路（source/destination）**  
   GT 中大量网络攻击类记录同时涉及 **source 与 destination**。若预测只输出**单一** `component`，而没有「链路级」或「优先故障端」约定，容易只命中其中一端或误指调用方。

---

## 3. Trace 侧：瓶颈与缺陷

1. **覆盖度与对齐问题**  
   推理链中频繁出现 **「无直接 Trace 异常 / TiDB 不在 Trace 列表」** 等说明：要么 Trace 数据未覆盖关键组件，要么 **Trace 摘要未与 Metric 中的异常组件对齐**，导致 Trace 步骤无法提供反证或确认。

2. **延迟异常与根因方向**  
   Trace 仅能证明「某条调用慢或超时」，但**慢在客户端排队、网络、还是服务端**需要与 Log/Metric 交叉；当前叙事里常见「下游延迟传播」类表述，易与 **network delay vs service cpu stress** 混淆（统计上可见 `network delay` 与 `cpu stress` 的交叉错配）。

3. **短故障或采样稀疏**  
   对 **pod kill、极短窗口** 类故障，Trace 可能几乎无异常 span，框架若缺少「**无 Trace 不等于无故障**」的显式规则，会拉低置信度或错误走向「无直接证据」。

---

## 4. Log 侧：瓶颈与缺陷

1. **「无 Log 异常」占比高**  
   大量步骤 3 为 **No anomalies detected in logs**，可能原因包括：日志采集未覆盖、关键词/模板未命中、或 **LLM 未充分消费原始日志片段**（仅依赖摘要）。

2. **JVM / 代码类故障依赖日志关键词**  
   GT 中 `jvm gc`、`jvm cpu`、`code error` 等往往与 **特定日志模式** 绑定；若日志证据未进入上下文，预测易退化为泛化的 **jvm exception** 或与网络类混淆。

3. **多服务报错时的根因归属**  
   上游大量 `timeout`、`connection refused` 时，Log 若只被用来「印证现象」而未做 **时间先后与调用方向** 约束，容易把 **现象服务** 当成 **根因服务**。

---

## 5. 推理与融合：瓶颈与缺陷

1. **故障类型标签空间与预测不一致**  
   GT 使用细粒度 `fault_type`（如 `target port misconfig`、`dns error`、`node disk fill`），而预测分布大量集中在 **pod failure / network delay / cpu stress** 等少数桶，说明 **推理头或后处理缺乏对细分类的可行区分**，或训练/提示未对齐标签体系。

2. **高先验的叙事模板**  
   对网络类故障，推理链多次采用「timeout → network delay」的叙事；对资源类则偏向「CPU/内存冲高 → stress」。这类模板在**区分度不足**时会系统性偏向少数 `reason`，与 GT 的多样性不匹配。

3. **「无直接证据」兜底风险**  
   当 Metric/Trace/Log 均偏弱时，预测使用 **no direct evidence** 并常把组件落在 **frontend** 等入口服务上；与 GT 中 **pod kill**（根因在业务服务而非入口）冲突明显，属于**兜底策略与真实注入点不一致**。

4. **多模态冲突未显式消解**  
   同一 `reasoning_trace` 中可能出现「Metric 强、Trace/Log 空」或「Log 指向 A、Metric 指向 B」；若缺少 **冲突检测与重试/投票**，最终判断会偏向某一步的片面观察。

---

## 6. 典型系统性误差模式（从统计归纳）

以下为 GT→预测 `fault_type`→`reason` 组合中出现频次较高的**代表性模式**（非完整列表），用于定位改进优先级：

- **network corrupt → network delay**：现象相似导致子类型混淆。  
- **network loss → network delay / pod failure**：丢包与超时、错误之间的归因不稳。  
- **target port misconfig → pod failure**：细类被并入更粗类。  
- **pod failure ↔ io fault**：存储/IO 异常与进程不可用叙事重叠。  
- **pod kill → no direct evidence / network delay**：短窗与证据不足时的误判。  
- **jvm gc / jvm cpu / jvm latency → jvm exception**：JVM 子类未拉开。  
- **node memory stress → node cpu stress / memory stress / node disk fill**：节点维度子类型混淆。

---

## 7. 改进方向（与评测无关的工程建议）

1. **标签与输出空间对齐**：推理输出增加与 GT 同构的细粒度枚举，或训练分类头对齐 `fault_type` 全集。  
2. **链路级根因**：对 `source`/`destination` 类 GT，输出 **边**（A→B）或 **主因端** 并给出规则。  
3. **节点–工作负载联合约束**：节点异常需校验同节点 Pod 列表再指向 TiDB/业务服务。  
4. **网络三分支特征**：在 Metric/Trace 特征层区分 delay / loss / corrupt（统计分布、重传、错误码族）。  
5. **短窗与无证据策略**：单独分支处理极短窗口，避免默认「无异常」。  
6. **Log/Trace 硬证据门控**：关键类型（JVM、code、DNS、端口）要求必须引用日志/配置片段再下结论。  
7. **冲突消解**：Metric vs Trace vs Log 不一致时触发二次检索或降级为「多假设」而非单点裁决。

---

*文档生成自对指定输出文件与 `groundtruth.jsonl` 的对比脚本统计与结构阅读，便于团队复盘架构瓶颈；若更换输出文件版本，请重新跑统计更新数字。*
