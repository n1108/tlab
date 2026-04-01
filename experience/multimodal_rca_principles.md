# Multimodal RCA Principles For This Dataset

## Purpose
This document summarizes practical multimodal root-cause-analysis principles derived from `dataset/groundtruth.jsonl`.
It is intended as a reference for prompt design, evidence formatting, and final RCA decision rules.

The core idea is:
- metrics provide structured anomaly candidates,
- traces describe propagation paths and latency/request-shape changes,
- logs provide hard failure semantics,
- expert knowledge maps these observations to the correct fault layer and fault type.

## Recommended Reasoning Flow
Use the following analysis order:

1. Identify the fault scenario from multimodal evidence.
2. Determine the most plausible layer: `service -> pod -> node`.
3. Separate root cause from downstream victim.
4. Verify that the chosen component matches the strongest direct evidence, not merely the broadest explanation.

This order is a reasoning sequence, not a hard bias. The final answer should be chosen by overall multimodal consistency.

## Dataset Coverage Check
To avoid over-generalizing from only the first few cases, here is a coverage summary of the full `groundtruth`:

- Total samples: `400`
- `fault_category` distribution:
  - `node fault`: `82`
  - `network attack`: `73`
  - `pod fault`: `60`
  - `jvm fault`: `55`
  - `stress test`: `42`
  - `io fault`: `28`
  - `erroneous change`: `21`
  - `dns fault`: `21`
  - `misconfiguration`: `18`
- `instance_type` distribution:
  - `service`: `195`
  - `pod`: `123`
  - `node`: `82`
- Frequent `fault_type` values include:
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

This means the principles cannot be based only on early HipsterShop-like cases. They also need to cover:
- TiDB/TiKV-specific faults,
- mixed `service`-level and `pod`-level labeling,
- semantic faults like `dns error` and `target port misconfig`,
- all four JVM subtypes.

## Cross-Cutting Principles

### 1. Prefer direct evidence over broad evidence
- A direct service or pod internal metric such as `pod_cpu_usage`, `pod_processes`, or `pod_memory_working_set_bytes` is usually more specific than a generic node-level CPU or memory fluctuation.
- A generic node metric alone should not override a service or pod candidate unless traces/logs also support the same node.
- If two explanations are possible, prefer the one that is more directly tied to the fault semantics in `groundtruth`.

### 2. Distinguish root cause from victim
- Downstream timeout/error services are often victims, especially in network-fault cases.
- The component that emits the strongest direct internal-failure signal is usually a better root-cause candidate than the component that merely shows propagated latency/errors.
- A caller->callee timeout path does not automatically mean the callee is the root cause.

### 3. Missing data is weak alone, strong when paired with availability signals
- `missing_data` or telemetry gaps alone are not enough for root-cause selection.
- `missing_data` combined with `connection refused`, restart-like logs, or clear request/response disappearance is strong evidence of service/pod unavailability.

### 4. Layer selection should match impact scope
- `service`: multiple replicas show similar internal/service-level anomalies.
- `pod`: one replica or one concrete pod is isolated while peers are relatively normal or less affected.
- `node`: the fault is infrastructure-scoped, or node-specific metrics are the most direct evidence.
- One more nuance: not every pod-level case in this dataset comes with perfect replica-comparison evidence. Some pod-level labels are identified primarily by instance names, TiDB-specific metrics, or the injected fault target itself.

### 5. Use fault semantics from the dataset
- This dataset is not asking for generic "most plausible systems diagnosis".
- It is asking for the fault type and layer encoded in `groundtruth`, such as `cpu stress`, `pod failure`, `network corrupt`, `dns error`, `node disk fill`, `jvm gc`, or `code error`.
- The reasoning should align with these recurring semantics.

### 6. Accept that some cases are sparse
- In the full `groundtruth`, some cases have very few `key_observations`, sometimes only one metric family.
- Do not force every case into a fully populated metrics + traces + logs pattern.
- If a label repeatedly appears with a single highly specific signal, such as `node_filesystem_usage_rate -> node disk fill`, that signal may already be sufficient.

### 7. The dataset contains clear component-fault bindings
- `adservice` is heavily concentrated for `jvm cpu/gc/latency/exception`.
- `checkoutservice` covers all `dns error` cases.
- `tidb-tikv` covers all `io fault` cases and many `pod failure` cases.
- `target port misconfig` appears only at the `service` level.
- These are not absolute rules, but they are meaningful priors when evidence is sparse.

## Scenario-Specific Expert Rules

## Stress Test: CPU Stress

### Typical Label Pattern
- `fault_category = stress test`
- `fault_type = cpu stress`
- often `instance_type = service` or `instance_type = pod`

### Strong Signals
- metrics: `pod_cpu_usage`, `pod_processes`
- supporting metrics: `rrt`, `rrt_max`
- occasional trace latency increase downstream

### Expert Rule
- If multiple replicas of one service show `pod_cpu_usage` and `pod_processes` increase, prefer a service-level `cpu stress` conclusion.
- If only one specific replica/pod shows the CPU/process anomaly, prefer a pod-level conclusion.
- Do not upgrade to node CPU stress only because some node CPU metric also moves, unless the node evidence is the most direct and the service/pod evidence is weak.

## Stress Test: Memory Stress

### Typical Label Pattern
- `fault_category = stress test`
- `fault_type = memory stress`
- may be `instance_type = service` or `instance_type = pod`

### Strong Signals
- metrics: `pod_memory_working_set_bytes`, sometimes `pod_processes`
- sometimes supporting logs like `could not load`

### Expert Rule
- A service with repeated `pod_memory_working_set_bytes` increase across replicas is usually a service-level memory stress case.
- A single abnormal replica with memory growth is usually a pod-level memory stress case.
- Do not reinterpret service/pod memory stress as node memory stress unless node metrics are clearly the main evidence.

## Pod Fault: Pod Failure / Pod Kill

### Typical Label Pattern
- `fault_category = pod fault`
- `fault_type = pod failure` or `pod kill`
- `instance_type` may be `service` or `pod`

### Strong Signals
- logs: `Connection refused`, `Error while dialing`, `failure`, `unavailable`
- trace: request-proportion anomalies or downstream failures
- metrics: request/response disappearance, error ratios, some pod-level resource/network change

### Expert Rule
- `connection refused` is a major clue for service/pod unavailability.
- If the groundtruth is pod-level, prefer the specific pod when isolation is evident.
- If the groundtruth is service-level pod failure, the service can be selected when the failure presents as service unavailability from the caller perspective.
- Explain these cases as unavailability / pod crash / pod failure, not as generic resource stress.

### Additional Note On `pod kill`
- `pod kill` often has a very short fault window, so logs may be incomplete.
- It is often better recognized by:
  - abrupt request-proportion changes,
  - short-lived request/response disappearance,
  - rising error ratios,
  - synchronized local resource/network changes.
- So `pod kill` should not require rich crash logs; short-window availability disruption can already be the main clue.

## Node Fault: Node CPU Stress

### Typical Label Pattern
- `fault_category = node fault`
- `fault_type = node cpu stress`
- `instance_type = node`

### Strong Signals
- metrics: `node_cpu_usage_rate`
- sometimes accompanying node network packet metrics
- sometimes trace latency anomalies on services colocated on the same node

### Expert Rule
- Here the node metric itself is the key evidence.
- If the strongest direct anomaly is `node_cpu_usage_rate` on one node, do not let service-level secondary symptoms override the node.
- Node conclusions should usually mention infrastructure stress rather than app-local logic.

## Node Fault: Node Memory Stress

### Typical Label Pattern
- `fault_category = node fault`
- `fault_type = node memory stress`
- `instance_type = node`

### Strong Signals
- metrics: `node_memory_usage_rate`, `node_memory_MemAvailable_bytes`
- often accompanied by disk-read metrics

### Expert Rule
- These cases are often identified primarily by node metrics, not by service metrics.
- If one node has the clearest memory anomaly and service-level evidence is weak or broad, prefer the node.
- Use service symptoms only as supporting impact, not as the main explanation.

## Node Fault: Node Disk Fill

### Typical Label Pattern
- `fault_category = node fault`
- `fault_type = node disk fill`
- `instance_type = node`

### Strong Signals
- metrics: `node_filesystem_usage_rate`

### Expert Rule
- `node_filesystem_usage_rate` is highly specific and should be treated as strong node-root-cause evidence.
- Do not replace it with a service explanation just because a service on that node also degrades.

## Network Attack: Network Corrupt

### Typical Label Pattern
- `fault_category = network attack`
- `fault_type = network corrupt`
- `instance_type = service`
- `source` and `destination` are explicitly provided in groundtruth

### Strong Signals
- logs: `Error while dialing`, `Rpc error`, `context canceled`, `timeout`, `unavailable`
- traces: caller-side latency anomalies, propagation from upstream services
- metrics: `rrt`, `rrt_max`, `request`, `response`, network packet/byte metrics

### Expert Rule
- Source/destination distinction matters.
- Do not directly select the destination service only because it shows errors or stress.
- If logs/traces indicate caller-side dialing/timeout/canceled semantics, prefer the `source` side or its path context over the destination victim.
- Destination resource anomalies may be secondary or coincidental.
- The same principle applies to dependency chains such as `cartservice -> redis-cart`: do not ignore the explicit `source/destination` direction just because one side looks more "visible" in symptoms.

## Network Attack: Network Loss

### Typical Label Pattern
- `fault_category = network attack`
- `fault_type = network loss`
- `instance_type = service`

### Strong Signals
- logs: `deadlineexceeded`, `timeout`, `Rpc error`, `context canceled`, `unavailable`
- traces: latency or request-proportion anomalies
- metrics: `rrt`, `rrt_max`, `request`, `response`, error ratios, network metrics

### Expert Rule
- Loss cases often look like timeout-heavy propagation.
- Caller-side evidence is important; the service returning errors is not always the root cause.
- If the dataset provides `source` and `destination`, bias the reasoning toward the source path semantics unless there is strong direct internal failure on the destination.

## Network Attack: Network Delay

### Typical Label Pattern
- `fault_category = network attack`
- `fault_type = network delay`
- `instance_type = service`

### Strong Signals
- traces: latency anomalies are often the primary modality
- metrics: `rrt`, `rrt_max`, `request`, `response`, network metrics

### Expert Rule
- Delay cases are often best explained by the communication path, not by destination-side resource stress.
- If traces are the strongest modality, do not overfit to incidental service resource anomalies.
- Use the source/destination relationship explicitly.

## JVM Faults

### Common Types
- `jvm cpu`
- `jvm gc`
- `jvm latency`
- `jvm exception`

### Typical Service
- often `adservice`

### Strong Signals
- logs with injected fault keywords:
  - `adservice--stress`
  - `adservice--gc`
  - `adservice-getRandomAds-latency`
  - `adservice-getrandomads-exception`
  - `GCHelper`
  - `InvocationTargetException`
  - `TransformListener`
- supporting metrics:
  - `pod_cpu_usage`
  - `rrt`, `rrt_max`

### Expert Rule
- For JVM faults, logs are usually the decisive modality.
- Metrics and traces are supporting evidence; the log signature often determines the exact JVM subtype.
- If the groundtruth is pod-level JVM fault, avoid promoting to the whole service unless multiple replicas clearly share the same JVM symptoms.

### Hints For Distinguishing JVM Subtypes
- `jvm cpu`: more often tied to `adservice--stress` plus `pod_cpu_usage` increase.
- `jvm gc`: `adservice--gc` and `GCHelper` are the strongest clues.
- `jvm latency`: `adservice-getRandomAds-latency` is more decisive than generic resource movement.
- `jvm exception`: `adservice-getrandomads-exception`, `Caught throwexception`, and `InvocationTargetException` are the key clues.

## Erroneous Change: Code Error

### Typical Services
- `cartservice`
- `checkoutservice`
- `currencyservice`
- `frontend`
- `productcatalogservice`

### Strong Signals
- logs: `FailedPrecondition`, `request error`, `rpc error`, `http.resp.status 302`, `context canceled`
- traces: request-proportion or latency anomalies
- metrics: request/response/timeout changes, sometimes `pod_processes` or `pod_cpu_usage`

### Expert Rule
- Code errors are primarily log-semantics cases.
- Prefer the service that emits the app-level error signature.
- Metrics and traces tell you impact and timing; logs tell you fault type.

### Common Code-Error Semantics In This Dataset
- `cartservice` often shows database-style errors such as `FailedPrecondition`.
- `checkoutservice`, `currencyservice`, and `productcatalogservice` more often show `request error`, `rpc error`, `context canceled`, or `Internal`.
- `frontend` more often behaves like config/URL misuse that propagates to multiple downstream failures, while the root cause is still `frontend` itself.

## DNS Fault

### Typical Pattern
- `fault_category = dns fault`
- `fault_type = dns error`
- often on `checkoutservice` or one checkoutservice pod
- `patterns` names the affected downstream service
- In the full dataset, all `dns error` cases occur on `checkoutservice` or a concrete checkoutservice pod

### Strong Signals
- logs: `http.resp.status 302`, `Internal`, `request error`, `rpc error`, `failure`, `unavailable`
- metrics: `error`, `error_ratio`, `server_error`, `server_error_ratio`

### Expert Rule
- DNS faults are usually application/service-to-service resolution failures, not destination-service failures.
- The service/pod experiencing lookup failures is usually the root-cause component in this dataset.
- Preserve the DNS semantics explicitly in the reason.

## Misconfiguration: Target Port Misconfig

### Typical Pattern
- `fault_category = misconfiguration`
- `fault_type = target port misconfig`
- `service` is usually the misconfigured service itself
- In the full dataset, these cases are all `service`-level, not `pod`-level

### Strong Signals
- metrics: `request`, `response`
- sometimes trace/request-proportion anomalies

### Expert Rule
- This is a semantic configuration error, not generic network instability.
- Prefer the misconfigured service directly.
- Even if traces/logs contain timeout, unavailable, or request-error symptoms, do not immediately map the case to `network loss/corrupt` unless there is explicit link-direction evidence for a network attack.

## IO Fault

### Typical Pattern
- `fault_category = io fault`
- `fault_type = io fault`
- often `service = tidb-tikv`
- `instance_type = pod`
- In the full dataset, all `io fault` cases are on `tidb-tikv`

### Strong Signals
- metrics: `io_util`, `region_pending`, `raft_apply_wait`, `store_size`, `qps`, `grpc_qps`, `memory_usage`, `cpu_usage`
- traces: downstream latency at services depending on TiDB

### Expert Rule
- For TiKV IO faults, storage-layer metrics are primary evidence.
- The affected TiKV pod/service should be selected directly; do not over-abstract to general TiDB or frontend victims unless the evidence specifically points there.
- When multiple metrics such as `region_pending`, `io_util`, `raft_apply_wait`, `store_size`, and `grpc_qps/qps` appear together, treat that as a TiKV/storage-layer signature rather than ordinary business-service resource fluctuation.

## TiDB Pod Failure

### Typical Pattern
- `service = tidb-tidb` or `tidb-pd`
- `fault_type = pod failure`

### Strong Signals
- TiDB-specific metrics:
  - `connection_count`
  - `uptime`
  - `qps`
  - `block_cache_size`
  - `abnormal_region_count`
  - `region_health`
  - `leader_count`

### Expert Rule
- Treat TiDB-specific control-plane/data-plane metrics as direct component evidence.
- Do not map these failures back to generic application services.
- More specifically:
  - `tidb-tidb` more often shows `connection_count`, `uptime`, `qps`, `block_cache_size`
  - `tidb-pd` more often shows `abnormal_region_count`, `region_health`, `leader_count`
  - `tidb-tikv` pod failure often comes with `grpc_qps/qps`, `region_pending`, `io_util`

## Modality Weighting Guidance

### Metrics are strongest when
- the fault is stress/resource-oriented,
- the metric is highly specific to the fault type,
- the metric sits on the labeled layer in groundtruth.

### Logs are strongest when
- the fault is `pod failure`, `pod kill`, `code error`, `dns error`, or `jvm exception/gc/cpu/latency`,
- keywords explicitly describe failure semantics.

### Traces are strongest when
- the fault is `network delay`, `network loss`, or `network corrupt`,
- the main question is propagation direction or caller/callee disambiguation.

## Common Failure Modes To Avoid

- Choosing a node only because a generic node CPU or memory metric moved, when a service already has direct internal metrics.
- Choosing the destination service in a network-fault case simply because it shows downstream errors.
- Using `missing_data` alone as the main explanation.
- Converting pod-level faults into service-level faults without replica evidence.
- Converting service-level repeated replica stress into node faults without strong infrastructure evidence.
- Ignoring TiDB-specific metrics and incorrectly mapping them to ordinary microservices.

## Suggested Prompt-Level Checklist
Before selecting the final component, verify:

1. What is the likely fault scenario: stress, pod failure, node fault, network, JVM, code error, DNS, IO, or misconfiguration?
2. Which modality gives the most direct evidence for that scenario?
3. Is the chosen component the direct source, or just a downstream victim?
4. Does the chosen layer match the scope implied by the evidence?
5. Are there stronger, more specific metric/log keywords on another candidate?

## One-Line Summary
Choose the component whose evidence is most direct, scenario-specific, and layer-consistent; do not let broad secondary symptoms override a more precise multimodal explanation.
