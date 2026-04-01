# Fine-Tuning Template For Multimodal RCA

## Goal
This document provides a practical template for preparing supervised fine-tuning samples for multimodal root-cause analysis on this dataset.

It is designed for cases where the model receives:
- fault description,
- metric evidence summary,
- trace evidence,
- log evidence,
- optional topology / deployment context.

The target behavior is:
- identify the correct root-cause component,
- describe the correct fault type / reasoning path,
- distinguish root cause from downstream victims,
- select the right layer: service / pod / node.

## Recommended Training Sample Schema

Each supervised sample should contain:

```json
{
  "uuid": "case uuid",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "natural-language anomaly description",
    "metrics": "precomputed metric summary text",
    "traces": "trace observations",
    "logs": "log observations",
    "topology": "optional topology / deployment note"
  },
  "output": {
    "component": "final root-cause component",
    "fault_type": "normalized fault type",
    "layer": "service | pod | node",
    "reason": "short final reason",
    "reasoning_steps": [
      "step 1 ...",
      "step 2 ...",
      "step 3 ...",
      "step 4 ..."
    ]
  }
}
```

## Preferred Output Style

- `component` should be the final answer component used for evaluation.
- `fault_type` should stay close to dataset semantics, such as:
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
- `layer` should match the component layer in groundtruth.
- `reasoning_steps` should emphasize evidence alignment, not verbose storytelling.

## Data Construction Principles

### 1. Keep metrics structured
- For service / pod stress cases, include the exact internal metrics such as:
  - `pod_cpu_usage`
  - `pod_processes`
  - `pod_memory_working_set_bytes`
- For node faults, include direct node metrics such as:
  - `node_cpu_usage_rate`
  - `node_memory_usage_rate`
  - `node_memory_MemAvailable_bytes`
  - `node_filesystem_usage_rate`

### 2. Keep network direction explicit
- For `network corrupt / loss / delay`, always preserve:
  - `source`
  - `destination`
  - key caller-side traces/logs
- Do not train the model to always choose the destination service.

### 3. Keep log semantics explicit
- For `pod failure / pod kill / code error / dns error / jvm*`, logs often define the fault type more directly than metrics.
- Preserve decisive keywords such as:
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

### 4. Preserve layer semantics
- service: multiple replicas share similar evidence
- pod: one concrete instance is isolated
- node: node metrics are the strongest direct evidence

## Optional Detailed Metric Extraction

If a training example needs more detailed metric evidence, use:

```bash
conda run -n agent python3 "unit_test/metric/pattern_analysis/extract_metric.py" \
  --uuid <UUID> \
  --component <COMPONENT> \
  --metric <METRIC> \
  --extend <MINUTES>
```

Notes:
- Run it in the `agent` conda environment.
- It writes time-series values to `metric-series.txt`.
- It also saves a line plot under `pattern-analysis/img/`.
- This is useful when you want to attach a compact narrative such as "sustained increase", "sharp spike", or "fault-window disappearance".

## High-Quality Training Examples

## Example 1: Service-Level CPU Stress

Groundtruth pattern:
- `uuid`: `345fbe93-80`
- `fault_type`: `cpu stress`
- `component`: `emailservice`

Training sample:

```json
{
  "uuid": "345fbe93-80",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "The system experienced an anomaly during the fault window.",
    "metrics": "emailservice shows pod_processes and pod_cpu_usage increasing across replicas; rrt/rrt_max are also relevant. Generic node CPU increase is present but less specific.",
    "traces": "No decisive caller-callee failure path. No strong evidence that another component is the source.",
    "logs": "No decisive app-failure logs.",
    "topology": "emailservice is a downstream service in checkout flow."
  },
  "output": {
    "component": "emailservice",
    "fault_type": "cpu stress",
    "layer": "service",
    "reason": "Replica-consistent pod_cpu_usage and pod_processes increase directly indicate service CPU stress.",
    "reasoning_steps": [
      "Service-level internal metrics are the strongest direct evidence.",
      "The anomaly appears across replicas, so service-level explanation is stronger than pod-local explanation.",
      "Node CPU movement is broader but less specific than emailservice internal metrics.",
      "Choose emailservice as the root cause."
    ]
  }
}
```

## Example 2: Network Corrupt With Source-Destination Direction

Groundtruth pattern:
- `uuid`: `74a44ae7-81`
- `fault_type`: `network corrupt`
- `source`: `checkoutservice`
- `destination`: `shippingservice`

Training sample:

```json
{
  "uuid": "74a44ae7-81",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "An abnormal event occurred between checkoutservice and shippingservice.",
    "metrics": "shippingservice shows memory increase and client_error_ratio increase; request/response and rrt-style service symptoms are present.",
    "traces": "checkoutservice/frontend path shows timeout-related anomalies toward shippingservice.",
    "logs": "Logs contain dialing failure, timeout, unavailable, and canceled semantics.",
    "topology": "checkoutservice calls shippingservice."
  },
  "output": {
    "component": "checkoutservice",
    "fault_type": "network corrupt",
    "layer": "service",
    "reason": "Caller-to-callee timeout and dialing failure semantics match a source-side network-corrupt fault.",
    "reasoning_steps": [
      "This is a network-fault scenario, so caller/destination direction matters.",
      "Trace and log evidence are stronger than destination-side resource symptoms.",
      "Shippingservice errors can be downstream symptoms rather than root cause.",
      "Choose checkoutservice as the source-side root cause."
    ]
  }
}
```

## Example 3: Service-Level Pod Failure

Groundtruth pattern:
- `uuid`: `38ee3d45-82`
- `fault_type`: `pod failure`
- `component`: `cartservice`

Training sample:

```json
{
  "uuid": "38ee3d45-82",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "A fault caused cart retrieval failures.",
    "metrics": "Cartservice request/response-related metrics disappear or become incomplete during the fault window.",
    "traces": "Frontend and checkoutservice show connection-refused style failures toward cartservice.",
    "logs": "Logs repeatedly show connection refused / unavailable semantics.",
    "topology": "frontend and checkoutservice depend on cartservice."
  },
  "output": {
    "component": "cartservice",
    "fault_type": "pod failure",
    "layer": "service",
    "reason": "Connection refused plus service disappearance indicates cartservice pod/service unavailability.",
    "reasoning_steps": [
      "Availability-style failure is the dominant scenario.",
      "Missing metrics alone are weak, but combined with connection refused they become strong unavailability evidence.",
      "The failure is perceived as cartservice unavailable from upstream callers.",
      "Choose cartservice with pod-failure semantics."
    ]
  }
}
```

## Example 4: Node Memory Stress

Groundtruth pattern:
- `uuid`: `b1ab098d-83`
- `fault_type`: `node memory stress`
- `component`: `aiops-k8s-06`

Training sample:

```json
{
  "uuid": "b1ab098d-83",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "A node-level anomaly occurred during the fault window.",
    "metrics": "aiops-k8s-06 shows node_memory_usage_rate / node_memory_MemAvailable_bytes anomalies as the most direct evidence.",
    "traces": "No stronger service-specific failure path dominates the case.",
    "logs": "No stronger app-level failure semantics.",
    "topology": "Multiple services may be impacted secondarily."
  },
  "output": {
    "component": "aiops-k8s-06",
    "fault_type": "node memory stress",
    "layer": "node",
    "reason": "Direct node memory metrics are the strongest and most specific evidence.",
    "reasoning_steps": [
      "Node memory metrics directly match the labeled fault type.",
      "Service-level symptoms are secondary and less specific.",
      "Trace/log evidence does not point to a competing service root cause.",
      "Choose aiops-k8s-06."
    ]
  }
}
```

## Example 5: Pod-Level Memory Stress

Groundtruth pattern:
- `uuid`: `31392fda-93`
- `fault_type`: `memory stress`
- `component`: `shippingservice-0`

Training sample:

```json
{
  "uuid": "31392fda-93",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "A localized memory anomaly occurred in shippingservice.",
    "metrics": "A single shippingservice replica shows pod_memory_working_set_bytes increase and related local signals.",
    "traces": "Downstream impact may exist, but no evidence that all replicas share the same failure.",
    "logs": "No stronger cross-replica failure semantics.",
    "topology": "shippingservice is downstream of checkoutservice."
  },
  "output": {
    "component": "shippingservice-0",
    "fault_type": "memory stress",
    "layer": "pod",
    "reason": "Single-replica memory increase indicates pod-local memory stress.",
    "reasoning_steps": [
      "The anomaly is isolated to one replica.",
      "There is not enough evidence for service-wide memory stress.",
      "Node-level interpretation would be broader but less specific.",
      "Choose shippingservice-0."
    ]
  }
}
```

## Example 6: JVM CPU Fault

Groundtruth pattern:
- `uuid`: `20622318-94`
- `fault_type`: `jvm cpu`
- `component`: `adservice`

Training sample:

```json
{
  "uuid": "20622318-94",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "An adservice JVM-related anomaly occurred.",
    "metrics": "adservice shows pod_cpu_usage increase and latency-related service symptoms.",
    "traces": "Frontend path latency increases downstream of adservice.",
    "logs": "Logs include adservice--stress, InvocationTargetException, and TransformListener-like JVM signatures.",
    "topology": "frontend calls adservice."
  },
  "output": {
    "component": "adservice",
    "fault_type": "jvm cpu",
    "layer": "service",
    "reason": "JVM stress logs plus pod_cpu_usage make adservice the direct CPU-fault source.",
    "reasoning_steps": [
      "JVM fault subtype is primarily determined by logs.",
      "adservice has direct pod_cpu_usage support.",
      "Trace latency is consistent with downstream impact from adservice.",
      "Choose adservice."
    ]
  }
}
```

## Example 7: DNS Error

Groundtruth pattern:
- `uuid`: `18b8cf77-318`
- `fault_type`: `dns error`
- `component`: `checkoutservice`

Training sample:

```json
{
  "uuid": "18b8cf77-318",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "Checkoutservice cannot correctly reach productcatalogservice.",
    "metrics": "checkoutservice error/server_error metrics rise.",
    "traces": "Possible frontend impact exists but is secondary.",
    "logs": "Resolution-like request failures and related app-side error semantics are present.",
    "topology": "checkoutservice depends on productcatalogservice."
  },
  "output": {
    "component": "checkoutservice",
    "fault_type": "dns error",
    "layer": "service",
    "reason": "Caller-side resolution failure semantics indicate checkoutservice DNS error.",
    "reasoning_steps": [
      "DNS faults should usually be assigned to the caller experiencing lookup failure.",
      "The destination service is not the primary root cause here.",
      "Error/server_error metrics support app-side lookup failure.",
      "Choose checkoutservice."
    ]
  }
}
```

## Example 8: Target Port Misconfiguration

Groundtruth pattern:
- `uuid`: `a42d2eb4-319`
- `fault_type`: `target port misconfig`
- `component`: `emailservice`

Training sample:

```json
{
  "uuid": "a42d2eb4-319",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "A service-port misconfiguration causes communication failure.",
    "metrics": "request/response behavior is abnormal for emailservice.",
    "traces": "Communication path fails, but not as generic packet loss/delay.",
    "logs": "Semantics are consistent with unreachable target port / wrong port mapping.",
    "topology": "emailservice participates in checkout flow."
  },
  "output": {
    "component": "emailservice",
    "fault_type": "target port misconfig",
    "layer": "service",
    "reason": "Port-level service misconfiguration directly explains request/response failure.",
    "reasoning_steps": [
      "This is a misconfiguration scenario rather than generic network instability.",
      "The service itself is the misconfigured object.",
      "Request/response abnormalities are consistent with wrong target port.",
      "Choose emailservice."
    ]
  }
}
```

## Example 9: TiKV IO Fault

Groundtruth pattern:
- `uuid`: `332adc3a-317`
- `fault_type`: `io fault`
- `component`: `tidb-tikv`

Training sample:

```json
{
  "uuid": "332adc3a-317",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "Storage-layer IO fault affects TiKV.",
    "metrics": "tidb-tikv shows io_util, region_pending, raft_apply_wait, snapshot/store metrics, and memory-related stress.",
    "traces": "Productcatalogservice path latency increases as downstream impact.",
    "logs": "No stronger competing service-level app failure.",
    "topology": "Business services depend on TiDB storage path."
  },
  "output": {
    "component": "tidb-tikv",
    "fault_type": "io fault",
    "layer": "pod",
    "reason": "Storage-specific TiKV IO metrics directly indicate an IO fault on tidb-tikv.",
    "reasoning_steps": [
      "Storage metrics are the most direct modality.",
      "Trace latency is downstream impact, not the primary fault location.",
      "This should not be mapped back to ordinary microservices.",
      "Choose tidb-tikv."
    ]
  }
}
```

## Example 10: Code Error

Groundtruth pattern:
- `uuid`: `50bce1c4-311`
- `fault_type`: `code error`
- `component`: `cartservice`

Training sample:

```json
{
  "uuid": "50bce1c4-311",
  "task": "Root cause analysis",
  "input": {
    "fault_description": "Cartservice code change causes runtime failures.",
    "metrics": "request/response and process/cpu-related symptoms appear around cartservice.",
    "traces": "Frontend is affected downstream.",
    "logs": "Logs include FailedPrecondition, request error, rpc error, and related DB failure semantics.",
    "topology": "frontend depends on cartservice."
  },
  "output": {
    "component": "cartservice",
    "fault_type": "code error",
    "layer": "service",
    "reason": "Application error logs directly indicate a cartservice code-level failure.",
    "reasoning_steps": [
      "Code-error semantics are log-driven.",
      "The error keywords are emitted by cartservice logic.",
      "Metrics and traces mainly show downstream impact.",
      "Choose cartservice."
    ]
  }
}
```

## Suggested Fine-Tuning Mix

For a balanced fine-tuning set, cover:
- service-level stress cases
- pod-level stress cases
- service-level pod failures
- pod-level pod failures
- node CPU/memory/disk faults
- network corrupt / loss / delay
- JVM cpu / gc / latency / exception
- DNS error
- target port misconfiguration
- code error
- TiDB pod failure / TiKV IO fault

## Final Recommendation
Do not train the model only on "answer format".
Train it on:
- directionality (`source` vs `destination`),
- layer specificity (`service` vs `pod` vs `node`),
- modality roles (metrics vs traces vs logs),
- fault semantics (resource stress vs unavailability vs network vs config vs JVM vs IO).
