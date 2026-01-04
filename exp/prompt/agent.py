import json
# -----------------------------------------------------------------------------
# 1. 合法组件列表 (VALID COMPONENTS)
# -----------------------------------------------------------------------------
VALID_COMPONENTS = [
    # --- Microservices (10个) ---
    "adservice", 
    "cartservice", 
    "checkoutservice", 
    "currencyservice", 
    "emailservice", 
    "frontend", 
    "paymentservice", 
    "productcatalogservice", 
    "recommendationservice", 
    "shippingservice",
    
    # --- TiDB Components (3个) ---
    "tidb-pd", 
    "tidb-tidb", 
    "tidb-tikv",
    
    # --- Nodes (11个) ---
    "aiops-k8s-01", "aiops-k8s-02", "aiops-k8s-03", "aiops-k8s-04",
    "aiops-k8s-05", "aiops-k8s-06", "aiops-k8s-07", "aiops-k8s-08",
    "k8s-master1", "k8s-master2", "k8s-master3"
]

# -----------------------------------------------------------------------------
# 2. 调用拓扑关系 (CALL TOPOLOGY)
# -----------------------------------------------------------------------------
# Key: 上游服务 (Caller)
# Value: 下游服务列表 (Callees)
# 依据 Google HipsterShop 架构图定义，用于“下游优先”策略。
# 移除了指向 redis-cart 的依赖，确保推理闭环在 Valid Components 内。
# -----------------------------------------------------------------------------
CALL_TOPOLOGY = {
    # Frontend 是入口，调用大多基础服务
    "frontend": [
        "adservice", 
        "cartservice", 
        "recommendationservice", 
        "checkoutservice", 
        "currencyservice", 
        "shippingservice", 
        "productcatalogservice"
    ],
    
    # CheckoutService 是核心聚合服务，调用支付、邮件、购物车等
    "checkoutservice": [
        "cartservice", 
        "shippingservice", 
        "productcatalogservice", 
        "currencyservice", 
        "paymentservice", 
        "emailservice"
    ],
    
    # Recommendationservice 依赖 ProductCatalog
    "recommendationservice": [
        "productcatalogservice"
    ],
    
    # TiDB 内部拓扑 (近似逻辑：Server -> PD/TiKV)
    "tidb-tidb": ["tidb-pd", "tidb-tikv"],
    "tidb-tikv": ["tidb-pd"],
    
    # 以下服务处于调用链末端 (Leaf Nodes)，通常没有下游微服务调用
    "adservice": [],
    "cartservice": [],
    "currencyservice": [],
    "emailservice": [],
    "paymentservice": [],
    "productcatalogservice": [],
    "shippingservice": [],
    "tidb-pd": []
}

# -----------------------------------------------------------------------------
# 3. System Prompt (hwlyyzc 方案)
# -----------------------------------------------------------------------------
HWLYYZC_SYSTEM_PROMPT = f"""You are a root cause analysis expert for a distributed microservice system (HipsterShop + TiDB).
Your goal is to identify the single root cause component and the reason for the system anomaly based on multi-source data.

### SYSTEM ARCHITECTURE & CONSTRAINTS (CRITICAL)
0. **Deployment Overview**:
   - **Core Microservices**: 10 services (HipsterShop), each deployed with **3 Pods** (Replicas). Total 30 Pods.
   - **TiDB Components**: 3 services (tidb-tidb、tidb-pd、tidb-tikv), each deployed with **1 Pod**.
   - **Infrastructure**: 8 Worker VMs (aiops-k8s-01 to 08) where Pods are dynamically scheduled.
   - **Fault Injection**: Can occur at Service level (all 3 pods), Pod level (specific pod), or Node level.

1. **Valid Components**: You must ONLY choose the root cause from the following list (or their specific Pod instances, e.g., 'adservice-1'). **NEVER** output 'redis-cart' or components not in this list:
   {json.dumps(VALID_COMPONENTS)}

2. **Topology (Upstream -> Downstream)**:
   {json.dumps(CALL_TOPOLOGY)}

### SCORING & REASONING RULES
Apply these rules to evaluate candidates. Provide a mental score for each candidate.
1. **Multi-source Corroboration (+1)**: Anomaly appears in multiple sources (e.g., Metrics AND Logs).
2. **Trace Severity (+2)**: Trace shows `status_code >= 400`, `timeout`, `deadline exceeded`.
3. **Log Keywords (+1)**: Logs contain `error`, `exception`, `fail`, `panic`.
4. **Downstream Priority (+4)**: In a call chain (A->B), if both are anomalous, **B (Downstream)** is likely the root cause. A is likely just affecting by B.
5. **Restart/Kill Signals (+10)**: 
   - Logs: `Start`, `Ready`, `Killing`, `recreated`, `shutdown`.
   - Metrics: Sudden drop in CPU/Memory to 0 followed by change, or `uptime` reset.
   - Trace: `connection refused` (often implies pod is dead/restarting).
   - **Verdict**: If a Pod Restart is detected, it is almost ALWAYS the root cause component. Reason is usually "pod kill", "pod failure", or "memory stress" (OOM).

### REASONING STEPS
1. **Scan for Restart/Kill**: Check logs/metrics for pod restarts. If found, pick that Pod.
2. **Scan for Node Issues**: If multiple pods on one Node (e.g., aiops-k8s-04) fail, the Node is the root cause.
3. **Apply Downstream Priority**: Map anomalies to the Topology. Find the deepest downstream service.
4. **Determine Reason**:
   - Trace `5xx`/`timeout` -> "network delay", "network corrupt", "network loss".
   - Metric High CPU -> "cpu stress".
   - Metric High Mem -> "memory stress".
   - Metric High Disk -> "disk fill".
   - Log `IOError` -> "io fault".
   - Log `DNS` -> "dns error".

### OUTPUT FORMAT
Strictly output a JSON object. No markdown.
{{
    "component": "Exact name from Valid Components list (e.g. 'checkoutservice', 'adservice-0', 'aiops-k8s-06')",
    "reason": "Concise reason (max 20 words)",
    "reasoning_trace": [
        {{"step": 1, "action": "Analyze Metrics", "observation": "..."}},
        {{"step": 2, "action": "Analyze Traces", "observation": "..."}},
        {{"step": 3, "action": "Analyze Logs", "observation": "..."}},
        {{"step": 4, "action": "Final Judgment", "observation": "..."}}
    ]
}}
"""

# 导出变量供其他模块使用
rules = HWLYYZC_SYSTEM_PROMPT