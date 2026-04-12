# LLM Inference Serving Optimization and Cloud Deployment Platform

## Project Plan (Final)

---

## 1. Project Description

This project addresses two distinct research questions:

**Part 1 (Local):** How do quantization (FP16/INT8/INT4) and concurrency (c=1/4/8/16) jointly affect vLLM serving performance on a constrained consumer GPU? Using industry-standard ShareGPT workloads, we benchmark across 13 configurations with a HuggingFace baseline as reference. Prometheus + Grafana provide real-time KV cache and throughput visibility.

**Part 2 (Cloud):** Given the same optimal INT4 vLLM configuration, which cloud platform delivers better performance per dollar — AWS SageMaker (A10G 24 GB, ~$1.41/hr) or Google Vertex AI (L4 24 GB, ~$0.98/hr)? The L4 has ~2× the INT8 compute of the A10G at 30% lower cost, making the outcome non-obvious. The "local vs cloud" framing was abandoned: comparing an RTX 2080 to an A10G proves only that better hardware is faster, which is trivial.

---

## 2. Local Experiments (Member A)

### 2.1 Data Matrix (13 + 1 runs)

|  | FP16 | INT8 (GPTQ) | INT4 (AWQ) |
|---|---|---|---|
| **c=1** | ✓ | ✓ | ✓ |
| **c=4** | ✓ | ✓ | ✓ |
| **c=8** | ✓ | ✓ | ✓ |
| **c=16** | ✓ | ✓ | ✓ |

Plus: HF Baseline × FP16 × c=1 (already completed).

### 2.2 What Each Dimension Tests

**Quantization (columns):**
- FP16: Full precision. `--enforce-eager` (no CUDA Graph), `--max-model-len 1024`. ~6 GB model, ~0.9 GB left for KV cache.
- INT8 (GPTQ): ~3-4 GB model. CUDA Graph re-enabled, `--max-model-len 2048`.
- INT4 (AWQ): ~2 GB model. CUDA Graph re-enabled, `--max-model-len 4096`. Maximum KV cache capacity.

**Concurrency (rows):**
- c=1: Sequential baseline. Isolates per-request latency.
- c=4: Light concurrency. Continuous batching starts showing benefits.
- c=8: Medium load. FP16 expected to saturate here (KV cache ~2,496 tokens).
- c=16: Heavy load. Only INT4 expected to handle gracefully.

### 2.3 Key Hypotheses

1. INT4 throughput exceeds FP16 at high concurrency — freed VRAM enables CUDA Graph + larger KV cache + longer max sequence length.
2. FP16 saturates early — throughput plateaus at c=4 or c=8 while latency spikes.
3. INT8 is the sweet spot — most of INT4's benefits with negligible quality loss.

### 2.4 Workload: ShareGPT

Source: `anon8231489123/ShareGPT_Vicuna_unfiltered` from HuggingFace. Industry-standard benchmark used by vLLM and SGLang in official evaluations.

| Property | Value |
|---|---|
| Input length | 10–2000 tokens, median ~200 |
| Output length | 10–1000 tokens, median ~150 |
| Sample size | 100 requests per run |

FP16 (`--max-model-len 1024`): Filter to input+output ≤ 1024 tokens. INT8/INT4: Can run broader range. The difference itself is a data point.

### 2.5 vLLM Launch Parameters

| Parameter | FP16 | INT8 | INT4 |
|---|---|---|---|
| `--dtype` | float16 | auto | auto |
| `--max-model-len` | 1024 | 2048 | 4096 |
| `--gpu-memory-utilization` | 0.85 | 0.90 | 0.90 |
| `--enforce-eager` | Yes | No | No |
| CUDA Graph | Disabled | Enabled | Enabled |

### 2.6 Observability: Prometheus + Grafana

Runs automatically via Docker Compose. Pre-configured dashboard captures:

| Panel | What It Reveals |
|---|---|
| KV Cache Utilization | Primary bottleneck — when this hits 99%, requests queue |
| Requests Running / Waiting | Saturation vs underutilization |
| Latency P50/P95/P99 | Server-side latency over time |
| Token Throughput | Tokens/sec — headline metric |
| Time to First Token | TTFT — critical for interactive apps |

Benchmark client shows "what happened." Grafana shows "why."

### 2.7 Metrics Collected Per Run

| Metric | Source |
|---|---|
| Avg / P50 / P95 / P99 latency | Benchmark client |
| Tokens/sec (per request + aggregate) | Benchmark client |
| Requests/sec | Benchmark client |
| Peak GPU memory | `nvidia-smi` |
| KV cache utilization over time | Prometheus / Grafana |
| Queue depth over time | Prometheus / Grafana |
| TTFT distribution | Prometheus / Grafana |

### 2.8 Results to Date

#### HF Baseline vs vLLM (FP16, c=1, n=20, fixed prompt)

| Metric | HF Baseline | vLLM | Delta |
|---|---|---|---|
| Avg latency | 6.496s | 4.337s | **-33.2%** |
| P50 latency | 6.335s | 4.305s | -32.0% |
| P95 latency | 7.331s | 4.666s | -36.3% |
| Avg tokens/s | 39.5 | 59.0 | **+49.4%** |
| Aggregate throughput | 698.4 tokens/s | 1,097.4 tokens/s | **+57.1%** |

Note: Run with fixed prompts, not ShareGPT. Will be re-validated with ShareGPT workload.

### 2.9 Local Execution Plan

| Step | Task | Time |
|---|---|---|
| 1 | Modify benchmark client for ShareGPT loading | 2-3 hours |
| 2 | Run FP16 × c=1/4/8/16 (ShareGPT, 100 requests each) | 2-3 hours |
| 3 | Restart vLLM with INT8 model, run c=1/4/8/16 | 2-3 hours |
| 4 | Restart vLLM with INT4 model, run c=1/4/8/16 | 2-3 hours |
| 5 | Data analysis + visualization | Half day |
| 6 | Presentation | 2-3 days |

---

## 3. Cloud Deployment (Teammate)

### 3.1 Objective

Deploy the locally-optimized vLLM INT4 configuration to **both** AWS SageMaker and Google Vertex AI using identical workloads and metrics. The core research question: given the same software stack, which platform delivers better tokens/dollar performance, and how do their operational characteristics (cold-start, auto-scaling speed, failure behavior) differ?

**Why this comparison is non-trivial:**
- Vertex AI L4 has ~2× the INT8 compute of SageMaker A10G, but costs 30% less
- Both GPUs have 24 GB VRAM — hardware is roughly matched in memory capacity
- Managed service overhead, framework versions, and scheduling differ between platforms
- The "correct" answer depends on whether you optimize for raw throughput, tail latency, or cost

### 3.2 SageMaker Deployment (Required)

#### Step 1: Deploy Endpoint (Day 1)

Use AWS's pre-built DJL-LMI container with vLLM backend. No custom Docker image needed.

```python
import sagemaker

container_uri = sagemaker.image_uris.retrieve(
    framework="djl-lmi", version="0.30.0", region="us-east-1"
)

model = sagemaker.Model(
    image_uri=container_uri,
    role=iam_role,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.2-3B-Instruct",  # or INT4 AWQ variant
        "OPTION_ROLLING_BATCH": "vllm",
        "TENSOR_PARALLEL_DEGREE": "max",
    }
)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",  # A10G 24GB, ~$1.41/hr
    endpoint_name="vllm-llama3-int4",
)
```

#### Step 2: Run Benchmark (Day 1-2)

Use the same ShareGPT benchmark script (change URL to SageMaker endpoint). Run:
- c=1, n=100 (ShareGPT)
- c=8, n=100 (ShareGPT)

Record: latency, throughput, cold-start time (time from `deploy()` call to first successful response).

#### Step 3: Auto-Scaling Test (Day 2)

Configure auto-scaling policy:

```python
client = boto3.client("application-autoscaling")

# Register scalable target: min=0, max=3 instances
client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,
    MaxCapacity=3,
)

# Scale based on invocations per instance
client.put_scaling_policy(
    PolicyName="invocation-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 10.0,  # target invocations per instance
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

Traffic pattern:
1. Send 0 QPS → 10 QPS burst
2. Sustain 10 QPS for 5 min
3. Drop to 0 QPS

Observe in CloudWatch: instance count changes, scale-up/scale-down latency, request failures during transitions.

#### Step 4: Cost Analysis (Day 2)

Calculate tokens/dollar:

```
tokens_per_dollar = total_tokens_generated / (instance_cost_per_second × wall_clock_time)
```

#### Step 5: Clean Up

**Delete endpoint immediately after experiments to stop billing.**

```python
sagemaker.Predictor(endpoint_name).delete_endpoint()
```

### 3.3 Vertex AI (Required — Part of Core Comparison)

Same flow on GCP, using identical workload and metrics as SageMaker:
1. Push vLLM INT4 container to Artifact Registry
2. Deploy to Vertex AI endpoint (`g2-standard-4`, L4 24GB, ~$0.98/hr)
3. Run same ShareGPT benchmark (c=1 and c=8, n=100)
4. Configure auto-scaling (min=0, max=3)
5. Record all metrics: latency P50/P95/P99, tokens/s, tokens/dollar, cold-start, scale 0→1 time

### 3.4 Expected Deliverable

| Metric | SageMaker A10G (INT4) | Vertex AI L4 (INT4) | Notes |
|---|---|---|---|
| GPU | A10G 24 GB | L4 24 GB | L4 has ~2× INT8 TOPS |
| Price | ~$1.41/hr | ~$0.98/hr | L4 is 30% cheaper |
| Cold-start | ? | ? | Model load from object storage |
| Latency P50 (c=1) | ? | ? | |
| Latency P95 (c=8) | ? | ? | Tail latency under load |
| Throughput (c=8) | ? | ? | tokens/s |
| Scale 0→1 time | ? | ? | Auto-scaling responsiveness |
| Scale 1→2 time | ? | ? | |
| **tokens/dollar** | ? | ? | **Primary comparison metric** |

Local RTX 2080 INT4 results (Member A's data) serve as the "constrained baseline" context, not as a direct comparison point.

### 3.5 Budget Estimate

| Platform | Estimated GPU hours | Cost |
|---|---|---|
| SageMaker (`ml.g5.xlarge`, A10G) | 4-6 hours | $6-9 |
| Vertex AI (`g2-standard-4`, L4) | 4-6 hours | $4-6 |
| **Total** | | **$10-15** |

---

## 4. Hardware & Environment

| Parameter | Value |
|---|---|
| Local GPU | NVIDIA RTX 2080 (8 GB, ~6.9 GB usable) |
| Architecture | Turing (SM 7.5) — INT8/INT4 via Tensor Cores |
| Model | meta-llama/Llama-3.2-3B-Instruct |
| Local serving | Docker Compose (vLLM + Prometheus + Grafana) |
| Cloud serving | SageMaker DJL-LMI container (vLLM backend) |
| Dependencies | uv (Python 3.10) |
| Attention backend | FlashInfer (FA2 requires SM 8.0+) |

---

## 5. Team Roles

### Member A — Local Experiments Lead

- Modify benchmark client for ShareGPT support
- Run all 13 local experiments (3 quantization × 4 concurrency)
- Manage Docker Compose (vLLM + Prometheus + Grafana)
- Data analysis and cross-experiment visualization
- Lead presentation narrative

### Member B — Cloud Comparison Lead

- Deploy INT4 vLLM to SageMaker (A10G), run c=1 and c=8 benchmarks, auto-scaling test
- Deploy INT4 vLLM to Vertex AI (L4), run identical benchmarks for direct comparison
- Calculate tokens/dollar for both platforms — primary deliverable
- Record cold-start and scale 0→1/1→2 times on both platforms
- CloudWatch (SageMaker) and Cloud Monitoring (Vertex AI) screenshots

### Member C — Data and Presentation Lead

- Prepare ShareGPT dataset (download, filter, format)
- Compile all results (local + cloud) into comparison tables
- Generate all charts (matplotlib/seaborn)
- Lead slide design and layout
- Coordinate rehearsal and timing

---

## 6. Timeline

| Step | Task | Owner | Time |
|---|---|---|---|
| 1 | ShareGPT benchmark script | A | 2-3 hours |
| 2 | FP16 × 4 concurrency levels | A | 2-3 hours |
| 3 | INT8 × 4 concurrency levels | A | 2-3 hours |
| 4 | INT4 × 4 concurrency levels | A | 2-3 hours |
| 5 | SageMaker deployment + benchmark | B | 1-2 days |
| 6 | SageMaker auto-scaling test | B | Half day |
| 7 | Vertex AI (if time permits) | B | 1-2 days |
| 8 | Data compilation + charts | C | 1 day |
| 9 | Presentation slides | All | 2-3 days |
| 10 | Rehearsal (×2) | All | 2 days |

---

## 7. Presentation Structure (15 slides, 20-25 min)

| Slide(s) | Content | Time |
|---|---|---|
| 1 | Title + Team | — |
| 2 | Motivation: Why LLM serving optimization matters | 2 min |
| 3 | Problem Statement: Resource constraints on consumer GPU | 2 min |
| 4 | System Architecture: Docker Compose + Prometheus + Grafana + SageMaker | 2 min |
| 5 | Methodology: ShareGPT workload, metrics, experiment matrix | 2 min |
| 6 | Baseline: HF vs vLLM | 2 min |
| 7-8 | Quantization Results: FP16 vs INT8 vs INT4 | 3 min |
| 9-10 | Concurrency Results + Grafana screenshots | 3 min |
| 11 | Cloud Deployment: Local vs SageMaker performance + cost | 2 min |
| 12 | Auto-Scaling behavior + CloudWatch screenshots | 1 min |
| 13 | Key Findings + Production Recommendations | 2 min |
| 14 | Future Work (SGLang, Triton, LoRA, Vertex AI, Nsight) | 1 min |
| 15 | Conclusion | 1 min |
| — | **Q&A** | **5+ min** |

---

## 8. Future Work (Not in Scope)

| Direction | Description |
|---|---|
| Additional frameworks | SGLang (RadixAttention), Triton + TensorRT-LLM |
| LoRA + quantization | Adapter overhead × rank × precision compatibility matrix |
| GPU kernel profiling | Nsight Systems trace analysis |
| Vertex AI | Cross-platform cloud comparison |
| Quality evaluation | Perplexity (WikiText-2), MMLU, HumanEval |
| Experiment tracking | MLflow + Hydra integration |
| Advanced workloads | Synthetic controlled-length, Poisson arrival patterns |

---

## 9. Repository Structure

```
CS6180-FinalProject/
├── README.md
├── CLAUDE.md
├── docker-compose.yml           # vLLM + Prometheus + Grafana
├── .env                         # HuggingFace token (git-ignored)
├── pyproject.toml
├── uv.lock
├── run_experiments.py           # Automated experiment runner
├── benchmark/
│   ├── client.py                # Async benchmark client (ShareGPT)
│   └── workloads/
│       └── sharegpt_filtered.json
├── serving/
│   └── baseline_hf.py           # HF + FastAPI baseline
├── cloud/
│   ├── sagemaker_deploy.py      # SageMaker deployment script
│   ├── sagemaker_autoscale.py   # Auto-scaling configuration
│   └── vertex_deploy.py         # Vertex AI (if done)
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── datasource.yml
│       │   └── dashboards/
│       │       └── dashboards.yml
│       └── dashboards/
│           └── vllm-dashboard.json
├── results/
│   ├── raw/                     # Benchmark CSVs
│   └── figures/                 # Charts
└── presentation/
    └── slides.pptx
```

---

## 10. Interview Narrative

"I designed and ran a two-part LLM serving study. In Part 1, I benchmarked vLLM across FP16, INT8, and INT4 quantization under four concurrency levels on a resource-constrained 8 GB consumer GPU, using industry-standard ShareGPT workloads. I found that INT4 quantization doesn't just save memory — it fundamentally shifts the system's operating regime: re-enabling CUDA Graph, expanding KV cache from ~2,500 to 16,000+ tokens, and moving the saturation point from c=4 to beyond c=10. Prometheus and Grafana instrumentation made these dynamics visible in real time.

In Part 2, I avoided the trivial 'consumer GPU vs data center GPU' comparison and instead ran an apples-to-apples cloud platform comparison: the same INT4 vLLM config on AWS SageMaker (A10G, $1.41/hr) vs Google Vertex AI (L4, $0.98/hr). The L4 has roughly 2× the INT8 compute at 30% lower cost — the question was whether that hardware advantage translates to proportionally better tokens/dollar in a managed serving environment, or whether platform overhead closes the gap."
