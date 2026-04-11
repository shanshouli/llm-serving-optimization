# LLM Inference Serving Optimization and Cloud Deployment Platform

Benchmarking vLLM inference serving across quantization levels (FP16/INT8/INT4) and concurrency under constrained GPU resources, then deploying the optimal configuration to AWS SageMaker for cloud validation.

## Project Overview

We study how quantization and concurrency affect LLM inference serving performance under constrained GPU resources. Using industry-standard ShareGPT workloads, we benchmark vLLM across three precision levels (FP16, INT8, INT4) and four concurrency levels (1, 4, 8, 16), with a naive HuggingFace baseline as reference. Real-time server-side metrics are captured via Prometheus + Grafana. We then deploy the optimal INT4 configuration to AWS SageMaker to evaluate cloud-specific behaviors: cold-start latency, auto-scaling, and cost-efficiency (tokens/dollar).

## Experiments

| # | Experiment | Description |
|---|---|---|
| 0 | **Baseline** | Hugging Face Transformers + FastAPI, single-request serving, no optimization |
| 1 | **Serving Framework** | Replace baseline with vLLM; measure improvements from PagedAttention and continuous batching |
| 2 | **Quantization** | Compare FP16 → INT8 (GPTQ) → INT4 (AWQ); measure speed/memory/quality tradeoff |
| 3 | **Concurrency** | Simulate concurrent users (1, 4, 8, 16 clients); measure throughput scaling and tail latency |
| 4 | **Cloud Deployment** | Deploy optimal INT4 config to AWS SageMaker; measure cold-start, auto-scaling, and tokens/dollar |

### Bonus (If Time Permits)

- Vertex AI deployment for cross-platform cloud comparison
- Perplexity evaluation on quantized models (WikiText-2)

## Data Matrix (Local Experiments)

|  | FP16 | INT8 (GPTQ) | INT4 (AWQ) |
|---|---|---|---|
| **c=1** | ✓ | ✓ | ✓ |
| **c=4** | ✓ | ✓ | ✓ |
| **c=8** | ✓ | ✓ | ✓ |
| **c=16** | ✓ | ✓ | ✓ |

Plus: HF Baseline × FP16 × c=1 (completed).

### Key Hypotheses

1. INT4 throughput exceeds FP16 at high concurrency — freed VRAM enables CUDA Graph + larger KV cache.
2. FP16 saturates early — throughput plateaus at c=4 or c=8 while latency spikes.
3. INT8 is the sweet spot — most of INT4's benefits with negligible quality loss.

## Workload: ShareGPT

Source: `anon8231489123/ShareGPT_Vicuna_unfiltered`. Industry-standard benchmark used in official vLLM evaluations.

| Property | Value |
|---|---|
| Input length | 10–2000 tokens, median ~200 |
| Output length | 10–1000 tokens, median ~150 |
| Sample size | 100 requests per run |

FP16 (`--max-model-len 1024`): Filtered to input+output ≤ 1024 tokens. INT8/INT4 can run the broader range.

## Results to Date

### HF Baseline vs vLLM (FP16, c=1, n=20, fixed prompt)

| Metric | HF Baseline | vLLM | Delta |
|---|---|---|---|
| Avg latency | 6.496s | 4.337s | **-33.2%** |
| P50 latency | 6.335s | 4.305s | -32.0% |
| P95 latency | 7.331s | 4.666s | -36.3% |
| Avg tokens/s | 39.5 | 59.0 | **+49.4%** |
| Aggregate throughput | 698.4 tokens/s | 1,097.4 tokens/s | **+57.1%** |

*Note: Run with fixed prompts. Will be re-validated with ShareGPT workload.*

## Model

**meta-llama/Llama-3.2-3B-Instruct**

| Precision | Approx. VRAM | `--max-model-len` | CUDA Graph |
|---|---|---|---|
| FP16 | ~6 GB | 1024 | Disabled (`--enforce-eager`) |
| INT8 (GPTQ) | ~3–4 GB | 2048 | Enabled |
| INT4 (AWQ) | ~2 GB | 4096 | Enabled |

## Hardware

| Device | Role |
|---|---|
| NVIDIA RTX 2080 (8 GB) | Primary — vLLM, quantization, benchmarks |
| Apple M2 (8 GB) | Benchmark scripting, evaluation pipeline |
| Apple M1 (8 GB) | Visualization, result analysis, presentation |
| AWS SageMaker (`ml.g5.xlarge`, A10G 24 GB) | Cloud deployment, auto-scaling, cost analysis |

### Known Constraints

- **Windows WDDM reserves ~1 GB VRAM** — only ~6.9 GB available on RTX 2080.
- **FP16 model** requires `--enforce-eager` and `--max-model-len 1024` to fit.
- **FlashAttention 2 not supported on RTX 2080** (SM 7.5 < 8.0) — vLLM falls back to FlashInfer automatically.

## Setup

### Prerequisites

- Docker Desktop with GPU support enabled
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- HuggingFace account with access to [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 1. Create Environment File

```
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

### 2. Start All Services (vLLM + Prometheus + Grafana)

```bash
docker compose up -d
```

### 3. Verify vLLM is Running

```bash
uv run python -c "import httpx; r = httpx.post('http://localhost:8000/v1/completions', json={'model':'meta-llama/Llama-3.2-3B-Instruct','prompt':'Hello','max_tokens':64}); print(r.json())"
```

### 4. Install Python Dependencies

```bash
uv add aiohttp httpx matplotlib seaborn pandas datasets
```

## Usage

### Run Benchmark (ShareGPT workload)

```bash
# 100 requests, 8 concurrent clients against vLLM
uv run benchmark/client.py --concurrency 8 --num-requests 100

# Run all experiments automatically
uv run run_experiments.py
```

### Switch Quantization Level

Stop the current container, update `docker-compose.yml` to point to the INT8/INT4 model, then restart:

```bash
docker compose down
# edit docker-compose.yml model and parameters
docker compose up -d
```

## Observability

Prometheus and Grafana run automatically via Docker Compose.

| Service | URL |
|---|---|
| vLLM API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana dashboard | http://localhost:3000 |

Key panels: KV Cache Utilization, Requests Running/Waiting, P50/P95/P99 Latency, Token Throughput, TTFT.

## Evaluation Metrics

| Metric | Source |
|---|---|
| Avg / P50 / P95 / P99 latency | Benchmark client |
| Tokens/sec (per request + aggregate) | Benchmark client |
| Requests/sec | Benchmark client |
| Peak GPU memory | `nvidia-smi` |
| KV cache utilization over time | Prometheus / Grafana |
| Queue depth over time | Prometheus / Grafana |
| TTFT distribution | Prometheus / Grafana |
| Cold-start latency | SageMaker deploy timing |
| Tokens/dollar | Cost analysis |

## Cloud Deployment (SageMaker)

Uses AWS DJL-LMI container with vLLM backend on `ml.g5.xlarge` (A10G 24 GB, ~$1.41/hr).

See `cloud/sagemaker_deploy.py` and `cloud/sagemaker_autoscale.py`.

**Delete endpoint immediately after experiments to stop billing.**

## Project Structure

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

## Team

| Member | Role |
|---|---|
| Member A | Local Experiments Lead — all 13 GPU runs, Docker Compose, data analysis, presentation narrative |
| Member B | Cloud Deployment Lead — SageMaker deployment, auto-scaling, cost analysis, CloudWatch monitoring |
| Member C | Data & Presentation Lead — ShareGPT dataset prep, charts, slide design, rehearsal coordination |
