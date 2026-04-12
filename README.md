# LLM Inference Serving Optimization and Cloud Platform Comparison

Two-part study: (1) optimize vLLM inference across quantization levels (FP16/INT8/INT4) and concurrency on a constrained consumer GPU; (2) deploy the optimal configuration to both AWS SageMaker and Google Vertex AI for a direct cloud platform comparison.

## Project Overview

This project answers two independent research questions:

**Part 1 — Local optimization (Member A):** How do quantization and concurrency jointly affect LLM serving performance under constrained GPU resources (RTX 2080, 8 GB)? We benchmark vLLM across FP16/INT8/INT4 × c=1/4/8/16 using ShareGPT workloads, with a HuggingFace baseline as reference. Prometheus + Grafana provide real-time KV cache and throughput visibility.

**Part 2 — Cloud platform comparison (Member B):** Given the same INT4 vLLM configuration, which cloud platform delivers better performance per dollar — AWS SageMaker (A10G, ~$1.41/hr) or Google Vertex AI (L4, ~$0.98/hr)? The L4 has ~2× the INT8 throughput of the A10G but costs 30% less, making the outcome non-obvious and practically relevant.

## Experiments

| # | Experiment | Description |
|---|---|---|
| 0 | **Baseline** | Hugging Face Transformers + FastAPI, single-request serving, no optimization |
| 1 | **Serving Framework** | Replace baseline with vLLM; measure improvements from PagedAttention and continuous batching |
| 2 | **Quantization** | Compare FP16 → INT8 (GPTQ) → INT4 (AWQ); measure speed/memory tradeoff on RTX 2080 |
| 3 | **Concurrency** | Simulate concurrent users (c=1/4/8/16); measure throughput scaling and tail latency |
| 4 | **Cloud Comparison** | Deploy INT4 vLLM to SageMaker (A10G) vs Vertex AI (L4); compare latency, throughput, tokens/dollar, cold-start, auto-scaling |

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
| NVIDIA RTX 2080 (8 GB) | Local experiments — vLLM, quantization, benchmarks |
| Apple M2 (8 GB) | Benchmark scripting, evaluation pipeline |
| Apple M1 (8 GB) | Visualization, result analysis, presentation |
| AWS SageMaker `ml.g5.xlarge` (A10G 24 GB, ~$1.41/hr) | Cloud platform A — latency, throughput, auto-scaling |
| Google Vertex AI `g2-standard-4` (L4 24 GB, ~$0.98/hr) | Cloud platform B — same metrics for direct comparison |

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

## Cloud Platform Comparison

Both platforms deploy the same INT4 AWQ vLLM configuration using ShareGPT workloads (c=1 and c=8, n=100 requests each).

| Metric | SageMaker A10G | Vertex AI L4 |
|---|---|---|
| GPU | A10G 24 GB (Ampere) | L4 24 GB (Ada Lovelace) |
| On-demand price | ~$1.41/hr | ~$0.98/hr |
| INT8 throughput | ~250 TOPS | ~485 TOPS |
| Latency P50/P95 | TBD | TBD |
| Tokens/s (c=8) | TBD | TBD |
| Tokens/dollar | TBD | TBD |
| Cold-start | TBD | TBD |
| Scale 0→1 | TBD | TBD |

See `cloud/sagemaker_deploy.py`, `cloud/sagemaker_autoscale.py`, and `cloud/vertex_deploy.py`.

**Delete all endpoints immediately after experiments to stop billing.**

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
| Member B | Cloud Comparison Lead — SageMaker + Vertex AI deployment, auto-scaling on both platforms, tokens/dollar cost analysis |
| Member C | Data & Presentation Lead — ShareGPT dataset prep, charts, slide design, rehearsal coordination |
