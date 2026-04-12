# Project Context for Claude Code

## What This Project Is

GenAI course final project with two distinct research questions:

**Part 1 (Local):** How do quantization (FP16/INT8/INT4) and concurrency (c=1/4/8/16) jointly affect vLLM serving performance on a constrained consumer GPU (RTX 2080, 8 GB)? Goal: find optimal configuration (expected: INT4).

**Part 2 (Cloud):** Given the same INT4 vLLM configuration, which cloud platform delivers better performance per dollar — AWS SageMaker (A10G 24 GB, ~$1.41/hr) or Google Vertex AI (L4 24 GB, ~$0.98/hr)? The L4 has ~2× the INT8 compute of the A10G but costs 30% less — the outcome is non-obvious and practically useful.

The "local vs cloud" framing was abandoned: comparing a consumer RTX 2080 to an A10G proves nothing beyond "better hardware is faster," which is trivial. SageMaker vs Vertex AI is a real engineering decision with non-obvious tradeoffs.

This also serves as a portfolio project targeting North America AI infrastructure / LLM inference engineer roles.

## Current Status (April 10, 2026)

### What's Done
- Docker Desktop + GPU passthrough verified on Windows 11
- vLLM 0.18.0 container running with Llama-3.2-3B-Instruct FP16
- HF baseline container working (serving/baseline_hf.py)
- First benchmark comparison complete: HF baseline vs vLLM FP16 c=1 (fixed prompts, n=20)
- uv initialized with Python 3.10, httpx installed
- README.md and CLAUDE.md updated to reflect final plan
- benchmark/client.py created (async benchmark with latency/throughput/P50/P95/P99 reporting, CSV output)
- Prometheus + Grafana monitoring stack designed (files ready, not yet deployed via compose)

### What's Not Done Yet
- Modify benchmark client for ShareGPT loading
- FP16 ShareGPT runs (c=1/4/8/16, 100 requests each)
- INT8 (GPTQ) model setup and runs
- INT4 (AWQ) model setup and runs
- SageMaker deployment — INT4, c=1 and c=8, auto-scaling (Member B)
- Vertex AI deployment — INT4, c=1 and c=8, auto-scaling (Member B, now required)
- tokens/dollar cost comparison across both cloud platforms (Member B)
- Data analysis and visualization (Member C)
- Presentation (all)

## Architecture

```
Windows 11 PowerShell
│
├── Docker Compose (manages all 3 services)
│   ├── vllm-server (GPU, port 8000) — vLLM + Llama-3.2-3B-Instruct
│   ├── prometheus (CPU only, port 9090) — scrapes vLLM /metrics every 5s
│   └── grafana (CPU only, port 3000) — dashboard at http://localhost:3000
│
└── Local Python (uv, project root)
    ├── benchmark/client.py — async load generator, ShareGPT workload
    ├── run_experiments.py — automated experiment runner
    ├── cloud/ — SageMaker/Vertex AI deployment scripts
    └── results/ — CSV/JSON experiment data
```

## Key Technical Constraints

### RTX 2080 (8 GB VRAM)
- Windows WDDM reserves ~1 GB → only ~6.9 GB usable
- FP16 model loads at 6.02 GB → almost no room for KV cache
- Must use `--enforce-eager` (disables CUDA Graph to save ~0.08 GB VRAM)
- Must use `--max-model-len 1024` (reduces KV cache requirement)
- FlashAttention 2 not supported (compute capability 7.5 < 8.0) → falls back to FlashInfer automatically
- `pin_memory=False` warning due to WSL detection inside Docker — minor perf impact, not fixable

### vLLM Launch Parameters Per Quantization Level

| Parameter | FP16 | INT8 (GPTQ) | INT4 (AWQ) |
|---|---|---|---|
| `--dtype` | float16 | auto | auto |
| `--max-model-len` | 1024 | 2048 | 4096 |
| `--gpu-memory-utilization` | 0.85 | 0.90 | 0.90 |
| `--enforce-eager` | Yes | No | No |
| CUDA Graph | Disabled | Enabled | Enabled |

### PowerShell Gotchas
- `curl` in PowerShell is an alias for `Invoke-WebRequest`, not real curl. Use `curl.exe` or Python httpx instead.
- PowerShell quote escaping breaks JSON payloads. Use Python for HTTP testing.
- After installing tools (uv, etc.), run `$env:Path = "C:\Users\zla77\.local\bin;$env:Path"` or restart shell.

## Experiment Plan

| # | Experiment | Owner | Status |
|---|---|---|---|
| 0 | HF Baseline: Transformers + FastAPI, c=1 | A | Done (fixed prompts; redo with ShareGPT) |
| 1 | vLLM FP16 × c=1/4/8/16 (ShareGPT, 100 req each) | A | TODO |
| 2 | vLLM INT8 (GPTQ) × c=1/4/8/16 | A | TODO |
| 3 | vLLM INT4 (AWQ) × c=1/4/8/16 | A | TODO |
| 4 | SageMaker (A10G): INT4, c=1 and c=8, auto-scaling, tokens/dollar | B | TODO |
| 5 | Vertex AI (L4): INT4, c=1 and c=8, auto-scaling, tokens/dollar | B | TODO |

## Results to Date

### HF Baseline vs vLLM (FP16, c=1, n=20, fixed prompt)

| Metric | HF Baseline | vLLM | Delta |
|---|---|---|---|
| Avg latency | 6.496s | 4.337s | -33.2% |
| P50 latency | 6.335s | 4.305s | -32.0% |
| P95 latency | 7.331s | 4.666s | -36.3% |
| Avg tokens/s | 39.5 | 59.0 | +49.4% |
| Aggregate throughput | 698.4 tok/s | 1,097.4 tok/s | +57.1% |

## File Structure

```
C:\shanshou\CS6180-FinalProject\
├── .env                          # HUGGING_FACE_HUB_TOKEN=hf_xxx (git-ignored)
├── .gitignore
├── pyproject.toml                # uv project config
├── uv.lock
├── README.md
├── CLAUDE.md
├── PROJECT_PLAN_FINAL.md         # Finalized project plan
├── docker-compose.yml            # vLLM + Prometheus + Grafana
├── run_experiments.py            # Automated experiment runner (TODO)
├── benchmark/
│   ├── client.py                 # Async benchmark client
│   └── workloads/
│       └── sharegpt_filtered.json
├── serving/
│   └── baseline_hf.py            # HF + FastAPI baseline server
├── cloud/
│   ├── sagemaker_deploy.py       # SageMaker deployment (Member B)
│   ├── sagemaker_autoscale.py    # Auto-scaling configuration (Member B)
│   └── vertex_deploy.py          # Vertex AI, if time permits (Member B)
├── evaluation/                   # Analysis scripts (TODO, Member C)
├── results/
│   ├── raw/                      # Benchmark CSVs
│   └── figures/                  # Charts (TODO, Member C)
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
└── presentation/                 # TODO: slides (all)
```

## Immediate Next Steps (Priority Order)

1. Modify benchmark/client.py to load ShareGPT workload
2. Deploy via docker compose (vLLM + Prometheus + Grafana)
3. Run FP16 × c=1/4/8/16 (ShareGPT, 100 requests each)
4. Restart vLLM with INT8 model, run c=1/4/8/16
5. Restart vLLM with INT4 model, run c=1/4/8/16
6. Member B: SageMaker INT4 deployment — c=1 and c=8, auto-scaling, record tokens/dollar
7. Member B: Vertex AI INT4 deployment — same workload, same metrics for direct comparison

## Team

- Member A (me): Local Experiments Lead. RTX 2080. Owns all 13 GPU runs, Docker Compose management, data analysis, presentation narrative.
- Member B: Cloud Deployment Lead. M2. SageMaker deployment, auto-scaling test, cost analysis, CloudWatch monitoring.
- Member C: Data & Presentation Lead. M1. ShareGPT dataset prep, charts (matplotlib/seaborn), slide design, rehearsal coordination.

## Code Style Requirements

- All code must include English comments at key and important places to ensure readability.

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 1 | Mar 24 – Mar 30 | Environment setup, baseline + vLLM comparison |
| 2 | Mar 31 – Apr 6 | Core experiments (quantization, concurrency) |
| 3 | Apr 7 – Apr 13 | Cloud deployment (SageMaker), cross-experiment analysis |
| 4 | Apr 14 – Apr 20 | Presentation preparation and rehearsal |
