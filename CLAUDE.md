# Project Context for Claude Code

## What This Project Is

GenAI course final project: benchmarking and optimizing LLM inference serving strategies under constrained GPU resources. The goal is to compare different serving backends, quantization levels, concurrency behavior, and multi-tenant LoRA adapter serving, then analyze the tradeoffs.

This also serves as a portfolio project targeting North America AI infrastructure / LLM inference engineer roles.

## Current Status (March 31, 2026)

### What's Done
- Docker Desktop + GPU passthrough verified on Windows 11
- vLLM 0.18.0 container running with Llama-3.2-3B-Instruct FP16
- First inference request confirmed working
- uv initialized with Python 3.10, httpx installed
- README.md created
- benchmark/client.py created (async benchmark with latency/throughput/P50/P95/P99 reporting, CSV output)
- Prometheus + Grafana monitoring stack designed (files ready, not yet deployed)

### What's Not Done Yet
- Benchmark data collection (vLLM vs HF baseline)
- HF baseline serving container
- Quantization experiments (INT8/INT4)
- Concurrency experiments
- LoRA adapter preparation and multi-LoRA serving
- Prometheus + Grafana deployment
- Analysis and visualization
- Presentation

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
    ├── benchmark/client.py — async load generator, hits localhost:8000
    ├── evaluation/ — analysis scripts (TODO)
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

### vLLM Launch Parameters (working config)
```
--model meta-llama/Llama-3.2-3B-Instruct
--dtype float16
--max-model-len 1024
--gpu-memory-utilization 0.85
--enforce-eager
```

### PowerShell Gotchas
- `curl` in PowerShell is an alias for `Invoke-WebRequest`, not real curl. Use `curl.exe` or Python httpx instead.
- PowerShell quote escaping breaks JSON payloads. Use Python for HTTP testing.
- After installing tools (uv, etc.), run `$env:Path = "C:\Users\zla77\.local\bin;$env:Path"` or restart shell.

## Experiment Plan

| # | Experiment | Status |
|---|---|---|
| 0 | Baseline: HF Transformers + FastAPI, single request | TODO |
| 1 | vLLM serving (PagedAttention, continuous batching) | Server running, benchmark TODO |
| 2 | Quantization: FP16 → INT8 (GPTQ) → INT4 (AWQ) | TODO |
| 3 | Concurrency: 1/4/8/16 simultaneous clients | TODO |
| 4 | Multi-LoRA adapter serving (3 adapters, routing, LRU cache) | TODO |

## File Structure (Current)

```
C:\shanshou\CS6180-FinalProject\
├── .env                          # HUGGING_FACE_HUB_TOKEN=hf_xxx (git-ignored)
├── .gitignore
├── pyproject.toml                # uv project config
├── uv.lock
├── README.md
├── docker-compose.yml            # vLLM + Prometheus + Grafana
├── benchmark/
│   └── client.py                 # Async benchmark client
├── serving/
│   └── baseline_hf.py            # TODO: HF baseline server
├── evaluation/                   # TODO: analysis scripts
├── adapters/                     # TODO: LoRA training/configs
├── results/
│   └── raw/                      # Benchmark CSVs
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
└── presentation/                 # TODO: slides
```

## Immediate Next Steps (Priority Order)

1. Stop standalone vLLM container → deploy via docker compose (vLLM + Prometheus + Grafana)
2. Run benchmark/client.py against vLLM, collect 50-request baseline data
3. Create serving/baseline_hf.py and run in a separate container
4. Run same benchmark against HF baseline
5. Compare results for April 1 status check

## Team

- Member A (me): Systems lead. RTX 2080 + M1 Pro. Owns all GPU experiments, vLLM deployment, routing layer, core analysis.
- Member B: Evaluation & tooling. M2. Benchmark scripting, evaluation pipeline, visualization.
- Member C: Data & presentation. M1. LoRA adapters, workload design, slides.

## Code Style Requirements

- All code must include English comments at key and important places to ensure readability.

## Timeline

| Week | Dates | Focus |
|---|---|---|
| 1 | Mar 24 – Mar 30 | Environment setup, baseline + vLLM comparison |
| 2 | Mar 31 – Apr 6 | Core experiments (quantization, concurrency), status check (Apr 1) |
| 3 | Apr 7 – Apr 13 | Exp 4 (multi-LoRA), cross-experiment analysis |
| 4 | Apr 14 – Apr 20 | Presentation preparation and rehearsal |
