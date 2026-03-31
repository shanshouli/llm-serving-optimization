# Efficient LLM Inference Serving under Resource Constraints

Benchmarking and optimizing LLM inference serving strategies for latency, throughput, and memory efficiency on consumer-grade GPUs.

## Project Overview

This project studies how different inference-serving strategies affect the deployment efficiency of large language models under constrained GPU resources. We compare a naive Hugging Face-based serving pipeline with optimized variants including vLLM-based serving, quantized inference (INT8/INT4), and concurrent request handling. We further explore adapter-aware serving by deploying multiple LoRA adapters on a shared base model and measuring adapter-switch overhead and caching effects.

All variants are evaluated on latency, throughput, GPU memory usage, and output quality across different workloads.

## Experiments

| # | Experiment | Description |
|---|---|---|
| 0 | **Baseline** | Hugging Face Transformers + FastAPI, single-request serving, no optimization |
| 1 | **Serving Framework** | Replace baseline with vLLM; measure improvements from PagedAttention and continuous batching |
| 2 | **Quantization** | Compare FP16 → INT8 (GPTQ) → INT4 (AWQ); measure speed/memory/quality tradeoff |
| 3 | **Concurrency** | Simulate concurrent users (1, 4, 8, 16 clients); measure throughput scaling and tail latency |
| 4 | **Adapter-Aware Serving** | Deploy shared base model + 3 LoRA adapters via vLLM multi-LoRA; compare static loading vs. dynamic loading; measure adapter-switch overhead |

### Bonus (If Time Permits)

- Cross-platform comparison: vLLM (CUDA) vs. llama.cpp (Apple Metal)
- Streaming response latency (TTFT vs. full generation)
- Long prompt vs. short prompt performance scaling

## Model

**meta-llama/Llama-3.2-3B-Instruct**

| Precision | Approx. VRAM |
|---|---|
| FP16 | ~6 GB |
| INT8 (GPTQ) | ~3–4 GB |
| INT4 (AWQ) | ~2 GB |
| LoRA adapter (each) | ~10–30 MB |

## Hardware

| Device | Role |
|---|---|
| NVIDIA RTX 2080 (8 GB) | Primary experiment machine — vLLM, quantization, multi-LoRA, benchmarks |
| Apple M1 Pro (16 GB) | LoRA adapter training (PEFT + MPS), llama.cpp cross-platform experiments |
| Apple M2 (8 GB) | Benchmark scripting, load generator, evaluation pipeline |
| Apple M1 (8 GB) | Visualization, result analysis, presentation |
| Google Colab (T4 16 GB) | Supplementary runs for reproducibility validation |

### Known Constraints

- **vLLM requires CUDA** — all vLLM experiments run exclusively on RTX 2080.
- **Windows WDDM reserves ~1 GB VRAM** — only ~6.9 GB available on the RTX 2080. FP16 model requires `--enforce-eager` and `--max-model-len 1024` to fit.
- **FlashAttention 2 not supported on RTX 2080** (compute capability 7.5 < 8.0) — vLLM falls back to FlashInfer automatically.

## Setup

### Prerequisites

- Docker Desktop with GPU support enabled
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- HuggingFace account with access to [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 1. Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. Create Environment File

Create a `.env` file in the project root:

```
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

### 3. Start vLLM Server

```powershell
docker run -d --gpus all `
  -v ${HOME}\.cache\huggingface:/root/.cache/huggingface `
  -p 8000:8000 `
  --name vllm-server `
  --env-file .env `
  vllm/vllm-openai:latest `
  --model meta-llama/Llama-3.2-3B-Instruct `
  --dtype float16 `
  --max-model-len 1024 `
  --gpu-memory-utilization 0.85 `
  --enforce-eager
```

Wait for startup (check logs):

```bash
docker logs -f vllm-server
# Ready when you see: "Started server process"
```

### 4. Verify vLLM is Running

```bash
uv run python -c "import httpx; r = httpx.post('http://localhost:8000/v1/completions', json={'model':'meta-llama/Llama-3.2-3B-Instruct','prompt':'Hello','max_tokens':64}); print(r.json())"
```

### 5. Install Python Dependencies

```bash
uv add aiohttp httpx matplotlib seaborn pandas
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token |
| **End-to-end latency** | Full request completion time |
| **Throughput** | Tokens/sec and requests/sec under load |
| **P95/P99 latency** | Tail latency under concurrent load |
| **GPU memory usage** | Peak VRAM consumption |
| **Output quality** | Perplexity on held-out set |
| **Adapter switch overhead** | Cold-load vs. warm-hit latency per adapter (Exp 4) |

### Workload Design

- **Single-user**: 100 sequential prompts, ~128 tokens input, 256 tokens output
- **Concurrent**: 1 / 4 / 8 / 16 simultaneous clients
- **Mixed-adapter** (Exp 4): 3 adapters, skewed distribution (70-20-10)

## Usage

### Run Benchmark

```bash
# Single client, 50 requests against vLLM
uv run benchmark/client.py 50 1

# 8 concurrent clients, 100 requests
uv run benchmark/client.py 100 8
```

### Switch Between Serving Backends

```powershell
# Stop vLLM
docker stop vllm-server

# Start HF baseline
docker run --gpus all `
  -v C:\shanshou\CS6180-FinalProject:/workspace `
  -v ${HOME}\.cache\huggingface:/root/.cache/huggingface `
  -p 8001:8001 `
  --name hf-baseline `
  --env-file .env `
  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime `
  bash -c "pip install transformers fastapi uvicorn accelerate && cd /workspace && uvicorn serving.baseline_hf:app --host 0.0.0.0 --port 8001"

# After baseline experiments, switch back
docker stop hf-baseline
docker start vllm-server
```

## Project Structure

```
CS6180-FinalProject/
├── README.md
├── .env                     # HuggingFace token (git-ignored)
├── pyproject.toml           # Python dependencies (managed by uv)
├── uv.lock                  # Dependency lockfile
├── benchmark/
│   ├── client.py            # Async benchmark client
│   ├── workloads/           # Prompt datasets and request patterns
│   └── metrics.py           # Metric collection and aggregation
├── serving/
│   ├── baseline_hf.py       # HuggingFace + FastAPI baseline server
│   └── router/              # Exp 4: adapter routing + LRU cache
│       ├── router.py
│       └── cache.py
├── evaluation/
│   ├── quality.py           # Perplexity / MMLU evaluation
│   └── analysis.py          # Result aggregation and chart generation
├── adapters/
│   ├── train_lora.py        # PEFT training script
│   └── configs/             # Adapter training configs
├── results/
│   ├── raw/                 # Raw experiment logs and CSVs
│   └── figures/             # Generated charts
└── presentation/
    └── slides.pptx
```
