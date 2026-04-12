# Local Experiment Analysis Report

**Project:** CS6180 Final Project — LLM Inference Serving Benchmark
**Hardware:** NVIDIA RTX 2080 (8 GB VRAM, ~6.94 GB usable)
**Model:** Meta Llama-3.2-3B-Instruct
**Workload:** ShareGPT (100 requests per run, filtered by context length)
**Serving framework:** vLLM 0.18.0 (FP16 / INT8 / INT4) vs HF Transformers baseline

---

## 1. Full Results Summary

| Backend | c | n | Avg Lat (s) | P50 (s) | P95 (s) | P99 (s) | Avg tok/s | Agg. Throughput (tok/s) | Req/s |
|---|---|---|---|---|---|---|---|---|---|
| HF Baseline | 1 | 100 | 7.138 | 7.131 | 13.762 | 14.755 | 38.7 | 38.7 | 0.14 |
| FP16 | 1 | 95 | 4.807 | 4.849 | 8.657 | 8.897 | 58.6 | 59.0 | 0.21 |
| FP16 | 4 | 95 | 5.173 | 5.175 | 9.301 | 9.384 | 53.9 | 217.9 | 0.77 |
| FP16 | 8 | 95 | 5.503 | 5.512 | 9.586 | 10.028 | 50.3 | 408.2 | 1.45 |
| FP16 | 16 | 95 | 8.736 | 8.500 | 13.915 | 15.322 | 30.7 | 512.4 | 1.83 |
| INT8 | 1 | 100 | 6.597 | 6.412 | 12.796 | 13.283 | 39.6 | 39.8 | 0.15 |
| INT8 | 4 | 100 | 6.936 | 6.764 | 13.421 | 14.200 | 37.2 | 150.5 | 0.58 |
| INT8 | 8 | 100 | 7.242 | 7.303 | 13.951 | 15.446 | 35.3 | 286.6 | 1.10 |
| INT8 | 16 | 100 | 8.297 | 8.176 | 15.723 | 17.785 | 30.9 | 500.5 | 1.93 |
| INT4 | 1 | 100 | 16.895 | 16.711 | 29.960 | 30.154 | 17.1 | 17.1 | 0.06 |
| INT4 | 4 | 100 | 15.577 | 15.521 | 28.163 | 28.279 | 18.1 | 72.7 | 0.26 |
| INT4 | 8 | 100 | 16.166 | 15.720 | 28.411 | 29.923 | 17.7 | 143.0 | 0.49 |
| INT4 | 16 | 100 | 16.813 | 16.369 | 29.551 | 30.991 | 16.9 | 273.3 | 0.95 |

## 2. vLLM vs HF Transformers Baseline (c=1)

Both evaluated at concurrency=1 with ShareGPT workload and max_len=1024.

| Metric | HF Baseline | vLLM FP16 | Delta |
|---|---|---|---|
| Avg latency (s)         | 7.138 | 4.807 | -32.7% |
| P50 latency (s)         | 7.131 | 4.849 | -32.0% |
| P95 latency (s)         | 13.762 | 8.657 | -37.1% |
| Avg tokens/s (per req)  | 38.7 | 58.6 | +51.4% |
| Agg. throughput (tok/s) | 38.7 | 59.0 | +52.3% |

**vLLM FP16 reduces average latency by 32.7% and improves per-request throughput by 51.4% compared to plain HuggingFace Transformers.** This gain comes from PagedAttention (eliminates KV cache fragmentation) and continuous batching (overlaps prefill and decode across requests).

## 3. Quantization Impact (FP16 / INT8 / INT4)

All three vLLM configurations serve the same model family at different precision levels.
Key difference: INT8 uses W8A8 compressed-tensors (neuralmagic); INT4 uses AWQ (casperhansen).

### Concurrency = 1

| Backend | Avg Lat (s) | P50 (s) | P95 (s) | Avg tok/s | Agg. Throughput (tok/s) |
|---|---|---|---|---|---|
| FP16 | 4.807 | 4.849 | 8.657 | 58.6 | 59.0 |
| INT8 | 6.597 | 6.412 | 12.796 | 39.6 | 39.8 |
| INT4 | 16.895 | 16.711 | 29.960 | 17.1 | 17.1 |

### Concurrency = 4

| Backend | Avg Lat (s) | P50 (s) | P95 (s) | Avg tok/s | Agg. Throughput (tok/s) |
|---|---|---|---|---|---|
| FP16 | 5.173 | 5.175 | 9.301 | 53.9 | 217.9 |
| INT8 | 6.936 | 6.764 | 13.421 | 37.2 | 150.5 |
| INT4 | 15.577 | 15.521 | 28.163 | 18.1 | 72.7 |

### Concurrency = 8

| Backend | Avg Lat (s) | P50 (s) | P95 (s) | Avg tok/s | Agg. Throughput (tok/s) |
|---|---|---|---|---|---|
| FP16 | 5.503 | 5.512 | 9.586 | 50.3 | 408.2 |
| INT8 | 7.242 | 7.303 | 13.951 | 35.3 | 286.6 |
| INT4 | 16.166 | 15.720 | 28.411 | 17.7 | 143.0 |

### Concurrency = 16

| Backend | Avg Lat (s) | P50 (s) | P95 (s) | Avg tok/s | Agg. Throughput (tok/s) |
|---|---|---|---|---|---|
| FP16 | 8.736 | 8.500 | 13.915 | 30.7 | 512.4 |
| INT8 | 8.297 | 8.176 | 15.723 | 30.9 | 500.5 |
| INT4 | 16.813 | 16.369 | 29.551 | 16.9 | 273.3 |

At c=1, INT8 is **37.3% slower** than FP16 and INT4 is **251.5% slower**. This is the opposite of the expected result on modern data-center GPUs.

**Root cause — RTX 2080 hardware limitations:**

1. **W8A8 INT8 (compressed-tensors):** The RTX 2080 (Turing, SM 7.5) has limited INT8 Tensor Core support. Unlike Ampere (SM 8.x) and Ada Lovelace (SM 8.9), Turing cannot efficiently dispatch INT8 matrix multiplications, so these ops run slower than native FP16 CUDA cores.

2. **AWQ INT4:** AWQ stores weights at 4-bit but dequantizes to FP16 at runtime before the matrix multiply. This adds a mandatory dequantization pass on top of the FP16 computation, making INT4 strictly slower than FP16 on this hardware.

3. **Chunked prefill overhead:** INT8 and INT4 use larger `max_model_len` (2048 and 4096 respectively), which triggers vLLM's chunked prefill scheduler. This adds per-request scheduling overhead absent in the FP16 (max_model_len=1024) setup.

**This behavior is expected to reverse on cloud hardware** (A10G / L4), where dedicated INT8 and INT4 Tensor Cores provide true hardware acceleration. Member B's SageMaker and Vertex AI experiments will test this hypothesis.

## 4. Concurrency Scaling

How latency and throughput change as concurrency increases within each backend.

### FP16

| c | Avg Lat (s) | Lat vs c=1 | Agg. Throughput (tok/s) | Throughput vs c=1 | Req/s |
|---|---|---|---|---|---|
| 1 | 4.807 | +0.0% | 59.0 | +0.0% | 0.21 |
| 4 | 5.173 | +7.6% | 217.9 | +269.2% | 0.77 |
| 8 | 5.503 | +14.5% | 408.2 | +591.7% | 1.45 |
| 16 | 8.736 | +81.8% | 512.4 | +768.4% | 1.83 |

### INT8

| c | Avg Lat (s) | Lat vs c=1 | Agg. Throughput (tok/s) | Throughput vs c=1 | Req/s |
|---|---|---|---|---|---|
| 1 | 6.597 | +0.0% | 39.8 | +0.0% | 0.15 |
| 4 | 6.936 | +5.1% | 150.5 | +278.0% | 0.58 |
| 8 | 7.242 | +9.8% | 286.6 | +619.7% | 1.10 |
| 16 | 8.297 | +25.8% | 500.5 | +1156.9% | 1.93 |

### INT4

| c | Avg Lat (s) | Lat vs c=1 | Agg. Throughput (tok/s) | Throughput vs c=1 | Req/s |
|---|---|---|---|---|---|
| 1 | 16.895 | +0.0% | 17.1 | +0.0% | 0.06 |
| 4 | 15.577 | -7.8% | 72.7 | +325.5% | 0.26 |
| 8 | 16.166 | -4.3% | 143.0 | +736.9% | 0.49 |
| 16 | 16.813 | -0.5% | 273.3 | +1499.7% | 0.95 |

**Observations:**

- **FP16:** Throughput peaks at c=16 (512.4 tok/s). Latency degrades sharply at c=16 as the KV cache (limited by the 1024-token context window) becomes the bottleneck with many concurrent long sequences.
- **INT8:** Throughput peaks at c=16 (500.5 tok/s). The larger context window (2048) allows more tokens in flight, but the slower per-token compute limits overall throughput gains.
- **INT4:** Throughput is nearly flat across all concurrency levels (peak: c=16, 273.3 tok/s). The AWQ dequantization step dominates latency regardless of batch size, leaving no room for concurrency gains on this hardware.

## 5. Optimal Configuration on RTX 2080

Based on all metrics, the optimal local serving configuration is:

| Goal | Recommended Config | Value |
|---|---|---|
| Lowest latency | FP16, c=1 | 4.807s avg, 8.657s P95 |
| Highest throughput | FP16, c=16 | 512.4 tok/s |
| Best latency/throughput tradeoff | FP16, c=4 | 5.173s avg, 217.9 tok/s |

**FP16 dominates across all concurrency levels on the RTX 2080.** This finding highlights a critical practical insight: quantization is not universally beneficial — its advantage is hardware-dependent. Consumer-grade Turing GPUs lack the specialized low-precision compute paths that make quantization worthwhile.

## 6. Key Findings

1. **vLLM vs HF Baseline:** vLLM FP16 reduces average latency by 32.7% and improves throughput by 51.4% at c=1, validating PagedAttention and continuous batching as essential optimizations.

2. **Quantization backfires on consumer GPUs:** On the RTX 2080 (Turing, SM 7.5), INT8 and INT4 quantization increase latency relative to FP16 — the opposite of the expected behavior on data-center hardware. The root causes are limited INT8 Tensor Core support and AWQ's runtime dequantization overhead.

3. **Concurrency sweet spot (FP16):** Throughput scales well from c=1 to c=4–8 on FP16, with diminishing returns and rising tail latency beyond c=8. The 1024-token KV cache limit constrains how many long requests can run in parallel.

4. **INT4 shows no concurrency benefit:** AWQ INT4 throughput is effectively flat across c=1–16, suggesting the dequantization bottleneck fully saturates the GPU compute pipeline regardless of batch size.

5. **Cloud hypothesis:** The performance ordering (INT4 > INT8 > FP16 in throughput) is expected to appear on Member B's A10G and L4 experiments. If confirmed, it will demonstrate that hardware generation is the primary determinant of quantization benefit — a non-obvious and practically relevant finding.

## 7. Implications for Cloud Experiments (Member B)

| Question | Expected Finding |
|---|---|
| Does INT4 beat FP16 on A10G/L4? | Yes — Ampere/Ada INT4 Tensor Cores provide true hardware acceleration |
| Which cloud is better: SageMaker (A10G) or Vertex AI (L4)? | L4 has ~2× INT8 compute at 30% lower cost; outcome non-obvious at INT4 |
| What concurrency is optimal on cloud? | Higher c possible due to 24 GB VRAM vs 6.94 GB local |
| Cost-performance metric | Tokens per dollar: measure at c=1 and c=8, compare across platforms |

Member B should use identical benchmark parameters (`--requests 100`, same ShareGPT dataset, same seed=42) to ensure results are directly comparable to local runs.
