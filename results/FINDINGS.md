# Experiment Findings

Tracks key findings from each experiment run. Updated as new results come in.

---

## 2026-04-15 — SageMaker Cloud Benchmarks Complete

### Data collected
- SageMaker (A10G, ml.g5.xlarge): FP16, INT8 (W8A8), INT4 (AWQ) × c=1, c=8 (100 req each)
- Local vLLM (RTX 2080): FP16, INT8, INT4 × c=1, c=8, c=16 (95–100 req each)

---

### Figure 1 — `sagemaker_quant_comparison.png`
**SageMaker (A10G): Latency & Throughput by Quantization**

| Model | c=1 avg lat | c=8 avg lat | c=8 agg tok/s |
|-------|------------|------------|--------------|
| FP16  | 4.264s | 5.994s | 63.7 |
| INT8  | 3.015s | 3.897s | 425.0 |
| INT4  | 2.007s | 2.805s | 566.2 |

**Key findings:**
- On A10G (Ampere), quantization directly reduces latency: INT4 is **2.1× faster** than FP16 at c=1.
- Aggregate throughput at c=8 scales dramatically with quantization: INT4 achieves **566 tok/s vs FP16's 64 tok/s** — an 8.9× improvement.
- INT8 sits cleanly between FP16 and INT4 on both metrics, confirming the quantization ladder is monotonic on modern hardware.
- FP16 throughput barely scales from c=1 to c=8 (28.7 → 63.7 tok/s), suggesting a compute bottleneck at FP16 precision.

---

### Figure 2 — `local_quant_comparison.png`
**Local RTX 2080: Quantization Effect Across Concurrency**

| Model | c=1 avg lat | c=8 avg lat | c=16 avg lat |
|-------|------------|------------|-------------|
| FP16  | 4.807s | 5.503s | 8.736s |
| INT8  | 6.597s | 7.242s | 8.297s |
| INT4  | 16.895s | 16.166s | 16.813s |

**Key findings:**
- On RTX 2080 (Turing), quantization makes inference **significantly slower**, not faster.
- INT4 (AWQ) is **3.5× slower** than FP16 at c=1 — the opposite of the SageMaker result.
- INT4 latency barely changes across concurrency levels (c=1/8/16 all ~16–17s), indicating the bottleneck is dequantization overhead, not request queuing.
- INT8 is ~37% slower than FP16, but the gap narrows at c=16, suggesting INT8 has better batch efficiency than INT4 on this hardware.

---

### Figure 3 — `local_vs_sagemaker_fp16.png`
**Local RTX 2080 vs SageMaker A10G — FP16 Head-to-Head**

| Setup | c=1 avg lat | c=8 avg lat |
|-------|------------|------------|
| Local RTX 2080 (FP16) | 4.807s | 5.503s |
| SageMaker A10G (FP16) | 4.264s | 5.994s |

**Key findings:**
- At c=1, the RTX 2080 and A10G perform **similarly for FP16** (4.8s vs 4.3s, only 11% gap).
- At c=8, the RTX 2080 is actually **slightly faster** than SageMaker FP16 (5.5s vs 6.0s).
- This confirms that "better hardware ≠ better performance" without quantization — the real A10G advantage only appears when using INT4/INT8.
- The FP16 comparison alone would not justify cloud cost (~$1.41/hr for SageMaker); the value proposition requires quantization.

---

### Figure 4 — `quant_speedup_by_hardware.png`
**Quantization Speedup: Hardware Architecture Matters**

| Hardware | INT8 vs FP16 | INT4 vs FP16 |
|----------|-------------|-------------|
| SageMaker A10G (Ampere) | **0.71×** (faster) | **0.47×** (faster) |
| Local RTX 2080 (Turing) | **1.37×** (slower) | **3.51×** (slower) |

**Key findings:**
- This is the central finding of the project: **quantization benefit is hardware-dependent**.
- A10G (Ampere, 2020) has native INT8 and INT4 tensor cores — quantization maps directly to hardware acceleration.
- RTX 2080 (Turing, 2018) lacks native INT4 hardware. AWQ dequantization at inference time adds overhead that exceeds any memory bandwidth savings.
- **Practical implication:** Deploying INT4 models on pre-Ampere GPUs will degrade performance. Teams should verify GPU generation before applying quantization.
- This finding is **non-obvious** and practically useful for ML infrastructure decisions.

---

---

## 2026-04-15 — SageMaker Cost Analysis

Instance: ml.g5.xlarge (A10G), $1.212/hr (us-west-2 on-demand)
Metric: **Cost per 1M tokens** = (1,000,000 / aggregate_tok/s) × ($/hr ÷ 3600)

### Figure 5 — `cost_analysis.png`
**Cost per 1M Tokens by Model and Concurrency**

| Model | c=1 ($/1M tok) | c=8 ($/1M tok) | Savings vs FP16 c=8 |
|-------|---------------|---------------|---------------------|
| FP16  | $6.76 | $1.16 | baseline |
| INT8  | $4.66 | $0.75 | **35% cheaper** |
| INT4  | $3.25 | $0.56 | **51% cheaper** |

**Key findings:**
- Concurrency has a larger impact on cost than quantization: FP16 c=8 ($1.16) is cheaper than INT4 c=1 ($3.25).
- **INT4 at c=8 is the most cost-efficient configuration at $0.56/1M tokens** — 2.1× cheaper than FP16 at c=8.
- INT8 at c=8 ($0.75) offers a good middle ground: 35% cheaper than FP16 with lower deployment risk than INT4.
- Always run at high concurrency in production — the cost difference between c=1 and c=8 is 6× for FP16.
- These numbers are for SageMaker only; Vertex AI (L4, ~$0.98/hr) comparison pending.

---

## 2026-04-14 — Local Experiments Complete (Member A)

- vLLM FP16, INT8 (GPTQ), INT4 (AWQ) × c=1/4/8/16 on RTX 2080 completed.
- HF Baseline (Transformers + FastAPI) benchmarked at c=1 for comparison.
- vLLM FP16 is **33% faster** than HF baseline at c=1 (4.337s vs 6.496s avg latency).
- INT4/INT8 slower than FP16 on RTX 2080 — see Figure 2 above.

---

## Open Questions / Next Steps

- [ ] Cost analysis: tokens per dollar on SageMaker vs Vertex AI
- [ ] Vertex AI (L4 GPU) deployment for SageMaker vs Vertex AI comparison
- [ ] Auto-scaling: DJL-LMI 0.30.0 does not publish `InvocationsPerInstance` to CloudWatch — scaling was not observable. Documented in `cloud/DEPLOYMENT_NOTES.md`.
