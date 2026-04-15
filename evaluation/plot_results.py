"""
Generate benchmark charts for the paper/presentation.
Run: uv run --python 3.11 python evaluation/plot_results.py
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/figures", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"latency": float(r["latency"]), "tokens": int(r["tokens"])})
    return rows

def summary(rows):
    lats = sorted(r["latency"] for r in rows)
    n = len(lats)
    total_tok = sum(r["tokens"] for r in rows)
    wall = sum(lats)
    return {
        "avg_lat": sum(lats) / n,
        "p50":     lats[int(n * 0.50)],
        "p95":     lats[int(n * 0.95)],
        "agg_tps": total_tok / wall,
        "avg_tps": sum(r["tokens"] / r["latency"] for r in rows) / n,
    }

# SageMaker data
sm = {
    ("FP16",  1): summary(load("results/raw/sagemaker_fp16_c1_n100.csv")),
    ("FP16",  8): summary(load("results/raw/sagemaker_fp16_c8_n100.csv")),
    ("INT8",  1): summary(load("results/raw/sagemaker_int8gptq_c1_n100.csv")),
    ("INT8",  8): summary(load("results/raw/sagemaker_int8gptq_c8_n100.csv")),
    ("INT4",  1): summary(load("results/raw/sagemaker_int4awq_c1_n100.csv")),
    ("INT4",  8): summary(load("results/raw/sagemaker_int4awq_c8_n100.csv")),
}

# Local data
local = {
    ("FP16",  1):  summary(load("results/raw/bench_c1_n95_maxlen1024.csv")),
    ("FP16",  8):  summary(load("results/raw/bench_c8_n95_maxlen1024.csv")),
    ("FP16", 16):  summary(load("results/raw/bench_c16_n95_maxlen1024.csv")),
    ("INT8",  1):  summary(load("results/raw/bench_c1_n100_maxlen2048.csv")),
    ("INT8",  8):  summary(load("results/raw/bench_c8_n100_maxlen2048.csv")),
    ("INT8", 16):  summary(load("results/raw/bench_c16_n100_maxlen2048.csv")),
    ("INT4",  1):  summary(load("results/raw/bench_c1_n100_maxlen4096.csv")),
    ("INT4",  8):  summary(load("results/raw/bench_c8_n100_maxlen4096.csv")),
    ("INT4", 16):  summary(load("results/raw/bench_c16_n100_maxlen4096.csv")),
}

COLORS = {"FP16": "#4C72B0", "INT8": "#55A868", "INT4": "#C44E52"}
MODELS = ["FP16", "INT8", "INT4"]

# ── Figure 1: SageMaker avg latency by quant × concurrency ────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SageMaker (A10G) — Latency & Throughput by Quantization", fontsize=13, fontweight="bold")

# Left: avg latency grouped bar
ax = axes[0]
x = np.arange(2)
width = 0.25
for i, m in enumerate(MODELS):
    vals = [sm[(m, 1)]["avg_lat"], sm[(m, 8)]["avg_lat"]]
    bars = ax.bar(x + i * width, vals, width, label=m, color=COLORS[m])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(["Concurrency = 1", "Concurrency = 8"])
ax.set_ylabel("Avg Latency (s)")
ax.set_title("Avg Latency (lower = better)")
ax.legend()
ax.set_ylim(0, 8)

# Right: aggregate throughput grouped bar
ax = axes[1]
for i, m in enumerate(MODELS):
    vals = [sm[(m, 1)]["agg_tps"], sm[(m, 8)]["agg_tps"]]
    bars = ax.bar(x + i * width, vals, width, label=m, color=COLORS[m])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(["Concurrency = 1", "Concurrency = 8"])
ax.set_ylabel("Aggregate Tokens/s")
ax.set_title("Aggregate Throughput (higher = better)")
ax.legend()

plt.tight_layout()
plt.savefig("results/figures/sagemaker_quant_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: sagemaker_quant_comparison.png")

# ── Figure 2: Local RTX 2080 — quant effect across concurrency ────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Local vLLM (RTX 2080) — Quantization Effect", fontsize=13, fontweight="bold")

concs = [1, 8, 16]
x = np.arange(len(concs))
width = 0.25

# Left: avg latency
ax = axes[0]
for i, m in enumerate(MODELS):
    vals = [local[(m, c)]["avg_lat"] for c in concs]
    bars = ax.bar(x + i * width, vals, width, label=m, color=COLORS[m])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}s", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width)
ax.set_xticklabels(["c=1", "c=8", "c=16"])
ax.set_xlabel("Concurrency")
ax.set_ylabel("Avg Latency (s)")
ax.set_title("Avg Latency — INT4/INT8 SLOWER than FP16!")
ax.legend()
ax.set_ylim(0, 22)

# Right: avg per-request tok/s
ax = axes[1]
for i, m in enumerate(MODELS):
    vals = [local[(m, c)]["avg_tps"] for c in concs]
    bars = ax.bar(x + i * width, vals, width, label=m, color=COLORS[m])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x + width)
ax.set_xticklabels(["c=1", "c=8", "c=16"])
ax.set_xlabel("Concurrency")
ax.set_ylabel("Avg Tokens/s per Request")
ax.set_title("Per-Request Throughput")
ax.legend()

plt.tight_layout()
plt.savefig("results/figures/local_quant_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: local_quant_comparison.png")

# ── Figure 3: SageMaker vs Local FP16 head-to-head ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["c=1", "c=8"]
sm_vals   = [sm[("FP16", c)]["avg_lat"]    for c in [1, 8]]
local_vals= [local[("FP16", c)]["avg_lat"] for c in [1, 8]]
x = np.arange(2)
w = 0.3
b1 = ax.bar(x - w/2, local_vals, w, label="Local RTX 2080 (FP16)", color="#4C72B0")
b2 = ax.bar(x + w/2, sm_vals,    w, label="SageMaker A10G (FP16)", color="#DD8452")
for bar, v in list(zip(b1, local_vals)) + list(zip(b2, sm_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Concurrency = 1", "Concurrency = 8"])
ax.set_ylabel("Avg Latency (s)")
ax.set_title("Local RTX 2080 vs SageMaker A10G — FP16 Latency Comparison")
ax.legend()
ax.set_ylim(0, 8)
plt.tight_layout()
plt.savefig("results/figures/local_vs_sagemaker_fp16.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: local_vs_sagemaker_fp16.png")

# ── Figure 4: Quantization benefit — GPU architecture matters ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Quantization Speedup: Hardware Matters\n(INT4 vs FP16, relative latency reduction)", fontsize=12, fontweight="bold")

# Left: SageMaker — INT4 helps
ax = axes[0]
fp16_lat = sm[("FP16", 1)]["avg_lat"]
vals = [(sm[(m, 1)]["avg_lat"] / fp16_lat) for m in MODELS]
bars = ax.bar(MODELS, vals, color=[COLORS[m] for m in MODELS])
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="FP16 baseline")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Relative Latency (1.0 = FP16)")
ax.set_title("SageMaker A10G (Ampere)\nQuantization HELPS ✅")
ax.set_ylim(0, 1.4)
ax.legend()

# Right: Local — INT4 hurts
ax = axes[1]
fp16_lat = local[("FP16", 1)]["avg_lat"]
vals = [(local[(m, 1)]["avg_lat"] / fp16_lat) for m in MODELS]
bars = ax.bar(MODELS, vals, color=[COLORS[m] for m in MODELS])
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="FP16 baseline")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Relative Latency (1.0 = FP16)")
ax.set_title("Local RTX 2080 (Turing)\nQuantization HURTS ❌")
ax.set_ylim(0, 4.5)
ax.legend()

plt.tight_layout()
plt.savefig("results/figures/quant_speedup_by_hardware.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: quant_speedup_by_hardware.png")

print("\nAll figures saved to results/figures/")
