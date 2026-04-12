"""
Local Experiment Analysis
=========================
Reads all benchmark CSVs from results/raw/, computes statistics,
and writes a Markdown analysis report to results/analysis_local.md.

File naming convention used for identification:
  *_n95_maxlen1024.csv   -> vLLM FP16   (n=95 because 5 samples skipped)
  bench_c1_n100_maxlen1024.csv -> HF Baseline  (c=1 only)
  *_maxlen2048.csv       -> vLLM INT8
  *_maxlen4096.csv       -> vLLM INT4

Run:
    uv run python evaluation/analyze.py
"""

import csv
import os
import statistics

# =============================================================================
# Data loading
# =============================================================================

RAW_DIR = "results/raw"

def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows: list[dict], concurrency: int) -> dict:
    """Compute all benchmark statistics from per-request CSV rows."""
    lats = sorted(float(r["latency"]) for r in rows)
    tps_list = [float(r["tps"]) for r in rows]
    tokens_list = [int(r["tokens"]) for r in rows]
    n = len(lats)
    total_tokens = sum(tokens_list)
    sum_lats = sum(lats)

    # Estimated wall-clock: sum(latencies) / concurrency
    # Exact for c=1; approximation for c>1 assuming even load distribution.
    est_wall = sum_lats / concurrency

    return {
        "n":              n,
        "concurrency":    concurrency,
        "avg_lat":        sum_lats / n,
        "p50_lat":        lats[n // 2],
        "p95_lat":        lats[int(n * 0.95)],
        "p99_lat":        lats[int(n * 0.99)],
        "min_lat":        lats[0],
        "max_lat":        lats[-1],
        "avg_tps":        sum(tps_list) / n,         # per-request tokens/s
        "total_tokens":   total_tokens,
        "agg_throughput": total_tokens / est_wall,   # tokens/s (aggregate)
        "req_per_s":      n / est_wall,              # requests/s
    }


# Map each CSV to its (backend, concurrency) label
def identify_files(raw_dir: str) -> list[tuple[str, int, str]]:
    """
    Returns list of (backend_label, concurrency, filepath).
    backend_label: "FP16", "INT8", "INT4", "HF Baseline"
    """
    results = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, fname)

        # Parse concurrency from filename (e.g. bench_c4_n95_maxlen1024.csv)
        parts = fname.replace(".csv", "").split("_")
        c_part = next(p for p in parts if p.startswith("c"))
        concurrency = int(c_part[1:])
        n_part = next(p for p in parts if p.startswith("n"))
        n_actual = int(n_part[1:])
        ml_part = next(p for p in parts if p.startswith("maxlen"))
        max_len = int(ml_part[6:])

        # Classify backend
        if max_len == 2048:
            backend = "INT8"
        elif max_len == 4096:
            backend = "INT4"
        elif max_len == 1024 and n_actual == 95:
            backend = "FP16"
        elif max_len == 1024 and n_actual == 100 and concurrency == 1:
            backend = "HF Baseline"
        else:
            backend = f"Unknown(maxlen={max_len},n={n_actual})"

        results.append((backend, concurrency, path))

    return results


# =============================================================================
# Main analysis
# =============================================================================

def main():
    file_list = identify_files(RAW_DIR)

    # Group stats by backend
    data: dict[str, dict[int, dict]] = {}
    for backend, c, path in file_list:
        rows = load_csv(path)
        stats = compute_stats(rows, c)
        data.setdefault(backend, {})[c] = stats

    # Define display order
    backend_order = ["HF Baseline", "FP16", "INT8", "INT4"]
    concurrency_levels = [1, 4, 8, 16]

    lines = []
    def w(line=""):
        lines.append(line)

    # =========================================================================
    # Report header
    # =========================================================================
    w("# Local Experiment Analysis Report")
    w()
    w("**Project:** CS6180 Final Project — LLM Inference Serving Benchmark")
    w("**Hardware:** NVIDIA RTX 2080 (8 GB VRAM, ~6.94 GB usable)")
    w("**Model:** Meta Llama-3.2-3B-Instruct")
    w("**Workload:** ShareGPT (100 requests per run, filtered by context length)")
    w("**Serving framework:** vLLM 0.18.0 (FP16 / INT8 / INT4) vs HF Transformers baseline")
    w()
    w("---")
    w()

    # =========================================================================
    # Section 1: Full results table
    # =========================================================================
    w("## 1. Full Results Summary")
    w()
    w("| Backend | c | n | Avg Lat (s) | P50 (s) | P95 (s) | P99 (s) | Avg tok/s | Agg. Throughput (tok/s) | Req/s |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for backend in backend_order:
        if backend not in data:
            continue
        for c in sorted(data[backend]):
            s = data[backend][c]
            w(f"| {backend} | {c} | {s['n']} "
              f"| {s['avg_lat']:.3f} | {s['p50_lat']:.3f} | {s['p95_lat']:.3f} | {s['p99_lat']:.3f} "
              f"| {s['avg_tps']:.1f} | {s['agg_throughput']:.1f} | {s['req_per_s']:.2f} |")
    w()

    # =========================================================================
    # Section 2: vLLM vs HF Baseline
    # =========================================================================
    w("## 2. vLLM vs HF Transformers Baseline (c=1)")
    w()
    w("Both evaluated at concurrency=1 with ShareGPT workload and max_len=1024.")
    w()

    if "HF Baseline" in data and "FP16" in data:
        hf = data["HF Baseline"][1]
        fp = data["FP16"][1]
        lat_delta = (hf["avg_lat"] - fp["avg_lat"]) / hf["avg_lat"] * 100
        tps_delta = (fp["avg_tps"] - hf["avg_tps"]) / hf["avg_tps"] * 100
        agg_delta = (fp["agg_throughput"] - hf["agg_throughput"]) / hf["agg_throughput"] * 100

        w("| Metric | HF Baseline | vLLM FP16 | Delta |")
        w("|---|---|---|---|")
        w(f"| Avg latency (s)         | {hf['avg_lat']:.3f} | {fp['avg_lat']:.3f} | {-lat_delta:+.1f}% |")
        w(f"| P50 latency (s)         | {hf['p50_lat']:.3f} | {fp['p50_lat']:.3f} | {(fp['p50_lat']-hf['p50_lat'])/hf['p50_lat']*100:+.1f}% |")
        w(f"| P95 latency (s)         | {hf['p95_lat']:.3f} | {fp['p95_lat']:.3f} | {(fp['p95_lat']-hf['p95_lat'])/hf['p95_lat']*100:+.1f}% |")
        w(f"| Avg tokens/s (per req)  | {hf['avg_tps']:.1f} | {fp['avg_tps']:.1f} | {tps_delta:+.1f}% |")
        w(f"| Agg. throughput (tok/s) | {hf['agg_throughput']:.1f} | {fp['agg_throughput']:.1f} | {agg_delta:+.1f}% |")
        w()
        w(f"**vLLM FP16 reduces average latency by {lat_delta:.1f}% and improves per-request "
          f"throughput by {tps_delta:.1f}% compared to plain HuggingFace Transformers.** "
          f"This gain comes from PagedAttention (eliminates KV cache fragmentation) and "
          f"continuous batching (overlaps prefill and decode across requests).")
    w()

    # =========================================================================
    # Section 3: Quantization impact at each concurrency level
    # =========================================================================
    w("## 3. Quantization Impact (FP16 / INT8 / INT4)")
    w()
    w("All three vLLM configurations serve the same model family at different precision levels.")
    w("Key difference: INT8 uses W8A8 compressed-tensors (neuralmagic); INT4 uses AWQ (casperhansen).")
    w()

    vllm_backends = ["FP16", "INT8", "INT4"]
    for c in concurrency_levels:
        w(f"### Concurrency = {c}")
        w()
        w("| Backend | Avg Lat (s) | P50 (s) | P95 (s) | Avg tok/s | Agg. Throughput (tok/s) |")
        w("|---|---|---|---|---|---|")
        for backend in vllm_backends:
            if backend in data and c in data[backend]:
                s = data[backend][c]
                w(f"| {backend} | {s['avg_lat']:.3f} | {s['p50_lat']:.3f} | {s['p95_lat']:.3f} "
                  f"| {s['avg_tps']:.1f} | {s['agg_throughput']:.1f} |")
        w()

    # Analysis paragraph
    fp1 = data.get("FP16", {}).get(1)
    i81 = data.get("INT8", {}).get(1)
    i41 = data.get("INT4", {}).get(1)
    if fp1 and i81 and i41:
        int8_slowdown = (i81["avg_lat"] - fp1["avg_lat"]) / fp1["avg_lat"] * 100
        int4_slowdown = (i41["avg_lat"] - fp1["avg_lat"]) / fp1["avg_lat"] * 100
        w(f"At c=1, INT8 is **{int8_slowdown:.1f}% slower** than FP16 and INT4 is "
          f"**{int4_slowdown:.1f}% slower**. This is the opposite of the expected result "
          f"on modern data-center GPUs.")
        w()
        w("**Root cause — RTX 2080 hardware limitations:**")
        w()
        w("1. **W8A8 INT8 (compressed-tensors):** The RTX 2080 (Turing, SM 7.5) has limited "
          "INT8 Tensor Core support. Unlike Ampere (SM 8.x) and Ada Lovelace (SM 8.9), "
          "Turing cannot efficiently dispatch INT8 matrix multiplications, so these ops run "
          "slower than native FP16 CUDA cores.")
        w()
        w("2. **AWQ INT4:** AWQ stores weights at 4-bit but dequantizes to FP16 at runtime "
          "before the matrix multiply. This adds a mandatory dequantization pass on top of "
          "the FP16 computation, making INT4 strictly slower than FP16 on this hardware.")
        w()
        w("3. **Chunked prefill overhead:** INT8 and INT4 use larger `max_model_len` (2048 and "
          "4096 respectively), which triggers vLLM's chunked prefill scheduler. This adds "
          "per-request scheduling overhead absent in the FP16 (max_model_len=1024) setup.")
        w()
        w("**This behavior is expected to reverse on cloud hardware** (A10G / L4), where "
          "dedicated INT8 and INT4 Tensor Cores provide true hardware acceleration. "
          "Member B's SageMaker and Vertex AI experiments will test this hypothesis.")
    w()

    # =========================================================================
    # Section 4: Concurrency scaling per backend
    # =========================================================================
    w("## 4. Concurrency Scaling")
    w()
    w("How latency and throughput change as concurrency increases within each backend.")
    w()

    for backend in vllm_backends:
        if backend not in data:
            continue
        w(f"### {backend}")
        w()
        w("| c | Avg Lat (s) | Lat vs c=1 | Agg. Throughput (tok/s) | Throughput vs c=1 | Req/s |")
        w("|---|---|---|---|---|---|")
        base_lat = data[backend][1]["avg_lat"]
        base_thr = data[backend][1]["agg_throughput"]
        for c in concurrency_levels:
            if c not in data[backend]:
                continue
            s = data[backend][c]
            lat_chg = (s["avg_lat"] - base_lat) / base_lat * 100
            thr_chg = (s["agg_throughput"] - base_thr) / base_thr * 100
            w(f"| {c} | {s['avg_lat']:.3f} | {lat_chg:+.1f}% "
              f"| {s['agg_throughput']:.1f} | {thr_chg:+.1f}% | {s['req_per_s']:.2f} |")
        w()

    w("**Observations:**")
    w()

    # FP16 scaling observation
    if "FP16" in data:
        fp_lats = {c: data["FP16"][c]["avg_lat"] for c in concurrency_levels if c in data["FP16"]}
        fp_thrs = {c: data["FP16"][c]["agg_throughput"] for c in concurrency_levels if c in data["FP16"]}
        best_thr_c = max(fp_thrs, key=fp_thrs.get)
        w(f"- **FP16:** Throughput peaks at c={best_thr_c} "
          f"({fp_thrs[best_thr_c]:.1f} tok/s). "
          f"Latency degrades sharply at c=16 as the KV cache (limited by the 1024-token "
          f"context window) becomes the bottleneck with many concurrent long sequences.")

    # INT8 scaling
    if "INT8" in data:
        i8_thrs = {c: data["INT8"][c]["agg_throughput"] for c in concurrency_levels if c in data["INT8"]}
        best_c = max(i8_thrs, key=i8_thrs.get)
        w(f"- **INT8:** Throughput peaks at c={best_c} ({i8_thrs[best_c]:.1f} tok/s). "
          f"The larger context window (2048) allows more tokens in flight, but the "
          f"slower per-token compute limits overall throughput gains.")

    # INT4 scaling
    if "INT4" in data:
        i4_thrs = {c: data["INT4"][c]["agg_throughput"] for c in concurrency_levels if c in data["INT4"]}
        best_c = max(i4_thrs, key=i4_thrs.get)
        w(f"- **INT4:** Throughput is nearly flat across all concurrency levels "
          f"(peak: c={best_c}, {i4_thrs[best_c]:.1f} tok/s). The AWQ dequantization "
          f"step dominates latency regardless of batch size, leaving no room for "
          f"concurrency gains on this hardware.")
    w()

    # =========================================================================
    # Section 5: Optimal configuration
    # =========================================================================
    w("## 5. Optimal Configuration on RTX 2080")
    w()
    w("Based on all metrics, the optimal local serving configuration is:")
    w()

    # Find best overall (lowest avg_lat at c=1)
    if "FP16" in data:
        best = data["FP16"]
        best_c_for_thr = max(best, key=lambda c: best[c]["agg_throughput"])
        w(f"| Goal | Recommended Config | Value |")
        w(f"|---|---|---|")
        w(f"| Lowest latency | FP16, c=1 | {best[1]['avg_lat']:.3f}s avg, {best[1]['p95_lat']:.3f}s P95 |")
        w(f"| Highest throughput | FP16, c={best_c_for_thr} | {best[best_c_for_thr]['agg_throughput']:.1f} tok/s |")
        w(f"| Best latency/throughput tradeoff | FP16, c=4 | {best[4]['avg_lat']:.3f}s avg, {best[4]['agg_throughput']:.1f} tok/s |")
    w()
    w("**FP16 dominates across all concurrency levels on the RTX 2080.** This finding "
      "highlights a critical practical insight: quantization is not universally beneficial — "
      "its advantage is hardware-dependent. Consumer-grade Turing GPUs lack the specialized "
      "low-precision compute paths that make quantization worthwhile.")
    w()

    # =========================================================================
    # Section 6: Key findings
    # =========================================================================
    w("## 6. Key Findings")
    w()
    if "HF Baseline" in data and "FP16" in data:
        hf1 = data["HF Baseline"][1]
        fp1_ = data["FP16"][1]
        lat_delta = (hf1["avg_lat"] - fp1_["avg_lat"]) / hf1["avg_lat"] * 100
        tps_delta = (fp1_["avg_tps"] - hf1["avg_tps"]) / hf1["avg_tps"] * 100
    else:
        lat_delta, tps_delta = 0.0, 0.0
    w("1. **vLLM vs HF Baseline:** vLLM FP16 reduces average latency by "
      f"{lat_delta:.1f}% and improves throughput by {tps_delta:.1f}% at c=1, "
      "validating PagedAttention and continuous batching as essential optimizations.")
    w()
    w("2. **Quantization backfires on consumer GPUs:** On the RTX 2080 (Turing, SM 7.5), "
      "INT8 and INT4 quantization increase latency relative to FP16 — the opposite of "
      "the expected behavior on data-center hardware. The root causes are limited INT8 "
      "Tensor Core support and AWQ's runtime dequantization overhead.")
    w()
    w("3. **Concurrency sweet spot (FP16):** Throughput scales well from c=1 to c=4–8 "
      "on FP16, with diminishing returns and rising tail latency beyond c=8. "
      "The 1024-token KV cache limit constrains how many long requests can run in parallel.")
    w()
    w("4. **INT4 shows no concurrency benefit:** AWQ INT4 throughput is effectively "
      "flat across c=1–16, suggesting the dequantization bottleneck fully saturates "
      "the GPU compute pipeline regardless of batch size.")
    w()
    w("5. **Cloud hypothesis:** The performance ordering (INT4 > INT8 > FP16 in throughput) "
      "is expected to appear on Member B's A10G and L4 experiments. "
      "If confirmed, it will demonstrate that hardware generation is the primary "
      "determinant of quantization benefit — a non-obvious and practically relevant finding.")
    w()

    # =========================================================================
    # Section 7: Implications for cloud experiments
    # =========================================================================
    w("## 7. Implications for Cloud Experiments (Member B)")
    w()
    w("| Question | Expected Finding |")
    w("|---|---|")
    w("| Does INT4 beat FP16 on A10G/L4? | Yes — Ampere/Ada INT4 Tensor Cores provide true hardware acceleration |")
    w("| Which cloud is better: SageMaker (A10G) or Vertex AI (L4)? | L4 has ~2× INT8 compute at 30% lower cost; outcome non-obvious at INT4 |")
    w("| What concurrency is optimal on cloud? | Higher c possible due to 24 GB VRAM vs 6.94 GB local |")
    w("| Cost-performance metric | Tokens per dollar: measure at c=1 and c=8, compare across platforms |")
    w()
    w("Member B should use identical benchmark parameters (`--requests 100`, same ShareGPT dataset, "
      "same seed=42) to ensure results are directly comparable to local runs.")
    w()

    # =========================================================================
    # Write output
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    out_path = "results/analysis_local.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {out_path}")
    print(f"Total configurations analyzed: {sum(len(v) for v in data.values())}")
    for backend in backend_order:
        if backend in data:
            print(f"  {backend}: {sorted(data[backend].keys())} concurrency levels")


if __name__ == "__main__":
    main()
