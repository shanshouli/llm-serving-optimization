"""
LLM Inference Benchmark Client (ShareGPT workload)

Sends requests from a ShareGPT dataset to an LLM serving endpoint and measures
performance: latency percentiles, tokens/sec, and aggregate throughput.

Usage:
    # Prepare dataset first (run once):
    uv run python benchmark/prepare_sharegpt.py

    # Then run benchmarks:
    uv run python benchmark/client.py --concurrency 1  --requests 100  # FP16
    uv run python benchmark/client.py --concurrency 8  --requests 100  # FP16
    uv run python benchmark/client.py --concurrency 1  --requests 100 --max-len 2048  # INT8
    uv run python benchmark/client.py --concurrency 8  --requests 100 --max-len 4096  # INT4

    # Against SageMaker (Member B):
    uv run python benchmark/client.py --url https://<endpoint>.sagemaker.amazonaws.com/...
                                       --concurrency 8 --requests 100 --max-len 4096
"""

import aiohttp
import asyncio
import time
import csv
import os
import json
import random
import argparse


# =============================================================================
# Dataset loading
# =============================================================================

def load_sharegpt(dataset_path: str, max_len: int, n: int, seed: int = 42) -> list[dict]:
    """
    Load ShareGPT samples from the preprocessed JSON file and filter by sequence length.

    Args:
        dataset_path: Path to sharegpt_filtered.json (created by prepare_sharegpt.py).
        max_len:      Maximum total tokens (input + output). Use 1024 for FP16,
                      2048 for INT8, 4096 for INT4.
        n:            Number of samples to return.
        seed:         Random seed for reproducibility.

    Returns:
        List of dicts with keys: prompt, input_tokens, output_tokens.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError:        If not enough samples pass the max_len filter.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run 'uv run python benchmark/prepare_sharegpt.py' first."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Filter: only keep samples whose estimated total length fits in the model's context.
    # Apply a 0.8 safety factor because estimate_tokens() uses word_count*1.3, which
    # underestimates actual Llama tokenizer counts by ~15-20%. Without the margin,
    # borderline samples can exceed the model's hard context limit and return HTTP 400.
    safe_max = int(max_len * 0.8)
    filtered = [
        s for s in all_samples
        if s["input_tokens"] + s["output_tokens"] <= safe_max
    ]

    if len(filtered) < n:
        raise ValueError(
            f"Only {len(filtered)} samples pass max_len={max_len} filter, "
            f"but {n} were requested. Reduce --requests or increase --max-len."
        )

    # Shuffle with a fixed seed so every run uses the same sample order.
    # This ensures FP16/INT8/INT4 results are comparable.
    rng = random.Random(seed)
    rng.shuffle(filtered)

    return filtered[:n]


# =============================================================================
# Core benchmark logic
# =============================================================================

async def single_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    sample: dict,
) -> dict:
    """
    Send one request to the LLM endpoint and measure latency.

    Args:
        session:   Shared HTTP connection pool.
        semaphore: Limits how many requests run simultaneously.
        api_url:   Full URL of the /v1/completions endpoint.
        model:     Model name string expected by the server.
        sample:    One ShareGPT sample (prompt + token length metadata).

    Returns:
        Dict with: latency (s), tokens (int), tps (float/s).
        Returns None if the server rejects the sample (HTTP 400 = prompt too long).
    """
    # max_tokens: use the ShareGPT reference output length, capped at 512.
    # This matches how vLLM's own benchmark_serving.py works — we tell the server
    # to generate the same amount of tokens as the original response, so comparisons
    # across quantization levels are apples-to-apples.
    max_tokens = min(sample["output_tokens"], 512)

    payload = {
        "model": model,
        "prompt": sample["prompt"],
        "max_tokens": max_tokens,
    }

    async with semaphore:
        # Track total elapsed time including any retry, so latency reflects real
        # client-side wait time even when a connection drop forces a retry.
        start = time.perf_counter()
        result = None

        for attempt in range(2):  # one retry on connection-level errors
            try:
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 400:
                        # Prompt exceeds actual context limit — token estimation was off
                        # (common for non-ASCII text: Chinese/Japanese have no spaces so
                        # word-based estimation drastically underestimates token count).
                        # Skip this sample silently; caller will report the skip count.
                        return None
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    result = await resp.json()
                break  # success — exit retry loop
            except aiohttp.ServerDisconnectedError:
                # A peer's 400 response can cause vLLM to close the shared HTTP
                # connection, causing other in-flight requests to see a disconnect.
                # Retry once with a fresh connection; give up if it fails again.
                if attempt == 1:
                    return None

        if result is None:
            return None

        latency = time.perf_counter() - start

    tokens = result["usage"]["completion_tokens"]
    return {
        "latency": latency,
        "tokens": tokens,
        "tps": tokens / latency,
    }


async def run_benchmark(
    n: int = 100,
    concurrency: int = 1,
    api_url: str = "http://localhost:18000/v1/completions",
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    dataset_path: str = "benchmark/workloads/sharegpt_filtered.json",
    max_len: int = 1024,
    output: str = None,
    seed: int = 42,
):
    """
    Run the full benchmark: load ShareGPT samples, send n requests at the given
    concurrency level, print summary statistics, and save raw data to CSV.

    Args:
        n:            Total requests to send.
        concurrency:  Max simultaneous requests (1 = sequential).
        api_url:      Completions endpoint URL.
        model:        Model name in the request payload.
        dataset_path: Path to sharegpt_filtered.json.
        max_len:      Max total tokens for sample filtering (1024/2048/4096).
        output:       CSV output path (auto-generated if None).
        seed:         Random seed for dataset sampling.
    """
    # Load dataset — different quantization levels use different max_len filters
    print(f"Loading {n} ShareGPT samples (max_len={max_len})...")
    samples = load_sharegpt(dataset_path, max_len=max_len, n=n, seed=seed)
    print(f"Loaded. Input tokens — median: "
          f"{sorted(s['input_tokens'] for s in samples)[len(samples)//2]}")

    # Semaphore gates how many requests run at once
    semaphore = asyncio.Semaphore(concurrency)

    # Reuse TCP connections to avoid per-request connection overhead
    connector = aiohttp.TCPConnector(limit=concurrency)

    wall_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            single_request(session, semaphore, api_url, model, sample)
            for sample in samples
        ]
        results = await asyncio.gather(*tasks)

    # Wall clock: actual elapsed time from first task start to last task finish.
    # Under concurrency, this is shorter than sum(latencies) because requests overlap.
    wall_clock = time.perf_counter() - wall_start

    # Filter out None results (samples skipped due to HTTP 400 / token estimation error)
    skipped = sum(1 for r in results if r is None)
    results = [r for r in results if r is not None]
    if skipped:
        print(f"Warning: {skipped} sample(s) skipped (prompt exceeded context limit).")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------

    latencies = sorted(r["latency"] for r in results)
    tps_list = [r["tps"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    n_actual = len(latencies)

    print(f"\n{'='*55}")
    print(f"Requests: {n_actual} | Concurrency: {concurrency} | max_len: {max_len}")
    print(f"{'='*55}")
    print(f"Avg latency:    {sum(latencies)/n_actual:.3f}s")
    print(f"P50 latency:    {latencies[n_actual//2]:.3f}s")
    print(f"P95 latency:    {latencies[int(n_actual*0.95)]:.3f}s")
    print(f"P99 latency:    {latencies[int(n_actual*0.99)]:.3f}s")
    print(f"Min latency:    {latencies[0]:.3f}s")
    print(f"Max latency:    {latencies[-1]:.3f}s")
    print(f"Avg tokens/s:   {sum(tps_list)/n_actual:.1f}  (per request)")
    print(f"Total tokens:   {total_tokens}")
    print(f"Throughput:     {total_tokens/wall_clock:.1f} tokens/s  (aggregate)")
    print(f"Requests/s:     {n_actual/wall_clock:.2f}")

    # -------------------------------------------------------------------------
    # Save raw per-request data to CSV
    # -------------------------------------------------------------------------

    os.makedirs("results/raw", exist_ok=True)
    outfile = output or f"results/raw/bench_c{concurrency}_n{n_actual}_maxlen{max_len}.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latency", "tokens", "tps"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {outfile}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference benchmark client (ShareGPT)")
    parser.add_argument("--requests",    type=int,   default=100,
                        help="Total requests to send (default: 100)")
    parser.add_argument("--concurrency", type=int,   default=1,
                        help="Concurrent requests (default: 1)")
    parser.add_argument("--url",         type=str,   default="http://localhost:18000/v1/completions",
                        help="Completions endpoint URL")
    parser.add_argument("--model",       type=str,   default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name in the request payload")
    parser.add_argument("--dataset",     type=str,   default="benchmark/workloads/sharegpt_filtered.json",
                        help="Path to sharegpt_filtered.json")
    parser.add_argument("--max-len",     type=int,   default=1024,
                        help="Max total tokens for filtering: 1024=FP16, 2048=INT8, 4096=INT4")
    parser.add_argument("--output",      type=str,   default=None,
                        help="CSV output path (auto-generated if not set)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for dataset sampling (default: 42)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        n=args.requests,
        concurrency=args.concurrency,
        api_url=args.url,
        model=args.model,
        dataset_path=args.dataset,
        max_len=args.max_len,
        output=args.output,
        seed=args.seed,
    ))
