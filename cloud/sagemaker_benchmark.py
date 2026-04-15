"""
SageMaker Benchmark Client
Runs ShareGPT benchmark against a deployed SageMaker endpoint.

Usage:
    python cloud/sagemaker_benchmark.py --concurrency 1 --requests 100
    python cloud/sagemaker_benchmark.py --concurrency 8 --requests 100
"""

import argparse
import asyncio
import csv
import json
import os
import random
import time
from pathlib import Path

import boto3

REGION = "us-west-2"
ENDPOINT_NAME = "vllm-llama3-int4"  # default; overridden by --endpoint CLI arg
RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw"
SHAREGPT_PATH = Path(__file__).parent.parent / "benchmark" / "workloads" / "sharegpt_filtered.json"

# Used only for warm-up requests (not counted in results)
FALLBACK_PROMPT = "Explain the concept of cloud computing in detail."


def load_samples(n: int, max_len: int = 1024, seed: int = 42) -> list[dict]:
    """Load ShareGPT samples with the same filtering and shuffle logic as
    benchmark/client.py, so cloud and local runs use identical prompts.

    Args:
        n:       Number of samples to return.
        max_len: Max total tokens (input + output). Use 1024 to match local FP16.
        seed:    Random seed — must match client.py default (42).
    """
    if not SHAREGPT_PATH.exists():
        raise FileNotFoundError(
            f"ShareGPT dataset not found: {SHAREGPT_PATH}\n"
            "Run 'python benchmark/prepare_sharegpt.py' first."
        )

    with open(SHAREGPT_PATH) as f:
        all_samples = json.load(f)

    # Apply same max_len filter as local client.py
    filtered = [
        s for s in all_samples
        if s["input_tokens"] + s["output_tokens"] <= max_len
    ]

    if len(filtered) < n:
        raise ValueError(
            f"Only {len(filtered)} samples pass max_len={max_len}, "
            f"but {n} were requested."
        )

    # Shuffle with fixed seed — identical to client.py so both sides use same prompts
    rng = random.Random(seed)
    rng.shuffle(filtered)

    print(f"Loaded {n} samples (max_len={max_len}, seed={seed})")
    return filtered[:n]


def invoke_endpoint(client, sample: dict) -> dict:
    # Cap max_new_tokens at 512, matching client.py: min(output_tokens, 512).
    # This replicates the realistic output length distribution from ShareGPT
    # rather than artificially fixing all responses to the same length.
    max_new_tokens = min(sample["output_tokens"], 512)

    payload = {
        "inputs": sample["prompt"],
        "parameters": {"max_new_tokens": max_new_tokens},
    }
    t0 = time.perf_counter()
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    latency = time.perf_counter() - t0
    result = json.loads(response["Body"].read().decode())

    generated = ""
    if isinstance(result, list) and result:
        generated = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        generated = result.get("generated_text", "")

    # DJL-LMI does not return usage.completion_tokens, so we estimate token
    # count using word count * 1.3 — the same approximation used in
    # prepare_sharegpt.py and vLLM's own benchmark tools.
    tokens = max(1, int(len(generated.split()) * 1.3))
    return {"latency": latency, "tokens": tokens, "tps": tokens / latency}


async def run_benchmark(concurrency: int, n_requests: int, max_len: int = 1024) -> tuple[list[dict], float]:
    samples = load_samples(n_requests, max_len=max_len)
    client = boto3.client("sagemaker-runtime", region_name=REGION)

    # Warm up: 2 requests excluded from statistics.
    # Cloud endpoint may need CUDA kernel init on first call; local vLLM is
    # pre-warmed by design, so warm-up is cloud-specific and does not bias comparison.
    print("Running 2 warm-up requests (excluded from results)...")
    loop = asyncio.get_running_loop()
    warmup_sample = {"prompt": FALLBACK_PROMPT, "output_tokens": 64}
    await loop.run_in_executor(None, invoke_endpoint, client, warmup_sample)
    await loop.run_in_executor(None, invoke_endpoint, client, warmup_sample)
    print("Warm-up done.")

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def bounded_invoke(sample):
        nonlocal completed
        async with semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, invoke_endpoint, client, sample)
            completed += 1
            print(f"  [{completed}/{n_requests}] latency={result['latency']:.2f}s tokens={result['tokens']}", flush=True)
            return result

    tasks = [bounded_invoke(s) for s in samples]
    wall_start = time.perf_counter()

    # return_exceptions=True: a single failed request won't crash the whole benchmark.
    # Failed results are filtered out before computing statistics.
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    wall_time = time.perf_counter() - wall_start

    # Separate successful results from exceptions and report failures
    results = [r for r in raw if isinstance(r, dict)]
    n_failed = len(raw) - len(results)
    if n_failed:
        print(f"WARNING: {n_failed}/{len(raw)} requests failed and were excluded.")

    return results, wall_time


def print_summary(results: list[dict], wall_time: float, concurrency: int):
    latencies = sorted(r["latency"] for r in results)
    tps_list = [r["tps"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)
    n = len(latencies)

    print(f"\n{'='*50}")
    print(f"Results: {n} requests, concurrency={concurrency}")
    print(f"{'='*50}")
    print(f"Avg latency:       {sum(latencies)/n:.3f}s")
    print(f"P50 latency:       {latencies[int(n*0.50)]:.3f}s")
    print(f"P95 latency:       {latencies[int(n*0.95)]:.3f}s")
    print(f"P99 latency:       {latencies[min(int(n*0.99), n-1)]:.3f}s")
    print(f"Avg tokens/s:      {sum(tps_list)/n:.1f}")
    print(f"Aggregate tokens/s:{total_tokens/wall_time:.1f}")
    print(f"Requests/s:        {n/wall_time:.2f}")
    print(f"Wall time:         {wall_time:.1f}s")


def save_csv(results: list[dict], concurrency: int, n_requests: int, quant: str = "fp16"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Include quantization in filename to avoid overwriting results across runs
    filename = RESULTS_DIR / f"sagemaker_{quant}_c{concurrency}_n{n_requests}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latency", "tokens", "tps"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {filename}")


async def main(concurrency: int, n_requests: int, max_len: int, quant: str, endpoint: str):
    global ENDPOINT_NAME
    ENDPOINT_NAME = endpoint  # override module-level constant with CLI arg
    print(f"Benchmarking SageMaker endpoint: {ENDPOINT_NAME}")
    print(f"Concurrency={concurrency}, Requests={n_requests}, max_len={max_len}, quant={quant}")
    results, wall_time = await run_benchmark(concurrency, n_requests, max_len)
    print_summary(results, wall_time, concurrency)
    save_csv(results, concurrency, n_requests, quant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max total tokens for sample filtering. Use 1024 to match local FP16.")
    parser.add_argument("--quant", type=str, default="fp16", choices=["fp16", "int4awq", "int8gptq"],
                        help="Quantization label used in output filename (fp16/int4awq/int8gptq).")
    parser.add_argument("--endpoint", type=str, default="vllm-llama3-int4",
                        help="SageMaker endpoint name to benchmark.")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.requests, args.max_len, args.quant, args.endpoint))
