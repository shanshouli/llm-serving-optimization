"""
Vertex AI Endpoint Benchmark Script
Mirrors sagemaker_benchmark.py exactly for direct comparison.

Uses the same ShareGPT workload, concurrency model, and metrics.
Calls Vertex AI endpoint via HTTP with Google auth (not boto3).

Usage:
    python cloud/vertex/vertex_benchmark.py --project YOUR_PROJECT_ID --endpoint ENDPOINT_ID
    python cloud/vertex/vertex_benchmark.py --project YOUR_PROJECT_ID --endpoint ENDPOINT_ID \
        --concurrency 8 --quant int4

Endpoint ID can be found in:
    results/raw/vertex_{quant}_endpoint_id.txt  (written by vertex_deploy.py)
"""

import argparse
import asyncio
import csv
import json
import os
import random
import time
from pathlib import Path

import google.auth
import google.auth.transport.requests
import httpx

REGION       = "us-central1"
RESULTS_DIR  = Path("results/raw")
SHAREGPT_PATH = "benchmark/workloads/sharegpt_filtered.json"

# Fallback prompt if ShareGPT not available
FALLBACK_PROMPT = "Explain the difference between machine learning and deep learning in 3 sentences."

# vLLM requires model field matching the loaded model name
MODEL_IDS = {
    "fp16": "meta-llama/Llama-3.2-3B-Instruct",
    "int8": "neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",
    "int4": "casperhansen/llama-3.2-3b-instruct-awq",
}


def get_auth_token() -> str:
    """Get a Google auth token for Vertex AI API calls."""
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token


def get_endpoint_url(project: str, endpoint_id: str) -> str:
    """Build the Vertex AI endpoint raw prediction URL.
    Uses :rawPredict to pass request directly to vLLM /v1/completions."""
    return (
        f"https://{REGION}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{REGION}/"
        f"endpoints/{endpoint_id}:rawPredict"
    )


def load_sharegpt_samples(n: int = 100, max_len: int = 1024, seed: int = 42) -> list[dict]:
    """Load ShareGPT samples — identical logic to sagemaker_benchmark.py for comparability."""
    if not os.path.exists(SHAREGPT_PATH):
        raise FileNotFoundError(
            f"ShareGPT dataset not found: {SHAREGPT_PATH}\n"
            "Run 'python benchmark/prepare_sharegpt.py' first."
        )

    with open(SHAREGPT_PATH) as f:
        all_samples = json.load(f)

    # Same max_len filter as sagemaker_benchmark.py and local client.py
    filtered = [
        s for s in all_samples
        if s["input_tokens"] + s["output_tokens"] <= max_len
    ]

    if len(filtered) < n:
        raise ValueError(
            f"Only {len(filtered)} samples pass max_len={max_len}, "
            f"but {n} were requested."
        )

    # Same fixed seed shuffle — ensures identical prompts across SageMaker and Vertex AI
    rng = random.Random(seed)
    rng.shuffle(filtered)
    print(f"Loaded {n} samples (max_len={max_len}, seed={seed})")
    return filtered[:n]


async def invoke_endpoint(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    sample: dict,
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> dict:
    """Send one request to Vertex AI endpoint and return latency + token count."""
    max_new_tokens = min(sample["output_tokens"], 512)

    # Official vLLM OpenAI server: /v1/completions endpoint
    # model field is required by vLLM and must match the loaded model name
    payload = {
        "model":      model_id,
        "prompt":     sample["prompt"],
        "max_tokens": max_new_tokens,
        "temperature": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI /v1/completions response: {"choices": [{"text": "..."}]}
        generated = data.get("choices", [{}])[0].get("text", "")
        # Estimate token count: word_count * 1.3 (same as sagemaker_benchmark.py)
        token_count = int(len(generated.split()) * 1.3)

        return {"latency": time.perf_counter() - t0, "tokens": token_count, "success": True}
    except Exception as e:
        return {"latency": time.perf_counter() - t0, "tokens": 0, "success": False, "error": str(e)[:120]}


async def run_benchmark(
    project: str,
    endpoint_id: str,
    concurrency: int,
    n_requests: int,
    max_len: int,
    quant: str = "fp16",
) -> tuple[list[dict], float]:
    """Run benchmark with given concurrency. Returns (results, wall_time)."""
    samples  = load_sharegpt_samples(n=n_requests, max_len=max_len)
    url      = get_endpoint_url(project, endpoint_id)
    model_id = MODEL_IDS[quant]

    # Refresh token (valid for 1 hour; enough for one benchmark run)
    token = get_auth_token()

    async with httpx.AsyncClient() as client:
        # Warm-up: 2 requests excluded from results
        print("Running 2 warm-up requests (excluded from results)...")
        warmup_sample = {"prompt": FALLBACK_PROMPT, "output_tokens": 64}
        for _ in range(2):
            await invoke_endpoint(client, url, token, warmup_sample, model_id)
        print("Warm-up done.")

        # Main benchmark: semaphore limits to 'concurrency' parallel requests
        sem      = asyncio.Semaphore(concurrency)
        results  = []
        failed   = 0

        async def bounded_request(i: int, sample: dict):
            nonlocal failed
            async with sem:
                r = await invoke_endpoint(client, url, token, sample, model_id)
                if r["success"]:
                    results.append({"latency": r["latency"], "tokens": r["tokens"]})
                    tps = r["tokens"] / r["latency"] if r["latency"] > 0 else 0
                    print(f"  [{i+1}/{n_requests}] latency={r['latency']:.2f}s tokens={r['tokens']}")
                else:
                    failed += 1
                    print(f"  [{i+1}/{n_requests}] FAILED: {r.get('error', '')}")

        wall_start = time.perf_counter()
        await asyncio.gather(*[
            bounded_request(i, samples[i % len(samples)])
            for i in range(n_requests)
        ])
        wall_time = time.perf_counter() - wall_start

    if failed:
        print(f"WARNING: {failed}/{n_requests} requests failed and were excluded.")
    return results, wall_time


def print_summary(results: list[dict], wall_time: float, concurrency: int):
    """Print benchmark summary — same format as sagemaker_benchmark.py."""
    if not results:
        print("No successful results.")
        return
    lats = sorted(r["latency"] for r in results)
    n    = len(lats)
    total_tokens = sum(r["tokens"] for r in results)
    tps_list     = [r["tokens"] / r["latency"] for r in results if r["latency"] > 0]

    print(f"\n{'='*50}")
    print(f"Results: {n} requests, concurrency={concurrency}")
    print(f"{'='*50}")
    print(f"Avg latency:       {sum(lats)/n:.3f}s")
    print(f"P50 latency:       {lats[int(n*0.50)]:.3f}s")
    print(f"P95 latency:       {lats[int(n*0.95)]:.3f}s")
    print(f"P99 latency:       {lats[min(int(n*0.99), n-1)]:.3f}s")
    print(f"Avg tokens/s:      {sum(tps_list)/n:.1f}")
    print(f"Aggregate tokens/s:{total_tokens/wall_time:.1f}")
    print(f"Requests/s:        {n/wall_time:.2f}")
    print(f"Wall time:         {wall_time:.1f}s")


def save_csv(results: list[dict], concurrency: int, n_requests: int, quant: str):
    """Save results to CSV — same schema as sagemaker_benchmark.py."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = RESULTS_DIR / f"vertex_{quant}_c{concurrency}_n{n_requests}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latency", "tokens", "tps"])
        writer.writeheader()
        writer.writerows([
            {"latency": r["latency"], "tokens": r["tokens"],
             "tps": r["tokens"] / r["latency"] if r["latency"] > 0 else 0}
            for r in results
        ])
    print(f"Saved: {filename}")


async def main(project: str, endpoint_id: str, concurrency: int,
               n_requests: int, max_len: int, quant: str):
    print(f"Benchmarking Vertex AI endpoint: {endpoint_id}")
    print(f"Project={project}, Concurrency={concurrency}, Requests={n_requests}, quant={quant}")
    results, wall_time = await run_benchmark(project, endpoint_id, concurrency, n_requests, max_len, quant)
    print_summary(results, wall_time, concurrency)
    save_csv(results, concurrency, n_requests, quant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",     required=True, help="GCP project ID")
    parser.add_argument("--endpoint",    required=True, help="Vertex AI endpoint ID (numeric)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests",    type=int, default=100)
    parser.add_argument("--max-len",     type=int, default=1024)
    parser.add_argument("--quant",       type=str, default="fp16",
                        choices=["fp16", "int8", "int4"],
                        help="Label for output filename")
    args = parser.parse_args()
    asyncio.run(main(args.project, args.endpoint, args.concurrency,
                     args.requests, args.max_len, args.quant))
