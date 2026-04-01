"""
LLM Inference Benchmark Client

This script sends a fixed number of identical prompts to an LLM serving endpoint
and measures performance metrics: latency, tokens/sec, and throughput.

Usage:
    uv run python benchmark/client.py [num_requests] [concurrency]

Examples:
    uv run python benchmark/client.py 50 1     # 50 requests, 1 at a time (sequential)
    uv run python benchmark/client.py 100 8    # 100 requests, 8 at a time (concurrent)
"""

import aiohttp
import asyncio
import time
import csv
import sys
import os
import argparse


# =============================================================================
# Core benchmark logic
# =============================================================================

async def single_request(session, semaphore, api_url, payload):
    """
    Send one request to the LLM endpoint and measure how long it takes.

    Args:
        session:   The shared HTTP connection pool (reused across requests).
        semaphore: Controls how many requests can run at the same time.
                   e.g., Semaphore(4) means at most 4 requests in flight.
        api_url:   The full URL of the completions endpoint.
        payload:   The JSON request body.

    Returns:
        A dict with:
          - latency: how many seconds this request took (end-to-end)
          - tokens:  how many tokens the model generated
          - tps:     tokens per second for this single request
    """
    # Wait here if we've hit the concurrency limit.
    # This is what controls "1 at a time" vs "8 at a time".
    async with semaphore:
        start = time.perf_counter()

        # Send the POST request to the serving endpoint
        async with session.post(api_url, json=payload) as resp:
            result = await resp.json()

        latency = time.perf_counter() - start

        # Extract the number of tokens generated from the response
        tokens = result["usage"]["completion_tokens"]

        return {
            "latency": latency,
            "tokens": tokens,
            "tps": tokens / latency,  # tokens per second
        }


async def run_benchmark(n=50, concurrency=1, api_url="http://localhost:8000/v1/completions",
                        model="meta-llama/Llama-3.2-3B-Instruct", output=None):
    """
    Run the full benchmark: send n requests with a given concurrency level,
    then print summary statistics and save raw data to CSV.

    Args:
        n:           Total number of requests to send.
        concurrency: How many requests to send at the same time.
                     1 = sequential (one after another)
                     8 = 8 requests in flight simultaneously
        api_url:     The completions endpoint URL.
        model:       Model name sent in the request payload.
        output:      CSV output path. Auto-generated if None.
    """
    # Every request uses the same prompt and token budget for a fair comparison.
    payload = {
        "model": model,
        "prompt": "Explain the concept of cloud computing in detail.",
        "max_tokens": 256,
    }

    # Semaphore limits how many requests run in parallel.
    # Think of it as "number of checkout lanes open at a store".
    semaphore = asyncio.Semaphore(concurrency)

    # TCPConnector reuses HTTP connections instead of opening a new one
    # for every request. The limit should match our concurrency.
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all request tasks upfront.
        # They won't all start immediately - the semaphore gates them.
        tasks = [single_request(session, semaphore, api_url, payload) for _ in range(n)]

        # asyncio.gather runs all tasks and waits for ALL of them to finish.
        # It returns a list of results in the same order.
        results = await asyncio.gather(*tasks)

    # -------------------------------------------------------------------------
    # Calculate summary statistics
    # -------------------------------------------------------------------------

    # Sort latencies to compute percentiles
    latencies = sorted(r["latency"] for r in results)
    tps_list = [r["tps"] for r in results]
    total_tokens = sum(r["tokens"] for r in results)

    # Wall clock = the longest single request time.
    # Under concurrency, total wall clock time is roughly max(latencies),
    # not sum(latencies), because requests overlap.
    wall_clock = max(r["latency"] for r in results)

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    print(f"\n{'='*50}")
    print(f"Requests: {n} | Concurrency: {concurrency}")
    print(f"{'='*50}")

    # Average: sum of all latencies / number of requests
    print(f"Avg latency:    {sum(latencies)/len(latencies):.3f}s")

    # P50 (median): the middle value when sorted. Half the requests are faster.
    print(f"P50 latency:    {latencies[len(latencies)//2]:.3f}s")

    # P95: 95% of requests are faster than this. Shows "typical worst case".
    print(f"P95 latency:    {latencies[int(len(latencies)*0.95)]:.3f}s")

    # P99: 99% of requests are faster than this. Shows tail latency.
    print(f"P99 latency:    {latencies[int(len(latencies)*0.99)]:.3f}s")

    print(f"Min latency:    {latencies[0]:.3f}s")
    print(f"Max latency:    {latencies[-1]:.3f}s")

    # Per-request tokens/s: how fast each individual request generated tokens.
    print(f"Avg tokens/s:   {sum(tps_list)/len(tps_list):.1f}")

    print(f"Total tokens:   {total_tokens}")

    # Aggregate throughput: total tokens generated / wall clock time.
    # This shows the system's overall capacity, not per-request speed.
    # Under concurrency, this should be HIGHER than avg tokens/s
    # because multiple requests are generating tokens in parallel.
    print(f"Throughput:     {total_tokens/wall_clock:.1f} tokens/s (aggregate)")

    # Requests per second: how many complete requests the system handles.
    print(f"Requests/s:     {n/wall_clock:.2f}")

    # -------------------------------------------------------------------------
    # Save raw data to CSV for later analysis and visualization
    # -------------------------------------------------------------------------

    os.makedirs("results/raw", exist_ok=True)
    outfile = output if output else f"results/raw/bench_c{concurrency}_n{n}.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latency", "tokens", "tps"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {outfile}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference benchmark client")
    parser.add_argument("--requests",    type=int,   default=50,
                        help="Total number of requests to send (default: 50)")
    parser.add_argument("--concurrency", type=int,   default=1,
                        help="Number of concurrent requests (default: 1)")
    parser.add_argument("--url",         type=str,   default="http://localhost:8000/v1/completions",
                        help="Completions endpoint URL")
    parser.add_argument("--model",       type=str,   default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name to include in the request payload")
    parser.add_argument("--output",      type=str,   default=None,
                        help="CSV output path (auto-generated if not set)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        n=args.requests,
        concurrency=args.concurrency,
        api_url=args.url,
        model=args.model,
        output=args.output,
    ))
