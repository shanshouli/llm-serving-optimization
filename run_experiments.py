"""
Automated Experiment Runner

Orchestrates the full benchmark pipeline sequentially:
  1. Start vLLM (port 8000) → run benchmark → stop
  2. Start HF baseline (port 8001) → run benchmark → stop

Because the RTX 2080 only has ~6.9 GB usable VRAM, both models cannot run
at the same time. This script stops one before starting the other.

Usage:
    # Exp 1: single-client baseline comparison
    uv run python run_experiments.py

    # Exp 3: concurrency sweep
    uv run python run_experiments.py --concurrency 1 4 8 16 --requests 50

    # Only run vLLM experiments (skip baseline)
    uv run python run_experiments.py --only vllm

    # Only run baseline experiments
    uv run python run_experiments.py --only baseline
"""

import argparse
import subprocess
import sys
import time

import httpx

# =============================================================================
# Configuration
# =============================================================================

VLLM_URL        = "http://localhost:8000"
BASELINE_URL    = "http://localhost:8001"
MODEL_NAME      = "meta-llama/Llama-3.2-3B-Instruct"

# vLLM loads fast (model already in GPU memory after first pull).
# Baseline takes longer: model load + no pipelining.
VLLM_HEALTH_TIMEOUT     = 300   # seconds
BASELINE_HEALTH_TIMEOUT = 600   # seconds


# =============================================================================
# Helpers
# =============================================================================

def wait_for_health(base_url: str, timeout: int, interval: int = 5) -> bool:
    """
    Poll <base_url>/health until it returns HTTP 200 or we hit the timeout.
    Returns True if healthy, False if timed out.
    """
    deadline = time.time() + timeout
    print(f"  Waiting for {base_url}/health (timeout: {timeout}s) ...", flush=True)

    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  Ready: {base_url}")
                return True
        except Exception:
            pass  # Service not up yet — keep polling
        time.sleep(interval)

    print(f"  ERROR: {base_url} did not become healthy within {timeout}s")
    return False


def compose(*args: str) -> None:
    """Run a docker compose command, raising on failure."""
    subprocess.run(["docker", "compose", *args], check=True)


def run_benchmark(base_url: str, model: str, concurrency: int,
                  n_requests: int, output: str) -> None:
    """
    Invoke benchmark/client.py with the given parameters.
    Results are written to the specified CSV output path.
    """
    subprocess.run([
        "uv", "run", "python", "benchmark/client.py",
        "--url",         f"{base_url}/v1/completions",
        "--model",       model,
        "--concurrency", str(concurrency),
        "--requests",    str(n_requests),
        "--output",      output,
    ], check=True)


# =============================================================================
# Experiment runners
# =============================================================================

def run_vllm_experiments(concurrency_levels: list[int], n_requests: int) -> None:
    """Start vLLM, run all concurrency levels, then stop it."""
    print("\n" + "="*60)
    print("EXPERIMENT: vLLM (PagedAttention + continuous batching)")
    print("="*60)

    print("\nStarting vLLM ...")
    compose("up", "-d", "vllm")

    if not wait_for_health(VLLM_URL, VLLM_HEALTH_TIMEOUT):
        compose("stop", "vllm")
        sys.exit(1)

    for c in concurrency_levels:
        output = f"results/raw/vllm_fp16_c{c}_n{n_requests}.csv"
        print(f"\n--- vLLM | concurrency={c} | requests={n_requests} ---")
        run_benchmark(VLLM_URL, MODEL_NAME, c, n_requests, output)

    print("\nStopping vLLM ...")
    compose("stop", "vllm")


def run_baseline_experiments(concurrency_levels: list[int], n_requests: int) -> None:
    """Start HF baseline, run all concurrency levels, then stop it."""
    print("\n" + "="*60)
    print("EXPERIMENT: HF Transformers Baseline (no optimization)")
    print("="*60)

    print("\nStarting HF baseline ...")
    # --profile baseline activates the service (it's excluded from default compose up)
    compose("--profile", "baseline", "up", "-d", "baseline")

    if not wait_for_health(BASELINE_URL, BASELINE_HEALTH_TIMEOUT):
        compose("stop", "baseline")
        sys.exit(1)

    for c in concurrency_levels:
        output = f"results/raw/hf_baseline_c{c}_n{n_requests}.csv"
        print(f"\n--- HF Baseline | concurrency={c} | requests={n_requests} ---")
        run_benchmark(BASELINE_URL, MODEL_NAME, c, n_requests, output)

    print("\nStopping HF baseline ...")
    compose("stop", "baseline")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference benchmark experiments")
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1],
        help="Concurrency levels to test, e.g. --concurrency 1 4 8 16"
    )
    parser.add_argument(
        "--requests", type=int, default=50,
        help="Number of requests per concurrency level (default: 50)"
    )
    parser.add_argument(
        "--only", choices=["vllm", "baseline"], default=None,
        help="Run only one backend instead of both"
    )
    args = parser.parse_args()

    print(f"Concurrency levels : {args.concurrency}")
    print(f"Requests per level : {args.requests}")
    print(f"Results directory  : results/raw/")

    if args.only != "baseline":
        run_vllm_experiments(args.concurrency, args.requests)

    if args.only != "vllm":
        run_baseline_experiments(args.concurrency, args.requests)

    print("\n" + "="*60)
    print("All experiments complete. Results saved to results/raw/")
    print("="*60)


if __name__ == "__main__":
    main()
