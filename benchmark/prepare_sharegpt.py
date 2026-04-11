"""
ShareGPT Dataset Preparation Script

Downloads the ShareGPT_Vicuna_unfiltered dataset from HuggingFace, extracts
the first human/assistant turn from each conversation, estimates token counts,
and saves the result to benchmark/workloads/sharegpt_filtered.json.

Run this once before benchmarking:
    uv run python benchmark/prepare_sharegpt.py

Output: benchmark/workloads/sharegpt_filtered.json
    Each entry: { "prompt": str, "input_tokens": int, "output_tokens": int }

The benchmark client (client.py) loads this file and filters by --max-len at
runtime, so one dataset file covers all three quantization levels:
    FP16:  filter to input + output <= 1024
    INT8:  filter to input + output <= 2048
    INT4:  filter to input + output <= 4096
"""

import json
import os
import random

from huggingface_hub import hf_hub_download


# =============================================================================
# Token estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token count: word count * 1.3.

    LLM tokenizers split contractions and punctuation into subword tokens,
    so raw word count underestimates. The 1.3x factor is a standard approximation
    used by vLLM's own benchmark tools. Good enough for filtering purposes.
    """
    return max(1, int(len(text.split()) * 1.3))


# =============================================================================
# Main preparation
# =============================================================================

def main(seed: int = 42):
    print("Downloading ShareGPT dataset from HuggingFace...")
    print("(This may take a few minutes on first run; cached afterwards.)\n")

    # Load HF token from .env so we get authenticated rate limits.
    # The token is optional but avoids 429 errors on slow connections.
    hf_token = None
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    hf_token = line.strip().split("=", 1)[1]
                    break

    # Use Aeala/ShareGPT_Vicuna_unfiltered — a public mirror with a stable file.
    # The original anon8231489123 repo removed its JSON files.
    local_path = hf_hub_download(
        repo_id="Aeala/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V4.3_unfiltered_cleaned_split.json",
        repo_type="dataset",
        token=hf_token,
    )

    with open(local_path, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Raw dataset size: {len(raw)} conversations")

    samples = []
    skipped = 0

    for item in raw:
        conversations = item.get("conversations", [])

        # Extract the first human → gpt turn pair only.
        # Using only the first turn keeps prompts self-contained (no prior context
        # needed) and matches how vLLM's benchmark_serving.py uses ShareGPT.
        for i, turn in enumerate(conversations):
            if turn.get("from") != "human":
                continue
            if i + 1 >= len(conversations):
                break
            next_turn = conversations[i + 1]
            if next_turn.get("from") != "gpt":
                break

            prompt = turn["value"].strip()
            reference = next_turn["value"].strip()

            # Skip empty or trivially short turns
            if len(prompt) < 20 or len(reference) < 20:
                skipped += 1
                break

            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(reference)

            # Discard extreme outliers that would skew the distribution.
            # Lower bound: prompts < 10 tokens are noise.
            # Upper bound: > 2000 input tokens exceed even INT4's context.
            if input_tokens < 10 or input_tokens > 2000:
                skipped += 1
                break
            if output_tokens < 10 or output_tokens > 1000:
                skipped += 1
                break

            samples.append({
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            break  # Only use first turn per conversation

    print(f"Extracted {len(samples)} valid samples ({skipped} skipped)\n")

    # Shuffle with a fixed seed. The benchmark client uses the same seed when
    # sampling, so FP16/INT8/INT4 runs see the same prompts in the same order.
    random.seed(seed)
    random.shuffle(samples)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    os.makedirs("benchmark/workloads", exist_ok=True)
    out_path = "benchmark/workloads/sharegpt_filtered.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")

    # -------------------------------------------------------------------------
    # Distribution summary
    # -------------------------------------------------------------------------

    input_lens  = sorted(s["input_tokens"]  for s in samples)
    output_lens = sorted(s["output_tokens"] for s in samples)
    total_lens  = sorted(s["input_tokens"] + s["output_tokens"] for s in samples)
    n = len(samples)

    print(f"\n{'='*45}")
    print(f"{'Metric':<30} {'Input':>6} {'Output':>7} {'Total':>7}")
    print(f"{'='*45}")
    print(f"{'Median':<30} {input_lens[n//2]:>6} {output_lens[n//2]:>7} {total_lens[n//2]:>7}")
    print(f"{'P95':<30} {input_lens[int(n*0.95)]:>6} {output_lens[int(n*0.95)]:>7} {total_lens[int(n*0.95)]:>7}")
    print(f"{'Max':<30} {input_lens[-1]:>6} {output_lens[-1]:>7} {total_lens[-1]:>7}")

    # Show how many samples are usable per quantization level
    print(f"\nSamples available per quantization level:")
    for label, max_len in [("FP16  (max_len=1024)", 1024),
                            ("INT8  (max_len=2048)", 2048),
                            ("INT4  (max_len=4096)", 4096)]:
        count = sum(1 for t in total_lens if t <= max_len)
        print(f"  {label}: {count:>5}  ({count/n*100:.1f}%)")

    print(f"\nDone. Run benchmark with:")
    print(f"  FP16: uv run python benchmark/client.py --max-len 1024 --concurrency 1")
    print(f"  INT8: uv run python benchmark/client.py --max-len 2048 --concurrency 1")
    print(f"  INT4: uv run python benchmark/client.py --max-len 4096 --concurrency 1")


if __name__ == "__main__":
    main()
