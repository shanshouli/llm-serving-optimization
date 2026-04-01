"""
HuggingFace Transformers Baseline Server

Serves Llama-3.2-3B-Instruct using plain HF Transformers (no PagedAttention,
no continuous batching). Exposes the same OpenAI-compatible /v1/completions
endpoint as vLLM so benchmark/client.py can target both without changes.

This is the "naive" baseline — requests are processed one at a time, which
shows what vLLM's optimizations are actually buying us.

Usage:
    python serving/baseline_hf.py

Environment variables:
    MODEL_NAME  — HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct)
    PORT        — Port to listen on (default: 8001)
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
PORT = int(os.getenv("PORT", 8001))

# Global model/tokenizer references, loaded once at startup
tokenizer = None
model = None

# Async lock: HF Transformers is not thread-safe for inference.
# This serializes requests so they don't corrupt each other's GPU state.
# vLLM handles this internally — this lock is the key difference in behavior.
inference_lock = asyncio.Lock()


# =============================================================================
# Request / Response schemas (mirrors the OpenAI /v1/completions format)
# =============================================================================

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256


class CompletionResponse(BaseModel):
    choices: list
    usage: dict


# =============================================================================
# App lifecycle: load model on startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, unload on shutdown."""
    global tokenizer, model

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # float16 to match vLLM's dtype; device_map="auto" puts it on the GPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    yield  # Server runs here

    # Cleanup on shutdown
    del model, tokenizer
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health():
    """Health check — returns ok once the model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    """
    OpenAI-compatible completions endpoint.

    Requests are serialized via inference_lock because plain HF generate()
    is not designed for concurrent calls. This matches how a developer would
    naively deploy a model without a dedicated inference server.
    """
    async with inference_lock:
        # Tokenize the prompt
        inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Run greedy generation (no sampling, same as vLLM default)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=False,
            )

        # Slice off the prompt tokens, decode only the generated part
        new_token_ids = output_ids[0][prompt_tokens:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        completion_tokens = len(new_token_ids)

    return {
        "choices": [{"text": generated_text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
