"""
SageMaker Endpoint Deployment Script
Deploys Llama-3.2-3B-Instruct with vLLM backend (INT4 AWQ) on ml.g5.xlarge.

Usage:
    python cloud/sagemaker_deploy.py --hf-token YOUR_HF_TOKEN
    python cloud/sagemaker_deploy.py --hf-token YOUR_HF_TOKEN --quantize none  # FP16 fallback
"""

import argparse
import time
import boto3
import sagemaker

ACCOUNT_ID = "697196251385"
REGION = "us-west-2"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
INSTANCE_TYPE = "ml.g5.xlarge"


def deploy(hf_token: str, quantize: str = "awq", endpoint_name: str = None):
    sess = sagemaker.Session(boto3.Session(region_name=REGION))

    container_uri = sagemaker.image_uris.retrieve(
        framework="djl-lmi", version="0.30.0", region=REGION
    )
    print(f"Container URI: {container_uri}")

    # Use the same pre-quantized models as local experiments for consistency:
    # - AWQ (INT4): casperhansen/llama-3.2-3b-instruct-awq (same as docker-compose.int4.yml)
    # - GPTQ (INT8): neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8 (same as docker-compose.int8.yml)
    # - FP16: meta-llama/Llama-3.2-3B-Instruct (base model)
    if quantize == "awq":
        hf_model_id = "casperhansen/llama-3.2-3b-instruct-awq"
    elif quantize == "gptq":
        hf_model_id = "neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8"
    else:
        hf_model_id = "meta-llama/Llama-3.2-3B-Instruct"

    env = {
        "HF_MODEL_ID": hf_model_id,
        "HF_TOKEN": hf_token,
        "OPTION_ROLLING_BATCH": "vllm",
        "TENSOR_PARALLEL_DEGREE": "max",
        "OPTION_MAX_MODEL_LEN": "4096" if quantize == "awq" else "2048" if quantize == "gptq" else "1024",
        "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
    }
    # Note: neuralmagic W8A8 model uses compressed-tensors format internally.
    # vLLM auto-detects quantization from model config — do NOT set OPTION_QUANTIZE,
    # as setting it to "gptq" causes a health check failure.

    # Derive endpoint name from quantization type if not explicitly provided
    if endpoint_name is None:
        endpoint_name = f"vllm-llama3-{'int4' if quantize == 'awq' else 'int8' if quantize == 'gptq' else 'fp16'}"

    model = sagemaker.Model(
        image_uri=container_uri,
        role=ROLE_ARN,
        env=env,
        sagemaker_session=sess,
    )

    print(f"Deploying {INSTANCE_TYPE} endpoint: {endpoint_name} ...")
    cold_start_begin = time.time()

    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
    )

    cold_start_sec = time.time() - cold_start_begin
    print(f"Endpoint ready. Cold-start time: {cold_start_sec:.1f}s")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Region: {REGION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--quantize", default="awq", choices=["awq", "gptq", "none"],
                        help="Quantization method (default: awq/INT4)")
    args = parser.parse_args()

    # Prefer env var over CLI arg to avoid token appearing in shell history
    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("Error: provide --hf-token or set HF_TOKEN environment variable")

    deploy(hf_token, args.quantize)
