"""
Vertex AI Endpoint Deployment Script
Deploys Llama-3.2-3B-Instruct with vLLM backend on Vertex AI (L4 GPU).

Matches SageMaker experiment for direct cost/performance comparison:
- FP16:  meta-llama/Llama-3.2-3B-Instruct
- INT8:  neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8
- INT4:  casperhansen/llama-3.2-3b-instruct-awq

Usage:
    python cloud/vertex/vertex_deploy.py --project YOUR_PROJECT_ID --hf-token YOUR_HF_TOKEN
    python cloud/vertex/vertex_deploy.py --project YOUR_PROJECT_ID --hf-token YOUR_HF_TOKEN --quantize int4
    python cloud/vertex/vertex_deploy.py --project YOUR_PROJECT_ID --hf-token YOUR_HF_TOKEN --quantize int8

Prerequisites:
    gcloud auth application-default login
    pip install google-cloud-aiplatform
"""

import argparse
import time
from google.cloud import aiplatform

# ── Constants ──────────────────────────────────────────────────────────────
REGION        = "us-central1"
MACHINE_TYPE  = "g2-standard-4"       # 1x L4 GPU, 4 vCPU, 16 GB RAM, ~$0.98/hr
ACCELERATOR   = "NVIDIA_L4"
ACCEL_COUNT   = 1

# Vertex AI Model Garden vLLM serving container (stable release)
VLLM_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20241211_0916_RC00"
)

# Model IDs match local and SageMaker experiments exactly
MODEL_IDS = {
    "fp16": "meta-llama/Llama-3.2-3B-Instruct",
    "int8": "neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",
    "int4": "casperhansen/llama-3.2-3b-instruct-awq",
}

# Max model length per quantization (same as SageMaker config)
MAX_MODEL_LEN = {
    "fp16": "1024",
    "int8": "2048",
    "int4": "4096",
}


def deploy(project: str, hf_token: str, quantize: str = "int4") -> str:
    """Deploy vLLM endpoint on Vertex AI. Returns endpoint resource name."""

    aiplatform.init(project=project, location=REGION)

    hf_model_id = MODEL_IDS[quantize]
    endpoint_name = f"vllm-llama3-{quantize}"

    print(f"Project:      {project}")
    print(f"Region:       {REGION}")
    print(f"Machine:      {MACHINE_TYPE} (1x {ACCELERATOR})")
    print(f"Model:        {hf_model_id}")
    print(f"Endpoint:     {endpoint_name}")
    print(f"Container:    {VLLM_IMAGE}")
    print()

    # Environment variables passed to the vLLM container
    env_vars = {
        "MODEL_ID":                   hf_model_id,
        "HUGGING_FACE_HUB_TOKEN":     hf_token,
        "DEPLOY_SOURCE":              "notebook",
        "VLLM_ARGS": (
            f"--max-model-len {MAX_MODEL_LEN[quantize]} "
            f"--gpu-memory-utilization 0.90 "
            f"--dtype auto"
        ),
    }

    # Upload model resource (points to HuggingFace, no actual artifact upload)
    print("Registering model resource...")
    model = aiplatform.Model.upload(
        display_name=endpoint_name,
        serving_container_image_uri=VLLM_IMAGE,
        serving_container_environment_variables=env_vars,
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_ports=[7080],
    )
    print(f"Model registered: {model.resource_name}")

    # Create endpoint
    print("Creating endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f"Endpoint created: {endpoint.resource_name}")

    # Deploy model to endpoint
    print(f"Deploying to {MACHINE_TYPE} with {ACCELERATOR}... (cold start ~10 min)")
    cold_start_begin = time.time()

    endpoint.deploy(
        model=model,
        deployed_model_display_name=endpoint_name,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR,
        accelerator_count=ACCEL_COUNT,
        traffic_percentage=100,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,   # wait until InService
    )

    cold_start_sec = time.time() - cold_start_begin
    print(f"\nEndpoint ready. Cold-start time: {cold_start_sec:.1f}s")
    print(f"Endpoint resource name: {endpoint.resource_name}")
    print(f"Save this for benchmarking: {endpoint.name}")

    # Save endpoint ID to file for benchmark script to read
    import os
    os.makedirs("results/raw", exist_ok=True)
    with open(f"results/raw/vertex_{quantize}_endpoint_id.txt", "w") as f:
        f.write(endpoint.name)
    print(f"Endpoint ID saved to results/raw/vertex_{quantize}_endpoint_id.txt")

    return endpoint.resource_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",   required=True, help="GCP project ID")
    parser.add_argument("--hf-token",  required=True, help="HuggingFace token")
    parser.add_argument("--quantize",  default="int4", choices=["fp16", "int8", "int4"],
                        help="Quantization: fp16 / int8 / int4 (default: int4)")
    args = parser.parse_args()
    deploy(args.project, args.hf_token, args.quantize)
