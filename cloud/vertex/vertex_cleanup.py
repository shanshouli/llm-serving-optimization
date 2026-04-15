"""
Vertex AI Cleanup Script
Deletes endpoint and model resources to stop billing.

Run immediately after benchmarks, or in an emergency if deployment hangs.

Usage:
    python cloud/vertex/vertex_cleanup.py --project YOUR_PROJECT_ID
    python cloud/vertex/vertex_cleanup.py --project YOUR_PROJECT_ID --quant int4
    python cloud/vertex/vertex_cleanup.py --project YOUR_PROJECT_ID --all
"""

import argparse
from google.cloud import aiplatform

REGION = "us-central1"


def delete_endpoints_by_name(project: str, display_name: str):
    """Delete all endpoints matching display_name."""
    aiplatform.init(project=project, location=REGION)
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{display_name}"')
    if not endpoints:
        print(f"  No endpoint found: {display_name}")
        return
    for ep in endpoints:
        print(f"  Undeploying all models from: {ep.resource_name}")
        ep.undeploy_all()
        print(f"  Deleting endpoint: {ep.resource_name}")
        ep.delete()
        print(f"  Done: {ep.display_name}")


def delete_models_by_name(project: str, display_name: str):
    """Delete all model resources matching display_name."""
    aiplatform.init(project=project, location=REGION)
    models = aiplatform.Model.list(filter=f'display_name="{display_name}"')
    if not models:
        print(f"  No model found: {display_name}")
        return
    for m in models:
        print(f"  Deleting model: {m.resource_name}")
        m.delete()
        print(f"  Done: {m.display_name}")


def cleanup(project: str, quant: str):
    """Delete endpoint + model for a given quantization config."""
    name = f"vllm-llama3-{quant}"
    print(f"\n--- Cleaning up: {name} ---")
    delete_endpoints_by_name(project, name)
    delete_models_by_name(project, name)
    print(f"Cleanup complete for {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--quant",   default=None, choices=["fp16", "int8", "int4"],
                        help="Which deployment to delete")
    parser.add_argument("--all",     action="store_true",
                        help="Delete ALL vllm-llama3-* endpoints and models")
    args = parser.parse_args()

    if args.all:
        for q in ["fp16", "int8", "int4"]:
            cleanup(args.project, q)
    elif args.quant:
        cleanup(args.project, args.quant)
    else:
        print("Specify --quant [fp16|int8|int4] or --all")
