# Member B — Cloud Deployment To-Do List

## Prerequisites (Before Starting)

- [ ] Pull latest from main branch — all benchmark scripts are already there
- [ ] Install Python dependencies: `uv add datasets aiohttp httpx boto3 sagemaker`
- [ ] Prepare ShareGPT dataset (if not already in repo):
  ```bash
  uv run python benchmark/prepare_sharegpt.py
  ```
  If `benchmark/workloads/sharegpt_filtered.json` is already in the repo after pulling, skip this step.
- [ ] Confirm AWS credentials are configured: `aws sts get-caller-identity`

---

## Step 1 — Deploy SageMaker Endpoint

Create `cloud/sagemaker_deploy.py` and deploy using AWS DJL-LMI container with vLLM backend:

```python
import sagemaker

container_uri = sagemaker.image_uris.retrieve(
    framework="djl-lmi", version="0.30.0", region="us-east-1"
)
model = sagemaker.Model(
    image_uri=container_uri,
    role=iam_role,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.2-3B-Instruct",
        "OPTION_ROLLING_BATCH": "vllm",
        "TENSOR_PARALLEL_DEGREE": "max",
    }
)
model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",   # A10G 24GB, ~$1.41/hr
    endpoint_name="vllm-llama3-int4",
)
```

Record: time from `deploy()` call to first successful response = **cold-start latency**.

---

## Step 2 — Run Benchmark (Same Script as Member A)

Point the existing benchmark client at the SageMaker endpoint. No code changes needed.

```bash
# c=1, 100 requests
uv run python benchmark/client.py \
  --url https://<endpoint>.sagemaker.amazonaws.com/v1/completions \
  --concurrency 1 --requests 100 --max-len 4096

# c=8, 100 requests
uv run python benchmark/client.py \
  --url https://<endpoint>.sagemaker.amazonaws.com/v1/completions \
  --concurrency 8 --requests 100 --max-len 4096
```

Results auto-saved to `results/raw/`.

---

## Step 3 — Configure Auto-Scaling

Create `cloud/sagemaker_autoscale.py`:

```python
import boto3

client = boto3.client("application-autoscaling")

client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,
    MaxCapacity=3,
)
client.put_scaling_policy(
    PolicyName="invocation-scaling",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 10.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

Traffic pattern to test:
1. Send 0 QPS → burst to 10 QPS
2. Sustain 10 QPS for 5 minutes
3. Drop to 0 QPS

Observe in CloudWatch: instance count timeline, scale-up time, scale-down time, any failed requests during transitions. Take screenshots.

---

## Step 4 — Cost Analysis

```
tokens_per_dollar = total_tokens_generated / (instance_cost_per_second × wall_clock_seconds)
instance_cost_per_second = 1.41 / 3600   # ml.g5.xlarge
```

---

## Step 5 — Clean Up (IMPORTANT)

**Delete endpoint immediately after experiments to stop billing.**

```python
import sagemaker
sagemaker.Predictor("vllm-llama3-int4").delete_endpoint()
```

---

## Step 6 — Vertex AI (If Time Permits)

1. Push DJL-LMI container image to Artifact Registry
2. Deploy to Vertex AI endpoint (`g2-standard-4`, L4 24GB, ~$0.98/hr)
3. Run same benchmark (c=1 and c=8)
4. Configure auto-scaling (min=0, max=3)
5. Delete endpoint after experiments

---

## Deliverables (Give to Member A/C)

Fill in this table with your measured numbers:

| Metric | SageMaker A10G (INT4) | Vertex AI L4 (if done) |
|---|---|---|
| Cold-start latency | ? | ? |
| Avg latency c=1 | ? | ? |
| P95 latency c=1 | ? | ? |
| Avg latency c=8 | ? | ? |
| P95 latency c=8 | ? | ? |
| Aggregate throughput c=8 | ? | ? |
| Scale 0→1 time | ? | N/A |
| Scale 1→2 time | ? | N/A |
| tokens/dollar | ? | ? |

Push `results/raw/` CSVs and CloudWatch screenshots to the repo.

---

## Budget Reminder

| Platform | Estimated Cost |
|---|---|
| SageMaker (`ml.g5.xlarge`) | ~$6–9 (4–6 GPU hours) |
| Vertex AI (if done) | ~$4–6 (4–6 GPU hours) |
