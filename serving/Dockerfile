# =============================================================================
# HuggingFace Baseline Server Image
#
# Built once with: docker compose build baseline
# Run with:        docker compose up -d baseline
#
# Uses PyTorch's official CUDA image so we don't need to install CUDA manually.
# =============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install inference dependencies
RUN pip install --no-cache-dir \
    transformers==4.44.0 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    accelerate==0.33.0 \
    huggingface_hub==0.24.0

# Copy only the serving code (not the whole project, to keep layers small)
COPY serving/baseline_hf.py /workspace/serving/baseline_hf.py

CMD ["python", "serving/baseline_hf.py"]
