FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA-enabled torch (CUDA 12.4, compatible with RTX 30/40 series)
# This wheel bundles all CUDA libs so no separate CUDA toolkit install is needed.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "from mistralai.client import Mistral" > /usr/local/lib/python3.13/site-packages/mistralai/__init__.py

# Download NLTK data required by haystack's EmbeddingBasedDocumentSplitter
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy source code (data/ & evaluation/results/ are mounted as volumes)
COPY config/     ./config/
COPY pipeline/   ./pipeline/
COPY orchestration/ ./orchestration/
COPY evaluation/ ./evaluation/
COPY logs/       ./logs/

CMD ["python", "-m", "orchestration.run_pipeline"]
