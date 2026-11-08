FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: git is needed by scripts/setup_target.sh to vendor the
# target app at build time.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first: embeddings don't need a GPU here, and the default
# build pulls in several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake both embedding models into the image so a run never needs to reach the
# sites they download from. HF_HUB_OFFLINE stops the library checking for updates.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"
ENV HF_HUB_OFFLINE=1

COPY . .
RUN bash scripts/setup_target.sh

# No default model/provider baked in -- the caller decides per invocation:
#   docker run --rm -e OPENAI_API_KEY=... -v "$PWD/reports:/app/reports" harness \
#       run --task rag_fusion_qa --provider openai --model gpt-6-luna
# To use a local Ollama model from inside the container, point at the host:
#   docker run --rm -e HARNESS_OLLAMA_URL=http://host.docker.internal:11434/v1 harness \
#       run --task rag_fusion_qa --provider ollama --model llama3.1:8b
ENTRYPOINT ["python", "-m", "harness.cli"]
CMD ["--help"]
