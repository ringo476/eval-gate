FROM python:3.12-slim

WORKDIR /app

# System deps: git is needed by scripts/setup_target.sh to vendor the
# target app at build time.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN bash scripts/setup_target.sh

# No default model/provider baked in -- the caller decides per invocation,
# e.g.:
#   docker run --rm -e ANTHROPIC_API_KEY=... harness run --provider anthropic --smoke 10
ENTRYPOINT ["python", "-m", "harness.cli"]
CMD ["--help"]
