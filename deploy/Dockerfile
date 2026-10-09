# YoHoGPT image: training, evaluation and the API gateway.
#   docker build -f deploy/Dockerfile -t yohogpt/yohogpt:0.1.0 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git curl docker.io && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY yohogpt ./yohogpt
COPY configs ./configs
COPY data/sources.yaml ./data/sources.yaml
COPY scripts ./scripts
RUN pip install -e ".[data,serve,export]"

RUN useradd -m -u 1000 yoho && mkdir -p /app/workspaces && chown -R yoho /app
USER yoho
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD curl -sf localhost:8080/health || exit 1
CMD ["yoho", "serve", "--backend", "openai", "--upstream", "http://vllm:8000", "--port", "8080"]
