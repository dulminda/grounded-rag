FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install .

# Bake a SciFact index into the image so `docker run` serves immediately.
ARG CONFIG=configs/bm25_tuned.yaml
RUN grounded data --dest /app/data/scifact \
 && grounded index --config ${CONFIG} --scifact /app/data/scifact --out /app/indexes/index.pkl \
 && rm -rf /app/data

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# Pass ANTHROPIC_API_KEY at runtime to enable /v1/answer and /v1/verify.
CMD ["grounded", "serve", "--index", "/app/indexes/index.pkl", "--host", "0.0.0.0", "--port", "8000"]
