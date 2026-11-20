# --- Build stage: resolve the locked dependencies with uv ---
FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- Runtime stage ---
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache \
    MODELS_ROOT=/app/models
WORKDIR /app
RUN groupadd -r app && useradd -m -r -g app app && mkdir -p /app/.cache && chown -R app:app /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app.py ./
COPY --chown=app:app fakenews ./fakenews
COPY --chown=app:app data/examples.json ./data/examples.json
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app models/models.yaml ./models/models.yaml
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
