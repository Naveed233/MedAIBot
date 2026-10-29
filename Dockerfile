# Build stage: resolve dependencies from uv.lock into a virtualenv.
FROM python:3.12-slim AS build
RUN pip install --no-cache-dir uv==0.13.6
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra openai --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra openai --no-editable

# Runtime stage: no build tools, non-root user.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY data/sources.json data/sources.json
COPY data/raw data/raw
RUN mkdir -p data/index .cache && chown -R app:app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
# Build the index into the image so containers start ready. Downloads the embedding model (about 64 MB).
RUN medaibot ingest
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["medaibot", "serve", "--host", "0.0.0.0", "--port", "8000"]
