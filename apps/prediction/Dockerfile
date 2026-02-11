FROM ghcr.io/astral-sh/uv:0.9.28-python3.12-trixie-slim

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy files
COPY src/*.py src/
COPY .python-version .
COPY pyproject.toml .

# Install dependencies and project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
