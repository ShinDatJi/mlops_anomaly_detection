FROM ghcr.io/astral-sh/uv:0.9.28-python3.12-trixie-slim


# Install curl
RUN apt-get update && apt-get install -y curl

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Copy files
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY src src

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
