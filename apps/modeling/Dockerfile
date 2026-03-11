FROM ghcr.io/astral-sh/uv:0.9.28-python3.12-trixie-slim

ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# copy files
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY src src

# Sync project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
