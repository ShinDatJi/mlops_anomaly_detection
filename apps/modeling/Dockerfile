FROM ghcr.io/astral-sh/uv:0.9.28-python3.12-trixie-slim AS modeling_base

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

# Sync project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

FROM modeling_base AS modeling_preprocessing
COPY src/preprocessing/*.py src/
CMD ["uv", "run", "src/preprocess_data.py"]

FROM modeling_base AS modeling_training
COPY src/training/*.py src/
CMD ["uv", "run", "src/train_model.py"]

FROM modeling_base AS modeling_evaluation
COPY src/evaluation/*.py src/
CMD ["uv", "run", "src/evaluate_model.py"]
