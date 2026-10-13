FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs build-essential ffmpeg libgl1 libglib2.0-0 ca-certificates && rm -rf /var/lib/apt/lists/*
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 HF_HUB_DISABLE_XET=1
WORKDIR /opt/project
COPY . .
RUN uv sync --frozen --no-dev --python 3.10 && uv cache clean
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/opt/project WANDB_MODE=disabled
CMD ["python", "scripts/dlc_run.py", "--help"]
