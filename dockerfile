FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
# torch 2.14.0+cu130 matches the lockfile (uv.lock) and requires NVIDIA driver
# >=580 (CUDA 13.0's minimum) and a Turing-or-newer GPU (compute capability
# >=7.5); CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta.
# If you deploy this image to older GPUs/drivers, pin the base image and the
# torch/torchvision install below back to a CUDA 12.4 build instead.

RUN apt update && apt upgrade -y && apt install -y git wget libgl1-mesa-glx libglib2.0-0

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

RUN git clone https://github.com/HeartWise-AI/Orion.git /opt/Orion
RUN git clone https://github.com/HeartWise-AI/HeartWise_StatPlots.git /opt/HeartWise_StatPlots

COPY uv.lock .
COPY pyproject.toml .
COPY docker_dependencies.txt .

COPY README.md .
COPY config/ config/
COPY dataloaders/ dataloaders/
COPY models/ models/
COPY projects/ projects/
COPY runners/ runners/
COPY scripts/ scripts/
COPY utils/ utils/

RUN pip install uv
RUN uv venv --system-site-packages /opt/venv
# uv.lock pins torch 2.14.0+cu130 / torchvision 0.29.0+cu130, matching this
# image's CUDA 13.0 base, so no separate torch pin/reinstall is needed here
# (previously this step reinstalled an older cu124 build over whatever uv
# sync resolved; that override is gone now that the lockfile and base image
# agree on CUDA 13).
RUN uv sync --python /opt/venv/bin/python

RUN uv pip install --python /opt/venv/bin/python -e /opt/HeartWise_StatPlots
RUN uv pip install --python /opt/venv/bin/python -e /opt/Orion
RUN uv pip install --python /opt/venv/bin/python -r docker_dependencies.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/Orion:$PYTHONPATH"

# Download model weights at build time (secret is mounted only during RUN, not persisted in image)
ARG DEEPCORO_MODELS
RUN test -n "${DEEPCORO_MODELS}" || \
    (echo "DEEPCORO_MODELS is required: use stenosis, mace, or stenosis,mace" >&2; exit 2)
RUN --mount=type=secret,id=api_key,target=/workspace/api_key.json \
    python utils/download_vasovision.py
RUN --mount=type=secret,id=api_key,target=/workspace/utils/api_key.json \
    cd utils && python download_pretrained_weights.py --models "${DEEPCORO_MODELS}"

CMD ["/bin/bash"]
