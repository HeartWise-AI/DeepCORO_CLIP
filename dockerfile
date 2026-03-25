FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

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
RUN uv sync --python /opt/venv/bin/python

# Pin torch to a CUDA 12.4 build compatible with the host driver stack.
RUN uv pip install --python /opt/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN uv pip install --python /opt/venv/bin/python -e /opt/HeartWise_StatPlots
RUN uv pip install --python /opt/venv/bin/python -e /opt/Orion
RUN uv pip install --python /opt/venv/bin/python -r docker_dependencies.txt

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/Orion:$PYTHONPATH"

# Download model weights at build time (secret is mounted only during RUN, not persisted in image)
RUN --mount=type=secret,id=api_key,target=/workspace/api_key.json \
    python utils/download_vasovision.py
RUN --mount=type=secret,id=api_key,target=/workspace/utils/api_key.json \
    cd utils && python download_pretrained_weights.py

CMD ["/bin/bash"]
