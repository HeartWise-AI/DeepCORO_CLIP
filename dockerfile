FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt update && apt upgrade -y && apt install -y wget libgl1-mesa-glx libglib2.0-0

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

COPY uv.lock .
COPY pyproject.toml .

COPY README.md .
COPY config/ config/
COPY dataloaders/ dataloaders/
COPY models/ models/
COPY projects/ projects/
COPY runners/ runners/
COPY scripts/ scripts/
COPY utils/ utils/

COPY api_key.json* .

RUN pip install uv
RUN uv sync --no-cache

CMD ["/bin/bash"]