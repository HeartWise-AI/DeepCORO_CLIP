FROM pytorch/pytorch:latest

WORKDIR /app

RUN apt update && apt upgrade -y && apt install -y git wget libgl1-mesa-glx libglib2.0-0

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# Clone the Orion repository and install it
RUN git clone https://github.com/HeartWise-AI/Orion.git

# Install HeartWise_StatPlots
RUN git clone https://github.com/HeartWise-AI/HeartWise_StatPlots.git

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

COPY api_key.json* .

RUN pip install uv
RUN uv sync

RUN uv pip install -e HeartWise_StatPlots
RUN uv pip install -r docker_dependencies.txt

CMD ["/bin/bash"]