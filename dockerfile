FROM pytorch/pytorch:latest

WORKDIR /app

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