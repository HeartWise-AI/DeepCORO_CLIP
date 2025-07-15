FROM pytorch/pytorch:latest

WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .

COPY README.md .
COPY config .
COPY dataloaders .
COPY models .
COPY projects .
COPY runners .
COPY scripts .
COPY utils .

RUN pip install uv
RUN uv sync

CMD ["/bin/bash"]