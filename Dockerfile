FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[data,alpaca,ml]"

COPY data/ ./data/
COPY retrain-model.sh ./retrain-model.sh
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/retrain-model.sh /app/entrypoint.sh

RUN mkdir -p /app/data/models

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["/app/entrypoint.sh"]
