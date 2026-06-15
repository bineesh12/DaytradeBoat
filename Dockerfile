FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir \
    "python-dotenv>=0.19" \
    "flask>=2.0,<3.1" \
    "werkzeug>=2.0,<3.1" \
    "requests>=2.28" \
    "yfinance>=0.2" \
    "alpaca-py>=0.30" \
    "numpy>=1.24,<2.0" \
    "pandas>=2.0,<2.3" \
    "xgboost==1.7.6" \
    "scikit-learn>=1.3,<1.7"

COPY src/ ./src/

RUN pip install --no-cache-dir --no-deps .

COPY retrain-model.sh ./retrain-model.sh
RUN chmod +x /app/retrain-model.sh

RUN mkdir -p /app/data/models

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "daytrading.runner"]
