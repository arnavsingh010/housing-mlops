FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir ".[api]"

# Train at build time so the image is self-contained (falls back to synthetic data if offline)
RUN housing-mlops train

# Hosts like Render pass the port in $PORT; default to 8000 locally
EXPOSE 8000
CMD ["sh", "-c", "uvicorn housing_mlops.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
