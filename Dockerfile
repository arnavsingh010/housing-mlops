FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

# Train at build time so the image is self-contained (falls back to synthetic data if offline)
RUN housing-mlops train

ENTRYPOINT ["housing-mlops"]
CMD ["predict", "--sample", "5"]
