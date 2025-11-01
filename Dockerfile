# minimal CPU image; switch to nvidia/cuda base if deploying on GPU
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# system deps (optional but useful for scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# copy only what's needed for the service
COPY src/nlp/summarizer/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# copy source tree (expects build context at repo root)
COPY src /app/src

EXPOSE 8000
CMD ["uvicorn", "src.nlp.summarizer.api:app", "--host", "0.0.0.0", "--port", "8000"]
