FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn==23.0.0

COPY datalens ./datalens
COPY static ./static
COPY data ./data
COPY app.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# One worker: datasets live in that process's memory. Threads keep the UI responsive during training.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:8000", "app:app"]
