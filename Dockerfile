FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Jangan membatasi OMP ke 2 — memperlambat matmul/CPU; atur paralelisme lewat ONNX_NUM_THREADS di app.py
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/

# Coolify/PaaS: ikuti env PORT agar tidak 502 Bad Gateway
ENV PORT=5000
EXPOSE 5000

# 1 worker = latensi lebih stabil di CPU (2 worker sering rebutan core). Naikkan via GUNICORN_WORKERS jika perlu throughput.
ENV GUNICORN_WORKERS=1

# Timeout 180s untuk unduh model ONNX pertama kali; exec = gunicorn sebagai PID 1
CMD exec gunicorn --workers ${GUNICORN_WORKERS:-1} --bind 0.0.0.0:$PORT --timeout 180 --preload app:app
