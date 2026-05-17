FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Atur paralelisme lewat TORCH_NUM_THREADS di app.py (jangan batasi OMP global)
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install torch CPU-only dulu (hemat ~1.5GB image vs wheel default berisi CUDA),
# lalu sisanya. Index extra hanya dipakai kalau paket tidak ada di PyPI.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1.0" \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/

# Coolify/PaaS: ikuti env PORT agar tidak 502 Bad Gateway
ENV PORT=3000
EXPOSE 3000

# 1 worker = latensi lebih stabil di CPU (2 worker sering rebutan core). Naikkan via GUNICORN_WORKERS jika perlu throughput.
ENV GUNICORN_WORKERS=1

# Timeout 180s untuk unduh model pertama kali; exec = gunicorn sebagai PID 1
# --max-requests recycling mengurangi memory creep dari aktivasi PyTorch (CPU arena).
# CATATAN: TIDAK pakai --preload. Kombinasi --preload + OpenMP/MKL (yang dipakai
# torch CPU) + fork() di Linux menyebabkan deadlock di worker (request hang
# selamanya). Tanpa --preload, model di-load di setiap worker secara independen.
CMD exec gunicorn \
    --workers ${GUNICORN_WORKERS:-1} \
    --bind 0.0.0.0:$PORT \
    --timeout 180 \
    --graceful-timeout 30 \
    --max-requests 200 \
    --max-requests-jitter 50 \
    app:app
