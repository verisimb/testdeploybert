FROM python:3.11-slim

WORKDIR /app

# Install dependencies sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY templates/ templates/

# Port: Coolify/PaaS sering set env PORT — ikut agar proxy tidak 502 Bad Gateway
ENV PORT=5000
EXPOSE 5000

# Shell form agar $PORT ter-expand; exec = PID 1 = gunicorn (signal shutdown benar)
CMD exec gunicorn --workers 2 --bind 0.0.0.0:$PORT --timeout 120 --preload app:app
