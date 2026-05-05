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

# Expose port
EXPOSE 5000

# Jalankan dengan gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "--preload", "app:app"]
