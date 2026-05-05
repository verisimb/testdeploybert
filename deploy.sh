#!/bin/bash
# Deploy script untuk Oracle ARM VPS (Ubuntu)
# Jalankan: bash deploy.sh

set -e
echo "=== Deploy Detektor Judi Online ==="

# 1. Update & install dependencies sistem
echo "[1/7] Install system dependencies..."
sudo apt update -qq
sudo apt install -y python3 python3-pip python3-venv nginx

# 2. Buat folder project
echo "[2/7] Setup project folder..."
mkdir -p ~/judi-detector/templates
cd ~/judi-detector

# 3. Copy file (pastikan sudah upload semua file ke VPS dulu)
# File yang dibutuhkan:
#   - app.py
#   - requirements.txt
#   - templates/index.html
#   - indobert-judionline/ (folder model dari Kaggle)

# 4. Buat virtual environment
echo "[3/7] Setup Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 5. Install Python packages
echo "[4/7] Install Python packages (ini bisa lama ~5-10 menit)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[5/7] Setup systemd service..."
sudo cp judi-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable judi-detector
sudo systemctl restart judi-detector

echo "[6/7] Setup Nginx..."
sudo cp nginx.conf /etc/nginx/sites-available/judi-detector
sudo ln -sf /etc/nginx/sites-available/judi-detector /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "[7/7] Buka port firewall..."
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT

echo ""
echo "✅ Deploy selesai!"
echo "   Cek status  : sudo systemctl status judi-detector"
echo "   Lihat log   : sudo journalctl -u judi-detector -f"
echo "   Akses web   : http://$(curl -s ifconfig.me)"
