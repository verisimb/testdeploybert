#!/bin/bash
# Deploy dengan Docker di Oracle ARM VPS (Ubuntu)
# Jalankan: bash deploy-docker.sh

set -e
echo "=== Deploy Detektor Judi Online (Docker) ==="

# 1. Install Docker
echo "[1/4] Install Docker..."
sudo apt update -qq
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update -qq
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Tambahkan user ke group docker agar tidak perlu sudo
sudo usermod -aG docker $USER

# 2. Install Nginx
echo "[2/4] Install & setup Nginx..."
sudo apt install -y nginx
sudo cp nginx.conf /etc/nginx/sites-available/judi-detector
sudo ln -sf /etc/nginx/sites-available/judi-detector /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

# 3. Buka port firewall
echo "[3/4] Buka port firewall..."
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT

# 4. Build & run Docker
echo "[4/4] Build & jalankan Docker (pertama kali bisa lama ~10 menit)..."
sudo docker compose up -d --build

echo ""
echo "✅ Deploy selesai!"
echo ""
echo "   Akses web    : http://$(curl -s ifconfig.me)"
echo ""
echo "── Perintah berguna ──"
echo "   Lihat log    : docker compose logs -f"
echo "   Stop         : docker compose down"
echo "   Restart      : docker compose restart"
echo "   Hapus semua  : docker compose down --rmi all -v"
