#!/bin/bash
# =============================================================================
# MIYA Service O'rnatuvchi
# Kompyuter yonganida MIYA avtomatik ishga tushadi
# =============================================================================

echo "============================================"
echo "  MIYA Service O'rnatish"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="$PROJECT_DIR/miya.service"

# logs papkasini yaratish
mkdir -p "$PROJECT_DIR/logs"

# Service faylni systemd ga nusxalash
echo "[1/4] Service faylni nusxalash..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/miya.service

# systemd ni yangilash
echo "[2/4] Systemd ni yangilash..."
sudo systemctl daemon-reload

# Serviceni yoqish (auto-start on boot)
echo "[3/4] Serviceni yoqish..."
sudo systemctl enable miya.service

# Hozir ishga tushirish
echo "[4/4] MIYA ni ishga tushirish..."
sudo systemctl start miya.service

sleep 3

# Status tekshirish
echo ""
echo "============================================"
STATUS=$(sudo systemctl is-active miya.service)
if [ "$STATUS" = "active" ]; then
    echo "  MIYA ishlayapti!"
    echo ""
    echo "  Web UI:  http://localhost:8000"
    echo "  Health:  http://localhost:8000/health"
    echo ""
    echo "  Foydali buyruqlar:"
    echo "    sudo systemctl status miya    — holat"
    echo "    sudo systemctl restart miya   — qayta ishga tushirish"
    echo "    sudo systemctl stop miya      — to'xtatish"
    echo "    journalctl -u miya -f         — loglarni ko'rish"
else
    echo "  MIYA ishga tushmadi. Logni tekshiring:"
    echo "    sudo systemctl status miya.service"
    echo "    cat $PROJECT_DIR/logs/backend.log"
fi
echo "============================================"
