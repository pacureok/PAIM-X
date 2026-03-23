#!/bin/bash
echo "========================================================"
echo "🛡️ INSTALACIÓN BLINDADA PAIM-X (PACURE LABS) 🛡️"
echo "========================================================"

# 1. Limpieza de librerías conflictivas de Kaggle
pip uninstall -y numpy torch torchvision torchaudio xformers av

# 2. Ecosistema NumPy y PyTorch Estable
echo "[1/4] Instalando Base NumPy..."
pip install --no-cache-dir "numpy<2.0.0"

echo "[2/4] Instalando Motor PyTorch y Xformers..."
pip install --no-cache-dir torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir xformers==0.0.27.post2 --only-binary=:all:

# 3. EL TRUCO MAGICO: Instalar 'av' moderno pre-compilado para evitar el error de C++
echo "[3/4] Instalando Motor AV y Audiocraft..."
pip install --no-cache-dir --only-binary=:all: av==12.3.0
pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

# 4. Herramientas restantes
echo "[4/4] Instalando módulos secundarios..."
pip install --no-cache-dir demucs encodec flashy hydra-core==1.3.2 hydra-colorlog julius num2words omegaconf pesq pystoi torchmetrics torchdiffeq yt-dlp librosa soundfile Pillow

echo "========================================================"
echo "✅ MOTOR AUDIO-VISUAL INSTALADO CORRECTAMENTE."
echo "========================================================"