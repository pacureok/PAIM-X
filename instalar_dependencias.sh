#!/bin/bash
echo "========================================================"
echo "🧹 LIMPIEZA Y RECONSTRUCCIÓN DE ENTORNO (PACURE LABS) 🚀"
echo "========================================================"

# 1. Desinstalar versiones conflictivas preinstaladas
pip uninstall -y numpy torch torchvision torchaudio xformers 

# 2. Instalar el 'Ecosistema NumPy 1.x' en orden estricto
echo "[1/4] Instalando Base NumPy 1.26..."
pip install --no-cache-dir "numpy<2.0.0"

echo "[2/4] Instalando PyTorch y Xformers compatibles..."
# Usamos Torch 2.4.0 que es estable con NumPy 1.x
pip install --no-cache-dir torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir xformers==0.0.27.post2 --only-binary=:all:

echo "[3/4] Instalando AudioCraft y dependencias de audio..."
pip install --no-cache-dir av==11.0.0  # Versión exacta que pide Audiocraft
pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

echo "[4/4] Instalando herramientas de procesamiento Pacure Labs..."
pip install --no-cache-dir demucs encodec flashy hydra-core==1.3.2 hydra-colorlog julius num2words omegaconf pesq pystoi torchmetrics torchdiffeq yt-dlp librosa soundfile Pillow

echo "========================================================"
echo "✅ ENTORNO RECONSTRUIDO. NUCLEO PAIM-X SINCRONIZADO."
echo "========================================================"