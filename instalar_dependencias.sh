#!/bin/bash
# instalar_dependencias.sh - PAIM-X (Powered by Pacure Labs)
# Configurado para compilar en arquitectura TPU (XLA)

echo "========================================================"
echo "🚀 INICIANDO INSTALACIÓN DE DEPENDENCIAS (PACURE LABS) 🚀"
echo "========================================================"

echo "[1/5] Actualizando repositorios del sistema..."
apt-get update -qq

echo "[2/5] Instalando herramientas base de procesamiento de audio (FFmpeg)..."
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev libpostproc-dev > /dev/null

echo "[3/5] Instalando PyTorch y el motor XLA (Optimizados para TPU)..."
# Instalamos la versión 2.4.0 que es la más estable para el compilador de la TPU
pip install torch~=2.4.0 torch_xla[tpu]~=2.4.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torchvision torchaudio

echo "[4/5] Instalando módulos base de PAIM-X (AudioCraft, Demucs y utilidades)..."
pip install av==12.3.0
pip install --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
pip install demucs encodec flashy>=0.0.1 hydra-core>=1.1 hydra_colorlog julius num2words omegaconf pesq pystoi torchdiffeq torchmetrics yt-dlp librosa soundfile Pillow

echo "[5/5] Ajustando compatibilidad de Numpy..."
pip install "numpy<2.0.0"

echo "========================================================"
echo "✅ ENTORNO LISTO. EL MOTOR PAIM-X PUEDE INICIAR."
echo "========================================================"