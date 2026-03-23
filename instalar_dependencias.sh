#!/bin/bash
echo "========================================================"
echo "🚀 REPARANDO DEPENDENCIAS PAIM-X (PACURE LABS) 🚀"
echo "========================================================"

# Instalación de FFmpeg en el sistema
apt-get update -qq && apt-get install -y ffmpeg > /dev/null

# 1. Forzar Numpy compatible antes que nada
pip install --no-cache-dir "numpy<2.0.0"

# 2. Instalar los módulos "olvidados" (Binarios solamente)
# 'av' es el que causó tu último error.
pip install --no-cache-dir --only-binary=:all: av==12.3.0 hydra-colorlog torchdiffeq

# 3. Instalar xformers (Binario)
pip install --no-cache-dir --only-binary=:all: xformers

# 4. AudioCraft (sin dependencias para que no intente bajar versiones viejas)
pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

# 5. El resto de herramientas de Pacure Labs
pip install --no-cache-dir demucs encodec flashy hydra-core julius num2words omegaconf pesq pystoi torchmetrics yt-dlp librosa soundfile Pillow

echo "========================================================"
echo "✅ TODO LISTO. EL MOTOR DEBERÍA ARRANCAR AHORA."
echo "========================================================"