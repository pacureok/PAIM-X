#!/bin/bash
echo "========================================================"
echo "🚀 INSTALACIÓN RÁPIDA PAIM-X (SOLO BINARIOS) 🚀"
echo "========================================================"

# Actualizar sistema de forma silenciosa
apt-get update -qq && apt-get install -y ffmpeg > /dev/null

# 1. Instalamos las versiones que YA TIENEN 'wheels' (archivos listos)
# No forzamos versiones viejas para evitar que se ponga a compilar.
pip install --no-cache-dir torch torchvision torchaudio 

# 2. xformers: Instalamos la versión más reciente que coincida con el torch instalado
# Usamos --only-binary para asegurar que no intente construir nada.
pip install --no-cache-dir --only-binary=:all: xformers

# 3. AudioCraft (lo instalamos sin dependencias para que no rompa nada)
pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

# 4. Resto de herramientas necesarias
pip install --no-cache-dir demucs encodec flashy hydra-core julius num2words omegaconf pesq pystoi torchmetrics yt-dlp librosa soundfile Pillow
pip install --no-cache-dir "numpy<2.0.0"

echo "========================================================"
echo "✅ ENTORNO LISTO SIN CONSTRUCCIÓN DE CÓDIGO."
echo "========================================================"