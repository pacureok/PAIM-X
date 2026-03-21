#!/bin/bash
echo "🚀 Instalando entorno PAIM-X (Pacure Labs)..."

# Actualizar sistema e instalar FFmpeg para audio
apt-get update -qq && apt-get install -y ffmpeg > /dev/null

# Instalar PyTorch para GPU y xformers (necesario para MusicGen)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.22.post7

# Instalar AudioCraft y herramientas de IA
pip install --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
pip install demucs encodec flashy>=0.0.1 hydra-core>=1.1 julius num2words omegaconf pesq pystoi torchmetrics yt-dlp librosa soundfile Pillow
pip install "numpy<2.0.0"

echo "✅ Instalación terminada."