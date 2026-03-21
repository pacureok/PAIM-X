#!/bin/bash
# instalar_dependencias.sh

apt-get update
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev libpostproc-dev

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install av==12.3.0
pip install --no-deps git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
pip install demucs encodec flashy>=0.0.1 hydra-core>=1.1 hydra_colorlog julius num2words omegaconf pesq pystoi torchdiffeq torchmetrics xformers==0.0.28.post3 yt-dlp librosa soundfile
pip install "numpy<2.0.0"
pip install Pillow