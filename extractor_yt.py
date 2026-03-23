import os
import subprocess
import yt_dlp
import librosa
import soundfile as sf
import torch

class ExtractorPAIM:
    def __init__(self, directorio_trabajo="./temp_paim"):
        self.dir_trabajo = directorio_trabajo
        os.makedirs(self.dir_trabajo, exist_ok=True)
        self.audio_bruto = os.path.join(self.dir_trabajo, "raw_video_audio.wav")
        self.dir_demucs = os.path.join(self.dir_trabajo, "demucs_out")

    def descargar_audio(self, url):
        print(f"📥 Descargando audio de YouTube...")
        opciones_ydl = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': self.audio_bruto.replace('.wav', ''),
            'quiet': True, 'no_warnings': True
        }
        with yt_dlp.YoutubeDL(opciones_ydl) as ydl:
            ydl.download([url])
        return self.audio_bruto

    def separar_pistas(self, ruta_audio):
        print("🎛️ Separando pistas...")
        # ¡Magia Multi-GPU! Si hay más de 1 GPU, mandamos Demucs a la GPU 1.
        dispositivo = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        print(f" -> Asignando Demucs a la tarjeta: {dispositivo}")
        
        comando = [
            "demucs", 
            "-d", dispositivo, # Forzamos el uso del acelerador específico
            "--out", self.dir_demucs, 
            "-n", "htdemucs", 
            ruta_audio
        ]
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        nombre = os.path.basename(ruta_audio).split('.')[0]
        return os.path.join(self.dir_demucs, "htdemucs", nombre, "other.wav")

    def aislar_impactos(self, ruta_sfx, num_samples=3):
        y, sr = librosa.load(ruta_sfx, sr=32000)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
        archivos = []
        for i, inicio in enumerate(onsets[:num_samples]):
            fin = inicio + sr  
            sample = y[inicio:fin]
            ruta = os.path.join(self.dir_trabajo, f"sfx_{i}.wav")
            sf.write(ruta, sample, sr)
            archivos.append(ruta)
        return archivos