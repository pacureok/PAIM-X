import os
import subprocess
import yt_dlp
import librosa
import soundfile as sf
import torch
import time
import requests

class ExtractorPAIM:
    def __init__(self, directorio_trabajo="./temp_paim"):
        self.dir_trabajo = directorio_trabajo
        os.makedirs(self.dir_trabajo, exist_ok=True)
        self.audio_bruto = os.path.join(self.dir_trabajo, "raw_video_audio.wav")
        self.dir_demucs = os.path.join(self.dir_trabajo, "demucs_out")

    def descargar_audio(self, url):
        print(f"📥 Descargando audio de YouTube...")
        
        if os.path.exists(self.audio_bruto):
            os.remove(self.audio_bruto)
            
        opciones_ydl = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': self.audio_bruto.replace('.wav', ''),
            'quiet': True, 
            'no_warnings': True,
            'extractor_args': {'youtube': {'client': ['android', 'ios', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
            }
        }

        # Intento 1: Descarga directa con yt-dlp
        try:
            print(" -> Método 1: Extracción directa...")
            with yt_dlp.YoutubeDL(opciones_ydl) as ydl:
                ydl.download([url])
            return self.audio_bruto
        except Exception as e:
            print(f" ⚠️ Método 1 falló (Posible bloqueo 403): {e}")
            print(" -> Iniciando Método 2: Extracción a través de API alternativa...")
            return self._descarga_alternativa(url)

    def _descarga_alternativa(self, url):
        """Plan B: Usa una API pública si YouTube bloquea la IP de Kaggle."""
        try:
            # Usamos la API pública de Cobalt para procesar el link
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            }
            data = {
                "url": url,
                "vCodec": "h264",
                "vQuality": "720",
                "aFormat": "wav",
                "filenamePattern": "basic",
                "isAudioOnly": True,
                "isAudioMuted": False,
                "dubLang": False,
                "disableMetadata": True,
            }
            
            # Petición a la API (usando uno de los nodos públicos de Cobalt)
            response = requests.post('https://api.cobalt.tools/api/json', headers=headers, json=data)
            
            if response.status_code == 200:
                resultado = response.json()
                if resultado.get("status") == "redirect" or resultado.get("status") == "stream":
                    link_descarga = resultado.get("url")
                    
                    # Descargar el archivo resultante
                    print(" -> Descargando archivo procesado...")
                    r_audio = requests.get(link_descarga, stream=True)
                    if r_audio.status_code == 200:
                        with open(self.audio_bruto, 'wb') as f:
                            for chunk in r_audio.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return self.audio_bruto
            
            # Si Cobalt falla, lanzamos un error que podamos entender
            raise Exception("La API alternativa no devolvió un link válido.")

        except Exception as api_error:
             print(f" ❌ Fallo total en la descarga: {api_error}")
             print(" 💡 SUGERENCIA: Sube el archivo .wav manualmente a la carpeta /kaggle/working/PAIM-X/temp_paim/ y ponle el nombre 'raw_video_audio.wav'")
             raise api_error


    def separar_pistas(self, ruta_audio):
        print("🎛️ Separando pistas...")
        dispositivo = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        print(f" -> Asignando Demucs a la tarjeta: {dispositivo}")
        
        comando = [
            "demucs", 
            "-d", dispositivo, 
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