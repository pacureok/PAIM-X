import os
import subprocess
import yt_dlp
import librosa
import soundfile as sf
import numpy as np

class ExtractorPAIM:
    def __init__(self, directorio_trabajo="./temp_paim"):
        """Inicializa los directorios temporales para no ensuciar el proyecto."""
        self.dir_trabajo = directorio_trabajo
        os.makedirs(self.dir_trabajo, exist_ok=True)
        self.audio_bruto = os.path.join(self.dir_trabajo, "raw_video_audio.wav")
        self.dir_demucs = os.path.join(self.dir_trabajo, "demucs_out")

    def descargar_audio(self, url):
        print(f"[1/4] Descargando audio de YouTube: {url}...")
        opciones_ydl = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': self.audio_bruto.replace('.wav', ''),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(opciones_ydl) as ydl:
            ydl.download([url])
        print(" -> Descarga completada.")
        return self.audio_bruto

    def separar_pistas(self, ruta_audio):
        print("[2/4] Ejecutando Demucs para separar efectos de sonido (esto tomará un momento)...")
        # Llamamos a demucs desde la línea de comandos para mayor estabilidad
        comando = [
            "demucs",
            "--out", self.dir_demucs,
            "-n", "htdemucs", # Modelo estándar y rápido
            ruta_audio
        ]
        
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Demucs separa en: vocals, drums, bass, other. 
        # Los efectos de los juegos suelen quedar en la pista "other" o "drums"
        nombre_archivo = os.path.basename(ruta_audio).split('.')[0]
        ruta_sfx = os.path.join(self.dir_demucs, "htdemucs", nombre_archivo, "other.wav")
        
        print(" -> Separación completada.")
        return ruta_sfx

    def aislar_impactos(self, ruta_sfx, num_samples=3):
        print("[3/4] Analizando picos de sonido para extraer samples limpios...")
        y, sr = librosa.load(ruta_sfx, sr=32000) # 32kHz es ideal para AudioCraft
        
        # Detectar los inicios de sonidos fuertes (onsets)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
        
        archivos_guardados = []
        # Tomar los primeros N impactos fuertes
        for i, inicio in enumerate(onsets[:num_samples]):
            # Cortar 1 segundo de audio a partir del impacto
            fin = inicio + sr  
            sample_cortado = y[inicio:fin]
            
            ruta_salida = os.path.join(self.dir_trabajo, f"sfx_limpio_{i}.wav")
            sf.write(ruta_salida, sample_cortado, sr)
            archivos_guardados.append(ruta_salida)
            
        print(f" -> Se extrajeron {len(archivos_guardados)} samples listos para usar.")
        return archivos_guardados

# ==========================================
# INTERFAZ PÚBLICA PARA EL USUARIO FINAL
# ==========================================

def procesar_prompt(texto_prompt, url_youtube):
    """
    Función principal que el público ejecutará.
    Recibe la variable de texto y la URL, y devuelve todo listo para la IA musical.
    """
    print(f"\n--- Iniciando Pipeline PAIM-X ---")
    print(f"Prompt recibido: '{texto_prompt}'")
    
    extractor = ExtractorPAIM()
    
    # Flujo de trabajo
    try:
        audio_original = extractor.descargar_audio(url_youtube)
        pista_sfx = extractor.separar_pistas(audio_original)
        samples_finales = extractor.aislar_impactos(pista_sfx)
        
        print("\n[4/4] ¡Éxito! Datos preparados para el modelo generativo.")
        
        # Aquí retornarías los datos para conectarlos a paim_x_core.py
        return {
            "texto": texto_prompt,
            "samples_audio": samples_finales
        }
        
    except Exception as e:
        print(f"\n[Error] Algo falló en la extracción: {e}")
        return None