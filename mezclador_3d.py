import numpy as np
import librosa
import soundfile as sf
import random

class MezcladorEspacialIA:
    def __init__(self, sample_rate=32000):
        self.sr = sample_rate

    def aplicar_paneo(self, audio_mono, direccion):
        """
        Convierte un sonido mono en estéreo posicional.
        direccion: 'L', 'R', 'L_to_R' (Efecto 3D/8D), 'R_to_L'
        """
        longitud = len(audio_mono)
        audio_stereo = np.zeros((2, longitud)) # [0] es Izquierda, [1] es Derecha

        if direccion == 'L':
            audio_stereo[0, :] = audio_mono
            audio_stereo[1, :] = audio_mono * 0.1 # Un poco de sangrado al otro oído
        elif direccion == 'R':
            audio_stereo[0, :] = audio_mono * 0.1
            audio_stereo[1, :] = audio_mono
        elif direccion == 'L_to_R':
            # Efecto 8D: El sonido viaja de Izquierda a Derecha
            curva_l = np.linspace(1.0, 0.0, longitud)
            curva_r = np.linspace(0.0, 1.0, longitud)
            audio_stereo[0, :] = audio_mono * curva_l
            audio_stereo[1, :] = audio_mono * curva_r
        elif direccion == 'R_to_L':
            # Efecto 8D: El sonido viaja de Derecha a Izquierda
            curva_l = np.linspace(0.0, 1.0, longitud)
            curva_r = np.linspace(1.0, 0.0, longitud)
            audio_stereo[0, :] = audio_mono * curva_l
            audio_stereo[1, :] = audio_mono * curva_r

        return audio_stereo

    def inyectar_sfx_inteligente(self, pista_principal, rutas_sfx, usar_3d=False):
        """
        La IA decide dónde colocar los efectos basándose en el ritmo de la canción.
        """
        print("\n[IA Espacial] Analizando la pista para decidir dónde colocar los SFX...")
        
        # 1. Cargar la canción generada
        y_cancion, _ = librosa.load(pista_principal, sr=self.sr, mono=False)
        
        # Si la canción es mono, la hacemos estéreo
        if y_cancion.ndim == 1:
            y_cancion = np.vstack((y_cancion, y_cancion))
            
        # 2. La IA detecta los "golpes" (beats) para que el SFX no suene fuera de tiempo
        y_mono = librosa.to_mono(y_cancion)
        tempo, beats = librosa.beat.beat_track(y=y_mono, sr=self.sr)
        tiempos_beats = librosa.frames_to_samples(beats)

        if not usar_3d or len(rutas_sfx) == 0:
            print("[IA Espacial] Efectos L/R desactivados. Dejando pista original.")
            return y_cancion

        # 3. La IA toma decisiones y mezcla
        opciones_paneo = ['L', 'R', 'L_to_R', 'R_to_L']
        
        for ruta_sfx in rutas_sfx:
            # Cargar el efecto ("bam", disparo, etc.)
            y_sfx, _ = librosa.load(ruta_sfx, sr=self.sr, mono=True)
            
            # La IA elige un beat aleatorio para poner el sonido
            beat_elegido = random.choice(tiempos_beats)
            
            # La IA decide cómo se moverá el sonido
            movimiento_elegido = random.choice(opciones_paneo)
            print(f"[IA Espacial] Decisión: Inyectando SFX en el sample {beat_elegido} moviéndose en modo '{movimiento_elegido}'")
            
            # Aplicar el efecto 3D/Espacial al SFX
            sfx_espacial = self.aplicar_paneo(y_sfx, movimiento_elegido)
            
            # Mezclar el SFX en la pista principal
            fin_sfx = beat_elegido + sfx_espacial.shape[1]
            
            # Evitar salirnos de la duración de la canción
            if fin_sfx < y_cancion.shape[1]:
                # Sumamos los audios (mezcla)
                y_cancion[:, beat_elegido:fin_sfx] += sfx_espacial * 0.8 # 0.8 controla el volumen del SFX

        # Normalizar para que no sature
        max_val = np.max(np.abs(y_cancion))
        if max_val > 1.0:
            y_cancion = y_cancion / max_val

        return y_cancion