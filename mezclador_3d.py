import numpy as np
import librosa
import soundfile as sf
import random
import os

class MezcladorEspacialIA:
    def __init__(self, sample_rate=32000):
        self.sr = sample_rate

    def normalizar(self, audio):
        """
        Evita que el audio se escuche 'roto' o distorsionado.
        Asegura que los picos no superen el límite digital de 0.95.
        """
        max_peak = np.max(np.abs(audio))
        if max_peak > 0.95:
            # Normalizamos con un pequeño margen de seguridad
            return audio / (max_peak + 1e-6) * 0.95
        return audio

    def aplicar_paneo(self, audio_mono, direccion):
        """
        Convierte un sonido mono en estéreo posicional o efectos 8D.
        """
        longitud = len(audio_mono)
        audio_stereo = np.zeros((2, longitud))

        if direccion == 'L':
            audio_stereo[0, :] = audio_mono
            audio_stereo[1, :] = audio_mono * 0.05 # Casi nada en el otro lado
        elif direccion == 'R':
            audio_stereo[0, :] = audio_mono * 0.05
            audio_stereo[1, :] = audio_mono
        elif direccion == 'L_to_R':
            curva_l = np.linspace(1.0, 0.0, longitud)
            curva_r = np.linspace(0.0, 1.0, longitud)
            audio_stereo[0, :] = audio_mono * curva_l
            audio_stereo[1, :] = audio_mono * curva_r
        elif direccion == 'R_to_L':
            curva_l = np.linspace(0.0, 1.0, longitud)
            curva_r = np.linspace(1.0, 0.0, longitud)
            audio_stereo[0, :] = audio_mono * curva_l
            audio_stereo[1, :] = audio_mono * curva_r

        return audio_stereo

    def mezclar_con_prioridad(self, base, inyectar, inicio_sample, vol=0.3):
        """
        Suma el audio inyectado a la base sin que se salga de los límites.
        """
        fin = inicio_sample + inyectar.shape[1]
        if fin < base.shape[1]:
            # Bajamos el volumen del SFX (0.3) para que no tape la música
            base[:, inicio_sample:fin] += inyectar * vol
        return base

    def procesar_master(self, ruta_musica, ruta_sfx_lista, ruta_voz=None, usar_3d=False):
        """
        La función maestra que une todo: Música + Voces + SFX con paneo.
        """
        print("\n[Mezclador V1.2] Iniciando masterización final...")
        
        # 1. Cargar la música generada (asegurar estéreo)
        y_musica, _ = librosa.load(ruta_musica, sr=self.sr, mono=False)
        if y_musica.ndim == 1: 
            y_musica = np.vstack((y_musica, y_musica))

        # 2. Reintegrar Voces (Si el usuario puso 'S')
        if ruta_voz and os.path.exists(ruta_voz):
            print("🎤 Mezclando voces originales con la música...")
            y_voz, _ = librosa.load(ruta_voz, sr=self.sr, mono=False)
            if y_voz.ndim == 1: 
                y_voz = np.vstack((y_voz, y_voz))
            
            # Ajustar duración para que coincidan
            min_len = min(y_musica.shape[1], y_voz.shape[1])
            y_musica[:, :min_len] += y_voz[:, :min_len] * 0.65 # Voz con buena presencia

        # 3. Inyectar efectos de sonido (SFX) con paneo 3D inteligente
        if usar_3d and ruta_sfx_lista:
            print(f"🎧 Aplicando efectos espaciales a {len(ruta_sfx_lista)} sonidos...")
            y_mono = librosa.to_mono(y_musica)
            # Detectar el ritmo para que los SFX caigan a tiempo
            _, beats = librosa.beat.beat_track(y=y_mono, sr=self.sr)
            tiempos_beats = librosa.frames_to_samples(beats)

            opciones_paneo = ['L', 'R', 'L_to_R', 'R_to_L']

            for ruta in ruta_sfx_lista:
                if os.path.exists(ruta):
                    sfx, _ = librosa.load(ruta, sr=self.sr, mono=True)
                    
                    # Elegir un momento rítmico aleatorio
                    beat = random.choice(tiempos_beats)
                    
                    # Elegir un modo de movimiento 3D
                    modo = random.choice(opciones_paneo)
                    print(f" -> SFX inyectado en modo '{modo}'")
                    
                    sfx_espacial = self.aplicar_paneo(sfx, modo)
                    y_musica = self.mezclar_con_prioridad(y_musica, sfx_espacial, beat, vol=0.35)

        # 4. Normalización Final para evitar que se escuche mal
        print("[Mezclador V1.2] Aplicando limitador y normalización...")
        return self.normalizar(y_musica)