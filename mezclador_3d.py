import numpy as np
import librosa
import soundfile as sf
import random
import os

class MezcladorEspacialIA:
    def __init__(self, sample_rate=32000):
        self.sr = sample_rate

    def normalizar(self, audio):
        """Evita la distorsión limitando los picos."""
        max_peak = np.max(np.abs(audio))
        if max_peak > 0.95:
            return audio / (max_peak + 1e-6) * 0.95
        return audio

    def aplicar_paneo(self, audio_mono, direccion):
        longitud = len(audio_mono)
        audio_stereo = np.zeros((2, longitud))
        if direccion == 'L':
            audio_stereo[0, :] = audio_mono
            audio_stereo[1, :] = audio_mono * 0.05
        elif direccion == 'R':
            audio_stereo[0, :] = audio_mono * 0.05
            audio_stereo[1, :] = audio_mono
        elif direccion == 'L_to_R':
            audio_stereo[0, :] = audio_mono * np.linspace(1.0, 0.0, longitud)
            audio_stereo[1, :] = audio_mono * np.linspace(0.0, 1.0, longitud)
        elif direccion == 'R_to_L':
            audio_stereo[0, :] = audio_mono * np.linspace(0.0, 1.0, longitud)
            audio_stereo[1, :] = audio_mono * np.linspace(1.0, 0.0, longitud)
        return audio_stereo

    def procesar_master(self, ruta_musica, ruta_sfx_lista, ruta_voz=None, usar_3d=False):
        print("\n[Mezclador V1.2] Masterizando...")
        y_musica, _ = librosa.load(ruta_musica, sr=self.sr, mono=False)
        if y_musica.ndim == 1: y_musica = np.vstack((y_musica, y_musica))

        if ruta_voz and os.path.exists(ruta_voz):
            y_voz, _ = librosa.load(ruta_voz, sr=self.sr, mono=False)
            if y_voz.ndim == 1: y_voz = np.vstack((y_voz, y_voz))
            min_len = min(y_musica.shape[1], y_voz.shape[1])
            y_musica[:, :min_len] += y_voz[:, :min_len] * 0.6

        if usar_3d and ruta_sfx_lista:
            y_mono = librosa.to_mono(y_musica)
            _, beats = librosa.beat.beat_track(y=y_mono, sr=self.sr)
            tiempos_beats = librosa.frames_to_samples(beats)
            opciones = ['L', 'R', 'L_to_R', 'R_to_L']
            for ruta in ruta_sfx_lista:
                sfx, _ = librosa.load(ruta, sr=self.sr, mono=True)
                beat = random.choice(tiempos_beats)
                sfx_p = self.aplicar_paneo(sfx, random.choice(opciones))
                fin = beat + sfx_p.shape[1]
                if fin < y_musica.shape[1]:
                    y_musica[:, beat:fin] += sfx_p * 0.3
        
        return self.normalizar(y_musica)