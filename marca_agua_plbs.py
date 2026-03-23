import numpy as np
import librosa
import soundfile as sf

class MarcaDeAguaPLBS:
    def __init__(self, sample_rate=32000):
        self.sr = sample_rate
        # Frecuencia inaudible / ultrasónica (justo por debajo del límite de 16kHz)
        self.frecuencia_oculta = 15500 

    def _generar_morse(self, mensaje_morse):
        """Genera las ondas de audio senoidales para el código Morse."""
        print(f"[Pacure Labs] Sintetizando código Morse espectral: {mensaje_morse}")
        
        # Tiempos del Morse (muy rápidos para no alargar la pista)
        duracion_punto = 0.08  # segundos
        duracion_raya = duracion_punto * 3
        pausa_intra = duracion_punto     # Entre símbolos de la misma letra
        pausa_inter = duracion_punto * 3 # Entre letras distintas
        
        audio_parts = []
        t_punto = np.linspace(0, duracion_punto, int(self.sr * duracion_punto), endpoint=False)
        t_raya = np.linspace(0, duracion_raya, int(self.sr * duracion_raya), endpoint=False)
        
        # Ondas de sonido a 15,500 Hz
        onda_punto = np.sin(2 * np.pi * self.frecuencia_oculta * t_punto)
        onda_raya = np.sin(2 * np.pi * self.frecuencia_oculta * t_raya)
        silencio_intra = np.zeros(int(self.sr * pausa_intra))
        silencio_inter = np.zeros(int(self.sr * pausa_inter))
        
        for char in mensaje_morse:
            if char == '.':
                audio_parts.extend([onda_punto, silencio_intra])
            elif char == '-':
                audio_parts.extend([onda_raya, silencio_intra])
            elif char == ' ':
                # Reemplazamos el último espacio corto por un espacio largo entre letras
                if audio_parts and np.array_equal(audio_parts[-1], silencio_intra):
                    audio_parts[-1] = silencio_inter
                    
        # Unimos todos los puntos, rayas y silencios en una sola pista
        return np.concatenate(audio_parts)

    def inyectar_firma(self, ruta_pista_final, texto="PLBS", volumen_firma=0.005):
        """Inyecta el Morse oculto SOLO al final de la canción."""
        
        # PLBS en código Morse
        morse_plbs = ".--. .-.. -... ..."
        
        # 1. Cargar la música generada
        y_cancion, _ = librosa.load(ruta_pista_final, sr=self.sr, mono=False)
        if y_cancion.ndim == 1:
            y_cancion = np.vstack((y_cancion, y_cancion))
            
        # 2. Generar el audio de la marca de agua (muy bajo volumen)
        onda_firma = self._generar_morse(morse_plbs) * volumen_firma
        
        longitud_cancion = y_cancion.shape[1]
        longitud_firma = len(onda_firma)
        
        print("[Pacure Labs] Inyectando firma esteganográfica al final de la pista...")
        
        # 3. Calcular la posición para que termine exactamente con la canción
        if longitud_cancion > longitud_firma:
            inicio_firma = longitud_cancion - longitud_firma
            # Sumamos la onda inaudible a ambos canales (Left y Right)
            y_cancion[0, inicio_firma:] += onda_firma
            y_cancion[1, inicio_firma:] += onda_firma
        else:
            # Si por error la canción es más corta que la firma, la ponemos desde el inicio
            y_cancion[0, :longitud_cancion] += onda_firma[:longitud_cancion]
            y_cancion[1, :longitud_cancion] += onda_firma[:longitud_cancion]
            
        return y_cancion