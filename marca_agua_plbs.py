import numpy as np
import soundfile as sf
import librosa
from PIL import Image, ImageDraw, ImageFont

class MarcaDeAguaPLBS:
    def __init__(self, sample_rate=32000):
        self.sr = sample_rate
        # Ponemos la marca en frecuencias muy agudas para que no arruine la música
        self.frec_min = 15000 
        self.frec_max = 20000
        
    def _crear_matriz_texto(self, texto="PLBS", ancho=200, alto=50):
        """Genera una imagen en blanco y negro con el texto y la convierte en matriz."""
        img = Image.new('L', (ancho, alto), color=0)
        dibujo = ImageDraw.Draw(img)
        
        # Usamos la fuente por defecto, dibujamos el texto en el centro
        # Ajustamos el tamaño para que ocupe bien el espacio
        dibujo.text((10, 10), texto, fill=255) 
        
        # Convertimos la imagen a un array de numpy (volteado para el espectrograma)
        matriz = np.array(img)[::-1, :] 
        return matriz / 255.0 # Normalizamos entre 0 y 1

    def _sintetizar_espectrograma(self, matriz_imagen, duracion_segundos):
        """Convierte los píxeles de la imagen en ondas senoidales."""
        t = np.linspace(0, duracion_segundos, int(self.sr * duracion_segundos), endpoint=False)
        audio_firma = np.zeros_like(t)
        
        alto, ancho = matriz_imagen.shape
        frecuencias = np.linspace(self.frec_min, self.frec_max, alto)
        
        # Tamaño de cada "píxel" en el tiempo
        muestras_por_columna = len(t) // ancho
        
        print(f"[Pacure Labs] Sintetizando frecuencias para la marca de agua...")
        for col in range(ancho):
            inicio = col * muestras_por_columna
            fin = inicio + muestras_por_columna
            tiempo_columna = t[inicio:fin]
            
            for fila in range(alto):
                intensidad = matriz_imagen[fila, col]
                if intensidad > 0.1: # Si hay un píxel blanco
                    # Generamos una onda a esa frecuencia específica
                    onda = intensidad * np.sin(2 * np.pi * frecuencias[fila] * tiempo_columna)
                    audio_firma[inicio:fin] += onda
                    
        return audio_firma

    def inyectar_firma(self, ruta_pista_final, texto="PLBS", volumen_firma=0.05):
        """Mezcla la pista original con la firma de Pacure Labs."""
        print(f"\n[Pacure Labs] Inyectando firma '{texto}' en el espectrograma...")
        
        # Cargar la música generada por PAIM-X
        y_cancion, _ = librosa.load(ruta_pista_final, sr=self.sr, mono=False)
        duracion = librosa.get_duration(y=y_cancion, sr=self.sr)
        
        # Generar el audio de la marca de agua
        matriz = self._crear_matriz_texto(texto)
        onda_firma = self._sintetizar_espectrograma(matriz, duracion)
        
        # Ajustar volumen para que sea "invisible" al oído pero visible en el espectrograma
        onda_firma = onda_firma * volumen_firma
        
        # Mezclar (Si la canción es estéreo, agregamos la firma a ambos canales)
        if y_cancion.ndim == 2:
            y_cancion[0, :] += onda_firma[:y_cancion.shape[1]]
            y_cancion[1, :] += onda_firma[:y_cancion.shape[1]]
        else:
            y_cancion += onda_firma[:y_cancion.shape[0]]
            
        return y_cancion