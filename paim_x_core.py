import torch
# Importamos la librería de XLA para TPUs
import torch_xla.core.xla_model as xm 
from audiocraft.models import MusicGen
import torchaudio

class PAIM_X:
    def __init__(self, model_size='facebook/musicgen-melody'):
        print("Inicializando PAIM-X...")
        
        # 1. Asignar el dispositivo a la TPU
        # xm.xla_device() busca automáticamente el núcleo TPU disponible
        self.device = xm.xla_device() 
        print(f"Dispositivo asignado: {self.device}")

        # 2. Cargar el modelo base en la TPU
        # Usamos la versión 'melody' porque permite inyectar audio + texto
        self.model = MusicGen.get_pretrained(model_size, device=self.device)
        
    def generar_track(self, texto_prompt, ruta_sample_juego, duracion_segundos=10):
        """
        Genera música usando un prompt de texto y un efecto de sonido como base rítmica/melódica.
        """
        self.model.set_generation_params(duration=duracion_segundos)
        
        # Cargar el efecto de sonido (SFX) extraído del juego
        sample_audio, sample_rate = torchaudio.load(ruta_sample_juego)
        
        # Mover el audio a la TPU para que coincida con el modelo
        sample_audio = sample_audio.to(self.device)
        
        print(f"Componiendo pista basada en: '{texto_prompt}'...")
        
        # La magia ocurre aquí: le pasamos el audio y el texto al mismo tiempo
        wav_generado = self.model.generate_with_chroma(
            descriptions=[texto_prompt],
            melody_wavs=sample_audio[None].expand(1, -1, -1),
            melody_sample_rate=sample_rate,
        )
        
        return wav_generado.cpu() # Lo devolvemos a la CPU para poder guardarlo

# Prueba rápida (cuando tengamos un audio)
if __name__ == "__main__":
    paim = PAIM_X()
    print("¡PAIM-X está listo para recibir audio en la TPU!")