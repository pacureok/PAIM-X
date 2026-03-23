import torch
from audiocraft.models import MusicGen
import torchaudio

class PAIM_X:
    def __init__(self, model_size='facebook/musicgen-melody'):
        # Detectar GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"PAIM-X cargando en: {self.device}")
        
        self.model = MusicGen.get_pretrained(model_size, device=self.device)
        
    def generar_track(self, texto_prompt, ruta_sample_juego, duracion_segundos=10):
        self.model.set_generation_params(duration=duracion_segundos)
        
        # Cargar audio base
        sample_audio, sample_rate = torchaudio.load(ruta_sample_juego)
        sample_audio = sample_audio.to(self.device)
        
        # Generar
        wav_generado = self.model.generate_with_chroma(
            descriptions=[texto_prompt],
            melody_wavs=sample_audio[None].expand(1, -1, -1),
            melody_sample_rate=sample_rate,
        )
        return wav_generado.cpu()