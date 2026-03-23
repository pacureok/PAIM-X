from audiocraft.models import musicgen
import torch
import gc
import torchaudio

class PAIM_X:
    def __init__(self, model_size='facebook/musicgen-melody'):
        self.limpiar_memoria()
        # El cerebro principal siempre vivirá en la GPU 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 [Pacure Labs] Motor Musical anclado en: {self.device}")
        self.model = musicgen.MusicGen.get_pretrained(model_size, device=self.device)
        
    def limpiar_memoria(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generar_track(self, texto_prompt, ruta_sample_juego, duracion_segundos=30):
        self.limpiar_memoria()
        
        # Ajustes de Alta Fidelidad para pistas largas
        self.model.set_generation_params(
            duration=duracion_segundos,
            use_sampling=True,
            top_k=250,
            cfg_coef=3.0 # Reducido para evitar ruido y notas superpuestas
        )
        
        melody_waveform, sr = torchaudio.load(ruta_sample_juego)
        melody_waveform = melody_waveform.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate_with_chroma(
                descriptions=[texto_prompt],
                melody_wavs=melody_waveform[None].expand(1, -1, -1),
                melody_sample_rate=sr,
                progress=False # Apagamos el log por defecto
            )
        self.limpiar_memoria()
        return outputs.cpu()