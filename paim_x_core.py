from audiocraft.models import musicgen
import torch
import gc

class PAIM_X:
    def __init__(self, model_size='facebook/musicgen-melody'):
        self.limpiar_memoria()
        # Seleccionamos la primera GPU para el modelo de música
        self.device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'
        print(f"🧠 [Pacure Labs] Cargando motor en {self.device}...")
        
        self.model = musicgen.MusicGen.get_pretrained(model_size, device=self.device)
        
    def limpiar_memoria(self):
        """Libera cache en todas las GPUs disponibles"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    def generar_track(self, texto_prompt, ruta_sample_juego, duracion_segundos=30):
        self.limpiar_memoria()
        
        # Configuración de alta fidelidad
        self.model.set_generation_params(
            duration=duracion_segundos,
            use_sampling=True,
            top_k=250,
            cfg_coef=4.0
        )
        
        # Carga del ADN melódico (Audio del juego)
        import torchaudio
        melody_waveform, sr = torchaudio.load(ruta_sample_juego)
        melody_waveform = melody_waveform.to(self.device)
        
        print(f"🎼 Componiendo {duracion_segundos} segundos...")
        
        with torch.no_grad():
            # Generación
            outputs = self.model.generate_with_chroma(
                descriptions=[texto_prompt],
                melody_wavs=melody_waveform[None].expand(1, -1, -1),
                melody_sample_rate=sr,
                progress=True
            )
            
        self.limpiar_memoria()
        return outputs.cpu()