import os
import sys
import time
import threading
import torch
import soundfile as sf
from extractor_yt import ExtractorPAIM
from paim_x_core import PAIM_X
from mezclador_3d import MezcladorEspacialIA
from marca_agua_plbs import MarcaDeAguaPLBS

class AnimacionCarga:
    def __init__(self, mensaje="Componiendo"):
        self.mensaje = mensaje
        self.animacion = ['|', '/', '-', '\\']
        self.corriendo = False
        self.hilo = None

    def girar(self):
        i = 0
        while self.corriendo:
            sys.stdout.write(f"\r{self.mensaje} {self.animacion[i % 4]} ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def iniciar(self):
        self.corriendo = True
        self.hilo = threading.Thread(target=self.girar)
        self.hilo.start()

    def detener(self):
        self.corriendo = False
        if self.hilo:
            self.hilo.join()
        sys.stdout.write(f"\r{self.mensaje} ✅ Hecho!        \n")
        sys.stdout.flush()

def lanzar_paim_x():
    print("\n" + "█"*50)
    print("      PACURE LABS - PAIM-X V1.3 MULTI-GPU")
    print("█"*50 + "\n")

    prompt = input("🎹 Prompt Musical: ")
    url = input("🔗 Link de YouTube: ")
    duracion_str = input("⏱️ Duración (Ej: 0:30 o 3:00): ")
    
    m, s = map(int, duracion_str.split(':'))
    segundos = (m * 60) + s

    inc_sfx = input("💥 ¿Inyectar SFX del juego? (S/N): ").lower() == 's'
    inc_voz = input("🗣️ ¿Mantener Voces del video? (S/N): ").lower() == 's'
    inc_3d  = input("🎧 ¿Efectos 3D L/R? (S/N): ").lower() == 's'

    print("\n" + "-"*50)
    
    extractor = ExtractorPAIM()
    audio_raw = extractor.descargar_audio(url)
    pista_sfx_full = extractor.separar_pistas(audio_raw)
    
    nombre_folder = os.path.basename(audio_raw).split('.')[0]
    ruta_voz = f"./temp_paim/demucs_out/htdemucs/{nombre_folder}/vocals.wav"
    samples_sfx = extractor.aislar_impactos(pista_sfx_full) if inc_sfx else []

    motor = PAIM_X()
    
    # --- ANIMACIÓN DE CARGA ---
    spinner = AnimacionCarga(f"🎼 Generando {duracion_str} min de audio en GPU 0")
    spinner.iniciar()
    base_tensor = motor.generar_track(prompt, audio_raw, segundos)
    spinner.detener()
    # --------------------------
    
    os.makedirs("./temp_paim", exist_ok=True)
    ruta_base = "./temp_paim/base.wav"
    sf.write(ruta_base, base_tensor[0].numpy().T, 32000)

    mezclador = MezcladorEspacialIA()
    audio_master = mezclador.procesar_master(ruta_base, samples_sfx, ruta_voz if inc_voz else None, inc_3d)

    firmador = MarcaDeAguaPLBS()
    ruta_p = "./temp_paim/pre.wav"
    sf.write(ruta_p, audio_master.T, 32000)
    
    # EL SECRETO: Volumen al 0.001 para que no haya zumbido "lin"
    final = firmador.inyectar_firma(ruta_p, texto="PLBS", volumen_firma=0.001)

    sf.write("PAIM_X_MASTER_FINAL.wav", final.T, 32000)
    print("\n✅ ¡LISTO! Archivo guardado con máxima fidelidad.")

if __name__ == "__main__":
    lanzar_paim_x()