import os
import torch
import soundfile as sf
from extractor_yt import ExtractorPAIM
from paim_x_core import PAIM_X
from mezclador_3d import MezcladorEspacialIA
from marca_agua_plbs import MarcaDeAguaPLBS

def lanzar_paim_x():
    print("\n" + "█"*40)
    print("      PACURE LABS - PAIM-X V1.2")
    print("█"*40 + "\n")

    # Pedir datos al usuario
    prompt = input("🎹 Escribe tu Prompt Musical: ")
    url = input("🔗 Link de YouTube: ")
    duracion_str = input("⏱️ Duración (Ej: 0:30 o 3:00): ")
    
    m, s = map(int, duracion_str.split(':'))
    segundos = (m * 60) + s

    inc_sfx = input("💥 ¿Inyectar SFX del juego? (S/N): ").lower() == 's'
    inc_voz = input("🗣️ ¿Mantener Voces del video? (S/N): ").lower() == 's'
    inc_3d  = input("🎧 ¿Efectos 3D L/R? (S/N): ").lower() == 's'

    # Motor
    extractor = ExtractorPAIM()
    audio_raw = extractor.descargar_audio(url)
    pista_sfx_full = extractor.separar_pistas(audio_raw)
    
    nombre_folder = os.path.basename(audio_raw).split('.')[0]
    ruta_voz = f"./temp_paim/demucs_out/htdemucs/{nombre_folder}/vocals.wav"
    samples_sfx = extractor.aislar_impactos(pista_sfx_full) if inc_sfx else []

    motor = PAIM_X()
    base_tensor = motor.generar_track(prompt, audio_raw, segundos)
    
    os.makedirs("./temp_paim", exist_ok=True)
    ruta_base = "./temp_paim/base.wav"
    sf.write(ruta_base, base_tensor[0].numpy().T, 32000)

    mezclador = MezcladorEspacialIA()
    audio_master = mezclador.procesar_master(ruta_base, samples_sfx, ruta_voz if inc_voz else None, inc_3d)

    firmador = MarcaDeAguaPLBS()
    ruta_p = "./temp_paim/pre.wav"
    sf.write(ruta_p, audio_master.T, 32000)
    final = firmador.inyectar_firma(ruta_p)

    sf.write("PAIM_X_MASTER_FINAL.wav", final.T, 32000)
    print("\n✅ ¡LISTO! Archivo: PAIM_X_MASTER_FINAL.wav")

if __name__ == "__main__":
    lanzar_paim_x()