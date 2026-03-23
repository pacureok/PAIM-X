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
    print("    SOPORTE MULTI-GPU ACTIVADO")
    print("█"*40 + "\n")

    # ENTRADAS DE USUARIO
    PROMPT = input("🎹 Prompt Musical: ")
    URL_YT = input("🔗 Link de YouTube: ")
    DURACION = input("⏱️ Duración (Ej: 3:00): ")
    m, s = map(int, DURACION.split(':'))
    SEGUNDOS = (m * 60) + s

    INC_SFX = input("💥 ¿Inyectar SFX del juego? (S/N): ").upper() == 'S'
    INC_VOZ = input("🗣️ ¿Mantener Voces del video? (S/N): ").upper() == 'S'
    INC_3D  = input("🎧 ¿Efectos Espaciales 3D? (S/N): ").upper() == 'S'

    # PROCESAMIENTO
    extractor = ExtractorPAIM()
    print("\n[1/4] Descargando y Separando audio (Usando GPU 1 para Demucs)...")
    # Forzamos a Demucs a usar la segunda GPU si existe
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if torch.cuda.device_count() > 1 else "0"
    
    audio_raw = extractor.descargar_audio(URL_YT)
    pista_sfx_completa = extractor.separar_pistas(audio_raw)
    
    # Rutas de pistas separadas (Demucs)
    nombre_folder = os.path.basename(audio_raw).split('.')[0]
    ruta_voz = f"./temp_paim/demucs_out/htdemucs/{nombre_folder}/vocals.wav"
    samples_sfx = extractor.aislar_impactos(pista_sfx_completa) if INC_SFX else []

    # GENERACIÓN (GPU 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("\n[2/4] Componiendo melodía (Usando GPU 0)...")
    motor = PAIM_X()
    base_tensor = motor.generar_track(PROMPT, audio_raw, SEGUNDOS)
    
    os.makedirs("./temp_paim", exist_ok=True)
    ruta_base = "./temp_paim/base.wav"
    sf.write(ruta_base, base_tensor[0].numpy().T, 32000)

    # MASTERIZACIÓN
    print("\n[3/4] Mezclando y aplicando efectos...")
    mezclador = MezcladorEspacialIA()
    voz_final = ruta_voz if INC_VOZ else None
    audio_master = mezclador.procesar_master(ruta_base, samples_sfx, voz_final, INC_3D)

    # FIRMA
    print("\n[4/4] Sellando Espectrograma...")
    firmador = MarcaDeAguaPLBS()
    ruta_pre_master = "./temp_paim/pre_master.wav"
    sf.write(ruta_pre_master, audio_master.T, 32000)
    master_final = firmador.inyectar_firma(ruta_pre_master)

    sf.write("PAIM_X_FINAL.wav", master_final.T, 32000)
    print("\n✅ PROCESO COMPLETADO. Master guardado como: PAIM_X_FINAL.wav")

if __name__ == "__main__":
    lanzar_paim_x()