import os
import sys
import torch
import soundfile as sf

# Importación de módulos de Pacure Labs
try:
    from extractor_yt import ExtractorPAIM
    from paim_x_core import PAIM_X
    from mezclador_3d import MezcladorEspacialIA
    from marca_agua_plbs import MarcaDeAguaPLBS
except ImportError as e:
    print(f"❌ Error de dependencias: {e}")
    print("Asegúrate de haber ejecutado 'bash instalar_dependencias.sh' primero.")
    sys.exit(1)

def paim_x_engine():
    print("\n" + "="*60)
    print("🚀 PAIM-X V1.0 - BY PACURE LABS 🚀")
    print("La próxima generación de audio generativo multimodal")
    print("="*60 + "\n")

    # --- CONFIGURACIÓN DE USUARIO ---
    # En entornos públicos, estos valores se pueden cambiar aquí:
    PROMPT = "Música épica de batalla, orquestal y frenética, estilo Dark Souls"
    URL_YT = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    MODO_3D = "SI"  # Opciones: "SI" / "NO"
    DURACION = 15    # Segundos
    # --------------------------------

    try:
        # 1. Extracción de samples del juego
        print("[PROCESO 1/4] Extrayendo ADN sonoro del video...")
        extractor = ExtractorPAIM()
        ruta_original = extractor.descargar_audio(URL_YT)
        pista_sfx = extractor.separar_pistas(ruta_original)
        samples = extractor.aislar_impactos(pista_sfx, num_samples=3)

        # 2. Composición con IA (GPU)
        print("\n[PROCESO 2/4] IA componiendo melodía original...")
        motor = PAIM_X()
        pista_base_tensor = motor.generar_track(PROMPT, samples[0], DURACION)
        
        ruta_temp = "./temp_paim/base_raw.wav"
        os.makedirs("./temp_paim", exist_ok=True)
        sf.write(ruta_temp, pista_base_tensor[0].numpy().T, 32000)

        # 3. Mezcla Espacial 3D/8D
        print("\n[PROCESO 3/4] Aplicando ingeniería de sonido espacial...")
        activar_3d = True if MODO_3D.upper() == "SI" else False
        mezclador = MezcladorEspacialIA(sample_rate=32000)
        audio_3d = mezclador.inyectar_sfx_inteligente(ruta_temp, samples[1:], activar_3d)
        
        ruta_3d = "./temp_paim/base_3d.wav"
        sf.write(ruta_3d, audio_3d.T, 32000)

        # 4. Firma de Pacure Labs
        print("\n[PROCESO 4/4] Sellando master con marca de agua PLBS...")
        firmador = MarcaDeAguaPLBS()
        master_final = firmador.inyectar_firma(ruta_3d, texto="PLBS")

        # Resultado final
        nombre_salida = "PAIM_X_MASTER_FINAL.wav"
        sf.write(nombre_salida, master_final.T, 32000)
        
        print("\n" + "="*60)
        print(f"✅ ¡ÉXITO! Tu pista está lista: {nombre_salida}")
        print("Gracias por usar tecnología de Pacure Labs.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error crítico en el motor: {e}")

if __name__ == "__main__":
    paim_x_engine()