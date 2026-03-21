import os
import soundfile as sf
import torch

# Importamos todos los módulos de Pacure Labs
from extractor_yt import ExtractorPAIM
from paim_x_core import PAIM_X
from mezclador_3d import MezcladorEspacialIA
from marca_agua_plbs import MarcaDeAguaPLBS

def ejecutar_pipeline_paim(prompt, url_yt, usar_3d, duracion):
    print("\n" + "="*50)
    print("🚀 INICIANDO PAIM-X V1 (POWERED BY PACURE LABS) 🚀")
    print("="*50)
    
    # --- PASO 1: Extracción ---
    extractor = ExtractorPAIM()
    audio_original = extractor.descargar_audio(url_yt)
    pista_sfx = extractor.separar_pistas(audio_original)
    samples_extraidos = extractor.aislar_impactos(pista_sfx, num_samples=3)
    
    if not samples_extraidos:
        print("❌ Error: No se pudieron extraer samples del video.")
        return

    # Usaremos el primer sample como base para la IA, y los demás para los efectos 3D
    sample_base = samples_extraidos[0]
    samples_para_efectos = samples_extraidos[1:]

    # --- PASO 2: Generación de Música en TPU ---
    print("\n[Motor de IA] Cargando modelo en TPU y componiendo...")
    motor_ia = PAIM_X()
    
    # Generamos la pista base usando el prompt y el primer sonido del juego
    pista_generada_tensor = motor_ia.generar_track(
        texto_prompt=prompt, 
        ruta_sample_juego=sample_base, 
        duracion_segundos=duracion
    )
    
    # Guardamos la pista cruda temporalmente
    ruta_cruda = "./temp_paim/pista_cruda.wav"
    sf.write(ruta_cruda, pista_generada_tensor[0].numpy().T, 32000)

    # --- PASO 3: Post-Procesamiento Espacial (3D/8D) ---
    mezclador = MezcladorEspacialIA(sample_rate=32000)
    activar_3d = True if usar_3d.upper() == "SI" else False
    
    pista_mezclada = mezclador.inyectar_sfx_inteligente(
        pista_principal=ruta_cruda,
        rutas_sfx=samples_para_efectos,
        usar_3d=activar_3d
    )
    
    ruta_mezclada = "./temp_paim/pista_mezclada.wav"
    sf.write(ruta_mezclada, pista_mezclada.T, 32000)

    # --- PASO 4: Sello de Pacure Labs (Marca de Agua) ---
    firmador = MarcaDeAguaPLBS(sample_rate=32000)
    audio_final_sellado = firmador.inyectar_firma(
        ruta_pista_final=ruta_mezclada,
        texto="PLBS",
        volumen_firma=0.04
    )
    
    # Guardar el archivo definitivo
    ruta_master = "PAIM_X_MASTER.wav"
    sf.write(ruta_master, audio_final_sellado.T, 32000)
    
    print("\n" + "="*50)
    print(f"✅ ¡PROCESO COMPLETADO! Pista guardada como: {ruta_master}")
    print("="*50)


# =====================================================================
# ÁREA DE USUARIO (COLAB / KAGGLE / CLOUD)
# Solo modifica estas variables y ejecuta la celda
# =====================================================================
if __name__ == "__main__":
    
    # 1. ¿Qué tipo de música quieres?
    PROMPT_USUARIO = "Música de jefe final, estilo orquestal oscuro, coros épicos, ritmo rápido"
    
    # 2. ¿De qué video de YouTube sacamos los efectos de sonido?
    URL_YOUTUBE = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Cambia por un gameplay
    
    # 3. ¿Quieres que la IA meta efectos 3D moviéndose de Izquierda a Derecha? ("SI" o "NO")
    APLICAR_EFECTOS_L_R = "SI"
    
    # 4. ¿Cuántos segundos quieres que dure la canción?
    DURACION_SEGS = 15

    # Ejecutar el motor
    ejecutar_pipeline_paim(
        prompt=PROMPT_USUARIO, 
        url_yt=URL_YOUTUBE, 
        usar_3d=APLICAR_EFECTOS_L_R, 
        duracion=DURACION_SEGS
    )