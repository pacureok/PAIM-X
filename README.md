# 🎧 PAIM-X: Generador de Audio Multimodal y Espacial
> Desarrollado por **[Pacure Labs]**

**PAIM-X** es un motor de inteligencia artificial generativa de audio de alto rendimiento, diseñado para componer música a partir de texto e integrar muestras reales de audio extraídas directamente de internet (como *gameplays* o videos). 

Está optimizado para ejecutarse en **TPUs (Tensor Processing Units)** mediante compilación XLA, lo que permite un procesamiento masivo y paralelo de los tensores de audio.

---

## 🧬 Sobre Pacure Labs
Pacure Labs es la organización base de Inteligencia Artificial, tecnología de vanguardia y herramientas de próxima generación. Nuestro objetivo es llevar los límites de la IA multimodal a las manos de creadores, desarrolladores y artistas.

## ✨ Características Principales

PAIM-X no es solo un modelo de lenguaje, es un *pipeline* completo de producción de audio compuesto por 4 módulos clave:

1. **Ingesta Dinámica (YouTube & Web):** Extrae audio directamente de URLs, utilizando inteligencia artificial (`Demucs`) para aislar los efectos de sonido (SFX) del ruido de fondo o las voces.
2. **Motor Generativo Autorregresivo:** Basado en arquitecturas *Transformer* (adaptado de AudioCraft), es capaz de fusionar *prompts* de texto ("Música épica de jefe final") con los *samples* extraídos para dictar la base rítmica o melódica.
3. **Mezclador de Inteligencia Espacial (Audio 3D/8D):** Un algoritmo de post-procesamiento que detecta los *beats* de la canción generada y posiciona dinámicamente los efectos de sonido en el espacio estéreo (paneo inteligente y movimiento de izquierda a derecha).
4. **Firma Espectral Esteganográfica:** Inyección automática de la marca de agua corporativa "**PLBS**" en las frecuencias ultra-altas (15kHz - 20kHz) del espectrograma, invisible al oído pero detectable bajo análisis forense de audio.

## 🚀 Arquitectura TPU y `torch_xla`

A diferencia de los modelos tradicionales que corren sobre la arquitectura CUDA de las GPUs, PAIM-X está escrito para compilarse usando **Accelerated Linear Algebra (XLA)**. 

Las TPUs de Google están diseñadas con arreglos sistólicos (*Systolic Arrays*) que procesan multiplicaciones de matrices de forma abrumadoramente rápida. PAIM-X utiliza `torch_xla` para enviar los grafos estáticos del modelo directamente a los núcleos de la TPU, reduciendo drásticamente los cuellos de botella durante la inferencia y el entrenamiento de audio de alta fidelidad.

## 🛠️ Instalación y Uso Rápido

Para probar PAIM-X en entornos de la nube con TPUs disponibles (como Google Colab o Kaggle):

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/pacureok/PAIM-X.git](https://github.com/pacureok/PAIM-X.git)
   cd PAIM-X