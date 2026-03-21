# Contribuir a PAIM-X 🚀

¡Gracias por tu interés en contribuir a PAIM-X, el motor de IA musical de Pacure Labs! 

Nos encanta recibir *Pull Requests* (PRs) de la comunidad. Ya sea que estés mejorando el modelo XLA para TPUs, optimizando la extracción de YouTube, o añadiendo nuevos efectos al mezclador 3D, tu ayuda es bienvenida.

## Cómo empezar
1. Haz un **Fork** de este repositorio.
2. Clona tu fork localmente.
3. Instala las dependencias ejecutando: `bash instalar_dependencias.sh`
4. Crea una rama para tu función: `git checkout -b feature/mi-nueva-funcion`

## Arquitectura del Proyecto
Si vas a modificar el código, ten en cuenta nuestro flujo de trabajo (Pipeline):
1. **`extractor_yt.py`**: Descarga y aísla efectos usando Demucs.
2. **`paim_x_core.py`**: El cerebro del Transformer compilado en TPU (basado en AudioCraft).
3. **`mezclador_3d.py`**: IA de post-procesamiento para paneo L/R y audio espacial 8D.
4. **`marca_agua_plbs.py`**: Esteganografía para inyectar la firma espectral de Pacure Labs.

## Directrices para Pull Requests
* Asegúrate de que el código corre sin errores en un entorno con TPU (Colab/Kaggle).
* No subas archivos de audio generados (`.wav` o `.mp3`) al repositorio.
* Mantén el código limpio y documentado en español.

¡Gracias por ayudar a construir el futuro del audio generativo!