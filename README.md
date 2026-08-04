# Segmentación por Clustering — Práctica de Ingeniería Mecatrónica

Implementación en Python del procedimiento completo: preparación de datos,
clustering (K-Means / Fuzzy C-Means / GMM), posprocesamiento y evaluación
con métricas internas (Silhouette, Davies-Bouldin) y externas (IoU, Dice).

## 1. Estructura del proyecto

```
segmentacion_clustering/
├── segmentacion.py          # funciones núcleo (todo el pipeline)
├── main.py                  # corre el experimento completo (k, espacios de color)
├── comparar_algoritmos.py   # compara K-Means vs Fuzzy C-Means vs GMM
├── imagenes/                # <-- coloca aquí tus 3-5 fotos (contexto mecatrónico/industrial)
├── mascaras_referencia/     # opcional: máscaras binarias para calcular IoU/Dice
└── resultados/              # figuras .png y metricas.csv generados
```

## 2. Instalación

```bash
pip install scikit-image scikit-learn scikit-fuzzy scipy matplotlib
```

## 3. Cómo mapea el código con el PROCEDIMIENTO de la guía

### Paso 1 — Revisión del estado del arte
No es código, pero aquí tienes 3 referencias recientes ya resumidas para tu informe
(sección 4 de este README).

### Paso 2 — Preparación de los datos
- `cargar_imagen()`: lee la imagen, la fuerza a 3 canales RGB y la redimensiona
  (para acotar el costo de clustering pixel a pixel).
- `convertir_espacio_color()`: convierte a RGB normalizado o a CIELab
  (`skimage.color.rgb2lab`), que separa mejor luminosidad (L) de cromaticidad (a,b)
  y suele mejorar la coherencia perceptual de los clusters de color.
- `construir_vector_caracteristicas()`: arma el vector por píxel
  `[c1, c2, c3, x, y]`, estandarizado (media 0, desviación 1) y con un
  `peso_espacial` ajustable para controlar cuánta influencia tienen las
  coordenadas frente al color.

### Paso 3 — Desarrollo de la aplicación
- `segmentar_kmeans()`, `segmentar_fuzzy_cmeans()`, `segmentar_gmm()`: los tres
  algoritmos de clustering solicitados (K-Means obligatorio; FCM/GMM opcional-comparativo).
- `main.py` recorre automáticamente **k = 2,3,4,5** y **espacios RGB/Lab**, genera
  las máscaras y las guarda como imágenes comparativas.
- `posprocesar_mascara()` aplica un filtro de mediana + eliminación de regiones
  pequeñas (posprocesamiento morfológico básico) para limpiar ruido tipo
  "sal y pimienta" en la segmentación.

### Paso 3 — Evaluación
- `calcular_metricas_internas()`: Silhouette y Davies-Bouldin (con submuestreo
  para imágenes grandes, ya que Silhouette es O(n²)).
- `calcular_iou_dice()` / `mejor_cluster_vs_referencia()`: si colocas una máscara
  binaria de referencia en `mascaras_referencia/<nombre_imagen>_mask.png`
  (blanco = objeto de interés), el script busca automáticamente qué clúster se
  parece más a esa referencia y calcula IoU y Dice.
- `main.py` guarda todo en `resultados/metricas.csv`, listo para pegar tablas en
  el informe IEEE.

### Paso 4 (Sustentación)
Usa `comparar_algoritmos.py` para tener evidencia visual y numérica (tiempo,
Silhouette, Davies-Bouldin) que te permita justificar en la sustentación por
qué elegiste cierto k, cierto espacio de color y cierto algoritmo.

## 4. Ejecución

**Experimento completo (barrido de k y espacio de color, con K-Means):**
```bash
python3 main.py
```
Si `imagenes/` está vacía, el script coloca 3 imágenes de muestra de
scikit-image solo para que puedas probar el pipeline antes de usar tus propias
fotos de piezas, PCB, banda transportadora, tornillos, etc.

**Comparación K-Means / Fuzzy C-Means / GMM sobre una imagen puntual:**
```bash
python3 comparar_algoritmos.py imagenes/mi_pieza.jpg --k 4 --espacio lab
```

**Con máscara de referencia para IoU/Dice:**
1. Guarda tu foto como `imagenes/pieza1.jpg`.
2. Guarda una máscara binaria (blanco = objeto) como
   `mascaras_referencia/pieza1_mask.png`.
3. Corre `python3 main.py`; el CSV incluirá las columnas `iou` y `dice`.

## 5. Referencias recientes para el Estado del Arte (informe IEEE)

A continuación tres líneas de trabajo recientes que puedes citar y resumir en
tu informe, mostrando qué mejoran respecto al clustering "puro" (color + espacio)
implementado aquí:

1. **Superpíxeles con información espacial explícita.** Métodos como SLIC
   (y variantes recientes como los superpíxeles "content-aware" de dos etapas)
   agrupan píxeles vecinos usando una distancia que combina color y posición,
   de forma similar a nuestro vector `[color, x, y]`, pero restringiendo la
   búsqueda a una vecindad espacial y ajustando el peso espacial de forma
   adaptativa según el contenido de la imagen, lo que produce regiones más
   compactas y con mejor adherencia a los bordes. *(Fuente: "Superpixels with
   Content-Awareness via a Two-Stage Generation Framework", MDPI Symmetry, 2024;
   y "A Comprehensive Review and New Taxonomy on Superpixel Segmentation", ACM
   Computing Surveys, 2024)*.

2. **Clustering guiado por embeddings/espacio latente.** En dominios con alta
   dimensionalidad (p. ej. imágenes hiperespectrales), en vez de agrupar en el
   espacio de color crudo, primero se extraen características espectro-espaciales
   (a veces con redes profundas) y luego se aplica clustering (k-means, subespacio,
   espectral) sobre ese espacio de características más informativo, mejorando la
   separación de regiones frente al uso directo de intensidades. *(Fuente:
   "Superpixel guided spectral-spatial feature extraction...", Scientific Reports,
   2025)*.

3. **Modelos fundacionales (foundation models) tipo SAM.** El Segment Anything
   Model (Meta AI, 2023) y sus variantes especializadas para inspección industrial
   reemplazan el clustering no supervisado por un modelo preentrenado con mil
   millones de máscaras que segmenta por medio de "prompts" (puntos, cajas,
   texto), logrando generalización zero-shot a nuevas piezas sin reentrenar,
   aunque a costa de mayor complejidad computacional y menor interpretabilidad
   que un k-means clásico. *(Fuente: "SAID: Segment All Industrial Defects with
   Scene Prompts", MDPI Sensors, 2025; "Exploring Few-Shot Defect Segmentation...
   with Vision Foundation Models", arXiv 2502.01216, 2025)*.

**Cómo contrastarlo con tu práctica (para el Análisis 2 del informe):**
K-Means/FCM/GMM sobre `[color, x, y]` son líneas base *interpretables, sin
entrenamiento y de bajo costo computacional* (segundos en CPU), adecuadas
para prototipado rápido en laboratorio. Los superpíxeles añaden restricción de
vecindad para regiones más compactas; los embeddings profundos separan mejor
clases con apariencia similar en color pero distinta textura; los foundation
models (SAM) generalizan a objetos no vistos sin reentrenar, pero requieren
GPU, son "caja negra" y su uso en un laboratorio con recursos limitados es
menos práctico que un clustering clásico bien ajustado.

## 6. Notas para el informe (costos y limitaciones)

- **Costo computacional:** el CSV de `main.py` reporta `tiempo_seg` por
  configuración; con imágenes de ~200 px de lado, K-Means tarda típicamente
  decenas/cientos de milisegundos; FCM y GMM suelen ser 2-5× más lentos (ver
  salida de `comparar_algoritmos.py`).
- **Limitaciones del clustering puro:** sensible a iluminación no uniforme,
  sombras y texturas complejas (puede fragmentar una misma superficie en varios
  clusters); no usa información semántica, solo similitud estadística de bajo
  nivel; el número de clusters *k* debe fijarse a priori (usa el barrido de
  `main.py` + Silhouette/Davies-Bouldin para justificarlo).
