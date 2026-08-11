# Explicación completa: fundamentos, teoría y workflow de la práctica de Segmentación por Clustering

## 1. ¿Qué problema se está resolviendo?

Segmentar una imagen significa decidir, para cada píxel, "¿a qué región pertenece?" (fondo, persona, mesa, computador, etc.) **sin que nadie diga de antemano cuáles son esas regiones**. Por eso es un problema *no supervisado*: el algoritmo no conoce las clases, solo agrupa píxeles que se parecen entre sí.

La idea central es convertir cada píxel en un **punto en un espacio matemático** (un vector de números) y luego usar clustering para agrupar esos puntos. Píxeles que caen cerca en ese espacio quedan en el mismo cluster, es decir, en la misma región.

## 2. El vector de características: el corazón de todo

Para el píxel `i` en la posición `(x, y)` con color, se construye:

```
x_i = [c1, c2, c3, β·x, β·y]
```

donde `c1, c2, c3` son o bien `[R, G, B]` o bien `[L*, a*, b*]`, y `β` (la constante `PESO_ESPACIAL`) controla cuánto "pesa" la posición frente al color.

### ¿Por qué estandarizar (Z-score)?

Porque R, G, B van de 0 a 255 y las coordenadas x, y pueden ir de 0 a 1280. Si no se normaliza, K-Means (que mide distancias euclidianas) le daría muchísimo más peso a la posición solo porque sus números son más grandes, no porque sea más importante. Al restar la media y dividir por la desviación estándar, todas las variables "compiten en igualdad de condiciones", y ahí `β` permite decidir a propósito cuánto peso extra se le da al espacio.

### RGB vs CIELab: la diferencia real

- **RGB** mezcla brillo y color en los tres canales a la vez. Dos colores pueden verse "distintos" numéricamente solo porque uno está más iluminado, aunque sean el mismo material.
- **CIELab** separa **L\*** (luminosidad) de **a\*, b\*** (cromaticidad pura: verde-rojo, azul-amarillo). Esto explica los resultados obtenidos: RGB agrupa principalmente por contraste de brillo (por eso k=2 le basta para separar "fondo claro" vs "objetos oscuros" con buen Silhouette), mientras que Lab necesita más clusters para expresar su ventaja, porque está capturando matices de color que RGB disuelve dentro del brillo.

## 3. K-Means: el algoritmo base

### Qué minimiza

La inercia (suma de distancias al cuadrado de cada punto a su centroide):

$$J = \sum_{j=1}^{k}\sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2$$

### Cómo lo hace (iterativo, 4 pasos)

1. Inicializa `k` centroides (se usa `k-means++`, una inicialización inteligente, no aleatoria pura).
2. Asigna cada píxel al centroide más cercano.
3. Recalcula cada centroide como el promedio de los píxeles que le tocaron.
4. Repite los pasos 2-3 hasta que los centroides casi no se muevan (`tol=1e-4`) o se llegue a `max_iter=300`.

Como el resultado depende de dónde arrancan los centroides, se usa `n_init=10`: el algoritmo corre 10 veces con distintos arranques y se queda con el de menor `J`. `random_state=42` es solo para que el experimento sea reproducible (mismos resultados cada vez que se corre).

### Analogía simple

Imagina que tiras `k` imanes sobre una nube de puntos. Cada punto se pega al imán más cercano; luego cada imán se mueve al centro de gravedad de los puntos que atrajo; se repite hasta que nada se mueve. Eso es K-Means.

## 4. Fuzzy C-Means y GMM: las variantes comparativas

La diferencia clave frente a K-Means es **qué tan "seguro" está el algoritmo de a qué cluster pertenece cada píxel**:

| Algoritmo | Tipo de pertenencia | Características |
|---|---|---|
| **K-Means** | Dura (0 o 1) | Un píxel es 100% del cluster A o 100% del cluster B. |
| **Fuzzy C-Means (FCM)** | Difusa (entre 0 y 1, suma 1) | Un píxel puede ser 70% "objeto" y 30% "sombra". Útil en bordes ambiguos, pero más lento (377 ms vs ~70-113 ms de K-Means en las pruebas). |
| **GMM (Gaussian Mixture Model)** | Probabilística | En vez de asumir clusters "esféricos" como K-Means, asume que cada cluster es una campana de Gauss (con forma/orientación elipsoidal propia) y calcula la probabilidad de que cada píxel venga de cada gaussiana. Más flexible geométricamente pero también más costoso y sensible a mala inicialización. |

## 5. Las métricas: cómo saber si la segmentación es "buena"

### Sin necesitar una respuesta correcta (métricas internas)

- **Silhouette (↑ mejor, rango -1 a 1):** para cada punto, compara "qué tan cerca está de su propio cluster" vs "qué tan cerca está del cluster más próximo ajeno". Si está bien metido en su cluster y lejos de los demás, el valor es cercano a 1. Mide **cohesión + separación**.
- **Davies-Bouldin (↓ mejor):** para cada par de clusters, compara la dispersión interna de cada uno contra qué tan separados están sus centros. Clusters compactos y bien separados dan un valor bajo.

### Con una máscara de referencia dibujada a mano (métricas externas)

- **IoU (Intersección sobre Unión):** área donde coinciden predicción y referencia, dividida entre el área total que ocupan entre las dos.
- **Dice:** similar pero pondera doble la intersección, es más generoso que IoU con aciertos parciales.

## 6. El workflow completo, paso a paso

```
1. cargar_imagen()                    → lee la foto, la pasa a RGB, la redimensiona (costo computacional)
2. convertir_espacio_color()          → RGB normalizado o CIELab (skimage.color.rgb2lab)
3. construir_vector_caracteristicas() → [color, β·x, β·y] estandarizado (Z-score)
4. segmentar_kmeans() / fcm / gmm     → agrupa los píxeles en k clusters
5. posprocesar_mascara()              → filtro de mediana + quita regiones diminutas (limpieza de ruido)
6. calcular_metricas_internas()       → Silhouette, Davies-Bouldin
7. mejor_cluster_vs_referencia()      → IoU, Dice (si hay máscara ground-truth)
8. main.py                            → repite todo esto para k=2,3,4,5 × {RGB, Lab} × cada imagen,
                                          guarda figuras y metricas.csv
```

Cada imagen termina convertida en una tabla enorme de puntos 5D, se le aplica clustering, y el resultado se "repinta" de vuelta como una imagen donde cada color representa un cluster distinto. Esa es la máscara de segmentación.

## 7. Cómo explicar los resultados específicos (para la sustentación)

Esto es lo más importante: entender el *porqué*, no solo el número.

**¿Por qué RGB con k=2 dio el mejor Silhouette (0.44-0.49)?**

Porque en las fotos de laboratorio la fuente de variación más grande es el contraste luz/sombra (paredes/techo claros vs. equipos/personas oscuros). Con solo 2 clusters, RGB separa perfectamente esa dicotomía de brillo, es una separación "fácil" y muy limpia, de ahí el Silhouette alto.

**¿Por qué Lab mejora progresivamente hasta k=3-4?**

Porque Lab ya no tiene esa ventaja fácil del brillo (la separó en el canal L* aparte). Necesita más clusters para ir descubriendo las diferencias de cromaticidad reales (tonos de piel, ropa, verde de fondo), por eso su Silhouette sube con k=3-4 en vez de bajar como en RGB.

**¿Por qué quitar (x,y) sube el Silhouette ~30-40%?**

Porque sin coordenadas, K-Means tiene *más libertad*: agrupa por color puro sin que la posición "estorbe" ni fragmente clusters que espacialmente están separados pero cromáticamente son iguales. Estadísticamente los grupos quedan más puros, de ahí el mejor Silhouette.

**Pero** (y esto es clave para no sonar contradictorio en la sustentación): un Silhouette más alto sin (x,y) **no significa una segmentación visualmente mejor**, significa clusters más "limpios" en el espacio de color, aunque estén dispersos por toda la imagen (ej. reflejos + ropa blanca + pared en un mismo cluster). Ahí está la tensión clave: pureza estadística vs. coherencia espacial real.

## 8. Comparación con el estado del arte, en una frase cada uno

| Método | En una frase |
|---|---|
| **K-Means (el implementado)** | Agrupa globalmente por color+posición, sin entrenar, muy rápido, muy interpretable (cada cluster = un centroide con 5 números que se puede leer). |
| **SLIC / superpíxeles** | Lo mismo pero restringido a vecindades locales y con peso color/espacio adaptativo por zona, lo que da mejor adherencia a bordes. |
| **Embeddings profundos** | Primero una red extrae características más "inteligentes" (no solo color crudo), luego se agrupa igual (a veces con K-Means) sobre ese espacio mejor. Cuesta más y necesita pesos preentrenados. |
| **SAM (foundation model)** | No agrupa: segmenta directamente vía "prompts" (puntos/cajas), generalizando a objetos nunca vistos, sin fijar k. Es muy costoso computacionalmente y es una caja negra. |

### La idea central a defender

El pipeline implementado es la línea base correcta para un laboratorio de mecatrónica: cero entrenamiento, corre en CPU en milisegundos, y cada decisión (k, espacio de color, peso espacial) es explicable con matemáticas simples. Los métodos modernos ganan en generalización y precisión, pero pagan con cómputo, datos y opacidad. Es un trade-off real, no un "K-Means es peor".
