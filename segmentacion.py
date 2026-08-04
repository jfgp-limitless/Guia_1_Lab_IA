"""
segmentacion.py
================
Funciones núcleo para la práctica de "Segmentación por Clustering"
(Ingeniería Mecatrónica).

Cubre los pasos 2 y 3 del PROCEDIMIENTO:
    2. Preparación de los Datos
    3. Desarrollo de la Aplicación (k-means, fuzzy c-means, GMM)
    3. Evaluación (silhouette, Davies-Bouldin, IoU, Dice)

Autor: (completar con nombre del estudiante)
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
from skimage import color, io, morphology, util
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.ndimage import median_filter

try:
    import skfuzzy as fuzz
    _TIENE_SKFUZZY = True
except ImportError:
    _TIENE_SKFUZZY = False


# ---------------------------------------------------------------------
# 1) PREPARACIÓN DE LOS DATOS
# ---------------------------------------------------------------------

def cargar_imagen(ruta, max_lado=300):
    """
    Carga una imagen desde disco, la convierte a RGB de 3 canales y la
    redimensiona (lado más largo = max_lado) para acotar el costo
    computacional del clustering pixel a pixel.
    """
    img = io.imread(ruta)
    if img.ndim == 2:  # escala de grises -> replicar a 3 canales
        img = color.gray2rgb(img)
    if img.shape[-1] == 4:  # RGBA -> RGB
        img = img[..., :3]

    h, w = img.shape[:2]
    escala = max_lado / max(h, w)
    if escala < 1.0:
        nueva_h, nueva_w = int(h * escala), int(w * escala)
        img = resize(img, (nueva_h, nueva_w), anti_aliasing=True)
        img = util.img_as_ubyte(img)
    return img


def convertir_espacio_color(img_rgb, espacio="lab"):
    """
    Convierte una imagen RGB (uint8, 0-255) a 'rgb' o 'lab' (CIELab).
    Devuelve un arreglo float con los 3 canales.
    """
    img_norm = img_rgb.astype(np.float64) / 255.0
    if espacio.lower() == "lab":
        return color.rgb2lab(img_norm)          # L in [0,100], a,b ~[-128,127]
    elif espacio.lower() == "rgb":
        return img_norm * 255.0                  # se deja en 0-255 para normalizar luego
    else:
        raise ValueError("espacio debe ser 'rgb' o 'lab'")


def construir_vector_caracteristicas(img_color, incluir_espacial=True,
                                      peso_espacial=1.0):
    """
    Construye el vector de características por píxel:
        [c1, c2, c3, x_norm, y_norm]   (si incluir_espacial=True)
        [c1, c2, c3]                    (si incluir_espacial=False)

    Todas las columnas se normalizan a media 0 / desviación estándar 1
    (estandarización), y las coordenadas espaciales se multiplican por
    'peso_espacial' para controlar cuánto influyen frente al color.

    Retorna:
        X : ndarray (n_pixeles, n_features) -> listo para clustering
        forma : (alto, ancho) para poder reconstruir la máscara
    """
    h, w, _ = img_color.shape
    canales = img_color.reshape(-1, 3).astype(np.float64)

    # Estandarización por canal (media 0, std 1)
    medias = canales.mean(axis=0)
    stds = canales.std(axis=0) + 1e-8
    canales_norm = (canales - medias) / stds

    if not incluir_espacial:
        return canales_norm, (h, w)

    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    coords_norm *= peso_espacial

    X = np.hstack([canales_norm, coords_norm])
    return X, (h, w)


# ---------------------------------------------------------------------
# 2) ALGORITMOS DE CLUSTERING
# ---------------------------------------------------------------------

def segmentar_kmeans(X, k, random_state=42):
    t0 = time.time()
    modelo = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    etiquetas = modelo.fit_predict(X)
    tiempo = time.time() - t0
    return etiquetas, modelo, tiempo


def segmentar_fuzzy_cmeans(X, k, m=2.0, error=1e-5, max_iter=200, seed=42):
    """
    Fuzzy C-Means usando scikit-fuzzy. Devuelve etiquetas duras
    (argmax de la matriz de pertenencia) y la matriz de pertenencia u.
    """
    if not _TIENE_SKFUZZY:
        raise ImportError("Instale scikit-fuzzy: pip install scikit-fuzzy")

    t0 = time.time()
    # skfuzzy espera datos con forma (n_features, n_muestras)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=k, m=m, error=error, maxiter=max_iter, seed=seed
    )
    etiquetas = np.argmax(u, axis=0)
    tiempo = time.time() - t0
    return etiquetas, u, fpc, tiempo


def segmentar_gmm(X, k, random_state=42):
    t0 = time.time()
    modelo = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=random_state)
    etiquetas = modelo.fit_predict(X)
    tiempo = time.time() - t0
    return etiquetas, modelo, tiempo


# ---------------------------------------------------------------------
# 3) POSPROCESAMIENTO
# ---------------------------------------------------------------------

def posprocesar_mascara(etiquetas_2d, tam_apertura=3, quitar_pequenas=True,
                         area_min=30):
    """
    Aplica un filtro de mediana (reduce el ruido tipo 'sal y pimienta'
    en la etiqueta por vecindad) y, opcionalmente, elimina regiones muy
    pequeñas por cada clase, uniéndolas a la clase vecina dominante.
    """
    suavizada = median_filter(etiquetas_2d, size=tam_apertura)

    if quitar_pequenas:
        salida = suavizada.copy()
        for clase in np.unique(suavizada):
            mascara_clase = suavizada == clase
            limpia = morphology.remove_small_objects(mascara_clase, area_min)
            # Los píxeles removidos se reasignan más adelante (quedan como 'huecos')
            salida[mascara_clase & ~limpia] = -1
        # Rellenar huecos con la etiqueta más cercana (dilatación morfológica simple)
        while np.any(salida == -1):
            huecos = salida == -1
            dilatada = morphology.dilation(np.where(salida == -1, -1, salida))
            salida[huecos] = dilatada[huecos]
            if np.array_equal(dilatada, salida):
                salida[salida == -1] = suavizada[salida == -1]
                break
        return salida
    return suavizada


# ---------------------------------------------------------------------
# 4) MÉTRICAS DE EVALUACIÓN
# ---------------------------------------------------------------------

def calcular_metricas_internas(X, etiquetas, max_muestras=5000, seed=42):
    """
    Calcula silhouette y Davies-Bouldin. Si hay más de max_muestras
    píxeles, se toma una submuestra aleatoria para silhouette
    (costo computacional cuadrático).
    """
    n = X.shape[0]
    resultado = {}

    if len(np.unique(etiquetas)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan}

    if n > max_muestras:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_muestras, replace=False)
        resultado["silhouette"] = silhouette_score(X[idx], etiquetas[idx])
    else:
        resultado["silhouette"] = silhouette_score(X, etiquetas)

    resultado["davies_bouldin"] = davies_bouldin_score(X, etiquetas)
    return resultado


def calcular_iou_dice(mascara_pred_binaria, mascara_referencia_binaria):
    """
    IoU (Jaccard) y coeficiente Dice entre dos máscaras binarias
    (True/False o 1/0) de la misma forma.
    """
    pred = mascara_pred_binaria.astype(bool)
    ref = mascara_referencia_binaria.astype(bool)

    interseccion = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    iou = interseccion / union if union > 0 else np.nan

    suma = pred.sum() + ref.sum()
    dice = (2 * interseccion) / suma if suma > 0 else np.nan
    return iou, dice


def mejor_cluster_vs_referencia(etiquetas_2d, mascara_referencia_binaria):
    """
    Dada una imagen de etiquetas (varias clases) y una máscara binaria de
    referencia (objeto de interés = 1), encuentra qué clúster se solapa
    mejor con la referencia y devuelve su IoU y Dice.
    """
    mejor_iou, mejor_dice, mejor_clase = -1, -1, None
    for clase in np.unique(etiquetas_2d):
        mascara_clase = etiquetas_2d == clase
        iou, dice = calcular_iou_dice(mascara_clase, mascara_referencia_binaria)
        if iou > mejor_iou:
            mejor_iou, mejor_dice, mejor_clase = iou, dice, clase
    return mejor_clase, mejor_iou, mejor_dice
