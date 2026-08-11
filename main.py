"""
main.py
=======
Ejecuta el pipeline completo de la práctica sobre las imágenes que estén
en la carpeta ./imagenes/ (formatos jpg/png). Si esa carpeta está vacía,
se usan 3 imágenes de ejemplo de scikit-image como placeholder para que
el estudiante pueda probar el código antes de tomar sus propias fotos
de un contexto mecatrónico/industrial (piezas, PCB, banda transportadora,
tornillos, etc.).

Ahora corre de forma COMPARATIVA los 3 algoritmos pedidos en la guía:
    - K-Means
    - Fuzzy C-Means
    - GMM (Gaussian Mixture Model)

Organiza los resultados así, para que sea fácil ver qué caso usó qué
algoritmo (todo se genera con un solo "python3 main.py"):

    resultados/
        <nombre_imagen>/
            kmeans/
                <nombre_imagen>_kmeans_comparacion.png   (figura k x espacio)
                metricas_kmeans.csv                       (una fila por k/espacio)
                metricas_kmeans.txt                       (resumen legible)
            fuzzy_cmeans/
                <nombre_imagen>_fuzzy_cmeans_comparacion.png
                metricas_fuzzy_cmeans.csv
                metricas_fuzzy_cmeans.txt
            gmm/
                <nombre_imagen>_gmm_comparacion.png
                metricas_gmm.csv
                metricas_gmm.txt
        metricas.csv   <- CSV global con TODAS las filas (imagen, algoritmo, espacio, k, métricas)
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio, data, util

from segmentacion import (
    cargar_imagen, convertir_espacio_color, construir_vector_caracteristicas,
    segmentar_kmeans, segmentar_fuzzy_cmeans, segmentar_gmm,
    posprocesar_mascara, calcular_metricas_internas, mejor_cluster_vs_referencia,
)

CARPETA_IMAGENES = "imagenes"
CARPETA_REFERENCIA = "mascaras_referencia"
CARPETA_RESULTADOS = "resultados"
VALORES_K = [2, 3, 4, 5]
ESPACIOS_COLOR = ["lab", "rgb"]
PESO_ESPACIAL = 1.0          # súbelo (p.ej. 2-3) para forzar regiones más compactas
MAX_LADO = 200                # tamaño máximo de imagen (controla el costo computacional)

# Algoritmos que se corren de forma comparativa en cada imagen.
# clave interna -> nombre "bonito" para títulos/figuras/txt
ALGORITMOS = {
    "kmeans": "K-Means",
    "fuzzy_cmeans": "Fuzzy C-Means",
    "gmm": "GMM",
}

CAMPOS_CSV = ["imagen", "algoritmo", "espacio_color", "k",
              "silhouette", "davies_bouldin", "tiempo_seg", "iou", "dice"]


def preparar_imagenes_ejemplo():
    """Si la carpeta de imágenes está vacía, coloca 3 imágenes de muestra
    (NO reemplazan tus propias fotos industriales: son solo un placeholder)."""
    os.makedirs(CARPETA_IMAGENES, exist_ok=True)
    if glob.glob(os.path.join(CARPETA_IMAGENES, "*")):
        return
    ejemplos = {
        "ejemplo_rocket.png": data.rocket(),
        "ejemplo_coffee.png": data.coffee(),
        "ejemplo_astronaut.png": data.astronaut(),
    }
    for nombre, arr in ejemplos.items():
        skio.imsave(os.path.join(CARPETA_IMAGENES, nombre), util.img_as_ubyte(arr))
    print("[i] No se encontraron imágenes propias: se copiaron 3 imágenes de "
          "muestra en ./imagenes/ solo para poder probar el pipeline. "
          "Reemplázalas por tus 3-5 fotos de un contexto mecatrónico/industrial.")


def cargar_mascara_referencia(nombre_base):
    """Busca en ./mascaras_referencia/ un archivo con el mismo nombre base
    de la imagen (p.ej. pieza1.png -> pieza1_mask.png), binario (0/255)."""
    posibles = glob.glob(os.path.join(CARPETA_REFERENCIA, f"{nombre_base}*"))
    if not posibles:
        return None
    m = skio.imread(posibles[0], as_gray=True)
    return m > 0.5


def ejecutar_algoritmo(algoritmo, X, k):
    """
    Corre el algoritmo pedido (clave interna de ALGORITMOS) sobre X con
    ese k. Devuelve (etiquetas_1d, tiempo_seg). Lanza ImportError si el
    algoritmo requiere una librería que no está instalada (scikit-fuzzy).
    """
    if algoritmo == "kmeans":
        etiquetas, _, tiempo = segmentar_kmeans(X, k)
    elif algoritmo == "fuzzy_cmeans":
        etiquetas, _, _, tiempo = segmentar_fuzzy_cmeans(X, k)
    elif algoritmo == "gmm":
        etiquetas, _, tiempo = segmentar_gmm(X, k)
    else:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}")
    return etiquetas, tiempo


def procesar_algoritmo_para_imagen(algoritmo, nombre_algo, nombre_base,
                                    img_rgb, mascara_ref, carpeta_salida,
                                    escritor_global):
    """
    Corre UN algoritmo (kmeans / fuzzy_cmeans / gmm) sobre UNA imagen,
    barriendo todos los VALORES_K y ESPACIOS_COLOR. Guarda:
        - la figura comparativa dentro de carpeta_salida
        - metricas_<algoritmo>.csv dentro de carpeta_salida
        - metricas_<algoritmo>.txt (resumen legible) dentro de carpeta_salida
        - además escribe cada fila en el CSV global (escritor_global)
    Devuelve la lista de filas (dicts) generadas, o None si el algoritmo
    no se pudo correr (p.ej. falta scikit-fuzzy).
    """
    filas = []

    fig, ejes = plt.subplots(len(ESPACIOS_COLOR), len(VALORES_K) + 1,
                              figsize=(4 * (len(VALORES_K) + 1), 4 * len(ESPACIOS_COLOR)))
    if len(ESPACIOS_COLOR) == 1:
        ejes = ejes[np.newaxis, :]

    try:
        for fila_idx, espacio in enumerate(ESPACIOS_COLOR):
            ejes[fila_idx, 0].imshow(img_rgb)
            ejes[fila_idx, 0].set_title(f"Original\n(espacio: {espacio})")
            ejes[fila_idx, 0].axis("off")

            img_color = convertir_espacio_color(img_rgb, espacio=espacio)
            X, forma = construir_vector_caracteristicas(
                img_color, incluir_espacial=True, peso_espacial=PESO_ESPACIAL
            )

            for col, k in enumerate(VALORES_K, start=1):
                etiquetas, tiempo = ejecutar_algoritmo(algoritmo, X, k)
                etiquetas_2d = etiquetas.reshape(forma)
                etiquetas_2d = posprocesar_mascara(etiquetas_2d)

                metricas = calcular_metricas_internas(X, etiquetas_2d.ravel())

                fila_csv = {
                    "imagen": nombre_base,
                    "algoritmo": nombre_algo,
                    "espacio_color": espacio,
                    "k": k,
                    "silhouette": round(metricas["silhouette"], 4),
                    "davies_bouldin": round(metricas["davies_bouldin"], 4),
                    "tiempo_seg": round(tiempo, 4),
                    "iou": "",
                    "dice": "",
                }

                if mascara_ref is not None:
                    _, iou, dice = mejor_cluster_vs_referencia(etiquetas_2d, mascara_ref)
                    fila_csv["iou"] = round(iou, 4)
                    fila_csv["dice"] = round(dice, 4)

                filas.append(fila_csv)
                escritor_global.writerow(fila_csv)

                ejes[fila_idx, col].imshow(etiquetas_2d, cmap="tab10")
                ejes[fila_idx, col].set_title(f"k={k} | sil={metricas['silhouette']:.2f}")
                ejes[fila_idx, col].axis("off")

    except ImportError as e:
        plt.close(fig)
        print(f"[!] {nombre_base} / {nombre_algo}: se omitió ({e}).")
        return None

    plt.suptitle(f"{nombre_algo} — {nombre_base}", fontsize=14)
    plt.tight_layout()
    ruta_fig = os.path.join(carpeta_salida, f"{nombre_base}_{algoritmo}_comparacion.png")
    plt.savefig(ruta_fig, dpi=130)
    plt.close(fig)

    # CSV específico de este algoritmo/imagen
    ruta_csv_algo = os.path.join(carpeta_salida, f"metricas_{algoritmo}.csv")
    with open(ruta_csv_algo, "w", newline="") as f:
        escritor_algo = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        escritor_algo.writeheader()
        for fila in filas:
            escritor_algo.writerow(fila)

    # TXT resumen legible (mejor combinación espacio/k según silhouette)
    ruta_txt_algo = os.path.join(carpeta_salida, f"metricas_{algoritmo}.txt")
    mejor = max(filas, key=lambda f: (f["silhouette"] if f["silhouette"] == f["silhouette"] else -1))
    with open(ruta_txt_algo, "w") as f:
        f.write(f"Resumen - {nombre_algo} - imagen: {nombre_base}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mejor configuración (mayor Silhouette):\n")
        f.write(f"  espacio_color = {mejor['espacio_color']}\n")
        f.write(f"  k             = {mejor['k']}\n")
        f.write(f"  silhouette    = {mejor['silhouette']}\n")
        f.write(f"  davies_bouldin= {mejor['davies_bouldin']}\n")
        f.write(f"  tiempo_seg    = {mejor['tiempo_seg']}\n")
        if mejor["iou"] != "":
            f.write(f"  iou           = {mejor['iou']}\n")
            f.write(f"  dice          = {mejor['dice']}\n")
        f.write("\nTodas las combinaciones probadas (espacio_color x k):\n\n")
        f.write(f"{'espacio':<10}{'k':<5}{'silhouette':<12}{'davies_bouldin':<16}{'tiempo_seg':<12}\n")
        for fila in filas:
            f.write(f"{fila['espacio_color']:<10}{fila['k']:<5}"
                    f"{fila['silhouette']:<12}{fila['davies_bouldin']:<16}{fila['tiempo_seg']:<12}\n")

    print(f"[ok] {nombre_base} / {nombre_algo}: figura y métricas guardadas en {carpeta_salida}/")
    return filas


def procesar_imagen(ruta_img, escritor_global):
    nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
    img_rgb = cargar_imagen(ruta_img, max_lado=MAX_LADO)
    mascara_ref = cargar_mascara_referencia(nombre_base)

    carpeta_imagen = os.path.join(CARPETA_RESULTADOS, nombre_base)
    os.makedirs(carpeta_imagen, exist_ok=True)

    for algoritmo, nombre_algo in ALGORITMOS.items():
        carpeta_algo = os.path.join(carpeta_imagen, algoritmo)
        os.makedirs(carpeta_algo, exist_ok=True)
        procesar_algoritmo_para_imagen(
            algoritmo, nombre_algo, nombre_base, img_rgb, mascara_ref,
            carpeta_algo, escritor_global,
        )


def main():
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    preparar_imagenes_ejemplo()

    rutas = sorted(glob.glob(os.path.join(CARPETA_IMAGENES, "*")))
    ruta_csv_global = os.path.join(CARPETA_RESULTADOS, "metricas.csv")

    with open(ruta_csv_global, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        for ruta in rutas:
            procesar_imagen(ruta, escritor)

    print(f"\n[ok] Métricas completas (todas las imágenes x algoritmos) guardadas en {ruta_csv_global}")
    print(f"[ok] Revisa resultados/<nombre_imagen>/<algoritmo>/ para figuras y métricas por caso.")


if __name__ == "__main__":
    main()