"""
main.py
=======
Ejecuta el pipeline completo de la práctica sobre las imágenes que estén
en la carpeta ./imagenes/ (formatos jpg/png). Si esa carpeta está vacía,
se usan 3 imágenes de ejemplo de scikit-image como placeholder para que
el estudiante pueda probar el código antes de tomar sus propias fotos
de un contexto mecatrónico/industrial (piezas, PCB, banda transportadora,
tornillos, etc.).

Genera en ./resultados/:
    - una figura .png por imagen con la comparación de k y espacios de color
    - metricas.csv con silhouette, Davies-Bouldin, tiempo de ejecución
      (y IoU/Dice si existe una máscara de referencia en ./mascaras_referencia/)
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


def procesar_imagen(ruta_img, escritor_csv):
    nombre_base = os.path.splitext(os.path.basename(ruta_img))[0]
    img_rgb = cargar_imagen(ruta_img, max_lado=MAX_LADO)
    mascara_ref = cargar_mascara_referencia(nombre_base)

    fig, ejes = plt.subplots(len(ESPACIOS_COLOR), len(VALORES_K) + 1,
                              figsize=(4 * (len(VALORES_K) + 1), 4 * len(ESPACIOS_COLOR)))
    if len(ESPACIOS_COLOR) == 1:
        ejes = ejes[np.newaxis, :]

    for fila, espacio in enumerate(ESPACIOS_COLOR):
        ejes[fila, 0].imshow(img_rgb)
        ejes[fila, 0].set_title(f"Original\n(espacio comparado: {espacio})")
        ejes[fila, 0].axis("off")

        img_color = convertir_espacio_color(img_rgb, espacio=espacio)
        X, forma = construir_vector_caracteristicas(
            img_color, incluir_espacial=True, peso_espacial=PESO_ESPACIAL
        )

        for col, k in enumerate(VALORES_K, start=1):
            etiquetas, _, tiempo = segmentar_kmeans(X, k)
            etiquetas_2d = etiquetas.reshape(forma)
            etiquetas_2d = posprocesar_mascara(etiquetas_2d)

            metricas = calcular_metricas_internas(X, etiquetas_2d.ravel())

            fila_csv = {
                "imagen": nombre_base,
                "espacio_color": espacio,
                "algoritmo": "kmeans",
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

            escritor_csv.writerow(fila_csv)

            ejes[fila, col].imshow(etiquetas_2d, cmap="tab10")
            ejes[fila, col].set_title(f"k={k} | sil={metricas['silhouette']:.2f}")
            ejes[fila, col].axis("off")

    plt.tight_layout()
    ruta_fig = os.path.join(CARPETA_RESULTADOS, f"{nombre_base}_comparacion.png")
    plt.savefig(ruta_fig, dpi=130)
    plt.close(fig)
    print(f"[ok] {nombre_base}: figura guardada en {ruta_fig}")


def main():
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    preparar_imagenes_ejemplo()

    rutas = sorted(glob.glob(os.path.join(CARPETA_IMAGENES, "*")))
    ruta_csv = os.path.join(CARPETA_RESULTADOS, "metricas.csv")

    with open(ruta_csv, "w", newline="") as f:
        campos = ["imagen", "espacio_color", "algoritmo", "k",
                  "silhouette", "davies_bouldin", "tiempo_seg", "iou", "dice"]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for ruta in rutas:
            procesar_imagen(ruta, escritor)

    print(f"\n[ok] Métricas completas guardadas en {ruta_csv}")


if __name__ == "__main__":
    main()
