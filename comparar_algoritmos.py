"""
comparar_algoritmos.py
=======================
Comparación cualitativa y cuantitativa entre K-Means, Fuzzy C-Means y GMM
sobre una misma imagen y un mismo k. Corresponde a la parte "opcional o
comparativa" del paso 3 del procedimiento.

Uso:
    python3 comparar_algoritmos.py imagenes/mi_imagen.jpg --k 4 --espacio lab
"""

import argparse
import os
import matplotlib.pyplot as plt

from segmentacion import (
    cargar_imagen, convertir_espacio_color, construir_vector_caracteristicas,
    segmentar_kmeans, segmentar_fuzzy_cmeans, segmentar_gmm,
    posprocesar_mascara, calcular_metricas_internas,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ruta_imagen")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--espacio", choices=["lab", "rgb"], default="lab")
    ap.add_argument("--peso_espacial", type=float, default=1.0)
    args = ap.parse_args()

    img_rgb = cargar_imagen(args.ruta_imagen, max_lado=200)
    img_color = convertir_espacio_color(img_rgb, espacio=args.espacio)
    X, forma = construir_vector_caracteristicas(
        img_color, incluir_espacial=True, peso_espacial=args.peso_espacial
    )

    resultados = {}

    et_km, _, t_km = segmentar_kmeans(X, args.k)
    resultados["K-Means"] = (et_km.reshape(forma), t_km)

    et_fcm, _, fpc, t_fcm = segmentar_fuzzy_cmeans(X, args.k)
    resultados["Fuzzy C-Means"] = (et_fcm.reshape(forma), t_fcm)

    et_gmm, _, t_gmm = segmentar_gmm(X, args.k)
    resultados["GMM"] = (et_gmm.reshape(forma), t_gmm)

    fig, ejes = plt.subplots(1, len(resultados) + 1, figsize=(4 * (len(resultados) + 1), 4))
    ejes[0].imshow(img_rgb)
    ejes[0].set_title("Original")
    ejes[0].axis("off")

    print(f"\nComparación de algoritmos | imagen={os.path.basename(args.ruta_imagen)} "
          f"| k={args.k} | espacio={args.espacio}\n")
    print(f"{'Algoritmo':<15}{'Tiempo (s)':<12}{'Silhouette':<12}{'Davies-Bouldin':<15}")

    for i, (nombre, (etiquetas_2d, tiempo)) in enumerate(resultados.items(), start=1):
        etiquetas_2d = posprocesar_mascara(etiquetas_2d)
        m = calcular_metricas_internas(X, etiquetas_2d.ravel())
        print(f"{nombre:<15}{tiempo:<12.4f}{m['silhouette']:<12.4f}{m['davies_bouldin']:<15.4f}")

        ejes[i].imshow(etiquetas_2d, cmap="tab10")
        ejes[i].set_title(nombre)
        ejes[i].axis("off")

    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    nombre_base = os.path.splitext(os.path.basename(args.ruta_imagen))[0]
    ruta_salida = f"resultados/{nombre_base}_comparacion_algoritmos.png"
    plt.savefig(ruta_salida, dpi=130)
    print(f"\n[ok] Figura guardada en {ruta_salida}")


if __name__ == "__main__":
    main()
