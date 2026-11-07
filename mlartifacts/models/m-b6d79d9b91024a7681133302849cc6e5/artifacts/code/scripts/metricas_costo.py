from __future__ import annotations
import numpy as np
import pandas as pd

# Validación de los valores a predecir
def _validar(y, probabilidades):
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    p = np.asarray(probabilidades, dtype=np.float64).reshape(-1)
    if y.size == 0 or len(y) != len(p):
        raise ValueError("Se requieren etiquetas y probabilidades de igual longitud no vacía.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("La etiqueta es_anomalia debe contener solo 0 y 1.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")
    return y, p

# Evaluación de costo de predecir mal
def evaluar_costo(y, probabilidades, umbral, costo_fn, costo_fp):

    y, p = _validar(y, probabilidades)
    if not np.isfinite(umbral) or umbral < 0:
        raise ValueError("El umbral de decisión debe ser no negativo.")
    if costo_fn < 0 or costo_fp < 0:
        raise ValueError("Los costos deben ser no negativos.")
    pred = p >= umbral
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    total = fn * float(costo_fn) + fp * float(costo_fp)
    
    return {"umbral": float(umbral),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": len(y),
            "costo_total": float(total),
            "costo_promedio": float(total / len(y))}


# Optimización del umbral
def optimizar_umbral(y, probabilidades, costo_fn, costo_fp):
    
    y, p = _validar(y, probabilidades)
    if costo_fn < 0 or costo_fp < 0:
        raise ValueError("Los costos deben ser no negativos.")

    indices = np.argsort(-p, kind="stable")
    p_ordenadas = p[indices]
    y_ordenadas = y[indices]
    tp_acumulados = np.cumsum(y_ordenadas == 1)
    fp_acumulados = np.cumsum(y_ordenadas == 0)
    finales_grupo = np.flatnonzero(np.r_[p_ordenadas[1:] != p_ordenadas[:-1], True])
    positivos = int(np.sum(y))
    negativos = len(y) - positivos

    umbrales = p_ordenadas[finales_grupo]
    tps = tp_acumulados[finales_grupo].astype(int)
    fps = fp_acumulados[finales_grupo].astype(int)

    # Escenario de ninguna alerta; cubre incluso probabilidades iguales a 1.
    umbral_sin_alertas = float(np.nextafter(1.0, np.inf))
    umbrales = np.r_[umbral_sin_alertas, umbrales]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    fns = positivos - tps
    tns = negativos - fps
    costos_totales = fns * costo_fn + fps * costo_fp

    tabla = pd.DataFrame({"umbral": umbrales, "tp": tps, "fp": fps, "fn": fns, "tn": tns,
                          "costo_total": costos_totales,
                          "costo_promedio": costos_totales / len(y)})

    # Orden descendente: idxmin conserva el umbral más alto en un empate.
    mejor = tabla.loc[tabla["costo_total"].idxmin()]
    resultado = evaluar_costo(y, p, float(mejor["umbral"]), costo_fn, costo_fp)
    return resultado, tabla
