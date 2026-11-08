from pathlib import Path
from statistics import mode
import sys
import numpy as np
import pandas as pd

# Constantes y columnas
MIN_TRANSACCIONES = 30
ESTABLECIMIENTOS = ["Supermercado", "Restaurante", "Gasolinera", "Farmacia", "Ropa",
                    "Tecnología", "Entretenimiento", "Transporte", "Otros"]
COLUMNAS_ENTRADA = ["cliente_id", "fecha_hora", "monto", "tipo_establecimiento"]
COLUMNAS_NO_CUMPLEN = ["cliente_id", "numero_transacciones"]
COLUMNAS_CUMPLEN = ["cliente_id", "fechas_hora", "montos", "tipos_establecimiento",
                    "numero_transacciones", "historico_establecimiento",
                    "promedio", "desviacion", "maximo", "minimo", "monto_hora",
                    "moda_establecimiento", "cantidad_establecimiento"]


# Función que busca la ruta del archivo.
def obtener_ruta_entrada(directorio_datos):
    
    ruta_anterior = directorio_datos / "datos_fuente.csv"
    if ruta_anterior.is_file():
        return ruta_anterior

    raise FileNotFoundError("No se encontró el archivo de entrada. Se buscó en:\n"
                            f" {ruta_anterior}")


# Se cargan las transacciones y se verifica que haya datos y que estén completos
def cargar_transacciones(ruta_entrada):
    
    datos = pd.read_csv(ruta_entrada, sep=None, engine="python", encoding="utf-8-sig")
    faltantes = [col for col in COLUMNAS_ENTRADA if col not in datos.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}.")
    if datos.empty:
        raise ValueError("El archivo de entrada no contiene transacciones.")

    datos = datos[COLUMNAS_ENTRADA].copy()
    if datos.isna().any().any():
        raise ValueError("El CSV contiene valores nulos en las columnas obligatorias.")

    # Asegura formatos correctos
    ids = pd.to_numeric(datos["cliente_id"], errors="raise")
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or (ids < 1).any():
        raise ValueError("La columna 'cliente_id' debe contener enteros positivos.")
    datos["cliente_id"] = ids.astype("int64")

    datos["fecha_hora"] = pd.to_datetime(datos["fecha_hora"], format="%d/%m/%Y %H:%M", errors="raise")
    datos["monto"] = pd.to_numeric(datos["monto"], errors="raise")
    if not np.isfinite(datos["monto"]).all():
        raise ValueError("La columna 'monto' debe contener valores numéricos finitos.")

    datos["tipo_establecimiento"] = datos["tipo_establecimiento"].astype(str).str.strip()
    categorias_desconocidas = sorted(set(datos["tipo_establecimiento"]) - set(ESTABLECIMIENTOS))

    if categorias_desconocidas:
        raise ValueError("Se encontraron tipos de establecimiento desconocidos: " f"{categorias_desconocidas}.")

    # Orden estable para mantener el orden original en caso de fechas idénticas.
    return datos.sort_values(["cliente_id", "fecha_hora"], ascending=[True, True], kind="stable").reset_index(drop=True)


def transformar_transacciones(datos):

    clientes_no_cumplen = []
    clientes_cumplen = []

    for cliente_id, historial in datos.groupby("cliente_id", sort=True):
        numero_transacciones = len(historial)

        # Se separa a los que tienen menos de 30 transaciones.
        if numero_transacciones < MIN_TRANSACCIONES:
            clientes_no_cumplen.append({"cliente_id": int(cliente_id),
                                        "numero_transacciones": numero_transacciones})
            continue

        # Histórico de las transacciones del cliente.
        frecuencias = historial["tipo_establecimiento"].value_counts(normalize=True)
        historico_establecimiento = [float(frecuencias.get(est, 0.0)) for est in ESTABLECIMIENTOS]

        # 30 transacciones más recientes, también en orden ascendente.
        ultimas_30 = historial.tail(MIN_TRANSACCIONES)
        fechas = ultimas_30["fecha_hora"].tolist()
        montos = [float(monto) for monto in ultimas_30["monto"]]
        tipos = ultimas_30["tipo_establecimiento"].tolist()

        # Creación de variables a
        horas_transcurridas = (fechas[-1] - fechas[0]).total_seconds() / 3600
        monto_hora = sum(montos) / horas_transcurridas if horas_transcurridas > 0 else np.nan

        clientes_cumplen.append({
            "cliente_id": int(cliente_id),
            "fechas_hora": [fecha.strftime("%Y-%m-%d %H:%M:%S") for fecha in fechas],
            "montos": montos,
            "tipos_establecimiento": tipos,
            "numero_transacciones": numero_transacciones,
            "historico_establecimiento": historico_establecimiento,
            "promedio": float(np.mean(montos)),
            "desviacion": float(np.std(montos, ddof=0)),
            "maximo": float(np.max(montos)),
            "minimo": float(np.min(montos)),
            "monto_hora": float(monto_hora),
            "moda_establecimiento": mode(tipos),
            "cantidad_establecimiento": len(set(tipos)),
        })

    no_cumplen = pd.DataFrame(clientes_no_cumplen, columns=COLUMNAS_NO_CUMPLEN)
    cumplen = pd.DataFrame(clientes_cumplen, columns=COLUMNAS_CUMPLEN)
    return no_cumplen, cumplen


def main() -> int:
    directorio_datos = Path(__file__).resolve().parent.parent / "datos"

    try:
        ruta_entrada = obtener_ruta_entrada(directorio_datos)
        transacciones = cargar_transacciones(ruta_entrada)
        no_cumplen, cumplen = transformar_transacciones(transacciones)

        directorio_datos.mkdir(parents=True, exist_ok=True)
        ruta_no_cumplen = directorio_datos / "clientes_no_cumplen.csv"
        ruta_cumplen = directorio_datos / "clientes_cumplen.csv"

        # Igual que en el notebook, las listas se escriben como cadenas en el CSV.
        no_cumplen.to_csv(ruta_no_cumplen, index=False, sep=";", encoding="utf-8")
        cumplen.to_csv(ruta_cumplen, index=False, sep=";", encoding="utf-8")

    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Error al transformar los datos: {error}", file=sys.stderr)
        return 1

    print(f"Archivo de entrada: {ruta_entrada}")
    print(f"Transacciones leídas: {len(transacciones)}")
    print(f"Clientes con menos de 30 transacciones: {len(no_cumplen)} → {ruta_no_cumplen}")
    print(f"Clientes con 30 o más transacciones: {len(cumplen)} → {ruta_cumplen}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
