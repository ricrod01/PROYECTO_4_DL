from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Parámetros, límites y variables posibles.
FECHA_INICIAL = pd.Timestamp("2026-06-01")
PROBABILIDAD_ANOMALIA = 0.10
MIN_TRANSACCIONES = 25
MAX_TRANSACCIONES = 40
ESTABLECIMIENTOS = ["Supermercado", "Restaurante", "Gasolinera", "Farmacia", "Ropa",
                    "Tecnología", "Entretenimiento", "Transporte", "Otros"]
FACTORES_MONTO = {
    "Supermercado": 1.30,
    "Restaurante": 0.80,
    "Gasolinera": 1.10,
    "Farmacia": 0.90,
    "Ropa": 1.40,
    "Tecnología": 2.50,
    "Entretenimiento": 0.70,
    "Transporte": 0.50,
    "Otros": 1.00}
MEDIAS_ESTABLECIMIENTOS = np.array([0.90, 0.65, 0.45, 0.30, 0.20, 0.12, 0.04, 0.015, 0.005])
COLUMNAS = ["cliente_id", "fecha_hora", "monto", "tipo_establecimiento"]


# Solicitud de número entero
def solicitar_entero(mensaje, minimo: int, maximo = None):
    entrada = input(mensaje).strip()
    try:
        numero = int(entrada)
    except ValueError as exc:
        raise ValueError("El valor debe ser un número entero.") from exc

    if numero < minimo or (maximo is not None and numero > maximo):
        if maximo is None:
            raise ValueError(f"El valor debe ser un entero mayor o igual a {minimo}.")
        raise ValueError(f"El valor debe ser un entero entre {minimo} y {maximo}.")
    return numero


def generar_transacciones_cliente(cliente_id: int, rng: np.random.Generator):
    n_transacciones = int(rng.integers(MIN_TRANSACCIONES, MAX_TRANSACCIONES + 1))

    # Perfil propio del cliente: monto típico y variabilidad.
    media_monto = rng.uniform(50, 1000)
    desviacion_monto = rng.uniform(20, min(media_monto / 1.5, 500))
    minimo_transferencia = rng.uniform(0.5, 15)
    monto_tipico = max(minimo_transferencia, rng.normal(loc=media_monto, scale=desviacion_monto))

    # Cada cliente inicia en un día diferente dentro de los primeros 40 días.
    desplazamiento_inicial = int(rng.integers(0, 40))
    fecha_actual = FECHA_INICIAL + pd.Timedelta(days=desplazamiento_inicial)

    # Distribución individual de tipos de establecimiento.
    posiciones = rng.permutation(len(ESTABLECIMIENTOS))
    pesos = np.zeros(len(ESTABLECIMIENTOS), dtype=float)
    for posicion, media in zip(posiciones, MEDIAS_ESTABLECIMIENTOS):
        pesos[posicion] = max(0.0, rng.normal(loc=media, scale=0.1 * media))
    probabilidades = pesos / pesos.sum()

    fechas_hora = []
    montos = []
    tipos_establecimiento = []

    for _ in range(n_transacciones):
        # Intervalos originales: de 1 a 2880 minutos entre transacciones.
        minutos = int(rng.integers(1, 2_881))
        fecha_actual += pd.Timedelta(minutes=minutos)

        establecimiento = rng.choice(ESTABLECIMIENTOS, p=probabilidades)
        variacion = rng.lognormal(mean=0, sigma=0.45)
        monto = round(monto_tipico * FACTORES_MONTO[establecimiento] * variacion, 2)

        fechas_hora.append(fecha_actual)
        montos.append(monto)
        tipos_establecimiento.append(establecimiento)

    # Se conserva la probabilidad (10 %) y los tres patrones de anomalía
    # del notebook, aunque las etiquetas NO se incluyen en el CSV solicitado.
    if rng.random() <= PROBABILIDAD_ANOMALIA:
        aleatorio_tipo = rng.random()

        if aleatorio_tipo <= 0.40:
            # Tres montos consecutivos inusualmente elevados.
            posicion = int(rng.integers(1, n_transacciones - 1))
            umbral = np.quantile(montos, 0.85)
            for indice in (posicion - 1, posicion, posicion + 1):
                factor_anomalo = rng.uniform(1.6, 3.0)
                montos[indice] = round(umbral * factor_anomalo, 2)

        elif aleatorio_tipo <= 0.75:
            # Tres transacciones a distancias de 1 a 15 minutos.
            posicion = int(rng.integers(1, n_transacciones - 1))
            fecha_central = fechas_hora[posicion]
            limite_antes = min(15, int((fecha_central - fechas_hora[posicion - 1]) / pd.Timedelta(minutes=1)))
            limite_despues = min(15, int((fechas_hora[posicion + 1] - fecha_central) / pd.Timedelta(minutes=1)))

            minutos_antes = int(rng.integers(1, limite_antes + 1))
            minutos_despues = int(rng.integers(1, limite_despues + 1))
            fechas_hora[posicion - 1] = fecha_central - pd.Timedelta(minutes=minutos_antes)
            fechas_hora[posicion + 1] = fecha_central + pd.Timedelta(minutes=minutos_despues)

        else:
            # De 3 a 5 establecimientos poco habituales para este cliente.
            indices_menores = np.argsort(probabilidades)[:3]
            cantidad = int(rng.integers(3, 6))
            indices_elegidos = rng.choice(indices_menores, size=cantidad, replace=True)
            establecimientos_raros = [ESTABLECIMIENTOS[i] for i in indices_elegidos]
            posiciones_anomalas = rng.choice(n_transacciones, size=cantidad, replace=False)

            for posicion, establecimiento in zip(posiciones_anomalas, establecimientos_raros):
                tipos_establecimiento[posicion] = establecimiento
                variacion = rng.lognormal(mean=0, sigma=0.45)
                montos[posicion] = round(monto_tipico * FACTORES_MONTO[establecimiento] * variacion, 2)

    # Una fila por transacción, sin listas anidadas.
    return [{"cliente_id": cliente_id,
             "fecha_hora": fecha,
             "monto": monto,
             "tipo_establecimiento": establecimiento,}
             for fecha, monto, establecimiento in zip(fechas_hora, montos, tipos_establecimiento)]


def generar_datos(n_clientes: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    registros = []
    for cliente_id in range(1, n_clientes + 1):
        registros.extend(generar_transacciones_cliente(cliente_id, rng))

    df = pd.DataFrame(registros, columns=COLUMNAS)
    df = df.sort_values("fecha_hora", ascending=True, kind="stable").reset_index(drop=True)
    return df


def main() -> int:
    try:
        n_clientes = solicitar_entero("Número de clientes (1 a 99): ", 1, 99)
        semilla = solicitar_entero("Semilla de aleatoriedad (entero >= 0): ", 0)
    except (ValueError, EOFError) as error:
        print(f"Error de entrada: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEjecución cancelada.", file=sys.stderr)
        return 130

    try:
        df = generar_datos(n_clientes, semilla)

        # La ubicación ../datos/ se calcula respecto del archivo .py.
        ruta_salida = Path(__file__).resolve().parent.parent / "datos" / "datos_fuente.csv"
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta_salida, index=False, sep=";", encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")

    except (OSError, ValueError) as error:
        print(f"Error al generar o guardar los datos: {error}", file=sys.stderr)
        return 1

    print(f"CSV generado: {ruta_salida}")
    print(f"Clientes: {n_clientes} | Transacciones: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
