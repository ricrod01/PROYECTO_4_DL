from __future__ import annotations
import ast
import hashlib
import importlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import sys

# Se reutiliza la transformación del repositorio; NO se reimplementa.
from scripts.transformacion_datos import (
    ESTABLECIMIENTOS,
    MIN_TRANSACCIONES,
    cargar_transacciones,
    transformar_transacciones,
)

BASE = Path(__file__).resolve().parent
MODELOS_BASE = BASE / "modelos_base"
ARCHIVO_METADATA = MODELOS_BASE / "metadata_entrenamiento.json"
MAX_TAMANO_MB = 30

st.set_page_config(
    page_title="Monitor de anomalías | Transacciones",
    page_icon="🔎",
    layout="wide",
)


def configurar_logger() -> logging.Logger:
    """Registra eventos agregados, sin montos ni identificadores de clientes."""
    logger = logging.getLogger("monitor_anomalias")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    salida_consola = logging.StreamHandler()
    salida_consola.setFormatter(formato)
    logger.addHandler(salida_consola)

    try:
        carpeta_logs = BASE / "logs"
        carpeta_logs.mkdir(exist_ok=True)
        salida_archivo = RotatingFileHandler(
            carpeta_logs / "app.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        salida_archivo.setFormatter(formato)
        logger.addHandler(salida_archivo)
    except OSError:
        logger.warning("No fue posible abrir el archivo de logs; se usará consola.")
    return logger


LOGGER = configurar_logger()


@st.cache_resource(show_spinner=False)
def obtener_modelo_aprobado():
    """Carga una única vez el modelo aprobado y su umbral de entrenamiento.

    `modelo_c.py` define `ruta_modelos` de forma relativa, por lo que se
    establece una ruta absoluta antes de instanciar el modelo. Su clase NO
    recibe el parámetro `directorio_modelos`.
    """
    if not ARCHIVO_METADATA.is_file():
        raise FileNotFoundError(f"No existe {ARCHIVO_METADATA}. Debes tener los modelos aprobados.")

    metadata = json.loads(ARCHIVO_METADATA.read_text(encoding="utf-8"))
    umbral = float(metadata["umbral_probabilidad"])
    if not np.isfinite(umbral) or umbral < 0:
        raise ValueError("El umbral almacenado en metadata es inválido.")

    # Importación diferida: no se carga TensorFlow hasta necesitar predicciones.
    ruta_scripts = str(BASE / "scripts")

    if ruta_scripts not in sys.path:
        sys.path.insert(0, ruta_scripts)

    modulo_modelo = importlib.import_module("modelo_c")
    modulo_modelo.ruta_modelos = str(MODELOS_BASE.resolve()) + os.sep

    modelo = modulo_modelo.ModeloEnsambladoFraude(umbral=umbral, batch_size=256)

    # Un recurso cacheado se comparte entre sesiones; el lock protege predict.
    return modelo, umbral, threading.Lock()


def transformar_csv_subido(contenido: bytes) -> dict:
    """Ejecuta las funciones originales y lee los 2 CSV generados.

    Los archivos se guardan en una carpeta temporal separada por solicitud;
    esto evita que la carga de un usuario sobrescriba la de otro.
    """
    if not contenido:
        raise ValueError("El archivo está vacío.")
    if len(contenido) > MAX_TAMANO_MB * 1024 * 1024:
        raise ValueError(f"El archivo excede el límite de {MAX_TAMANO_MB} MB.")

    with tempfile.TemporaryDirectory(prefix="transacciones_") as temporal:
        carpeta = Path(temporal)
        ruta_entrada = carpeta / "datos_fuente.csv"
        ruta_cumplen = carpeta / "clientes_cumplen.csv"
        ruta_no_cumplen = carpeta / "clientes_no_cumplen.csv"

        ruta_entrada.write_bytes(contenido)
        # Ambas funciones pertenecen a scripts/transformacion_datos.py.
        transacciones = cargar_transacciones(ruta_entrada)
        no_cumplen, cumplen = transformar_transacciones(transacciones)

        # Mismo formato de salida que el `main()` del script original.
        no_cumplen.to_csv(ruta_no_cumplen, index=False, sep=";", encoding="utf-8")
        cumplen.to_csv(ruta_cumplen, index=False, sep=";", encoding="utf-8")

        # Lectura real de los archivos solicitados, no solo de variables en RAM.
        tabla_no = pd.read_csv(ruta_no_cumplen, sep=";", encoding="utf-8")
        tabla_si = pd.read_csv(ruta_cumplen, sep=";", encoding="utf-8")

        return {
            "transacciones": int(len(transacciones)),
            "cumplen": tabla_si,
            "no_cumplen": tabla_no,
            "csv_cumplen": ruta_cumplen.read_bytes(),
            "csv_no_cumplen": ruta_no_cumplen.read_bytes(),
        }


def inferir_clientes(cumplen: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Aplica el ensamble a los clientes elegibles; devuelve 1 fila/cliente."""
    modelo, umbral, bloqueo = obtener_modelo_aprobado()
    if cumplen.empty:
        return pd.DataFrame(), umbral

    with bloqueo:
        predicciones = modelo.predict(cumplen)
    predicciones = predicciones.reset_index(drop=True)

    if len(predicciones) != len(cumplen):
        raise RuntimeError("La cantidad de predicciones no coincide con los clientes.")

    prob = pd.to_numeric(predicciones["probabilidad_mayor"], errors="raise")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("El modelo devolvió probabilidades fuera del intervalo [0,1].")

    # El cheque indica *anomalía detectada*, tal como solicitó el usuario.
    detectada = prob >= umbral
    tabla = pd.DataFrame(
        {
            "cliente_id": cumplen["cliente_id"].astype("int64").to_numpy(),
            "probabilidad_anomalia": prob.to_numpy(dtype=float),
            "anomalia_detectada": detectada.to_numpy(dtype=bool),
            "resultado": np.where(detectada, "✅", "❌"),
            "tipo_anomalia": predicciones["prediccion_final"].to_numpy(),
        }
    )
    return tabla, umbral


def analizar_archivo(contenido: bytes, sesion: str) -> dict:
    """Orquesta la transformación, la lectura de salidas y la inferencia."""
    inicio = time.perf_counter()
    resultado = transformar_csv_subido(contenido)
    LOGGER.info(
        "evento=transformacion_ok sesion=%s transacciones=%d cumplen=%d no_cumplen=%d",
        sesion,
        resultado["transacciones"],
        len(resultado["cumplen"]),
        len(resultado["no_cumplen"]),
    )

    resultado["predicciones"] = pd.DataFrame()
    resultado["umbral"] = None
    resultado["error_modelo"] = None
    if not resultado["cumplen"].empty:
        try:
            tabla, umbral = inferir_clientes(resultado["cumplen"])
            resultado["predicciones"] = tabla
            resultado["umbral"] = umbral
            LOGGER.info(
                "evento=inferencia_ok sesion=%s clientes=%d alertas=%d tiempo_total_s=%.3f",
                sesion,
                len(tabla),
                int(tabla["anomalia_detectada"].sum()),
                time.perf_counter() - inicio,
            )
        except Exception as error:
            resultado["error_modelo"] = str(error)
            LOGGER.exception("evento=inferencia_error sesion=%s", sesion)
    else:
        LOGGER.info("evento=sin_clientes_elegibles sesion=%s", sesion)
    return resultado


def grafica_pie(etiquetas: list[str], cantidades: list[int], colores: dict):
    """Construye un gráfico de pie con número y porcentaje por categoría."""
    tabla = pd.DataFrame({"Estado": etiquetas, "Clientes": cantidades})
    fig = px.pie(
        tabla,
        names="Estado",
        values="Clientes",
        color="Estado",
        color_discrete_map=colores,
        hole=0.36,
    )
    fig.update_traces(
        textinfo="value+percent",
        textposition="inside",
        hovertemplate="%{label}: %{value} clientes (%{percent})<extra></extra>",
    )
    fig.update_layout(margin=dict(l=5, r=5, t=15, b=5), legend_title_text="")
    return fig


def preparar_graficos_cliente(fila: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recrea las 30 operaciones y compara frecuencias de negocios.

    Diferencia absoluta en puntos porcentuales, igual a la lógica del RF.
    """
    fechas = ast.literal_eval(fila["fechas_hora"])
    montos = ast.literal_eval(fila["montos"])
    tipos = ast.literal_eval(fila["tipos_establecimiento"])
    historico = np.asarray(ast.literal_eval(fila["historico_establecimiento"]), dtype=float)

    if not (len(fechas) == len(montos) == len(tipos) == MIN_TRANSACCIONES):
        raise ValueError("El registro seleccionado no tiene exactamente 30 transacciones.")
    if len(historico) != len(ESTABLECIMIENTOS):
        raise ValueError("El histórico no tiene las nueve categorías requeridas.")

    movimientos = pd.DataFrame(
        {
            "Fecha y hora": pd.to_datetime(fechas, format="%Y-%m-%d %H:%M:%S"),
            "Monto": np.asarray(montos, dtype=float),
            "Tipo de negocio": tipos,
        }
    ).sort_values("Fecha y hora")

    observado = (
        pd.Series(tipos)
        .value_counts(normalize=True)
        .reindex(ESTABLECIMIENTOS, fill_value=0)
        .to_numpy(dtype=float)
    )
    diferencias = pd.DataFrame(
        {
            "Tipo de negocio": ESTABLECIMIENTOS,
            "Histórico (%)": historico * 100,
            "Observado últimas 30 (%)": observado * 100,
            "Diferencia absoluta (p.p.)": np.abs(observado - historico) * 100,
        }
    )
    return movimientos, diferencias


def mostrar_dashboard(resultado: dict) -> None:
    cumplen = resultado["cumplen"]
    no_cumplen = resultado["no_cumplen"]
    predicciones = resultado["predicciones"]
    total_clientes = len(cumplen) + len(no_cumplen)

    st.subheader("Resumen general")
    a, b, c, d = st.columns(4)
    a.metric("Clientes analizados", f"{total_clientes:,}")
    b.metric("Cumplen (≥30)", f"{len(cumplen):,}")
    c.metric("No cumplen (<30)", f"{len(no_cumplen):,}")
    d.metric("Transacciones recibidas", f"{resultado['transacciones']:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Clientes que cumplen el mínimo")
        st.plotly_chart(
            grafica_pie(
                ["Cumplen", "No cumplen"],
                [len(cumplen), len(no_cumplen)],
                {"Cumplen": "#138A72", "No cumplen": "#D97757"},
            ),
            use_container_width=True,
        )
    with col_b:
        st.markdown("#### Anomalías identificadas por el modelo")
        if not predicciones.empty:
            n_anomalias = int(predicciones["anomalia_detectada"].sum())
            st.plotly_chart(
                grafica_pie(
                    ["Con anomalía", "Sin anomalía"],
                    [n_anomalias, len(predicciones) - n_anomalias],
                    {"Con anomalía": "#D94B5A", "Sin anomalía": "#138A72"},
                ),
                use_container_width=True,
            )
            st.caption("Este gráfico incluye solo clientes con al menos 30 transacciones.")
        elif resultado["error_modelo"]:
            st.error("No se pudieron calcular las predicciones del modelo.")
        else:
            st.info("No hay clientes elegibles para generar predicciones.")

    st.divider()
    st.subheader("Clientes que no cumplen el mínimo")
    if no_cumplen.empty:
        st.success("Todos los clientes tienen al menos 30 transacciones.")
    else:
        st.dataframe(
            no_cumplen[["cliente_id", "numero_transacciones"]].rename(
                columns={"cliente_id": "ID cliente", "numero_transacciones": "N.º transacciones"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.download_button(
        "Descargar clientes_no_cumplen.csv",
        data=resultado["csv_no_cumplen"],
        file_name="clientes_no_cumplen.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Resultados del modelo por cliente")
    if resultado["error_modelo"]:
        st.error(
            "La transformación terminó, pero la inferencia falló. "
            f"Detalle: {resultado['error_modelo']}"
        )
    elif not predicciones.empty:
        tabla_vista = pd.DataFrame(
            {
                "ID cliente": predicciones["cliente_id"],
                "Probabilidad de anomalía (%)": (
                    predicciones["probabilidad_anomalia"] * 100
                ).round(2),
                "Anomalía": predicciones["resultado"],
                "Tipo detectado": predicciones["tipo_anomalia"],
            }
        ).sort_values("Probabilidad de anomalía (%)", ascending=False)
        st.caption(
            f"Umbral de decisión del modelo aprobado: {resultado['umbral']:.4f}. "
            "✅ = anomalía detectada; ❌ = no se supera el umbral."
        )
        st.dataframe(tabla_vista, hide_index=True, use_container_width=True)
        st.download_button(
            "Descargar predicciones.csv",
            data=predicciones.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="predicciones.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay resultados de predicción.")

    st.divider()
    st.subheader("Detalle de un cliente que cumple")
    if cumplen.empty:
        st.info("No hay clientes con 30 o más transacciones para filtrar.")
        return

    opciones = sorted(cumplen["cliente_id"].astype(int).tolist())
    cliente_elegido = st.selectbox(
        "Filtrar por ID de cliente",
        options=opciones,
        format_func=lambda valor: f"Cliente {valor}",
    )
    fila = cumplen.loc[cumplen["cliente_id"] == cliente_elegido].iloc[0]

    if not predicciones.empty:
        registro = predicciones.loc[predicciones["cliente_id"] == cliente_elegido].iloc[0]
        indicador_1, indicador_2 = st.columns(2)
        indicador_1.metric("Probabilidad de anomalía", f"{registro['probabilidad_anomalia']:.2%}")
        indicador_2.metric("Resultado", f"{registro['resultado']} {registro['tipo_anomalia']}")

    try:
        movimientos, diferencias = preparar_graficos_cliente(fila)
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.markdown("#### Montos de las últimas 30 transacciones")
            fig_lineas = px.line(
                movimientos,
                x="Fecha y hora",
                y="Monto",
                markers=True,
                hover_data={"Tipo de negocio": True},
            )
            fig_lineas.update_layout(margin=dict(l=5, r=5, t=12, b=5))
            fig_lineas.update_xaxes(tickformat="%d/%m/%Y %H:%M")
            st.plotly_chart(fig_lineas, use_container_width=True)

        with derecha:
            st.markdown("#### Diferencia por tipo de negocio")
            fig_barras = px.bar(
                diferencias,
                x="Tipo de negocio",
                y="Diferencia absoluta (p.p.)",
                hover_data=["Histórico (%)", "Observado últimas 30 (%)"],
                text="Diferencia absoluta (p.p.)",
            )
            fig_barras.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_barras.update_layout(
                margin=dict(l=5, r=5, t=18, b=45),
                xaxis_tickangle=-45,
                yaxis_title="Diferencia absoluta (puntos porcentuales)",
            )
            st.plotly_chart(fig_barras, use_container_width=True)

        st.caption(
            "El histórico representa la distribución de negocios de TODAS las transacciones "
            "del cliente en el archivo cargado (incluye las últimas 30). "
            "El observado usa solo las últimas 30. Se grafica su diferencia absoluta, "
            "como en el preprocesamiento del Random Forest."
        )
        with st.expander("Ver las 30 transacciones del cliente"):
            st.dataframe(movimientos, hide_index=True, use_container_width=True)
    except (ValueError, TypeError, SyntaxError) as error:
        LOGGER.exception("evento=grafico_error sesion=%s", st.session_state["sesion_id"])
        st.error(f"No fue posible visualizar el detalle del cliente: {error}")


# -------------------------- Interfaz --------------------------
if "sesion_id" not in st.session_state:
    st.session_state["sesion_id"] = uuid.uuid4().hex[:12]

st.title("🔎 Monitor de anomalías en transacciones")
st.write("Carga un CSV de transacciones para validar qué clientes cumplen con el mínimo "
         "de 30 operaciones y detectar comportamientos anómalos mediante el ensamble CNN–LSTM–RF.")

with st.sidebar:
    st.header("Carga de datos")
    archivo = st.file_uploader("Subir archivo .csv", type=["csv"])
    st.caption("Columnas necesarias: `cliente_id`, `fecha_hora`, `monto`, "
               "`tipo_establecimiento`. Se aceptan separadores coma o punto y coma.")
    procesar = st.button("Procesar archivo", type="primary", disabled=archivo is None)
    st.info("El modelo evalúa un historial de 30 transacciones por cliente, no cada transacción individual.")

if archivo is None:
    st.info("Sube un CSV desde el panel lateral para comenzar.")
    st.stop()

contenido = archivo.getvalue()
huella = hashlib.sha256(contenido).hexdigest()

if procesar:
    try:
        with st.spinner("Transformando datos y ejecutando el modelo..."):
            analisis = analizar_archivo(contenido, st.session_state["sesion_id"])
        st.session_state["analisis"] = analisis
        st.session_state["huella_archivo"] = huella
        if analisis["error_modelo"]:
            st.warning("La transformación concluyó, pero no se pudo ejecutar el modelo.")
        else:
            st.success("Archivo procesado correctamente.")
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.session_state.pop("analisis", None)
        st.session_state.pop("huella_archivo", None)
        LOGGER.warning("evento=archivo_invalido sesion=%s motivo=%s", st.session_state["sesion_id"], type(error).__name__)
        st.error(f"No se pudo procesar el CSV: {error}")
    except Exception:
        st.session_state.pop("analisis", None)
        st.session_state.pop("huella_archivo", None)
        LOGGER.exception("evento=error_procesamiento sesion=%s", st.session_state["sesion_id"])
        st.error("Ocurrió un error inesperado al procesar el archivo. Revisa logs/app.log.")

if (st.session_state.get("huella_archivo") != huella or "analisis" not in st.session_state):
    st.info("Presiona «Procesar archivo» para generar el análisis de este CSV.")
    st.stop()

mostrar_dashboard(st.session_state["analisis"])
