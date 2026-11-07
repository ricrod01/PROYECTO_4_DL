# Librerías
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import joblib
import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split,)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metricas_costo import evaluar_costo, optimizar_umbral
from modelo_c import (EscaladorGlobalSecuencias, calcular_diferencias_establecimientos,
                              convertir_fechas_a_minutos, convertir_montos)

# Anomalías y variables usadas
TIPOS_ANOMALIA = ("transaccion_corta", "orden_no_congruente", "establecimiento_raro")
VARIABLES = ("fechas_hora", "montos", "tipos_establecimiento", "historico_establecimiento")

# Ruta del proyecto
def ruta_proyecto(ruta):
    path = Path(ruta).expanduser()
    return path if path.is_absolute() else ROOT / path


# Genera la huella de los archivos de datos usados en el entrenamiento
def sha256_archivo(ruta):
    huella = hashlib.sha256()
    with ruta.open("rb") as contenido:
        for bloque in iter(lambda: contenido.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()

# Versiones de librerías usadas
def versiones_paquetes():
    paquetes = ("tensorflow", "mlflow", "numpy", "pandas", "scikit-learn", "joblib", "keras")
    resultado = {"python": platform.python_version(), "plataforma": platform.platform()}
    for paquete in paquetes:
        try:
            resultado[paquete] = version(paquete)
        except PackageNotFoundError:
            resultado[paquete] = "no_disponible"
    return resultado

# Configuración de lo que se guarda en mlflow.
def configurar_mlflow(config):
    base_datos = ruta_proyecto(config["mlflow"]["base_datos"]).resolve()
    base_datos.parent.mkdir(parents=True, exist_ok=True)
    artifacts = ruta_proyecto(config["mlflow"]["directorio_artefactos"]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{base_datos.as_posix()}")
    nombre = config["mlflow"]["experimento"]
    if mlflow.get_experiment_by_name(nombre) is None:
        mlflow.create_experiment(nombre, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(nombre)

# Definición de semilla
def definir_semilla(semilla, determinismo_gpu):
    os.environ["PYTHONHASHSEED"] = str(semilla) 
    random.seed(semilla)
    np.random.seed(semilla)
    tf.keras.utils.set_random_seed(semilla)
    if determinismo_gpu:
        tf.config.experimental.enable_op_determinism()


# Se cargan los datos de entrenamiento
def cargar_datos(config):
    ruta = ruta_proyecto(config["datos"]["ruta_csv"]).resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta el CSV: {ruta}. Copia allí tus datos antes de entrenar.")
    datos = pd.read_csv(ruta, sep=config["datos"]["separador"])
    obligatorias = {*VARIABLES, "tipo_anomalia", "es_anomalia"}
    faltantes = obligatorias - set(datos.columns)
    if faltantes:
        raise ValueError(f"Al CSV le faltan columnas: {sorted(faltantes)}")

    filas_iniciales = len(datos)

    # Excluyen filas sin tipo de anomalía.
    datos = datos.dropna(subset=["tipo_anomalia", "es_anomalia"]).copy()

    # Evaluación de la calidad de los datos
    if datos.empty:
        raise ValueError("El CSV no contiene registros útiles.")
    if datos[list(VARIABLES)].isna().any().any():
        raise ValueError("Existen nulos en las columnas de secuencias o establecimientos.")
    if not datos["tipo_anomalia"].isin(("Normal", *TIPOS_ANOMALIA)).all():
        raise ValueError("Hay tipos de anomalía distintos de Normal y los tres tipos conocidos.")
    etiquetas = pd.to_numeric(datos["es_anomalia"], errors="raise")
    if not etiquetas.isin([0, 1]).all():
        raise ValueError("La etiqueta es_anomalia debe ser 0 (normal) o 1 (anomalía).")
    datos["es_anomalia"] = etiquetas.astype(np.int32)
    if not (((datos["tipo_anomalia"] == "Normal") == (datos["es_anomalia"] == 0)).all()):
        raise ValueError("tipo_anomalia y es_anomalia no son consistentes.")
    datos = datos.reset_index(drop=True)

    # Resumen de los datos
    resumen = {"filas_originales": filas_iniciales,
               "filas_validas": len(datos),
               "filas_descartadas": filas_iniciales - len(datos),
               "conteo_clases": {str(k): int(v) for k, v in datos["tipo_anomalia"].value_counts().items()},
               "sha256_csv": sha256_archivo(ruta)}
    return datos, ruta, resumen

# Permite la separación en entrenamiento y prueba
def particionar(datos, config):
    semilla = int(config["semilla"])
    entrenamiento_y_validacion, prueba = train_test_split(datos, test_size=float(config["datos"]["fraccion_prueba"]),
                                                          random_state=semilla, stratify=datos["tipo_anomalia"])
    entrenamiento, validacion = train_test_split(entrenamiento_y_validacion,
                                                 test_size=float(config["datos"]["fraccion_validacion_del_resto"]),
                                                 random_state=semilla, stratify=entrenamiento_y_validacion["tipo_anomalia"])
    return tuple(d.reset_index(drop=True) for d in (entrenamiento, validacion, prueba))

# Se selecciona el tipo de anomalía (ya que cada submodelo se centra un uno)
def seleccionar_clases(df, tipo):
    return df.loc[df["tipo_anomalia"].isin(("Normal", tipo))].reset_index(drop=True)

# Creación de pipeline de la CNN-1D 
def crear_pipeline_cnn() -> Pipeline:
    return Pipeline([("diferencias_minutos", FunctionTransformer(convertir_fechas_a_minutos, validate=False)),
                     ("transformacion_logaritmica", FunctionTransformer(np.log1p, validate=False)),
                     ("normalizacion", StandardScaler(copy=True, with_mean=True, with_std=True))])

# Creación de pipeline de la LSTM
def crear_pipeline_lstm() -> Pipeline:
    return Pipeline([("convertir_a_numerico", FunctionTransformer(convertir_montos, validate=False)),
                     ("transformacion_logaritmica", FunctionTransformer(np.log1p, validate=False)),
                     ("normalizacion_global", EscaladorGlobalSecuencias())])

# Creación del submodelo CNN-1D
def crear_cnn(p):
    
    modelo = tf.keras.Sequential([tf.keras.layers.Input(shape=(29, 1)),
                                  tf.keras.layers.Conv1D(filters=int(p["filters_1"]), kernel_size=3, strides=1,
                                                         padding="same", activation="relu"),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
                                  tf.keras.layers.Conv1D(filters=int(p["filters_2"]), kernel_size=3, strides=1,
                                                         padding="same", activation="relu"),
                                  tf.keras.layers.BatchNormalization(),
                                  tf.keras.layers.GlobalMaxPooling1D(),
                                  tf.keras.layers.Dense(units=int(p["dense_units"]), activation="relu"),
                                  tf.keras.layers.Dropout(rate=float(p["dropout_rate"])),
                                  tf.keras.layers.Dense(units=1, activation="sigmoid"),], name="cnn_fechas")
    compilar_red(modelo, float(p["learning_rate"]))
    return modelo

# Creación del submodelo LSTM
def crear_lstm(p):
    """Dos capas LSTM (no se cambia la cantidad), luego una Dense oculta."""
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 1), name="secuencia_montos"),
        tf.keras.layers.LSTM(units=int(p["lstm_units_1"]), activation="tanh",
                             recurrent_activation="sigmoid", return_sequences=True,
                             dropout=0.0, recurrent_dropout=0.0, name="lstm_1"),
        tf.keras.layers.LayerNormalization(name="normalizacion_lstm_1"),
        tf.keras.layers.Dropout(rate=float(p["dropout_rate"]), name="dropout_1"),
        tf.keras.layers.LSTM(units=int(p["lstm_units_2"]), activation="tanh",
                             recurrent_activation="sigmoid", return_sequences=False,
                             dropout=0.0, recurrent_dropout=0.0, name="lstm_2"),
        tf.keras.layers.Dense(units=int(p["dense_units"]), activation="relu", name="capa_densa"),
        tf.keras.layers.Dense(units=1, activation="sigmoid", name="score_fraude"),
    ], name="lstm_montos")
    compilar_red(modelo, float(p["learning_rate"]))
    return modelo

# Función que compila el modelo
def compilar_red(modelo, learning_rate):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.50),
                 tf.keras.metrics.Precision(name="precision", thresholds=0.50),
                 tf.keras.metrics.Recall(name="recall", thresholds=0.50),
                 tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
                 tf.keras.metrics.AUC(name="pr_auc", curve="PR")],)

# Métricas que se almacenarán
def metricas_clasificacion(y, p, umbral = 0.5):
    reales = np.asarray(y, dtype=np.int32)
    scores = np.asarray(p, dtype=np.float64)
    predichos = scores >= umbral
    metricas = {"accuracy": float(accuracy_score(reales, predichos)),
                "precision": float(precision_score(reales, predichos, zero_division=0)),
                "recall": float(recall_score(reales, predichos, zero_division=0)),
                "f1": float(f1_score(reales, predichos, zero_division=0)),
                "pr_auc": float(average_precision_score(reales, scores))}
    
    if len(np.unique(reales)) > 1:
        metricas["roc_auc"] = float(roc_auc_score(reales, scores))
    return metricas

# Función que registra las métricas
def registrar_metricas(metricas, prefijo):
    mlflow.log_metrics({f"{prefijo}_{k}": float(v) for k, v in metricas.items()})


# Función que hace el ajuste de los submodelos CNN y LSTM
def buscar_red(nombre, entrenamiento, validacion, config, constructor, constructor_pipeline, columna):

    # Modelo y su tipo de anomalía
    tipo = {"cnn": "transaccion_corta", "lstm": "orden_no_congruente"}[nombre]

    # Separación de datos de entrenamiento, validación y de variable independiente
    train_especifico = seleccionar_clases(entrenamiento, tipo)
    val_especifico = seleccionar_clases(validacion, tipo)
    y_train = train_especifico["es_anomalia"].to_numpy(dtype=np.int32)
    y_val = val_especifico["es_anomalia"].to_numpy(dtype=np.int32)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError(f"El {nombre} requiere ambas clases en train y validación.")

    # Se construye el pipeline y se entrena
    pipeline = constructor_pipeline()
    x_train = pipeline.fit_transform(train_especifico[columna]).astype(np.float32)[..., np.newaxis]
    x_val = pipeline.transform(val_especifico[columna]).astype(np.float32)[..., np.newaxis]

    # Clases y balanceo según los pesos
    clases = np.array([0, 1], dtype=np.int32)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y_train)
    pesos_clases = {int(c): float(p) for c, p in zip(clases, pesos)}

    # Definición de resultados a llenar
    mejor_score = -np.inf
    mejor_p = None
    mejores_pesos = None
    mejor_resultado = None
    rejilla = list(ParameterGrid(config[nombre]["rejilla"]))
    if not rejilla:
        raise ValueError(f"La rejilla de {nombre} no tiene combinaciones.")

    # Se entrena el modelo
    for i, p in enumerate(rejilla, start=1):
        tf.keras.backend.clear_session()
        definir_semilla(int(config["semilla"]), False)
        modelo = constructor(p)
        entrenamiento_cfg = config["entrenamiento"]

        # Configuración de parada temprana y adaptación de la tasa de aprendizaje
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                patience=int(entrenamiento_cfg["paciencia_early_stopping"]),
                min_delta=float(entrenamiento_cfg["min_delta"]),
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max",
                factor=float(entrenamiento_cfg["factor_reduccion_lr"]),
                patience=int(entrenamiento_cfg["paciencia_reduccion_lr"]),
                min_lr=float(entrenamiento_cfg["learning_rate_minimo"]),
            ),
        ]

        # Entrenamiento del modelo y almacenamiento de resultados y variables
        with mlflow.start_run(run_name=f"{nombre}_rejilla_{i:03d}", nested=True):
            mlflow.set_tag("algoritmo", nombre.upper())
            mlflow.log_params({f"hp_{k}": v for k, v in p.items()})
            mlflow.log_params({"semilla": int(config["semilla"]),
                               "epocas_maximas": int(entrenamiento_cfg["epocas_maximas"]),
                               "n_train": len(y_train), "n_validacion": len(y_val)})
            mlflow.log_dict({"pesos_clases": pesos_clases}, "pesos_clases.json")
            historia = modelo.fit(x_train, y_train,
                                  validation_data=(x_val, y_val),
                                  epochs=int(entrenamiento_cfg["epocas_maximas"]),
                                  batch_size=int(p["batch_size"]),
                                  class_weight=pesos_clases,
                                  callbacks=callbacks,
                                  shuffle=True,
                                  verbose=0)
            
            # Cada métrica por época y el histórico son parte de la huella.
            for metrica, valores in historia.history.items():
                for epoca, valor in enumerate(valores, start=1):
                    mlflow.log_metric(f"epoca_{metrica}", float(valor), step=epoca)
            mlflow.log_dict({k: [float(v) for v in valores]
                             for k, valores in historia.history.items()}, "historia.json")
            prob_val = modelo.predict(x_val, batch_size=int(p["batch_size"]), verbose=0).reshape(-1)
            evaluacion = metricas_clasificacion(y_val, prob_val)
            registrar_metricas(evaluacion, "validacion")
            mlflow.log_metric("epocas_entrenadas", len(historia.epoch))
            score = evaluacion["pr_auc"]
            mlflow.set_tag("seleccionado", "por_comparar")

        # Se almacena el mejor resultado
        print(f"[{nombre} {i}/{len(rejilla)}] PR-AUC validación: {score:.5f}")
        if score > mejor_score:
            mejor_score = score
            mejor_p = dict(p)
            mejores_pesos = [np.array(w, copy=True) for w in modelo.get_weights()]
            mejor_resultado = dict(evaluacion)
        del modelo

    # Se limpia la sesión y se obtiene el mejor modelo
    tf.keras.backend.clear_session()
    definir_semilla(int(config["semilla"]), False)
    modelo_ganador = constructor(mejor_p)
    modelo_ganador.set_weights(mejores_pesos)
    return modelo_ganador, pipeline, mejor_p, mejor_resultado

# Función que permite entrenar el mejor RandomForest
def buscar_rf(entrenamiento, validacion, config):

    # Separación de datos de entrenamiento, validación y de variable independiente
    train_rf = seleccionar_clases(entrenamiento, "establecimiento_raro")
    val_rf = seleccionar_clases(validacion, "establecimiento_raro")
    y_train = train_rf["es_anomalia"].to_numpy(dtype=np.int32)
    y_val = val_rf["es_anomalia"].to_numpy(dtype=np.int32)
    folds = int(config["rf"]["cv_pliegues"])
    if min(np.bincount(y_train, minlength=2)) < folds:
        raise ValueError("No hay suficientes ejemplos por clase para los folds del RF.")

    # Pipeline
    pipeline = Pipeline([("preprocesamiento", Pipeline([("diferencias_establecimientos", 
                                                         FunctionTransformer(calcular_diferencias_establecimientos,
                                                                             validate=False))])),
                         ("random_forest", RandomForestClassifier(criterion="gini", bootstrap=True,
                                                                  random_state=int(config["semilla"]), n_jobs=1))])
    
    # Rejilla, validación cruzada y GridSearch
    rejilla = config["rf"]["rejilla"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(config["semilla"]))
    busqueda = GridSearchCV(estimator=pipeline, param_grid=rejilla,
                            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                            refit="pr_auc", cv=cv, n_jobs=int(config["rf"]["n_jobs_grid"]),
                            verbose=0, return_train_score=True, error_score="raise")

    # Entrenamiento del modelo y almacenamiento de resultados y variables
    with mlflow.start_run(run_name="rf_gridsearch_cv", nested=True):
        mlflow.set_tag("algoritmo", "RandomForest")
        mlflow.log_params({"cv_pliegues": folds,
                           "n_jobs_grid": int(config["rf"]["n_jobs_grid"]),
                           "semilla": int(config["semilla"]), "n_train": len(y_train)})
        busqueda.fit(train_rf[list(VARIABLES[2:])], y_train)
        mlflow.log_metric("mejor_pr_auc_cv", float(busqueda.best_score_))
        mlflow.log_params({f"mejor_{k}": v for k, v in busqueda.best_params_.items()})
        with tempfile.TemporaryDirectory() as tmp:
            resultados = pd.DataFrame(busqueda.cv_results_)
            resultados.to_csv(Path(tmp) / "resultados_gridsearch.csv", index=False)
            mlflow.log_artifact(str(Path(tmp) / "resultados_gridsearch.csv"))

    # Se almacena el mejor resultado
    modelo = busqueda.best_estimator_
    p_val = modelo.predict_proba(val_rf[list(VARIABLES[2:])])[:, list(modelo.classes_).index(1)]
    met = metricas_clasificacion(y_val, p_val)
    registrar_metricas(met, "rf_validacion")
    mlflow.log_metric("rf_mejor_pr_auc_cv", float(busqueda.best_score_))

    return modelo, dict(busqueda.best_params_), met

# Función que permite predecir las probabilidades de cada modelo
def probabilidades_ensamble(df, cnn, lstm, rf, pipeline_cnn, pipeline_lstm, batch_size = 256):

    x_cnn = pipeline_cnn.transform(df["fechas_hora"]).astype(np.float32)[..., np.newaxis]
    x_lstm = pipeline_lstm.transform(df["montos"]).astype(np.float32)[..., np.newaxis]
    p_cnn = cnn.predict(x_cnn, batch_size=batch_size, verbose=0).reshape(-1)
    p_lstm = lstm.predict(x_lstm, batch_size=batch_size, verbose=0).reshape(-1)
    p_rf = rf.predict_proba(df[list(VARIABLES[2:])])[:, list(rf.classes_).index(1)]
    
    return np.column_stack([p_cnn, p_lstm, p_rf]).astype(np.float64)

# Función que guarda el modelo si es aprobado
def guardar_aprobados(config, run_id, cnn, lstm, rf, pipeline_cnn, pipeline_lstm, metadatos):

    # Primero se guardan los modelos con su versión
    versionado = ruta_proyecto(config["salida"]["modelos_versionados"]) / run_id
    if versionado.exists():
        raise FileExistsError(f"El directorio de modelos de este run ya existe: {versionado}")
    versionado.mkdir(parents=True)
    cnn.save(versionado / "modelo_c_cnn.keras")
    lstm.save(versionado / "modelo_c_lstm.keras")
    joblib.dump(pipeline_cnn, versionado / "pipeline_c_cnn.joblib", compress=3)
    joblib.dump(pipeline_lstm, versionado / "pipeline_c_lstm.joblib", compress=3)
    joblib.dump(rf, versionado / "modelo_c_rf.joblib", compress=3)
    (versionado / "metadata_entrenamiento.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    mlflow.log_artifacts(str(versionado), artifact_path="modelos_aprobados")

    # Los archivos ya han quedado versionados y en los artefactos del run
    compatibles = ruta_proyecto(config["salida"]["modelos_compatibles"])
    compatibles.mkdir(parents=True, exist_ok=True)
    for fichero in versionado.iterdir():
        temporal = compatibles / (fichero.name + ".parcial")
        shutil.copy2(fichero, temporal)
        temporal.replace(compatibles / fichero.name)

    # Almacenando para que sea compatible con MLFlow
    for nombre, funcion in (
        ("cnn_aprobada", lambda: mlflow.keras.log_model(cnn, name="cnn_aprobada")),
        ("lstm_aprobada", lambda: mlflow.keras.log_model(lstm, name="lstm_aprobada")),
        ("random_forest_aprobado", lambda: mlflow.sklearn.log_model(
            rf, name="random_forest_aprobado", code_paths=[str(ROOT / "scripts")]))):
        try:
            funcion()
        except Exception as exc:
            mlflow.set_tag(f"aviso_{nombre}", f"Fallo log_model; archivos nativos guardados: {type(exc).__name__}")
            print(f"Aviso: no se pudo registrar el sabor MLflow {nombre}: {exc}")
    print(f"Aprobado: modelos versionados en {versionado}")
    print(f"Inferencia compatible desde: {compatibles}")


# Función que permite realizar todo el entrenamiento de todos los submodelos.
def entrenar(config: dict, ruta_config: Path) -> None:

    # Configuración de las constantes
    configurar_mlflow(config)
    definir_semilla(int(config["semilla"]), bool(config["entrenamiento"]["determinismo_gpu"]))
    datos, ruta_datos, informacion = cargar_datos(config)
    train, val, test = particionar(datos, config)
    costos = config["costo"]
    limite = float(costos["costo_promedio_maximo"])
    if limite < 0:
        raise ValueError("El costo_promedio_maximo no puede ser negativo.")

    # Entrenamiento de MLFlow
    with mlflow.start_run(run_name="ensamble_modelo_C") as run:
        mlflow.set_tags({"modelo": "C", "estructura": "CNN_1D+LSTM+RandomForest",
                         "moneda_costo": "GTQ", "criterio_aprobacion": "costo_promedio_validacion"})
        mlflow.log_params({"semilla": int(config["semilla"]),
                           "costo_fn": float(costos["falso_negativo"]),
                           "costo_fp": float(costos["falso_positivo"]),
                           "costo_maximo": limite, "n_train": len(train),
                           "n_validacion": len(val), "n_test": len(test),
                           "sha256_datos": informacion["sha256_csv"]})
        mlflow.log_dict(config, "config.json")
        mlflow.log_artifact(str(ruta_config), artifact_path="config_original")
        mlflow.log_dict(informacion, "resumen_datos.json")
        mlflow.log_dict(versiones_paquetes(), "versiones_entorno.json")
        mlflow.log_dict({"entrenamiento": train["tipo_anomalia"].value_counts().to_dict(),
                         "validacion": val["tipo_anomalia"].value_counts().to_dict(),
                         "prueba": test["tipo_anomalia"].value_counts().to_dict()},
                        "distribuciones_particiones.json")

        # Rutas necesarias para el etrenamiento
        for ruta_codigo in [ROOT / "scripts/entrenar_modelos.py", ROOT / "scripts/modelo_c.py",
                            ROOT / "scripts/metricas_costo.py", ROOT / "requirements.txt"]:
            mlflow.log_artifact(str(ruta_codigo), artifact_path="codigo_fuente")

        # Submodelos
        cnn, pipe_cnn, hiper_cnn, val_cnn = buscar_red(
            "cnn", train, val, config, crear_cnn, crear_pipeline_cnn, "fechas_hora")
        lstm, pipe_lstm, hiper_lstm, val_lstm = buscar_red(
            "lstm", train, val, config, crear_lstm, crear_pipeline_lstm, "montos")
        rf, hiper_rf, val_rf = buscar_rf(train, val, config)

        # Registro de métricas
        mlflow.log_dict({"cnn": hiper_cnn, "lstm": hiper_lstm, "rf": hiper_rf},
                        "mejores_hiperparametros.json")
        registrar_metricas(val_cnn, "cnn_validacion")
        registrar_metricas(val_lstm, "lstm_validacion")
        registrar_metricas(val_rf, "rf_validacion")

        # Cálculo de probabilidades de anomalías
        probs_val = probabilidades_ensamble(val, cnn, lstm, rf, pipe_cnn, pipe_lstm)
        scores_val = probs_val.max(axis=1)

        # Ajuste del umbral
        ajuste = config["umbral_probabilidad"]
        if ajuste["modo"] == "optimizar":
            resultado_val, tabla_umbrales = optimizar_umbral(
                val["es_anomalia"], scores_val,
                float(costos["falso_negativo"]), float(costos["falso_positivo"]))
            with tempfile.TemporaryDirectory() as tmp:
                ruta_tabla = Path(tmp) / "costos_por_umbral_validacion.csv"
                tabla_umbrales.to_csv(ruta_tabla, index=False)
                mlflow.log_artifact(str(ruta_tabla), artifact_path="evaluacion")
            umbral = resultado_val["umbral"]
        elif ajuste["modo"] == "fijo":
            umbral = float(ajuste["valor_fijo"])
            if not 0 <= umbral <= 1:
                raise ValueError("El umbral de probabilidad fijo debe pertenecer a [0, 1].")
            resultado_val = evaluar_costo(val["es_anomalia"], scores_val,
                                           umbral, float(costos["falso_negativo"]),
                                           float(costos["falso_positivo"]))
        else:
            raise ValueError("umbral_probabilidad.modo debe ser 'optimizar' o 'fijo'.")


        # Se registran métricas con viejo y nuevo umbral
        registrar_metricas(resultado_val, "validacion")
        registrar_metricas(metricas_clasificacion(val["es_anomalia"], scores_val, umbral), "ensamble_validacion")

        # Se aprueba el modelo si cumple con el límite de costo promedio
        aprobado = resultado_val["costo_promedio"] <= limite
        mlflow.log_metric("modelos_guardados", int(aprobado))
        mlflow.set_tag("aprobacion", "aprobado" if aprobado else "rechazado")
        mlflow.log_dict({"evaluacion_validacion": resultado_val, "aprobado": aprobado,
                         "costo_promedio_maximo": limite}, "evaluacion/criterio_aprobacion.json")
        print(f"Costo promedio en validación: Q{resultado_val['costo_promedio']:.2f}")
        print(f"Umbral de probabilidad elegido: {umbral:.6f}")

        # Métricas del conjunto de prueba
        probs_test = probabilidades_ensamble(test, cnn, lstm, rf, pipe_cnn, pipe_lstm)
        scores_test = probs_test.max(axis=1)
        resultado_test = evaluar_costo(test["es_anomalia"], scores_test,
                                       umbral, float(costos["falso_negativo"]),
                                       float(costos["falso_positivo"]))
        registrar_metricas(resultado_test, "test")
        registrar_metricas(metricas_clasificacion(test["es_anomalia"], scores_test, umbral),
                            "ensamble_test")
        mlflow.log_dict(resultado_test, "evaluacion/evaluacion_test.json")
        mlflow.log_dict({
            "cnn": {
                "test": metricas_clasificacion(
                    seleccionar_clases(test, "transaccion_corta")["es_anomalia"],
                    probs_test[test["tipo_anomalia"].isin(("Normal", "transaccion_corta")).to_numpy(), 0])
            },
            "lstm": {
                "test": metricas_clasificacion(
                    seleccionar_clases(test, "orden_no_congruente")["es_anomalia"],
                    probs_test[test["tipo_anomalia"].isin(("Normal", "orden_no_congruente")).to_numpy(), 1])
            },
            "rf": {
                "test": metricas_clasificacion(
                    seleccionar_clases(test, "establecimiento_raro")["es_anomalia"],
                    probs_test[test["tipo_anomalia"].isin(("Normal", "establecimiento_raro")).to_numpy(), 2])
            },
        }, "evaluacion/metricas_submodelos_test.json")
        print(f"Costo promedio en test: Q{resultado_test['costo_promedio']:.2f}")
        print(f"Estado de la versión: {'APROBADA' if aprobado else 'RECHAZADA'}")

        # Se guardan los modelos aprobados
        if aprobado:
            guardar_aprobados(
                config, run.info.run_id, cnn, lstm, rf, pipe_cnn, pipe_lstm,
                {"run_id": run.info.run_id, "umbral_probabilidad": umbral,
                 "evaluacion_validacion": resultado_val, "evaluacion_test": resultado_test,
                 "sha256_csv": informacion["sha256_csv"],
                 "mejores_hiperparametros": {"cnn": hiper_cnn, "lstm": hiper_lstm, "rf": hiper_rf}},
            )
        else:
            print("No se guardaron modelos. Los parámetros, métricas y logs sí están en MLflow.")
        print(f"MLflow run_id: {run.info.run_id}")

# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del Modelo C con MLflow")
    parser.add_argument("--config", default=str(ROOT / "config.json"),
                        help="Ruta al JSON de semillas, rejillas, unidades y costos.")
    argumentos = parser.parse_args()
    ruta_config = Path(argumentos.config).resolve()
    with ruta_config.open(encoding="utf-8") as archivo:
        config = json.load(archivo)
    entrenar(config, ruta_config)


if __name__ == "__main__":
    main()
