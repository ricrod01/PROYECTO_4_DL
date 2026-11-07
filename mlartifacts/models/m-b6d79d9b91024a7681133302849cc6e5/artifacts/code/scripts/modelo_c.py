import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

ruta_modelos = '../modelos_base/'


def convertir_montos(datos):

    # Se genera una serie.
    if isinstance(datos, pd.DataFrame):
        columna = datos.iloc[:, 0]
    else:
        columna = datos

    # Se separan los datos para convertirlos en listas de números.
    montos_separados = (columna.astype('string').str.strip().str.slice(1, -1).str.replace("'", '', regex=False).str.split(r',\s*', expand=True, regex=True))

    return montos_separados.to_numpy(dtype=np.float32)


class EscaladorGlobalSecuencias(BaseEstimator, TransformerMixin):

    # Entrenador del modelo.
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)

        self.media_ = X.mean()
        self.desviacion_ = X.std(ddof=0)

        if self.desviacion_ == 0:
            self.desviacion_ = 1.0

        return self

    # Transformador del modelo.
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.media_) / self.desviacion_).astype(np.float32)
    

def convertir_fechas_a_minutos(columna_fechas):
    # Separa las 30 fechas en columnas.
    fechas_texto = (columna_fechas.astype('string').str.strip().str.slice(1, -1).str.replace("'", '', regex=False).str.split(r',\s*', expand=True, regex=True))

    # El formato explícito evita que Pandas intente inferirlo.
    fechas_datetime = fechas_texto.apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='raise')

    # Matriz de numpy.
    fechas_numpy = fechas_datetime.to_numpy(dtype="datetime64[ns]")

    # Diferencia en minutos entre transacciones consecutivas.
    diferencias_minutos = (np.diff(fechas_numpy, axis=1) / np.timedelta64(1, "m"))

    return diferencias_minutos.astype(np.float32)


ESTABLECIMIENTOS = ['Supermercado', 'Restaurante', 'Gasolinera', 'Farmacia', 'Ropa',
                    'Tecnología', 'Entretenimiento', 'Transporte', 'Otros']

def calcular_diferencias_establecimientos(datos):

    datos = datos.reset_index(drop=True).copy()

    # Convierte el string en una lista de establecimientos.
    tipos = (datos['tipos_establecimiento'].astype('string').str.strip().str.slice(1, -1).str.replace("'", '', regex=False).str.split(r",\s*", regex=True))

    # Lleva las listas a formato largo.
    tipos_largos = tipos.explode().str.strip()

    # Cuenta cuántas veces aparece cada establecimiento por fila.
    frecuencias_actuales = pd.crosstab(index=tipos_largos.index, columns=tipos_largos)

    # Garantiza el orden exacto de las nueve categorías.
    frecuencias_actuales = frecuencias_actuales.reindex(index=datos.index, columns=ESTABLECIMIENTOS, fill_value=0)

    # Convierte los conteos en frecuencias relativas.
    cantidades_transacciones = tipos.str.len()
    frecuencias_actuales = frecuencias_actuales.div(cantidades_transacciones, axis=0)

    # Convierte el histórico de string a una matriz numérica.
    frecuencias_historicas = (datos['historico_establecimiento'].astype('string').str.strip().str.slice(1, -1).str.replace("'", "", regex=False).str.split(r",\s*", expand=True, regex=True))
    frecuencias_historicas = frecuencias_historicas.astype(np.float32)
    frecuencias_historicas.columns = ESTABLECIMIENTOS

    # Diferencias absolutas entre frecuencia actual e histórica.
    diferencias = np.abs(frecuencias_actuales.to_numpy(dtype=np.float32) - frecuencias_historicas.to_numpy(dtype=np.float32))
    nombres_columnas = [f'diferencia_{establecimiento.lower()}' for establecimiento in ESTABLECIMIENTOS]

    # DataFrame con la diferencia de frecuencias.
    df_nuevo = pd.DataFrame(diferencias, columns=nombres_columnas, index=datos.index, dtype=np.float32)

    # Se guardan los históricos.
    for x in range(len(ESTABLECIMIENTOS)):
        df_nuevo['historico_' + ESTABLECIMIENTOS[x]] = frecuencias_historicas[ESTABLECIMIENTOS[x]]
    
    return df_nuevo

# Compatibilidad de las funciones con los nombres.
def registrar_compatibilidad_joblib():
    import __main__

    elementos = {'convertir_fechas_a_minutos': convertir_fechas_a_minutos,
                 'convertir_montos': convertir_montos,
                 'EscaladorGlobalSecuencias': EscaladorGlobalSecuencias,
                 'calcular_diferencias_establecimientos': calcular_diferencias_establecimientos}

    for nombre, elemento in elementos.items():
        setattr(__main__, nombre, elemento)

# Modelo.
class ModeloEnsambladoFraude:

    CLASES = np.array(['transaccion_corta', 'orden_no_congruente', 'establecimiento_raro'])

    def __init__(self, umbral=0.5, batch_size=256):
        
        self.umbral = float(umbral)
        self.batch_size = int(batch_size)

        # Compatibilidad con los joblib guardados desde el notebook.
        registrar_compatibilidad_joblib()

        # Modelos y pipelines.
        self.modelo_cnn = tf.keras.models.load_model(f'{ruta_modelos}modelo_c_cnn.keras', compile=False)
        self.modelo_lstm = tf.keras.models.load_model(f'{ruta_modelos}modelo_c_lstm.keras', compile=False)
        self.pipeline_cnn = joblib.load(f'{ruta_modelos}pipeline_c_cnn.joblib')
        self.pipeline_lstm = joblib.load(f'{ruta_modelos}pipeline_c_lstm.joblib')
        self.modelo_rf = joblib.load(f'{ruta_modelos}modelo_c_rf.joblib')

    def set_umbral(self, umbral_nuevo):
        self.umbral = float(umbral_nuevo)

    def predict_proba(self, datos):

        # CNN
        X_cnn = self.pipeline_cnn.transform(datos['fechas_hora'])
        X_cnn = (X_cnn.astype(np.float32)[..., np.newaxis])
        probabilidad_cnn = self.modelo_cnn.predict(X_cnn, batch_size=self.batch_size, verbose=0).reshape(-1)

        # LSTM
        X_lstm = self.pipeline_lstm.transform(datos['montos'])
        X_lstm = (X_lstm.astype(np.float32)[..., np.newaxis])
        probabilidad_lstm = self.modelo_lstm.predict(X_lstm, batch_size=self.batch_size, verbose=0).reshape(-1)

        # Random Forest
        X_rf = datos[['tipos_establecimiento', 'historico_establecimiento']]
        probabilidades_rf_completas = (self.modelo_rf.predict_proba(X_rf))

        # Busca explícitamente la posición de la clase positiva.
        indice_clase_positiva = np.where(self.modelo_rf.classes_ == 1)[0]
        probabilidad_rf = probabilidades_rf_completas[:, indice_clase_positiva[0]]

        return np.column_stack([probabilidad_cnn, probabilidad_lstm, probabilidad_rf]).astype(np.float32)

    def predict(self, datos):

        probabilidades = self.predict_proba(datos)

        # Etiqueta binaria de cada submodelo.
        etiquetas_individuales = (probabilidades >= self.umbral).astype(np.int32)

        # Modelo que produjo la probabilidad más alta.
        indice_mayor = np.argmax(probabilidades, axis=1)
        probabilidad_mayor = np.max(probabilidades, axis=1)

        # 0 representa Normal.
        codigo_final = np.where(probabilidad_mayor >= self.umbral, indice_mayor + 1, 0)
        etiquetas_finales = np.full(len(datos), 'Normal', dtype=object)

        hay_anomalia = codigo_final != 0

        etiquetas_finales[hay_anomalia] = self.CLASES[codigo_final[hay_anomalia] - 1]

        return pd.DataFrame(
            {
                'prob_transaccion_corta': probabilidades[:, 0],
                'etiqueta_transaccion_corta': etiquetas_individuales[:, 0],

                'prob_orden_no_congruente': probabilidades[:, 1],
                'etiqueta_orden_no_congruente': etiquetas_individuales[:, 1],

                'prob_establecimiento_raro': probabilidades[:, 2],
                'etiqueta_establecimiento_raro': etiquetas_individuales[:, 2],

                'probabilidad_mayor': probabilidad_mayor,
                'codigo_final': codigo_final,
                'prediccion_final': etiquetas_finales
            },
            index=datos.index)