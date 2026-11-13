# PROYECTO_4_DL


## Problemática

La identificación de anomalías en transacciones es fundamental para evitar el descontento de los clientes, evitar daños en la reputación y el ahorro de costos debido al extorno de la suma total o parcial de las anomalías.


Actualmente, no se cuenta con ninguna forma de evitar frenar las transacciones fraudulentas antes de que el cliente se diera cuenta y notificara, haciendo que sienta poca seguridad hacia la empresa y provocando abandono de los productos que maneja en esta entidad. Por lo cual, se debe realizar la implementación de algún modelo que permita reconocer las anomalías en transacciones con el fin de minimizar cualquier tipo de pérdida que estas puedan generar.


## Supuestos

### Modelos
Para este proyecto se ha usado como base el modelo C del Proyecto 1 del curso de Deep Learning. Este modelo se divide en tres partes:


**Red CNN-1D:** identifica las anomalías de transacciones cortas.


**Red LSTM:** identifica las anomalías en el orden de los montos de las transacciones.


**Random Forest:** identifica las anomalías por tipo de negocio.



### Datos
Cada fila del conjunto de datos es una transacción identificada con el ID del cliente. Se tienen clientes con transacciones variables en el año de datos elegido para el modelo, de manera que solo se analizarán los clientes con 30 transacciones; en caso de que haya clientes con un número mayor, se seleccionarán las últimas 30 transacciones.


Datos de transacciones que incluyen las variables: 


  -Fecha 

  
  -Hora

  
  -Monto de la transacción 

  
  -Tipo de negocio ("Supermercado", "Restaurante", "Gasolinera", "Farmacia", "Ropa", "Tecnología", "Entretenimiento", "Transporte", "Otros")

### Usuario
El usuario coloca un formato válido de *CSV* al modelo y le coloca las columnas necesarias en el formato de los supuestos de datos descritos anteriormente.

### Entorno

-Se tiene la capacidad computacional necesaria para correr el modelo.


-Las computadoras generan el ambiente del Docker.

## Arquitectura del proyecto

```text
PROYECTO_4_DL/
├── datos/
│   └── datos_sinteticos.csv
├── docs/
├── logs/
├── mlartifacts/
├── modelos/
├── modelos_base/
├── notebooks/
├── scripts/
│   └── modelo_c.py
├── app.py
├── mlflow.db
├── PLANIFICACION.md
├── README.md
├── Dockerfile
├── compose.yaml
└── requirements.txt
```


## Cambios del modelo

No se realizarán cambios en el modelo, esto debido a que el modelo ganador del proyecto 1, fue creado específicamente para predecir eficientemente cada tipo de anomalía considerada en los datos sintéticos. El modelo fue considerado exitoso, por lo que no será reentrenado ni se agregarán nuevas componentes (ya que tampoco se agregarán nuevas anomalías).


## Hipótesis

Creemos que un modelo que implemente:


  -Una red CNN-1D para identificar las transacciones cortas.

  
  -Una red LSTM para identificar aquellas anomalías en el orden de las transacciones.

  
  -Random Forest que permite identificar las anomalías por tipo de negocio.

mejorará la predicción de las anomalías, debido a que cada sub-modelo estará especializado en un tipo de anomalía porque es más específico y aprende a reconocer los patrones que surgen de las anomalías consideradas.

## Diagrama del sistema


## Levantar el contenedor

Para construir la imagen del contenedor.

```bash
docker compose build
```

## Actualización del modelo

### Requisitos

- Python **3.13.7 de 64 bits** (recomendado).
- Ubicar el CSV real en `datos/datos_sinteticos.csv` (separador `;`) o editar `datos.ruta_csv` en `config.json`.
- Columnas mínimas: `fechas_hora`, `montos`, `tipos_establecimiento`,
`historico_establecimiento`, `tipo_anomalia`, `es_anomalia`.
- Cada registro debe tener 30 fechas, 30 montos, 30 establecimientos y un histórico de nueve frecuencias, serializados como listas.

### Ejecutar

```bash
docker compose up -d mlflow
docker compose run --rm entrenamiento
```

Interfaz de MLflow: <http://localhost:5000>

Para volver a entrenar, modificar `scripts/config.json` y volver a ejecutar:

```powershell
docker compose run --rm entrenamiento
```

Ver registros y detener el servidor:

```powershell
docker compose logs -f mlflow
docker compose
```

### Configuración

- `semilla`: controla las particiones, la inicialización de redes y Random Forest.
- `cnn.rejilla`: cambia filtros y **unidades de la capa densa**, tasas de aprendizaje,
  dropout y batch size. Las dos capas Conv1D quedan fijas.
- `lstm.rejilla`: cambia **nodos de ambas capas LSTM** y de la capa densa. Las dos
  capas LSTM quedan fijas.
- `rf.rejilla`: parámetros de `GridSearchCV` de sklearn (con prefijo
  `random_forest__`). El ejemplo reproduce los valores del notebook original.
- `entrenamiento`: épocas máximas, EarlyStopping y reducción de learning rate.
- `umbral_probabilidad.modo`: `optimizar` busca el corte de menor costo sobre
  validación; `fijo` usa `valor_fijo` entre 0 y 1.
- `costo.falso_negativo`: Q4,200 y `costo.falso_positivo`: Q180, según el notebook.
- `costo.costo_promedio_maximo`: **Q15** (valor inicial). La aprobación ocurre si
  el costo en validación es **menor o igual** a Q15. No significa 15 % ni probabilidad 15.

**Fórmula:** `(FN * 4200 + FP * 180) / N` (quetzales por registro evaluado).

### Separación de datos y selección

1. División estratificada por `tipo_anomalia`: 20 % test, 16 % validación y 64 % train.
2. Cada submodelo entrena solo contra ejemplos `Normal` y su tipo de anomalía,
   tal como el notebook original; el preprocesamiento se ajusta exclusivamente a train.
3. Cada combinación CNN/LSTM genera un **run hijo** en MLflow; se elige la mejor
   por PR-AUC de validación. La rejilla RF usa 5 pliegues estratificados y PR-AUC.
4. Se combinan las tres probabilidades con `max`, sin aprender un meta-modelo.
5. El umbral de decisión se optimiza o fija **usando validación**.
6. Solo si `(costo_promedio_validacion <= costo_promedio_maximo)`, se guardan los
   tres modelos y los dos pipelines. **Test no interviene en esta autorización.**
7. Test se evalúa una única vez con el umbral ya seleccionado para reportar resultados.

> Precaución: al usar una única validación para elegir hiperparámetros y el umbral,
> puede existir optimismo en la métrica de validación; la medida de generalización
> es el conjunto de prueba. Con datos reales se recomienda validación temporal/por
> cliente y una partición de calibración adicional para evitar fuga entre historiales.

### Evidencias guardadas

MLflow persistirá en `mlflow.db` (backend) y `mlartifacts/` (artefactos):

- Run principal por entrenamiento, y runs hijos de cada combinación CNN/LSTM y de RF.
- Archivo `config.json` original, código fuente, `requirements.txt`, versiones de
  dependencias, SHA-256 del CSV, tamaño y distribución de los splits.
- Hiperparámetros, pesos de clase, pérdidas y métricas por época, historiales JSON.
- Tabla CSV de todas las combinaciones de RF y, para umbral optimizado, tabla CSV
  de costos por umbral. No se generan gráficos.
- Métricas PR-AUC, ROC-AUC, precisión, recall, F1, costo total, costo medio y
  conteos TP/TN/FP/FN en validación y prueba; etiqueta `aprobado` o `rechazado`.

**Si aprueba**, se guardan además:

```text
modelos/<mlflow_run_id>/
    modelo_c_cnn.keras
    modelo_c_lstm.keras
    pipeline_c_cnn.joblib
    pipeline_c_lstm.joblib
    modelo_c_rf.joblib
    metadata_entrenamiento.json
```

La misma versión aprobada se publica en `modelos_base/`, manteniendo la compatibilidad
con `scripts/modelo_c.py`. Los artefactos también aparecen dentro del run en
`modelos_aprobados/`, y se loguean los modelos nativos de MLflow por separado.

**Si falla la aprobación**, los experimentos y métricas **sí** quedan registrados
pero **no se crean ni se reemplazan archivos de modelos**.

### Predicción de la versión aprobada

```python
from scripts.modelo_c import ModeloEnsambladoFraude
import json
import pandas as pd

with open("modelos_base/metadata_entrenamiento.json", encoding="utf-8") as f:
    metadata = json.load(f)

modelo = ModeloEnsambladoFraude(
    umbral=metadata["umbral_probabilidad"],
    directorio_modelos="modelos_base",
)
datos = pd.read_csv("datos/datos_sinteticos.csv", sep=";")
resultado = modelo.predict(datos)
print(resultado[["probabilidad_mayor", "prediccion_final"]].head())
```

El umbral óptimo puede ser ligeramente mayor que 1.0 si la política de menor
costo consiste en no emitir ninguna alerta; es un valor de corte interno válido
cuando las probabilidades están en [0,1].


## Aplicación

Desde la raíz del repositorio:

```bash
docker compose up -d aplicacion
Start-Process "http://localhost:8501"
```

Abrir <http://localhost:8501>.

Si se desea abrir en línea:
Abror <https://proyecto4dl-lxeuudguyfpka8dpapymw2.streamlit.app>


## Uso de Inteligencia Artificial

## Declaración del uso de Inteligencia Artificial

Durante el desarrollo del proyecto **Monitor de Anomalías en Transacciones**, se utilizaron herramientas de Inteligencia Artificial (IA) generativa como apoyo en distintas etapas de programación, implementación y despliegue de la solución.

El uso de IA se concentró principalmente en las siguientes actividades:

1. **Generación y transformación de datos:** apoyo en la elaboración de código Python para generar datos, realizar procesos de limpieza, transformación y preparación de la información utilizada en el proyecto.

2. **Flujo de entrenamiento de modelos:** asistencia en la estructuración y programación del flujo de entrenamiento, incluyendo el preprocesamiento de datos, la configuración de modelos de aprendizaje automático y aprendizaje profundo, así como los procedimientos de entrenamiento y evaluación.

3. **Desarrollo del tablero interactivo:** apoyo en la implementación de la aplicación web mediante Streamlit, incluyendo la construcción de componentes visuales, la carga de archivos, la presentación de resultados y la integración de los modelos predictivos.

4. **Contenerización de la aplicación:** asistencia en la configuración de Docker, mediante la creación y adaptación de los archivos necesarios para empaquetar la aplicación, gestionar sus dependencias y facilitar su ejecución en un entorno reproducible.

La Inteligencia Artificial se utilizó como una **herramienta de asistencia técnica para la generación, revisión y optimización de código**. Su participación contribuyó a agilizar el desarrollo y la resolución de problemas de implementación.