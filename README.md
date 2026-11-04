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




### Usuario
El sistema antifraude del Banco está construido sobre variables agregadas por ventana de tiempo: 

  -Monto promedio de las últimas 24 horas. 

  
  -Número de transacciones por hora.

  
  -Monto máximo del día.

  
  -Diversidad de comercios.


### Entorno
**El conjunto de datos**: 100,000 datos, de los cuales 89,936 eran transacciones normales, 3,994 eran de anomalías debido al orden no congruente. 
3,611 eran de anomalías debido a transacciones cortas (muchas transacciones en poco tiempo) y 2,459 eran anomalías debido a compras en establecimientos no comunes para el cliente.

**Datos de entrenamiento**: las 30 transacciones más recientes de cada cliente. Para cada transacción se contaba con el monto, la fecha y hora y el tipo de establecimiento en el que fue registrada la transacción. 


## Arquitectura del proyecto

```text
PROYECTO_4_DL/
├── datos/
│   └── datos_sinteticos.csv
├── docs/
├── logs/
├── mlartifacts/
├── modelos_base/
├── modelos/
├── reportes/
├── resultados/
├── notebooks/
├── scripts/
│   └── modelo_c.py
├── app.py
├── README.md
├── Dockerfile
├── compose.yaml
└── requirements.txt
```


## Cambios del modelo

No se realizarán cambios en el modelo, esto debido a que el modelo ganador del proyecto 1, fue creado específicamente para predecir eficientemente cada tipo de anomalía considerada en los datos sintéticos. El modelo fue considerado exitoso, por lo que no será reentrenado ni se agregarán nuevas componentes (ya que tampoco se agregarán nuevas anomalías).


## Hipótesis



## Diagrama del sistema


## Levantar el repositorio


## Actualización del modelo

