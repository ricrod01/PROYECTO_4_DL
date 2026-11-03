# PROYECTO_4_DL


## Problemática

La identificación de anomalías en transacciones es fundamental para evitar el descontento de los clientes, evitar daños en la reputación y el ahorro de costos debido al extorno de la suma total o parcial de las anomalías.


Actualmente, no se cuenta con ninguna forma de evitar frenar las transacciones fraudulentas antes de que el cliente se diera cuenta y notificara, haciendo que sienta poca seguridad hacia la empresa y provocando abandono de los productos que maneja en esta entidad. Por lo cual, se debe realizar la implementación de algún modelo que permita reconocer las anomalías en transacciones con el fin de minimizar cualquier tipo de pérdida que estas puedan generar.


## Supuestos

### Modelos
**Random Forest**:


**CNN**:


**LSTM**: Los datos están estructurados en un orden definido.



### Usuario

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



## Hipótesis



## Diagrama del sistema


## Levantar el repositorio


## Actualización del modelo

