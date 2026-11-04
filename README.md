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

