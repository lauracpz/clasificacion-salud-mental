# Clasificación de salud mental con regresión logística

Proyecto académico de aprendizaje supervisado que compara dos modelos de regresión logística para clasificar la etiqueta Depression del conjunto de datos Exploring Mental Health Data. El modelo base usa tres variables y el modelo completo usa diez.

El [análisis del problema](docs/analisis-del-problema.md) explica el objetivo, las variables, la preparación de datos, la interpretación del modelo y sus limitaciones. La [guía del código](docs/explicacion_linea_por_linea.md) describe la estructura del programa módulo por módulo.

> **Uso responsable:** este proyecto es académico. Sus predicciones no son diagnósticos ni deben usarse para tomar decisiones clínicas o personales.

## Qué hace

- Comprueba que el CSV contenga las columnas necesarias.
- Unifica presión y satisfacción académica o laboral en dos variables comunes.
- Codifica algunas respuestas categóricas y transforma los rangos de duración del sueño a valores numéricos aproximados.
- Elimina las filas incompletas para las variables del análisis.
- Divide los datos de forma estratificada: 80 % para entrenamiento y 20 % para validación.
- Compara un modelo base de tres variables con un modelo completo de diez variables.
- Escala los predictores dentro de cada modelo y calcula accuracy, balanced accuracy, precisión, recall, F1 y ROC AUC.
- Guarda las métricas, los coeficientes estandarizados y las matrices de confusión.

## Estructura

~~~text
clasificacion-salud-mental/
├── data/
│   └── raw/                         # train.csv local; no se sube a Git
├── docs/
│   ├── analisis-del-problema.md
│   └── explicacion_linea_por_linea.md
├── outputs/                         # archivos generados; no se suben a Git
├── src/
│   └── mental_health_classification/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py                   # argumentos y punto de entrada
│       └── models.py                # preparación, entrenamiento y evaluación
├── .gitignore
├── pyproject.toml
└── README.md
~~~

## Datos

El programa usa train.csv de [Exploring Mental Health Data, Kaggle Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11). Descarga el archivo desde Kaggle y guárdalo como data/raw/train.csv. El CSV se mantiene local y no se redistribuye como parte de este repositorio.

El programa necesita la etiqueta Depression, codificada como 0 o 1. Ignora las columnas que no forman parte de los predictores definidos, como nombres, ciudad, profesión, titulación o hábitos alimentarios.

## Instalación y ejecución

Se requiere Python 3.10 o posterior. Desde la carpeta del repositorio, crea un entorno virtual e instala el proyecto.

~~~powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
~~~

En macOS o Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
~~~

Con train.csv en la ruta predeterminada, ejecuta el paquete:

~~~powershell
py -m mental_health_classification
~~~

También puedes indicar otra ruta para los datos o los resultados:

~~~powershell
py -m mental_health_classification --data "ruta\train.csv" --output-dir "outputs"
~~~

El programa guarda metricas.csv, coeficientes.csv y matrices_confusion.png en outputs/. Los resultados se generan localmente y no se incluyen en Git.

También se instala el comando clasificacion-salud-mental, que ejecuta el mismo flujo.

## Interpretación

- **Accuracy** es la proporción total de predicciones correctas.
- **Balanced accuracy** promedia el acierto obtenido en cada clase.
- **Precisión** indica qué proporción de las predicciones positivas fue correcta.
- **Recall** indica qué proporción de los positivos reales detectó el modelo.
- **F1** resume precisión y recall.
- **ROC AUC** mide la capacidad de ordenar casos positivos por encima de casos negativos a través de distintos umbrales.

Los coeficientes guardados corresponden al modelo completo y a predictores escalados. Su signo indica una asociación dentro del modelo; no demuestra causalidad. La magnitud no se debe interpretar como una importancia clínica.

## Limitaciones

La evaluación se realiza con una sola partición del archivo de entrenamiento. No incluye validación externa, validación cruzada, calibración de probabilidades ni selección del umbral. El rendimiento puede cambiar en otras muestras o poblaciones. El proyecto no constituye una herramienta clínica.
