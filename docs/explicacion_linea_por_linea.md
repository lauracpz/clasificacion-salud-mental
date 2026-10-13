# Explicación del código por módulos

El proyecto sigue una estructura de paquete. La preparación y evaluación están en src/mental_health_classification/models.py; las opciones de ejecución están en cli.py; y __main__.py permite iniciar el programa como un módulo de Python.

## __main__.py: ejecutar el paquete

- La importación de main trae el punto de entrada definido en cli.py.
- La condición if __name__ == "__main__" comprueba que el paquete se esté ejecutando directamente.
- La llamada a main() inicia la lectura de argumentos, carga el CSV y coordina el experimento.

## cli.py: rutas y flujo de ejecución

### Imports y ruta del proyecto

- argparse permite aceptar opciones como --data y --output-dir.
- Path representa las rutas de los archivos de forma compatible entre Windows, macOS y Linux.
- pandas lee el CSV en una tabla de datos.
- Desde models.py se importan la columna objetivo y las funciones de preparación y entrenamiento.
- PROJECT_ROOT calcula la raíz del repositorio partiendo de la ubicación de cli.py. Así los valores predeterminados no dependen de la carpeta desde la que se lanza el comando.

### parse_arguments()

- Crea un analizador para las opciones de la terminal.
- --data permite elegir el CSV. Si se omite, se usa data/raw/train.csv.
- --output-dir permite elegir la carpeta para los resultados. Si se omite, se usa outputs/.
- type=Path convierte las rutas recibidas a objetos de ruta.
- parse_args() lee las opciones y las devuelve en un objeto con atributos data y output_dir.

### main()

1. Obtiene las rutas con parse_arguments().
2. Comprueba que el archivo de datos exista; si falta, muestra un error con la ruta esperada.
3. Lee el archivo con pd.read_csv().
4. Llama a prepare_dataset() para validar, codificar y limpiar las columnas.
5. Calcula e imprime cuántas filas se descartaron.
6. Muestra la distribución porcentual de las clases 0 y 1.
7. Llama a run_experiment() para entrenar, evaluar y guardar los resultados.

## models.py: análisis y modelos

### Imports y parámetros

- re interpreta las expresiones de duración del sueño.
- Path tipa la carpeta de salida.
- Matplotlib y Seaborn construyen el gráfico de matrices de confusión.
- NumPy y pandas gestionan valores y tablas.
- Scikit-learn proporciona la regresión logística, la partición, la tubería de preprocesamiento y las métricas.
- TARGET identifica la columna que el modelo debe predecir.
- RANDOM_STATE fija la semilla para que la partición sea reproducible.
- VALIDATION_SIZE reserva el 20 % de las filas para validación.
- DECISION_THRESHOLD convierte probabilidades en predicciones de clase; 0,50 es el umbral usado por este programa.
- BASELINE_FEATURES lista los tres predictores del modelo base.
- ALL_FEATURES lista los diez predictores del modelo completo.
- FEATURE_SETS asocia un nombre legible a cada lista de predictores.
- REQUIRED_COLUMNS define las columnas originales que deben existir en el CSV.
- CATEGORY_MAPPINGS transforma las categorías de texto seleccionadas en códigos numéricos.

### map_sleep_duration(value)

1. Si el valor está ausente, devuelve NaN.
2. Si ya es un número, lo convierte a decimal y lo conserva.
3. Si es texto, elimina espacios sobrantes y pasa el texto a minúsculas.
4. Busca expresiones como “more than N hours”, “less than N hours” o “N-M hours”.
5. En un rango N-M, el código usa N, el límite inferior. Para “more than N” usa N + 1; para “less than N”, N - 1.
6. Si no reconoce el formato, devuelve NaN, que después se trata como dato ausente.

### prepare_dataset(raw_data)

1. Compara las columnas del archivo con REQUIRED_COLUMNS y TARGET.
2. Si falta alguna, detiene el proceso con un mensaje que indica los nombres.
3. Copia la tabla para no modificar el objeto de entrada.
4. Construye Work/Study Pressure con el valor numérico disponible entre presión académica y laboral.
5. Construye Work/Study Satisfaction de forma equivalente con satisfacción académica y laboral.
6. Normaliza y codifica género, situación laboral o académica, pensamientos suicidas e historial familiar.
7. Convierte la duración del sueño y la etiqueta a valores numéricos.
8. Marca como ausentes las etiquetas que no sean 0 o 1 y convierte los predictores a números.
9. Conserva las diez variables y la etiqueta; elimina filas con valores ausentes.
10. Si no queda ninguna fila, lanza un error; en caso contrario, devuelve la tabla preparada.

### build_model()

- Crea una Pipeline con StandardScaler seguido de LogisticRegression.
- La tubería ajusta el escalador únicamente con los datos de entrenamiento cuando se llama a fit.
- max_iter=2_000 permite más iteraciones para que el algoritmo encuentre una solución.
- RANDOM_STATE hace repetible la inicialización aleatoria cuando el estimador la usa.

### evaluate_model(model, features, target, feature_columns)

1. Calcula la probabilidad estimada para la clase positiva con predict_proba().
2. Convierte las probabilidades a 0 o 1 usando DECISION_THRESHOLD.
3. Construye la matriz de confusión en el orden de clases 0 y 1.
4. Calcula accuracy, balanced accuracy, precisión, recall, F1 y ROC AUC.
5. Devuelve el diccionario de métricas y la matriz.

ROC AUC usa las probabilidades, no las predicciones ya convertidas a 0 y 1. zero_division=0 define el valor de precisión o recall cuando una clase no recibe predicciones.

### save_confusion_matrices(matrices, output_dir)

- Crea un panel por modelo.
- Recorre los paneles y las matrices recibidas.
- Dibuja cada matriz como mapa de calor con los conteos y etiquetas de las dos clases.
- Añade títulos y nombres de ejes.
- Guarda matrices_confusion.png y cierra la figura para liberar memoria.

### run_experiment(dataset, output_dir)

1. Separa predictores y etiqueta.
2. Comprueba que existan dos clases y que haya al menos dos observaciones de cada una.
3. Divide los datos en entrenamiento y validación con una partición estratificada.
4. Comprueba que la validación contenga ambas clases, requisito para ROC AUC.
5. Recorre los dos conjuntos de variables. Para cada uno crea una tubería y la ajusta solo con los datos de entrenamiento.
6. Evalúa cada modelo con los mismos datos de validación y guarda sus métricas y matriz de confusión.
7. Crea la carpeta de salida si no existe y guarda las métricas en metricas.csv.
8. Guarda el gráfico de matrices de confusión.
9. Extrae del modelo completo los coeficientes de la regresión logística. Como el modelo incluye escalado, estos coeficientes están asociados a predictores estandarizados.
10. Ordena los coeficientes por magnitud absoluta y los guarda en coeficientes.csv.
11. Imprime tamaños de muestra, umbral, métricas y ruta de salida.
12. Devuelve la tabla de métricas.

## Ejecución

El punto de entrada __main__.py permite ejecutar el paquete desde la raíz del repositorio:

~~~powershell
py -m mental_health_classification
~~~

También se pueden cambiar las rutas:

~~~powershell
py -m mental_health_classification --data "ruta\train.csv" --output-dir "outputs"
~~~

Las métricas y los coeficientes describen asociaciones dentro del conjunto de datos y del ajuste realizado. No demuestran causalidad ni constituyen un diagnóstico.
