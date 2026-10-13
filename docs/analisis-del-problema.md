# Análisis del problema de clasificación de depresión

## 1. Objetivo y alcance

El objetivo es estimar la probabilidad de que una observación tenga la etiqueta Depression y clasificarla como 0 (no) o 1 (sí). Se usa regresión logística, un método supervisado apropiado para una respuesta binaria y que permite examinar la dirección de las asociaciones entre predictores y etiqueta.

El proyecto tiene un propósito académico. Las predicciones describen patrones del conjunto de datos y no son diagnósticos; las asociaciones estimadas no demuestran causalidad.

## 2. Datos y variables

La implementación usa train.csv del conjunto [Exploring Mental Health Data, Kaggle Playground Series S4E11](https://www.kaggle.com/competitions/playground-series-s4e11). La etiqueta objetivo es Depression.

| Variable | Significado y codificación |
| --- | --- |
| Gender | Género; Female = 0, Male = 1. |
| Age | Edad, convertida a valor numérico. |
| Working Professional or Student | Situación laboral o académica; Student = 0, Working Professional = 1. |
| Have you ever had suicidal thoughts ? | Pensamientos suicidas declarados; No = 0, Yes = 1. |
| Work/Study Hours | Horas diarias de trabajo o estudio, convertidas a valor numérico. |
| Financial Stress | Nivel numérico de estrés financiero. |
| Family History of Mental Illness | Antecedentes familiares; No = 0, Yes = 1. |
| Sleep Duration | Duración del sueño, convertida a horas aproximadas cuando aparece como texto. |
| Work/Study Pressure | Presión académica o laboral. |
| Work/Study Satisfaction | Satisfacción académica o laboral. |

Se omiten campos de alta cardinalidad o que no son necesarios para la comparación, como nombre, ciudad, titulación, profesión y hábitos alimentarios.

## 3. Preparación de los datos

- Se normalizan los textos categóricos para que diferencias de mayúsculas y espacios no afecten a la codificación.
- Las respuestas categóricas seleccionadas se convierten a 0 y 1.
- Si hay presión académica y laboral en columnas separadas, se toma el primer valor numérico disponible para formar Work/Study Pressure. Se aplica el mismo criterio para combinar satisfacción académica y laboral.
- Las duraciones del sueño ya numéricas se conservan. Para un rango de texto N-M se usa N, el límite inferior; “more than N hours” se convierte en N + 1 y “less than N hours” en N - 1.
- Las etiquetas que no sean 0 o 1, las categorías desconocidas y los valores numéricos inválidos se tratan como ausentes.
- Se eliminan las filas que tengan un valor ausente en cualquiera de las diez variables o en Depression.

## 4. Modelo y protocolo de evaluación

La regresión logística estima la probabilidad de la clase positiva mediante la función logística:

~~~text
P(Depression = 1 | X) = 1 / (1 + exp(-(beta_0 + beta_1*x_1 + ... + beta_k*x_k)))
~~~

Cada beta expresa cómo cambia el log-odds estimado al variar un predictor, manteniendo los demás constantes. El signo positivo indica una asociación con mayores log-odds; el signo negativo, con menores log-odds. La escala del predictor afecta a la magnitud del coeficiente.

El código compara dos configuraciones:

- **Modelo base:** pensamientos suicidas, duración del sueño y estrés financiero.
- **Modelo completo:** las diez variables de la tabla anterior.

Para cada modelo, el programa separa 80 % de los datos para entrenamiento y 20 % para validación, conserva aproximadamente la proporción de clases mediante una partición estratificada y fija la semilla aleatoria en 42. El escalado y la regresión logística se ajustan dentro de una tubería: así, el escalador aprende solo de los datos de entrenamiento.

El umbral de clasificación es 0,5. El programa calcula accuracy, balanced accuracy, precisión, recall, F1, ROC AUC y matrices de confusión. Las métricas y los coeficientes de una ejecución se guardan en outputs/.

## 5. Lectura de resultados

### Asociaciones estimadas

En un ajuste con las diez variables en sus escalas originales, las direcciones estimadas fueron:

| Variable | Coeficiente | Interpretación de la dirección |
| --- | ---: | --- |
| Gender | 0,05 | Asociación positiva pequeña con la categoría codificada como 1. |
| Age | -0,17 | Asociación negativa. |
| Working Professional or Student | -1,91 | La categoría 1 (profesional) tiene menor log-odds estimado que la categoría 0 (estudiante), manteniendo los demás predictores constantes. |
| Have you ever had suicidal thoughts ? | 2,48 | Asociación positiva marcada en ese ajuste. |
| Work/Study Hours | 0,13 | Asociación positiva pequeña. |
| Financial Stress | 0,55 | Asociación positiva. |
| Family History of Mental Illness | 0,23 | Asociación positiva. |
| Sleep Duration | -0,13 | Asociación negativa. |
| Work/Study Pressure | 0,72 | Asociación positiva. |
| Work/Study Satisfaction | -0,40 | Asociación negativa. |

Estos coeficientes describen un ajuste con predictores sin estandarizar. El programa actual vuelve a entrenar los modelos después de escalar las variables, por lo que sus coeficientes pueden ser distintos. No se deben comparar coeficientes de escalas diferentes como si sus magnitudes fueran directamente equivalentes.

### Efecto del umbral

En una evaluación con partición 70/30 se observó el siguiente intercambio entre acierto de cada clase y accuracy:

| Umbral | Acierto en la clase 0 | Acierto en la clase 1 | Accuracy |
| ---: | ---: | ---: | ---: |
| 0,5 | 96,75 % | 79,66 % | 93,59 % |
| 0,2 | 91,70 % | 91,82 % | 91,72 % |

Al bajar el umbral, más casos positivos superan la regla de clasificación: aumenta el recall de la clase 1, aunque se reduce el acierto de la clase 0 y la accuracy global. La elección del umbral depende del coste de falsos negativos frente al de falsos positivos.

Estas cifras ilustran una evaluación con otro protocolo. La implementación de este repositorio usa una partición 80/20 y el umbral 0,5; por tanto, no deben presentarse como resultados generados por el código actual. Sus métricas aparecerán en outputs/metricas.csv después de ejecutarlo con el CSV correspondiente.

### Métricas complementarias

- **Accuracy** resume la proporción total de predicciones correctas. Puede ocultar un mal rendimiento en la clase minoritaria.
- **Balanced accuracy** promedia la tasa de acierto de cada clase.
- **Precisión** mide qué proporción de las predicciones positivas fue correcta.
- **Recall** mide qué proporción de los casos positivos reales detectó el modelo.
- **F1** combina precisión y recall en una sola medida.
- **ROC AUC** evalúa cómo ordena el modelo casos positivos por encima de negativos para distintos umbrales.

La matriz de confusión completa estas métricas al mostrar verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.

## 6. Limitaciones

- La evaluación del código depende de una única partición aleatoria; no incluye validación cruzada ni evaluación externa.
- La validación se usa para comparar los dos modelos, así que sus métricas no equivalen a una evaluación final independiente.
- El umbral se mantiene en 0,5; no se selecciona según un coste explícito de error.
- El desbalance entre clases puede hacer que accuracy parezca alta aunque recall de la clase positiva sea menor.
- La representatividad y procedencia de los datos limitan la generalización a otras muestras o poblaciones.
- La variable de pensamientos suicidas es sensible y muy cercana al resultado estudiado. Su capacidad predictiva no la convierte en una herramienta de diagnóstico.
- Los datos observacionales pueden mostrar asociaciones sin permitir inferir causalidad.

## 7. Conclusión

El proyecto muestra un flujo de clasificación binaria con limpieza de variables, codificación categórica, escalado, comparación de dos conjuntos de predictores y evaluación con métricas complementarias. Las señales más marcadas en el ajuste descrito aparecen en pensamientos suicidas, presión laboral o académica y estrés financiero; sueño, satisfacción y edad presentan asociaciones negativas. Estas tendencias dependen del conjunto y del ajuste, por lo que deben interpretarse con cautela y no como relaciones causales o clínicas.
