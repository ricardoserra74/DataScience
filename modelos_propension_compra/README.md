## DEFINICIÓN DEL PROYECTO<BR>
Vamos a desarrollar un modelo que prediga si un usuario va a convertir o no, en función de unas variables de campaña y de usuario. <br>
Para este proyecto, desarrollaremos 3 enfoques: <br>
1- Enfoque de modelo estadístico: al tratarse de una regresión logística, emplearemos los paquetes estadísticos xxx para crear el modelo de regresión, probarlo y validarlo. <br>
2- Enfoque de machine learning: emplearemos librerías de machine learning que nos indiquen cuál es el mejor modelo a aplicar para este problema. <br>
3- Enfoque de deep learning: emplearemos una red neuronal para obtener el modelo. <br>
4- Compararemos finalmente los 3 resultados. <br>

La fase de análisis de los datos inicial será común para los tres enfoques, y posteriormente se crearán datasets específicos que cumplan los requisitos de cada enfoque. <br>

El modelo se puede ejecutar y probar aquí -->> https://huggingface.co/spaces/Ricardoserra/model-lightgbm-conversion-digital-mkt <br><br>

### 1- Algunas gráficas de la fase de análisis: ###
---
![ Distribución de variables numéricas](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/analisis1.png) <br><br>
---
![ Distribución de variables numéricas respecto a la y](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/analisis2.png) <br><br>
---
![ Relaciones de las variables con la variable y](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/analisis3.png) <br><br>
---

### 2- Listado de Modelos ordenados por la métrica "Accuracy": ###
---
![ Distintos modelos](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/models1.png) <br><br>

### 3- Modelo elegido: ###
---
![ Modelo LGBM Classifier](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/models2.png) <br><br>

### 4- Métricas de evaluación del modelo elegido: ###
---
![ Curva ROC](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/models3.png) <br><br>

### 5- Explicación de una predicción: ###
---
![ Explicación de una predicción](https://github.com/ricardoserra74/DataScience/blob/main/modelos_propension_compra/img/models4.png) <br>
