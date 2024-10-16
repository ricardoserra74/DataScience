"""
## DEFINICIÓN DEL PROYECTO <BR>
Vamos a desarrollar un modelo que prediga si un usuario va a convertir o no, en función de unas variables de campaña y de usuario. <br>
Para este proyecto, desarrollaremos 3 enfoques: <br>
1- Enfoque de modelo estadístico: al tratarse de una regresión logística, emplearemos los paquetes estadísticos xxx para crear el modelo de regresión, probarlo y validarlo. <br>
2- Enfoque de machine learning: emplearemos librerías de machine learning que nos indiquen cuál es el mejor modelo a aplicar para este problema. <br>
3- Enfoque de deep learning: emplearemos una red neuronal para obtener el modelo. <br>
4- Compararemos finalmente los 3 resultados. <br>

La fase de análisis de los datos inicla será común para los tres enfoques, y posteriormente se crearán datasets específicos que cumplan los requisitos de cada enfoque. <br>
## 0- IMPORTACIÓN DE LIBRERÍAS Y FUNCIONES<br>
"""

## LIBRERÍAS BÁSICAS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import plotly.express as px
import scipy.stats as stats
from collections import Counter
import pickle

# LIBRERÍAS PARA HACER TRAMIFICACIONES, AGRUPACIONES Y TRANSFORMACIONES
from optbinning import Scorecard, BinningProcess, OptimalBinning, MulticlassOptimalBinning
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import label_binarize


# LIBRERÍAS PARA MODELADOS
import pycaret
from pycaret.classification import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from relativeImp import relativeImp
from statsmodels.formula.api import logit 
import statsmodels.api as sm
import patsy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTETomek


# LIBRERÍAS PARA VISUALIZACIONES
from explainerdashboard import ClassifierExplainer, RegressionExplainer
import gradio as gr
import plotly.graph_objects as go
import matplotlib.pyplot as pyplot
import seaborn as sns
import ydata_profiling
import lime
from lime import lime_tabular

pd.set_option("display.max_rows", None, "display.max_columns", None)


# FUNCIÓN V DE CRAMER : empleamos imput como dataset de variables y varObj como variable objetivo
# Las tablas de contingencia miden la asociación entre dos variables categóricas, así que como tenemos variables numéricas, las tramificamos en 5 grupos.
def cramers_v(var1, varObj):
    
    if not var1.dtypes == 'category':
        #bins = min(5,var1.value_counts().count())
        var1 = pd.cut(var1, bins = 5)
    if not varObj.dtypes == 'category': #np.issubdtype(varObj, np.number):
        #bins = min(5,varObj.value_counts().count())
        varObj = pd.cut(varObj, bins = 5)
        
    data = pd.crosstab(var1, varObj).values
    vCramer = stats.contingency.association(data, method = 'cramer')
    return vCramer



## FUNCIÓN MEJOR TRANSFORMACIÓN ##
# Busca la transformación de variables input de intervalo que maximiza la VCramer o 
# la correlación tipo Pearson con la objetivo
def mejorTransf (vv,target, name=False, tipo = 'cramer', graf=False):
    
    # Escalado de datos (evitar fallos de tamaño de float64 al hacer exp de número grande..cosas de python)
    vv = pd.Series(scale(vv), name=vv.name)
    # Traslación a valores positivos de la variable (sino falla log y las raíces!)
    vv = vv + abs(min(vv))+0.0001
      
    # Definimos y calculamos las transformaciones típicas  
    transf = pd.DataFrame({vv.name + '_ident': vv, vv.name + '_log': np.log(vv), vv.name + '_exp': np.exp(vv), vv.name + '_sqrt': np.sqrt(vv), 
                         vv.name + '_sqr': np.square(vv), vv.name + '_cuarta': vv**4, vv.name + '_raiz4': vv**(1/4)})
      
    # Distinguimos caso cramer o caso correlación
    if tipo == 'cramer':
      # Aplicar la función cramers_v a cada transformación frente a la respuesta
      tablaCramer = pd.DataFrame(transf.apply(lambda x: cramers_v(x,target)),columns=['VCramer'])
      
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCramer,x=tablaCramer.VCramer,title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # Identificar mejor transformación
      best = tablaCramer.query('VCramer == VCramer.max()').index
      ser = transf[best[0]].squeeze()
    
    if tipo == 'cor':
      # Aplicar coeficiente de correlación a cada transformación frente a la respuesta
      tablaCorr = pd.DataFrame(transf.apply(lambda x: np.corrcoef(x,target)[0,1]),columns=['Corr'])
      # Si queremos gráfico, muestra comparativa entre las posibilidades
      if graf: px.bar(tablaCorr,x=tablaCorr.Corr,title='Relaciones frente a ' + target.name).update_yaxes(categoryorder="total ascending").show()
      # identificar mejor transformación
      best = tablaCorr.query('Corr.abs() == Corr.abs().max()').index
      ser = transf[best[0]].squeeze()
  
    # Aquí distingue si se devuelve la variable transformada o solamente el nombre de la transformación
    if name:
      return(ser.name)
    else:
      return(ser)


# Función para generar la fórmula de la regresión logística
def ols_formula(df, dependent_var, *excluded_cols):
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)


# Función para dibujar la curva ROC
#Generamos un clasificador sin entrenar , que asignará 0 a todo
def curva_roc(ytest,ypred,model,xtest):
    ns_probs = [0 for _ in range(len(ytest))]
    # Predecimos las probabilidades
    lr_probs = model.predict_proba(xtest)

    #Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
    lr_probs = lr_probs[:, 1]

    # Calculamos el AUC
    ns_auc = roc_auc_score(ytest, ns_probs)
    lr_auc = roc_auc_score(ytest, lr_probs)
    # Imprimimos en pantalla
    print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
    print('Regresión Logística: ROC AUC=%.3f' % (lr_auc))
    # Calculamos las curvas ROC
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
    # Pintamos las curvas ROC
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenar')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Regresión Logística')
    # Etiquetas de los ejes
    pyplot.xlabel('Tasa de Falsos Positivos')
    pyplot.ylabel('Tasa de Verdaderos Positivos')
    pyplot.legend()
    pyplot.show()


#definimos funcion para mostrar los resultados
def mostrar_matriz(y_test, pred_y):
    LABELS= ['No convierte','Convierte']
    conf_matrix = confusion_matrix(y_test, pred_y)
    pyplot.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    pyplot.title("Confusion matrix")
    pyplot.ylabel('True class')
    pyplot.xlabel('Predicted class')
    pyplot.show()
    print (classification_report(y_test, pred_y))

## 0- IMPORTACIÓN DE DATOS<br>

## IMPORTAMOS LOS DATOS DE LAS VARIABLES Y LA OBJETIVO
dataset = pd.read_csv('digital_marketing_campaign_dataset.csv', sep=",")
dataset.describe()

"""
## 1- ANÁLISIS DE VARIABLES: <br>
En este apartado analizaremos: <br>
1- Tipos de variables: tenemos 14 variables numéricas y 6 categóricas tipo object. <br>
2- Valores mal codificados. <br>
3- Valores fuera de rango. <br>
4- Variables con valores infrarrepresentados: ver si aplicamos undersampling o oversampling, para la Y <br>
5- Outliers:  <br>
6- Missings o valores perdidos. <br>
Adjuntamos en el ejercicio la salida de la librería ydata_profiling, que resumimos a continuación.
"""

dataset.info()
# Utilizamos la librería ydata_profiling para hacer un análisis descriptivo de las variables
profile = ydata_profiling.ProfileReport(dataset)

# Nos guardamos el html para adjuntarlo al proyecto
profile.to_file(output_file='Analisis_descriptivo.html')

profile
dataset.Conversion.value_counts(normalize=True)
## 1.1 Resumen de las variables: <br>
"""
Exploramos el dataset para comprender su composición: disponemos de 8000 obervaciones, sin valores faltantes ni duplicados.<br>
- CustomerId: número identificativo del cliente, inicialmente lo dejamos (por si tenemos que mergear algún dataset), pero lo sacaremos de los modelos por no aportar información.<br>
- Age: edad del cliente, numérica. Valores comprendidos entre 18 y 69, con una media de 43,6. Presenta concentración en los valores mínimo y máximo. <br>
- Gender: género, tipo object con dos valores: Male, Female. El género Female está más representado en el dataset, con un 60,5% de los registros.<br>
- Income: renta anual del cliente en dólares. Valores entre 20.000 y 150.000. <br>
- CampaignChannel: canal de la campaña, tipo object con 5 valores diferentes: Referral, PPC, Email, SEO, Social Media. La representación de cada uno en el dataset está en torno al 20% <br>
- CampaignType: objetivo de la campaña, tipo object con 4 valores diferentes: Awareness, Consideration, Conversion, Retention. La representación de cada valor en el dataset está en torno al 25%. <br>
- AdSpend: cantidad gastada en la campaña de marketing en dólares, numérico. Valores comprendidos entre 100$ y 10.000$, con media de 5.000$<br>
- ClickThroughRate: tasa de click de los clientes sobre los impactos de la campaña, numérico. Es un porcentaje, expresado sobre la unidad, con valores entre 0.01 y 0.29, con promedio de 0.15 <br>
- ConversionRate: tasa de click a la que compran los clientes, número de clicks. Porcentaje sobre la unidad, con valores entre 0.01 y 0.19m, con promedio de 0.10 <br>
- AdvertisingPlatform: plataforma de marketing. Contiene un único valor, lo eliminaremos del modelo al no aportar información. <br>
- AdvertisingTool: herramienta de marketing. Contiene un único valor, lo eliminaremos del modelo al no aportar información. <br>
- WebsiteVisits: visitas a la web. Numérico, con valores entre 0 y 49, y un promedio de 24.75 visitas. Hay una incongruencia con valores=0 (suponen el 2%) ya que estos registros sí que tienen informadas las variables PagesPerVisit y TimeOnsite. Probaremos a imputar estos valores<br>
- PagesPerVisit: promedio de páginas visitadas por sesión. Numérico, con valores entre 1 y 10, con promedio de 5. <br>
- TimeOnSite: promedio de tiempo en la web, por visita, en minutos. Numérico, con valores entre 0.5 y 15, con promedio de 7.7 <br>
- SocialShare: número de veces que el contenido de la campaña fue compartido en redes sociales. Numérico, con valores entre 0 y 100, con promedio de 50. Tenemos un 1,2% de ceros. <br>
- EmailOpens: número de veces que los emails de marketing fueron abiertos. Valores entre 0 y 19, con un promedio de 9.5. El dataset tiene un 5% de ceros. Este dato es incongruente, porque estos registros sí tienen informada la variable EmailClicks. Veremos cómo imputar.<br>
- EmailClicks: número de veces que el se clicó en el email de marketing. Valores entre 0 y 9, con promedio de 4,4. Aquí tenemos un 10% de ceros. <br>
- PreviousPurchases: número de compras anteriores realizadas por el cliente. Valores entre 0 y 10, con promedio de 4,5. El dataset tiene un 10,5% de ceros. <br>
- LoyaltyPoints: número de puntos de fidelización acumulados por el cliente. Valores entre 0 y 5000, con promedio de 2.490 <br>
- Conversion: variable dicotómica, qie indica si el cliente compró (1) o no (0). Presenta un importante desbalanceo: el 87% de los registros sí compraron, frente al 12,7% que no efectuaron compra. Tenemos que aplicar técnicas de balanceo.
"""


## 1.2 Transformaciones iniciales de variables <br>

# Eliminamos CustomerId, AdvertisingPlatform y AdvertisingTool
dataset.drop(['AdvertisingPlatform','AdvertisingTool'], axis = 1, inplace=True)

# Renombramos la variable objetivo: para facilitar el trabajo posterior
dataset.rename(columns={'Conversion': 'y'}, inplace=True)

# Transformamos las variables categóricas
dataset['CampaignChannel'] = dataset['CampaignChannel'].astype('category')
dataset['CampaignType'] = dataset['CampaignType'].astype('category')
dataset['Gender'] = dataset['Gender'].astype('category')

dataset[dataset['WebsiteVisits'] > 0].head()

"""
## 1.3 Análisis Gráfico de las variables
### 1.3.1- Distribución de variables numéricas <br>
Volvemos a representar las variables para evaluarlas conjuntamente: <br>
Confirmamos que las distribuciones son uniformes, confirmamos más peso de los 0 en el género, y confirmamos el desbalanceo de la variable respuesta.
"""

fig, axes = pyplot.subplots(nrows=5, ncols=4, figsize=(10, 8))
axes = axes.flat
columnas_numeric = dataset.select_dtypes(include=['float64', 'int64']).columns
for i, colum in enumerate(columnas_numeric):
    sns.distplot(dataset[colum], hist= True, rug = True,
                 kde_kws = {'shade': True, 'linewidth': 1}, ax  = axes[i] )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
fig.tight_layout()
pyplot.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de variables numéricas', fontsize = 10, fontweight = "bold");

"""
### 1.3.2- Distribución de las variables numéricas frente a la variable objetivo <br>
Teniendo en cuenta el desbalanceo de la variable objetivo, vemos las distribuciones de las variables numéricas respecto a la y<br>
Normalizamos las gráficas para una mejor visualización, entendiendo el ya existente desbalanceo. Como resumen: <br>
* Edad: hay menos densidad para los no compradores en las edades entorno a los 40, y entre 65 y 70 años. <br>
* Género: menor densidad de hombres, y menor densidad de no compradores. <br>
* Ingresos: distribución ligeramente superior en los usuarios compradores. <br>
* Gasto de campaña: los no compradores son asimétricos en torno a los 2000$, y el gasto es notablemente superior en los compradores, desde los 6000$ <br>
* ClickThroughRate: los no compradores son asimétricos en torno a ratios de click bajos. Parece que los compradores tienen ratios de click más altos.<br>
* ConversionRate: los no compradores son asimétricos en torno a ratios de conversión bajos. Parece que los compradores tienen ratios de conversión más altos.<br>
* WebSiteVisits: los no compradores tienen altas densidades en torno a bajas visitas a la web. Parece que los compradores visitan más la web.<br>
* PagesPerVisit: misma tendencia que indicadores anteriores: los no compradores tienen más densidades en menos páginas por visita. <br>
* TimeOnsite: los compradores pasan más tiempo en el site que los no compradores. <br>
* SocialShares: las densidades son parecidas entre compradores y no compradores, aunque los no compradores tienen un pico alrededor de los 50. <br>
* EmailOpens: los compradores tienen mayores densidades de aperturas altas que los no compradores. <br>
* EmailClicks, PreviousPurchases y LoyaltyPoints: lso compradores tienen densidades más altas en los valores más altos. <br>
"""

#  plot Numerical Data
a = 4  # 4 filas
b = 4  # 4 columnas
c = 1  # contador de plot

fig, axes = pyplot.subplots(nrows=4, ncols=4, figsize=(15, 13))
axes = axes.flat

for i in columnas_numeric:
    pyplot.subplot(a, b, c)
    pyplot.xlabel(i)
    sns.kdeplot(data = dataset, x = i, hue ='y', fill=True, common_norm=False, alpha=0.6, gridsize=80)
    c = c + 1

pyplot.tight_layout()
pyplot.subplots_adjust(top = 0.9)
fig.suptitle('Distribución de variables numéricas respecto a la y', fontsize = 20, fontweight = "bold");

"""
###  1.3.3.- Distribución de variables categóricas: <br>
* Gender: para el género "Male", los valores de la y están menos representados que para el género "Female". <br>
* CampaignChannels: los valores se distribuyen de una manera más o menos homogénea, entre el 19% y el 21% de registros en cada categoría. <br>
* CampaignType: también se distribuyen de una manera equilibrada, entre el 24% y el 26% de registros por categoría. <br>
* En cuanto a cómo se distribuyen las categorías en función de la variable objetivo, la categoría de CampaignType = Conversion tiene infrarrepresentación para los valores de y=0 respecto al resto de categorías: 138/1.016 = 13.6%, frente al resto de categorías que están en torno al 28%. 
"""

columnas_cat = dataset[['Gender', 'CampaignChannel', 'CampaignType' ]]

fig , ax = pyplot.subplots(1,3,figsize = (17,5))   
for i , subplots in zip (columnas_cat, ax.flatten()):  
  sns.countplot(x=columnas_cat[i], hue = dataset['y'], data = dataset, ax = subplots, palette = None, alpha=0.8)
pyplot.show()
print('Distribución de la y por género: \n')
pd.crosstab(dataset.y, dataset.Gender, normalize=True)

print('Distribución de la y por CampaignChannel: \n')
pd.crosstab(dataset.y, dataset.CampaignChannel, normalize=True)
print('Distribución de la y por CampaignType: \n')
pd.crosstab(dataset.y, dataset.CampaignType, normalize=True)
## 1.4 Incidencia y gestión de outliers <br>
* No se observan valores atípicos en las variables numéricas
# Creamos dos variables de control para evaluar los efectos de las variables predictoras sobre la variable objetivo

dataset['aleatorio'] = np.random.uniform(0,1,size=dataset.shape[0])
dataset['aleatorio2'] = np.random.uniform(0,1,size=dataset.shape[0])
# Los valores atípicos y faltantes se trabajan solo de las variables input, no de la objetivo. Dividimos el dataset, que luego también usaremos
varObj = dataset.y 
imput = dataset.drop(['CustomerID', 'y'],axis=1)

# No existen asimetrías en las variables numéricas, con lo que no tenemos problemas de valores atípicos.
imput.select_dtypes(include=np.number).apply(lambda x: x.skew())
## 1.5 Existencia de valores perdidos: <br>
# Como hemos comprobado anteriormente con la librería ydata_profiling, no existen valores faltantes.
# Proporción de missings por variable 
dataset.apply(lambda x: x.isna().sum()/dataset.shape[0]*100)

"""
## 1.6 Imputaciones: <br>
Existen 362 registros que tienen EmailOpens = 0 y EmailClicks distintos de cero, lo que es incongruente. Para corregir esos registros, recalculamos el campo EmailOpens ayudándonos de la columna ClickTroughRate, de manera que EmailOpens(0) = EmailClicks / ClickTroughRate. <br>
Finalmente, nos quedamos con 41 registros que tienen EmailOpens = 0 y EmailClicks = 0 <br>
"""


# Al aplicar dicha transformación, el número de valores posibles del campo EmailOpens ha aumentado, y se han generado valores extremos. Para corregir estos outliers, les asignamos el mayor valor de la serie, que es 19.
# Consultamos
dataset.loc[(dataset['EmailOpens'] == 0) & (dataset['EmailClicks'] == 0)].count()

# Aplicamos la fórmula de imputación de los valores incongruentes para EmailOpens=0
dataset.loc[dataset['EmailOpens'] == 0, 'EmailOpens'] = round((dataset.EmailClicks / dataset.ClickThroughRate), 0)
dataset.describe()
ax = sns.boxplot(x="EmailOpens", data=dataset)

# Para corregir esos registros, les asignamos el valor máximo de la variable EmailOpens antes de realizar la transformación: 19
dataset['EmailOpens'].loc[(dataset["EmailOpens"] > 19)] = 19

ax = sns.boxplot(x="EmailOpens", data=dataset)

"""
## 1.7- Análisis inicial de las relaciones de las variables con la objetivo: V de Cramer<br>
La V de Cramer mide la asociación entre dos variables nominales, y su valor está entre 0 y 1 (existiendo poca asociación entre las variables para valores cercanos a cero, y alta asociación para valores cercanos a 1). No obstante, para las variables numéricas, las tramificamos en 5 tramos, para que sean tratadas como categóricas.<br>

Tras ejecutar una tabla de contingencia con el método V de Cramer, parece que la variable que más influye en la conversion es "PreviousPurchases", seguida de "PagesPerVisit", "EmailOpens", "ConversionRate", "TimeOnsite", "EmailClicks", "LoyaltyPoints", "ClickThroughRate", WebSiteVisits" y "Adspend". <br>
Las variables que menos relación tienen, son "Age", "Income", "Campaignchannel", "SocialShares" y "Gender". <br>
"Social Shares" y "Gender" están por debajo de las variables aleatorias, con lo que vamos a prescindir de ellas. <br>

Ejecutamos también las correlaciones entre las variables numéricas y la variable objetivo: aunque varían los valores, el resultado confirma la variables obtenidas con la V de Cramer.
"""

dataset.info()
# Aplicamos la función  V DE CREAMER al input completo contra la objetivo
tablaCramer = pd.DataFrame(imput.apply(lambda x: cramers_v(x,varObj)),columns=['VCramer'])

# Obtenemos el gráfico de importancia de las variables frente a la objetivo continua según vcramer
px.bar(tablaCramer,x=tablaCramer.VCramer, title='Relaciones frente a la conversión', text_auto=True).update_yaxes(categoryorder="total ascending", automargin="left").show()

# Matriz de correlaciones
corr = dataset.select_dtypes(include=np.number).corr()
corr.style.background_gradient(cmap='YlOrRd').format(precision=3)

dataset.info()
# Borramos las variables aleatorias
dataset = dataset.drop(['aleatorio', 'aleatorio2'], axis=True)

"""
# 2- TRANSFORMACIONES DE VARIABLES 
## 2.1 Transformaciones en las variables numéricas: <br> 
- Ejecutamos la función precargada que nos busca la mejor transformación de las variables numéricas, para maximizar las relaciones con la variable objetivo. Mediante el gráfico de barras, vemos que no cambian significativamente las distribuciones de las variables aplicando transformaciones matemáticas, con lo que no las transformaremos.
# Aplicar a las variables continuas la mejor transformación según cramer frente a varObjBin
"""

transf_cramer = imput.select_dtypes(include=np.number).apply(lambda x: mejorTransf(x,varObj, tipo='cramer'))
transf_cramer_names = imput.select_dtypes(include=np.number).apply(lambda x: mejorTransf(x,varObj,tipo='cramer', name=True))
transf_cramer.columns = transf_cramer_names.values
transf_cramer.head()
# Generar input con tranformaciones
imput_transf = imput.join(transf_cramer)

# Aplicar la función al input completo contra la objetivo
tablaCramer = pd.DataFrame(imput_transf.apply(lambda x: cramers_v(x,varObj)),columns=['VCramer'])

# Obtener el gráfico de importancia de las variables frente a la objetivo continua según vcramer
px.bar(tablaCramer,x=tablaCramer.VCramer,title='Relaciones frente a Compra tras efectuar Transformaciones en las variables', text_auto=True).update_yaxes(categoryorder="total ascending", automargin="left").show()

"""
# 3- ENFOQUE ESTADÍSTICO: MODELIZACIÓN DE UNA REGRESIÓN LOGÍSTICA <br>
Antes de empezar con la partición del dataset en entrenamiento y test, recordamos que la variable objetivo presenta un importante desbalance en favor del valor 1. Esto es muiy importante, ya que el modelo tendrá problemas en detectar los valores 0.
"""

dataset.y.value_counts(normalize=True)

"""
## 3.1- Partición en training y test: <br>
El dataset de entrenamiento tiene 6400 registros, tanto para las x como para las y. <br>
El dataset de test tiene 1600 registros. <br>
La proporción de los valores de la variable objetivo en Training y en Test se ha mantenido casi al 100%. <br>
"""

# Borramos las variables aleatorias
imput = imput.drop(['aleatorio', 'aleatorio2'], axis=True)
imput.head()
# Creamos 4 objetos: predictores para training y test y variable objetivo para training y test. 
X_train, X_test, y_train, y_test = train_test_split(imput, varObj, test_size=0.2, random_state=1234)

# Comprobamos dimensiones
print('Tamaño dataset Training:', X_train.shape, y_train.shape)
print('Tamaño dataset Test:', X_test.shape, y_test.shape)
print('Proporción de la variable y en Training: \n', y_train.value_counts(normalize=True))
print('Proporción de la variable y en Test: \n', y_test.value_counts(normalize=True))
# Juntamos los sets de entrenamiento de las variables explicativas y la objetivo:
data_train = X_train.join(y_train)
data_test = X_test.join(y_test)

data_train.head()
## 3.2- Creación del modelo completo
# Aplicamos a fórmula de modelo completo
formC=ols_formula(data_train,'y')
formC
# One-hot encoding: convertimos las variables categóricas en numéricas, y generamos las matrices de diseño según la fórmula de modelo completo.
y_tr, X_tr = patsy.dmatrices(formC, data_train, return_type='dataframe')

# Generamos las matrices de diseño según la fórmula de modelo completo para test
y_tst, X_tst = patsy.dmatrices(formC, data_test, return_type='dataframe')
X_tst.head()


"""
## 3.3- Métricas de evaluación del modelo: <br>
Tras ejecutar el modelo en training, hacemos las predicciones en test y obtenemos las métricas que nos van a ayudar en la elección de nuestro modelo: 
* PseudoR: 0.1988, sería equiparable a un un 40% en una regresión lineal. El modelo explica el 40% de la variabilidad de la y.
* La exactitud (accuracy) del modelo para clasificar correctamente es de un 90%. 
* La matriz de confusión nos da: 47 verdaderos 0, 1398 verdaderos 1, 142 falsos 1, 13 falsos 0
* Para la clase 0 (No conversión):
    * Precisión de la clase 0: predicciones correctas de la clase 0 / total de predicciones de la clase 0 --> 78%. El modelo ha predicho 60 y ha acertado 47.
    * Recall: predicciones correctas de la clase 0 / registros reales de la clase 0 --> 25%. El modelo ha predicho bien 47 de 189 registros con 0.
    * F1-Score: 38% --> el modelo no detecta bien la clase 0.
* Para la clase 1 (Conversión):
    * Precisión de la clase 1: predicciones correctas de la clase 1 / total de predicciones de la clase 1 --> 91%. El modelo ha predicho 1540 y ha acertado 1398.
    * Recall: predicciones correctas de la clase 1 / registros reales de la clase 1 --> 99%. El modelo ha predicho bien 1398 de 1412 registros con 1.
    * F1-Score: 95% --> el modelo detecta muy bien la clase 1

* Área bajo la curva roc: 1> 0.81> 0.5 --> modelo relativamente bueno. Es mejor que el "no modelo".
* No obstante, nos encontramos con una capacidad de predicción de los 0 casi nula.

Los 4 escenarios que se nos presentan son los siguientes: <br>
* Alta precision y alto recall: el modelo maneja perfectamente esa clase. Se trata de un buen modelo de clasificación para poder implementar en nuestro proyecto.
* Alta precision y bajo recall: el modelo no detecta la clase «perseguida» muy bien, pero cuando lo hace es altamente confiable.
* Baja precisión y alto recall: El algoritmos detecta bien la clase «perseguida» pero también incluye muestras de la/s otra/s clase/s.
* Baja precisión y bajo recall: El modelo no logra clasificar la clase correctamente. No sería bueno seguir adelante con la implementación de dicho modelo.

Como resumen: nos encontramos con un modelo que tiene:
* Para las "No conversiones": alta precisión y bajo recall: el modelo no detecta la clase bien, pero cuando lo hace, es altamente confiable.
* Para las "Conversiones": alta precisión y alto recall: el modelo maneja muy bien la clase=1
# Definición de modelo: hay muchos solver, python usa por defecto éste. Aplicamos el solver newton-cg porque es el que nos da valores más altos de las métricas
modelo1 = LogisticRegression(solver='newton-cg', max_iter=1000, penalty=None)
"""


# Arreglar y para que le guste a sklearn...numeric, creo un data set nuevo
y_tr_ = y_tr.y.ravel()

# Ajuste de modelo
modelLog = modelo1.fit(X_tr,y_tr)

# Accuracy del modelo en training: el score da la tasa de acierto: el modelo clafisica correctamente el 87,5% de las observaciones
acc = modelLog.score(X_tr,y_tr)
print('Precisión del modelo: el modelo clasifica correctamente el porcentaje de las observaciones', acc)

# Predicciones en test
y_pred = modelLog.predict(X_tst)

mostrar_matriz(y_tst, y_pred)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_tr_, modelLog.predict_proba(X_tr)[:, 1]))
curva_roc(y_tst, y_pred, modelLog, X_tst)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(modelLog.score(X_tr, y_tr_)))
print('Test set score: {:.4f}'.format(modelLog.score(X_tst, y_tst)))

# Ajusto regresión de ejemplo para sacar los coeficientes. Lo hacemos una única vez con este modelo sólo, por no saturar
modelLogCompleto = logit(formC,data=data_train).fit()
modelLogCompleto.summary()

"""
## 3.4- Proceso de backward para retirar variables del modelo y mejorarlo: MODELO 2
Eliminamos las variables que no tiene un p-valor significativo, y volvemos a ejecutar el modelo en training, hacemos las predicciones en test y obtenemos las métricas que nos van a ayudar en la elección de nuestro modelo: La variable CampaignChannel es la que provoca esta mínima disminución.
* PseudoR: 0.1964, sería equiparable a un un 39% en una regresión lineal. El modelo explica el 39% de la variabilidad de la y, inferior al modelo anterior.
* La exactitud (accuracy) del modelo para clasificar correctamente es de un 90%: se mantiene respecto al modelo anterior. 
* La matriz de confusión nos da: 44 verdaderos 0, 1403 verdaderos 1, 145 falsos 1, 8 falsos 0
* Para la clase 0 (No conversión):
    * Precisión de la clase 0: predicciones correctas de la clase 0 / total de predicciones de la clase 0 --> 85%. El modelo ha predicho 52 y ha acertado 44.
    * Recall: predicciones correctas de la clase 0 / registros reales de la clase 0 --> 23%. El modelo ha predicho bien 44 de 189 registros con 0. Cae 2 pp respecto al modelo anterior.
    * F1-Score: 37% --> el modelo detecta 1pp peor la clase 0 que el modelo anterior.
* Para la clase 1 (Conversión):
    * Precisión de la clase 1: predicciones correctas de la clase 1 / total de predicciones de la clase 1 --> 91%. El modelo ha predicho 1548 y ha acertado 1403. Mejora respecto al modelo anterior.
    * Recall: predicciones correctas de la clase 1 / registros reales de la clase 1 --> 99%. El modelo ha predicho bien 1403 de 1411 registros con 1.
    * F1-Score: 95% --> el modelo detecta bien la calse 1.

* Área bajo la curva roc: 1> 0.81> 0.5 --> modelo relativamente bueno. Es mejor que el "no modelo" y muy levemente que el modelo anterior.
* No obstante, seguimos con una capacidad de predicción de los ceros muy baja. Haremos oversampling con este modelo.
"""


# Creamos datasets nuevos eliminando las variables no representativas
data_train2 = data_train.drop(['Age','Gender', 'Income', 'SocialShares', 'CampaignChannel'], axis = 1)
data_test2 = data_test.drop(['Age','Gender', 'Income', 'SocialShares', 'CampaignChannel'], axis = 1)


# Aplicamos a fórmula de modelo eliminando las variables que descartamos
form2=ols_formula(data_train2,'y')
form2
# One-hot encoding: convertimos las variables categóricas en numéricas, y g eneramos las matrices de diseño según la fórmula de modelo completo.
y_tr2, X_tr2 = patsy.dmatrices(form2, data_train2, return_type='dataframe')

# Generamos las matrices de diseño según la fórmula de modelo completo para test
y_tst2, X_tst2 = patsy.dmatrices(form2, data_test2, return_type='dataframe')
X_tr2.head()

# Arreglar y para que le guste a sklearn...numeric, creo un data set nuevo
y_tr_2 = y_tr2.y.ravel()

# Ajuste de modelo
modelLog2 = modelo1.fit(X_tr2,y_tr_2)

# Accuracy del modelo en training: el score da la tasa de acierto: el modelo clafisica correctamente el 87,5% de las observaciones
acc2 = modelLog2.score(X_tr2,y_tr_2)
print('Precisión del modelo: el modelo clasifica correctamente el porcentaje de las observaciones', acc2)

# Predicciones en test
y_pred2 = modelLog2.predict(X_tst2)

mostrar_matriz(y_tst2, y_pred2)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_tr_2, modelLog2.predict_proba(X_tr2)[:, 1]))
curva_roc(y_tst2, y_pred2, modelLog2, X_tst2)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(modelLog2.score(X_tr2, y_tr_2)))
print('Test set score: {:.4f}'.format(modelLog2.score(X_tst2, y_tst2)))

"""
## 3.5 Proceso con hiperparámetros: MODELO 3
Empleamos la librería GridSearchCV para que nos combine parámetros para solver, penalización y peso de las clases:
* La exactitud (accuracy) del modelo para clasificar correctamente es de un 85%: disminuye respecto a modelos anteriores.
* La matriz de confusión nos da: 30 verdaderos 0, 1326 verdaderos 1, 159 falsos 1, 85 falsos 0
* Para la clase 0 (No conversión):
    * Precisión de la clase 0: predicciones correctas de la clase 0 / total de predicciones de la clase 0 --> 26%. El modelo ha predicho 115 y ha acertado 30.
    * Recall: predicciones correctas de la clase 0 / registros reales de la clase 0 --> 16%. El modelo ha predicho bien 30 de 189 registros con 0. Empeora.
    * F1-Score: 20% --> el modelo detecta peor la clase 0 que los modelos anteriores.
* Para la clase 1 (Conversión):
    * Precisión de la clase 1: predicciones correctas de la clase 1 / total de predicciones de la clase 1 --> 89%. El modelo ha predicho 1485 y ha acertado 1326. Empeora respecto al modelo anterior.
    * Recall: predicciones correctas de la clase 1 / registros reales de la clase 1 --> 94%. El modelo ha predicho bien 1326 de 1411 registros con 1.
    * F1-Score: 92% --> el porcentaje mejora respecto a modelos anteriores.

* Área bajo la curva roc: 1> 0.62> 0.5 --> modelo malo, se aproxima al no modelo.
* No obstante, seguimos con una capacidad de predicción de los ceros baja.
"""

# Creo un diccionario de métodos, que el modelo vaya combinando parámetros. Entrenamos con el dataset2, sin las variables innecesarias.

grid_param = {
    'solver': ['lbfgs', 'newton_cg', 'sag', 'liblinear'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'class_weight': ['balanced', 'unbalanced'],
    }

model_grid = GridSearchCV(estimator=modelo1, ##indicamos modelo
                     param_grid=grid_param,# indicamos hiperparámetros
                     scoring='f1',# métrica que queremos mejorar
                     cv=5,# validación cruzada: trocea el dataset en 5, y prueba 4 training y 1 en test cada vez.
                     n_jobs=-1)# 

model_grid.fit(X_tr2, y_tr_2)
print(model_grid.best_params_)
print(model_grid.best_score_)
modelo3 = LogisticRegression(solver='sag', max_iter=1000, class_weight='balanced', penalty='l2')


# Ajuste de modelo
modelLoghiper = modelo3.fit(X_tr2,y_tr_2)

# Accuracy del modelo en training: el score da la tasa de acierto: el modelo clafisica correctamente el 87,5% de las observaciones
acc_hiper = modelLoghiper.score(X_tr2,y_tr_2)
print('Precisión del modelo: el modelo clasifica correctamente el porcentaje de las observaciones', acc_hiper)

# Predicciones en test
y_pred_hiper = modelLoghiper.predict(X_tst2)

mostrar_matriz(y_tst2, y_pred_hiper)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_tr_2, modelLoghiper.predict_proba(X_tr2)[:, 1]))
curva_roc(y_tst2, y_pred_hiper, modelLoghiper, X_tst2)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(modelLoghiper.score(X_tr2, y_tr_2)))
print('Test set score: {:.4f}'.format(modelLoghiper.score(X_tst2, y_tst2)))

"""
## 3.6- Balanceo del conjunto de datos: SAMPLIG CON SMOTE TOMEK: MODELO 4
Probamos a incluir sobremuestreo y submuestreo con la librería SMOTE para oversampling, y TOMEK para undersampling, para que balancee la muestra.
* La exactitud (accuracy) del modelo para clasificar correctamente es de un 73%: disminuye 17 puntos respecto a modelos anteriores.
* La matriz de confusión nos da: 145 verdaderos 0, 1024 verdaderos 1, 44 falsos 1, 387 falsos 0
* Para la clase 0 (No conversión):
    * Precisión de la clase 0: predicciones correctas de la clase 0 / total de predicciones de la clase 0 --> 27%. El modelo ha predicho 532 y ha acertado 145.
    * Recall: predicciones correctas de la clase 0 / registros reales de la clase 0 --> 77%. El modelo ha predicho bien 145 de 189 registros con 0. Mejora.
    * F1-Score: 41% --> el modelo detecta mejor la clase 0 que los modelos anteriores.
* Para la clase 1 (Conversión):
    * Precisión de la clase 1: predicciones correctas de la clase 1 / total de predicciones de la clase 1 --> 96%. El modelo ha predicho 1068 y ha acertado 1024. 
    * Recall: predicciones correctas de la clase 1 / registros reales de la clase 1 --> 73%. El modelo ha predicho bien 1024 de 1411 registros con 1.
    * F1-Score: 82% --> el porcentaje ha empeorado respecto a modelos anteriores.

* Área bajo la curva roc: 1> 0.81> 0.5 --> modelo relativamente bueno. Es mejor que el "no modelo" y muy levemente que el modelo anterior.
* Ninguna de las estrategias ha mejorado ostensiblemente la explicación del modelo.
"""


os_us = SMOTETomek(sampling_strategy='auto')
X_train_tomek, y_train_tomek = os_us.fit_resample(X_tr2, y_tr_2)

print ("Distribución antes de resampling {}".format(X_tr2.shape))
print ("Distribución después de resampling {}".format(X_train_tomek.shape))
# Arreglar y para que le guste a sklearn...numeric, creo un data set nuevo
# y_tr_tomek = y_train_tomek.y.ravel()

# Ajuste de modelo
modelLogtomek = modelo1.fit(X_train_tomek,y_train_tomek)

# Accuracy del modelo en training: el score da la tasa de acierto: el modelo clafisica correctamente el 87,5% de las observaciones
acc_tomek = modelLogtomek.score(X_train_tomek,y_train_tomek)
print('Precisión del modelo: el modelo clasifica correctamente el porcentaje de las observaciones', acc_tomek)

# Predicciones en test
y_pred_tomek = modelLogtomek.predict(X_tst2)

mostrar_matriz(y_tst2, y_pred_tomek)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_train_tomek, modelLogtomek.predict_proba(X_train_tomek)[:, 1]))
curva_roc(y_tst2, y_pred_tomek, modelLogtomek, X_tst2)
dataset.info()
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(modelLogtomek.score(X_train_tomek, y_train_tomek)))
print('Test set score: {:.4f}'.format(modelLogtomek.score(X_tst2, y_tst2)))

"""
# 4- ENFOQUE 2: MACHINEL LEARNING: ME QUEDO AQUÍ:
## 4.1- Selección inicial de modelos
# Creamos un dataset dummy, donde hayamos transformado las variables categóricas, para poder trabajar con él cuando vayamos a tunear hiperparámetros desde fuera de Pycaret (Pycaret hace transformación de variables)
"""

dataset_dum = dataset.drop(['CustomerID'], axis=1)

dataset_dum.Gender.replace(('Male','Female'),
                      (0,1),inplace=True)

dataset_dum.CampaignType.replace(('Conversion','Awareness', 'Consideration', 'Retention'),
                      (1,2,3,4),inplace=True)

dataset_dum.CampaignChannel.replace(('Referral','PPC', 'Email', 'SEO', 'Social Media'),
                      (1,2,3,4,5),inplace=True)



Gender_df = pd.get_dummies(dataset_dum.Gender, prefix='Gender')
Gender_df['Gender_1'] = Gender_df['Gender_1'].astype('int')
Gender_df['Gender_0'] = Gender_df['Gender_0'].astype('int')


CampaignChannel_df = pd.get_dummies(dataset_dum.CampaignChannel, prefix='Channels')
CampaignChannel_df['Channels_3'] = CampaignChannel_df['Channels_3'].astype('int')
CampaignChannel_df['Channels_2'] = CampaignChannel_df['Channels_2'].astype('int')
CampaignChannel_df['Channels_1'] = CampaignChannel_df['Channels_1'].astype('int')
CampaignChannel_df['Channels_4'] = CampaignChannel_df['Channels_4'].astype('int')
CampaignChannel_df['Channels_5'] = CampaignChannel_df['Channels_5'].astype('int')

CampaignType_df = pd.get_dummies(dataset_dum.CampaignType, prefix='Objetivo')
CampaignType_df['Objetivo_1'] = CampaignType_df['Objetivo_1'].astype('int')
CampaignType_df['Objetivo_2'] = CampaignType_df['Objetivo_2'].astype('int')
CampaignType_df['Objetivo_3'] = CampaignType_df['Objetivo_3'].astype('int')
CampaignType_df['Objetivo_4'] = CampaignType_df['Objetivo_4'].astype('int')


dataset_dum = pd.concat([Gender_df, CampaignChannel_df, CampaignType_df, dataset_dum], axis=1)


pd.options.display.max_columns = None
dataset_dum.head(5)

# Borramos variables para evitar colinealidad
dataset_dum = dataset_dum.drop(['Gender', 'Gender_1', 'Channels_1', 'Objetivo_4', 'CampaignChannel', 'CampaignType'], axis=1)
# Dividimos el dataset, que luego también usaremos
varObj_dum = dataset_dum.y 
imput_dum = dataset_dum.drop(['y'],axis=1)
# Creamos training y test para usar con Pycaret (en principio no van a ahcer falta)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(imput, varObj, test_size=0.2, random_state=1234)
# Creamos training y test para usar sin Pycaret, para todos los modelos que generemos de aquí en adelante:
X_train_dum, X_test_dum, y_train_dum, y_test_dum = train_test_split(imput_dum, varObj_dum, test_size=0.2, random_state=1234)
# Con la librería Pycaret obtenemos los modelos: en el setup se hacen automáticamente todas las transformaciones de las variables, el split en train y test, le meto balanceo
# Por defecto: el train size es 70%, aplica data_split_stratify
model_setup = setup(data=dataset_dum, target='y', session_id=1234, fix_imbalance = True)

models()
## los primeros modelos son los ensemble: .
best_models = compare_models()
# Devuelve el mejor modelo
model_setup.automl()


## 4.2- Creamos el modelo LGBMC
# Creamos el modelo de LIGHTGBM
model_lightgbm = create_model('lightgbm')
print(model_lightgbm)
# Imprimimos matriz de confusión
plot_model(model_lightgbm, plot = "confusion_matrix", plot_kwargs={'percent' : False})
plot_model(model_lightgbm, plot = 'auc')
# Obtenemos evaluación del modelo
evaluate_model(model_lightgbm)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_lightgbm.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_lightgbm.score(X_test_dum, y_test_dum)))


## 4.3- Creamos modelo lightgbm tuneado
# Le incremento los estimadores y aplico balanceo
import lightgbm as lgb
lgbm = lgb.LGBMClassifier(boosting_type='gbdt', class_weight='balanced', colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,
               random_state=1234, reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)
model_lgbm_dum = lgbm.fit(X_train_dum, y_train_dum)
model_lgbm_dum.score(X_test_dum, y_test_dum)
y_pred_lgbm_dum = model_lgbm_dum.predict(X_test_dum)
mostrar_matriz(y_test_dum, y_pred_lgbm_dum)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_train_dum, model_lgbm_dum.predict_proba(X_train_dum)[:, 1]))
curva_roc(y_test_dum, y_pred_lgbm_dum, model_lgbm_dum, X_test_dum)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_lgbm_dum.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_lgbm_dum.score(X_test_dum, y_test_dum)))

## 4.4- Modelo GBC
# Creamos el modelo de GBC
model_gbc = create_model('gbc')
# no ha hecho nada con los hiperparámetros. Podemos tunearlo
print(model_gbc)
# Imprimimos matriz de confusión
plot_model(model_gbc, plot = "confusion_matrix", plot_kwargs={'percent' : False })
plot_model(model_gbc, plot = 'auc')
# Lanzamos el widget completo de evaluación del modelo. Algunas llamadas dan error.
evaluate_model(model_gbc)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_gbc.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_gbc.score(X_test_dum, y_test_dum)))
# dashboard(model_gbc)


## 4.5- Tuneamos hiperparámetros de GBC
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='log_loss', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=500, n_iter_no_change=None,
                           random_state=1234, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
model_gbc_dum = gbc.fit(X_train_dum, y_train_dum)
model_gbc_dum.score(X_test_dum, y_test_dum)
y_pred_gbc_dum = model_gbc_dum.predict(X_test_dum)
mostrar_matriz(y_test_dum, y_pred_gbc_dum)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_train_dum, model_gbc_dum.predict_proba(X_train_dum)[:, 1]))
curva_roc(y_test_dum, y_pred_gbc_dum, model_gbc_dum, X_test_dum)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_gbc_dum.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_gbc_dum.score(X_test_dum, y_test_dum)))


## 4.6- Modelo XGBOOST
# Creamos el modelo de GBC
model_xgboost = create_model('xgboost')
# no ha hecho nada con los hiperparámetros. Podemos tunearlo
print(model_xgboost)
# Imprimimos matriz de confusión
plot_model(model_xgboost, plot = "confusion_matrix", plot_kwargs={'percent' : False})
plot_model(model_xgboost, plot = 'auc')
# Lanzamos el widget completo de evaluación del modelo. Algunas llamadas dan error.
evaluate_model(model_xgboost)

# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_xgboost.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_xgboost.score(X_test_dum, y_test_dum)))
# dashboard(model_xgboost)


## 4.7- Modelo XGBOOST AJUSTADO
# Añadimos max_Depth=60 y estimadores
xgbc = XGBClassifier(base_score=None, booster='gbtree', callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device='cpu', early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=60, max_leaves=None,
              min_child_weight=None, monotone_constraints=None,
              multi_strategy=None, n_estimators=500, n_jobs=-1,
              num_parallel_tree=None, objective='binary:logistic', random_state=1234)
model_xgbc_dum = xgbc.fit(X_train_dum, y_train_dum)
model_xgbc_dum.score(X_test_dum, y_test_dum)
y_pred_xgbc_dum = model_xgbc_dum.predict(X_test_dum)
mostrar_matriz(y_test_dum, y_pred_xgbc_dum)
# Extraemos el Area bajo la curva ROC, porque ayuda mucho cuando hay desbalance entre 0 y 1
print('Curva ROC: ', metrics.roc_auc_score(y_train_dum, model_xgbc_dum.predict_proba(X_train_dum)[:, 1]))
curva_roc(y_test_dum, y_pred_xgbc_dum, model_xgbc_dum, X_test_dum)
# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(model_xgbc_dum.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_xgbc_dum.score(X_test_dum, y_test_dum)))

"""
# 5. ENFOQUE DEEP LEARNING
En este punto derivamos al notebook en Google Collab
# 6. EXPLICACIÓN DE UNA PREDICCIÓN <br>
¿Qué ha tenido en cuenta el modelo para predecir que la bomba en la posición de registro 5.300 tiene el 97% de probabilidades de funcionar? <br>
- Que el grupo de extracción es el 1. <br>
- Que la clase-tipo de extracción sea el 0. <br>
- Que la calidad del agua sea 6. <br>
- Que la fuente sea el 8. <br>
- Que el instalador sea aquél con una frecuencia de 21756 <br>

y, ¿Qué ha tenido en cuenta para NO incluir el registro en la clase de function? 3%: <br>
- Que el año de construcción sea 2009. <br>
- Que la cantidad sea 1. <br>
- Que han pasado 1016 días en que se registró la bomba. <br>
- Que la longitud sea 220.77. <br>
- que el fundador es aquél con una frecuencia 110. <br>
"""


# Usamos la librería Lime para explicar los resultados
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_dum.values, mode='classification',training_labels=y_train_dum,feature_names=X_train_dum.columns)


exp = explainer.explain_instance(X_train_dum.iloc[123], model_lightgbm.predict_proba, num_features=21)
exp.show_in_notebook(show_all=True)
exp.as_pyplot_figure()
html = exp.as_html()

# Guardar el HTML en un archivo
with open("lime_explanation.html", "w") as file:
    file.write(html)


# 7. PREDICCIONES Y EXPORTACIÓN DEL MODELO
# Guardamos el modelo para poder llamarlo después sin tener que re-entrenarlo
pkl_filename = "model_lightgbm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_lightgbm, file) # type: ignore

def predict_conversion(Gender_0,Channels_3,Channels_2,Channels_4,Channels_5,Objetivo_2,Objetivo_3,Objetivo_1,Age,Income,AdSpend,ClickThroughRate,ConversionRate,WebsiteVisits,PagesPerVisit,TimeOnSite,SocialShares,
                            EmailOpens,EmailClicks,PreviousPurchases,LoyaltyPoints):
    # Load model
    with open("model_lightgbm.pkl", "rb") as f:
        model_final = pickle.load(f)

    input_data = [Gender_0,Channels_3,Channels_2,Channels_4,Channels_5,Objetivo_2,Objetivo_3,Objetivo_1,Age,Income,AdSpend,ClickThroughRate,ConversionRate,WebsiteVisits,PagesPerVisit,TimeOnSite,SocialShares,
                            EmailOpens,EmailClicks,PreviousPurchases,LoyaltyPoints]

    input_df = pd.DataFrame([input_data], columns=['Gender_0', 'Channels_3', 'Channels_2', 'Channels_4', 'Channels_5', 'Objetivo_2', 'Objetivo_3', 'Objetivo_1', 'Age', 'Income', 'AdSpend', 'ClickThroughRate',
    'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints'])
    
    # Make prediction: ver cómo sacar las probabilidades y pintarlas que queda mejor
    pred = model_final.predict_proba(input_df)
    prediccion_final = {"Probabilidad de SÍ compra": np.round(float(pred[0][1]),2), "Probabilidad de NO compra": np.round(float(pred[0][0]),2)}
        
    return prediccion_final
    #return pred
desc = "Esta aplicación calcula la probabilidad de que un usuario realice una compra en base a las variables a continuación: " \
       "Todos los campos son *Obligatorios* para una correcta predicción"

iface = gr.Interface(fn=predict_conversion,
                     inputs=[gr.Number(label="¿Es hombre? Sí=1 / No=0"),
                             gr.Number(label="¿Es campaña de Email? Sí=1 / No=0"),
                             gr.Number(label="¿Es campaña de PPC? Sí=1 / No=0"),
                             gr.Number(label="¿Es campaña de SEO? Sí=1 / No=0"),
                             gr.Number(label="¿Es campaña de Social Media? Sí=1 / No=0"),
                             gr.Number(label="¿El objetivo es Awareness? Sí=1 / No=0"),
                             gr.Number(label="¿El objetivo es Consideration? Sí=1 / No=0"),
                             gr.Number(label="¿El objetivo es Conversion? Sí=1 / No=0"),
                             gr.Number(label="¿Qué edad tiene?"),
                             gr.Number(label="¿Qué ingresos tiene?"),
                             gr.Number(label="¿Cuál ha sido el gasto en publicidad?"),
                             gr.Number(label="¿Cuál es el CTR? número con decimales"),
                             gr.Number(label="¿Cuál es la tasa de conversión? número con decimales"),
                             gr.Number(label="¿Cuántas visitas a la web ha hecho?"),
                             gr.Number(label="¿Cuántas páginas por visita ha hecho?"),
                             gr.Number(label="¿Cuál ha sido el tiempo en página en minutos?"),
                             gr.Number(label="¿Cuántos social shares ha hecho?"),
                             gr.Number(label="¿Cuántas aperturas de email ha hecho?"),
                             gr.Number(label="¿Cuántos clicks en los emails ha hecho?"),
                             gr.Number(label="¿Cuántas compras previas ha hecho?"),
                             gr.Number(label="¿Cuántos puntos de fidelización tiene?")],
                              outputs= gr.Label("RESULTADOS:"),
                              
                       title="Modelo de predicción de compra en campañas de Marketing Digital",
                       description=desc,
                       theme=gr.themes.Soft())
iface.launch(share=True)

# Librería que proporciona una explicación del modelo. Adjuntamos html en el proyecto.
# dashboard(model_lightgbm)
