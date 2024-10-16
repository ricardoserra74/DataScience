## IMPORTAMOS LOS DATOS DE LAS VARIABLES Y LA OBJETIVO
dataset = pd.read_csv('digital_marketing_campaign_dataset.csv', sep=",")

# Utilizamos la librería ydata_profiling para hacer un análisis descriptivo de las variables
profile = ydata_profiling.ProfileReport(dataset)

# Nos guardamos el html para adjuntarlo al proyecto
profile.to_file(output_file='Analisis_descriptivo.html')
profile

# Eliminamos CustomerId, AdvertisingPlatform y AdvertisingTool
dataset.drop(['AdvertisingPlatform','AdvertisingTool'], axis = 1, inplace=True)

# Renombramos la variable objetivo: para facilitar el trabajo posterior
dataset.rename(columns={'Conversion': 'y'}, inplace=True)

# Transformamos las variables categóricas
dataset['CampaignChannel'] = dataset['CampaignChannel'].astype('category')
dataset['CampaignType'] = dataset['CampaignType'].astype('category')
dataset['Gender'] = dataset['Gender'].astype('category')

# ANÁLISIS GRÁFICO DE LAS VARIABLES

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
fig.suptitle('Distribución de variables numéricas', fontsize = 10, fontweight = "bold")

# DISTRIBUCIÓN DE LAS VARIABLES NUMÉRICAS RESPECTO A LA Y

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
fig.suptitle('Distribución de variables numéricas respecto a la y', fontsize = 20, fontweight = "bold")

# DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS

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

# Creamos dos variables de control para evaluar los efectos de las variables predictoras sobre la variable objetivo
dataset['aleatorio'] = np.random.uniform(0,1,size=dataset.shape[0])
dataset['aleatorio2'] = np.random.uniform(0,1,size=dataset.shape[0])

# OUTLIERS
# Los valores atípicos y faltantes se trabajan solo de las variables input, no de la objetivo. Dividimos el dataset, que luego también usaremos
varObj = dataset.y 
imput = dataset.drop(['CustomerID', 'y'],axis=1)
# No existen asimetrías en las variables numéricas, con lo que no tenemos problemas de valores atípicos.
imput.select_dtypes(include=np.number).apply(lambda x: x.skew())

# VALORES FALTANTES, MISSINGS 
dataset.apply(lambda x: x.isna().sum()/dataset.shape[0]*100)#Proporción de missings por variable 
dataset.apply(lambda x: x.isna().sum()/dataset.shape[0]*100)

# RELACIONES ENTRE LAS VARIABLES Y LA VARIABLE OBJETIVO
# Aplicamos la función  V DE CREAMER al input completo contra la objetivo
tablaCramer = pd.DataFrame(imput.apply(lambda x: cramers_v(x,varObj)),columns=['VCramer'])

# Obtenemos el gráfico de importancia de las variables frente a la objetivo continua según vcramer
px.bar(tablaCramer,x=tablaCramer.VCramer, title='Relaciones frente a la conversión', text_auto=True).update_yaxes(categoryorder="total ascending", automargin="left").show()

# Matriz de correlaciones
corr = dataset.select_dtypes(include=np.number).corr()
corr.style.background_gradient(cmap='YlOrRd').format(precision=3)




