# Borramos las variables aleatorias
imput = imput.drop(['aleatorio', 'aleatorio2'], axis=True)

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

# Aplicamos a fórmula de modelo completo
formC=ols_formula(data_train,'y')
formC

# One-hot encoding: convertimos las variables categóricas en numéricas, y generamos las matrices de diseño según la fórmula de modelo completo.
y_tr, X_tr = patsy.dmatrices(formC, data_train, return_type='dataframe')

# Generamos las matrices de diseño según la fórmula de modelo completo para test
y_tst, X_tst = patsy.dmatrices(formC, data_test, return_type='dataframe')

# Definición de modelo: hay muchos solver, python usa por defecto éste. Aplicamos el solver newton-cg porque es el que nos da valores más altos de las métricas
modelo1 = LogisticRegression(solver='newton-cg', max_iter=1000, penalty=None)

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




