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

