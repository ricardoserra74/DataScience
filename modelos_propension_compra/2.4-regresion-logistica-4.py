# Creamos un modelo con oversampling y undersamplig, SMOTE y TOMEK

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

# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting

print('Training set score: {:.4f}'.format(modelLogtomek.score(X_train_tomek, y_train_tomek)))
print('Test set score: {:.4f}'.format(modelLogtomek.score(X_tst2, y_tst2)))
