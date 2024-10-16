# Creamos un tercer modelo con hiperparámetros
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

# Aplicamos el resultado al nuevo modelo
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

