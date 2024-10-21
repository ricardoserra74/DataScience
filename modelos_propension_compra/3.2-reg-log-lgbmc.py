# Creamos el modelo de LIGHTGBM
model_lightgbm = create_model('lightgbm')
print(model_lightgbm)

# Imprimimos matriz de confusión y curva ROC
plot_model(model_lightgbm, plot = "confusion_matrix", plot_kwargs={'percent' : False})
plot_model(model_lightgbm, plot = 'auc')

# Obtenemos evaluación del modelo
evaluate_model(model_lightgbm)

# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting
print('Training set score: {:.4f}'.format(model_lightgbm.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_lightgbm.score(X_test_dum, y_test_dum)))
