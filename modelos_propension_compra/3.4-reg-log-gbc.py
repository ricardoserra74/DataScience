# Creamos el modelo de GBC
model_gbc = create_model('gbc')

# no ha hecho nada con los hiperparámetros. Podemos tunearlo
print(model_gbc)

# Imprimimos matriz de confusión y curva ROC
plot_model(model_gbc, plot = "confusion_matrix", plot_kwargs={'percent' : False })
plot_model(model_gbc, plot = 'auc')

# Lanzamos el widget completo de evaluación del modelo. Algunas llamadas dan error.
evaluate_model(model_gbc)

# comprobamos si hay overfitting: la precisión es muy comparable en ambos sets, así que concluimos que no hay overfitting
print('Training set score: {:.4f}'.format(model_gbc.score(X_train_dum, y_train_dum)))
print('Test set score: {:.4f}'.format(model_gbc.score(X_test_dum, y_test_dum)))

