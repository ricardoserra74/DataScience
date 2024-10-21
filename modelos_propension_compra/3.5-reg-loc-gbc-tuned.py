# Creamos un modelo GBC en el que ajustamos algunos hiperparámetros:
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

