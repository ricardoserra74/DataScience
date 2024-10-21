# Creamos un modelo nuevo con hiperparámetros ajustados: añadimos max_Depth=60 y estimadores
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

