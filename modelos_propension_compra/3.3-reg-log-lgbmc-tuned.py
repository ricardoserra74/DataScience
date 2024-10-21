# Cfreamos un modelo LGBMC donde tuneamos los parámetros: Le incremento los estimadores y aplico balanceo
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


