from functools import partial

import numpy as np
import optuna
import sklearn
from sklearn.model_selection import ShuffleSplit, cross_validate
from xgboost import XGBClassifier

from clinical_predictions.utils import balanced_subsample


def objective(trial, X_train, y_train):
    classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'XGBoost'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_float('svc_c', 1e-10, 1e10, log=True)
        model = sklearn.svm.SVC(C=svc_c, gamma='auto')
    elif classifier_name == 'Lasso':
        alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        model = sklearn.linear_model.Lasso(alpha=alpha)
    elif classifier_name == 'RandomForest':
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        model = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)
    else:
        param = {
            # 'objective': 'binary:logistic',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'n_estimators': trial.suggest_int('n_estimators', 2, 16, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 3),
            # 'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            # 'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        }
        model = XGBClassifier(**param)

    subsample = trial.suggest_categorical('subsample', [True, False])
    if subsample:
        subsample_indexs = balanced_subsample(y_train)
        X_train = X_train.loc[subsample_indexs]
        y_train = y_train.loc[subsample_indexs]

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'precision'])
    prec_alpha = 1
    mean_f1_prec_score = np.mean(scores['test_accuracy'] + scores['test_precision'] * prec_alpha)

    trial.set_user_attr(key="best_booster", value=model)
    return mean_f1_prec_score


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def get_best_model_with_optuna(X_train, y_train, n_trials=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, X_train=X_train, y_train=y_train), n_trials=n_trials, callbacks=[callback])
    best_model = study.user_attrs["best_booster"]
    best_trail = study.best_trial
    return best_model, best_trail
