from functools import partial

import numpy as np
import optuna
import sklearn
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import ShuffleSplit, cross_validate
from xgboost import XGBClassifier


def objective(trial, X_train, y_train):
    use_feature_selection = False
    classifier_name = trial.suggest_categorical('classifier', ['XGBoost',
                                                               'LogisticRegression'])  # ['LogisticRegression']) # ['RandomForest', 'XGBoost', 'LogisticRegression'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_float('svc_c', 1e-1, 1e3, log=True)
        model = sklearn.svm.SVC(C=svc_c, gamma='auto')
    elif classifier_name == 'RandomForest':
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 2, 16, log=True)
        model = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    elif classifier_name == 'LogisticRegression':
        # use_feature_selection = True
        logistic_regression_c = trial.suggest_float('logistic_regression_c', 1e-4, 1e1, log=True)
        logr_penalty = trial.suggest_categorical('logr_penalty', ["l1", "l2"])
        model = sklearn.linear_model.LogisticRegression(C=logistic_regression_c, penalty=logr_penalty,
                                                        solver='liblinear')
    else:
        param = {
            'booster': trial.suggest_categorical('xgb_booster', ['gbtree', 'dart']),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 2, 16, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 1, 4),
            # 'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        }
        model = XGBClassifier(**param)

    # subsample = trial.suggest_categorical('subsample', [True, False])
    # if subsample:
    #     subsample_indexs = balanced_subsample(y_train)
    #     X_train = X_train.loc[subsample_indexs]
    #     y_train = y_train.loc[subsample_indexs]

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    if use_feature_selection:
        fix_feature_selection = trial.suggest_categorical('fix_feature_selection', [True, False])
        n_features_to_select = 'auto'
        if fix_feature_selection:
            alpha_features_to_select = trial.float("alpha_features_to_select", 0.1, 0.9)
        sfs = SequentialFeatureSelector(model, scoring='f1_weighted', n_features_to_select=alpha_features_to_select,
                                        cv=cv, n_jobs=4)
        X_train = sfs.fit_transform(X_train, y_train)

    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=['accuracy', 'precision', 'f1', 'f1_weighted', 'f1_macro'])
    prec_alpha = 0.8
    mean_f1_prec_score = np.mean(scores['test_f1_weighted'] + scores['test_precision'] * prec_alpha)

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
