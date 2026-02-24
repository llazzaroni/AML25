import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression

SEED = 25
np.random.seed(SEED)

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def feature_engineering_spearman(
    X_train, y_train,
    top_k_corr=200,                  #numbers of features kept by correlation with target
    rf_keep=200,                     # numbers of features kept by RF importance
    rf_n_estimators=1000,
    rf_max_depth=None,
    rf_min_samples_leaf=1,
    rf_max_features="sqrt",      
    corr_feature_max=0.99,        
    low_var_threshold=0.1,

):

    X_train = pd.DataFrame(X_train).copy()
    y_train = pd.Series(np.asarray(y_train).reshape(-1), index=X_train.index)


    #median imputation
    med = X_train.median(axis=0)
    X_train_vt = X_train.fillna(med)
    vt_cols = X_train.columns
    
    mean = X_train_vt.mean()
    std = X_train_vt.std(ddof=0)

    X_train_scaled = (X_train_vt - mean) / std

    imputer = KNNImputer(n_neighbors=1, weights='distance')
    #imputer = IterativeImputer(estimator=SVR(kernel='rbf', C=52, gamma="scale"), initial_strategy='median', max_iter=10)
    X_train_vt = pd.DataFrame(
        imputer.fit_transform(X_train_scaled),
        columns=vt_cols, index=X_train.index
    )

    #pearson = X_train_vt.corrwith(y_train, method='pearson').abs()
    spearman = X_train_vt.corrwith(y_train, method='spearman').abs()
    #mi = pd.Series(mutual_info_regression(X_train_vt, y_train, random_state=SEED), index=X_train_vt.columns)

    #combined = (pearson.rank() + spearman.rank()) / 2
    top_features = spearman.nlargest(top_k_corr).index

    #print(f"[SELECT] Kept top-{len(corr_keep_cols)} features by |Pearson r| with target (requested {top_k_corr}).")


    X_train_decorr = X_train_vt.loc[:,top_features]

    # RandomForest-based selection, keep top 'rf_keep' features
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features, 
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train_decorr, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train_decorr.columns).sort_values(ascending=False)

    if rf_keep is None or rf_keep <= 0:
        final_cols = importances.index.tolist()
    else:
        final_cols = importances.head(min(rf_keep, len(importances))).index.tolist()
    
    return final_cols


def feature_engineering_mi(
    X_train, y_train,
    top_k_corr=200,                  #numbers of features kept by correlation with target
    rf_keep=200,                     # numbers of features kept by RF importance
    rf_n_estimators=1000,
    rf_max_depth=None,
    rf_min_samples_leaf=1,
    rf_max_features="sqrt",      
    corr_feature_max=0.99,        
    low_var_threshold=0.1,

):

    X_train = pd.DataFrame(X_train).copy()
    y_train = pd.Series(np.asarray(y_train).reshape(-1), index=X_train.index)


    #median imputation
    med = X_train.median(axis=0)
    X_train_vt = X_train.fillna(med)
    vt_cols = X_train.columns
    
    mean = X_train_vt.mean()
    std = X_train_vt.std(ddof=0)

    X_train_scaled = (X_train_vt - mean) / std

    imputer = KNNImputer(n_neighbors=1, weights='distance')
    #imputer = IterativeImputer(estimator=SVR(kernel='rbf', C=52, gamma="scale"), initial_strategy='median', max_iter=10)
    X_train_vt = pd.DataFrame(
        imputer.fit_transform(X_train_scaled),
        columns=vt_cols, index=X_train.index
    )

    #pearson = X_train_vt.corrwith(y_train, method='pearson').abs()
    #spearman = X_train_vt.corrwith(y_train, method='spearman').abs()
    mi = pd.Series(mutual_info_regression(X_train_vt, y_train, random_state=SEED), index=X_train_vt.columns)

    #combined = (pearson.rank() + spearman.rank()) / 2
    top_features = mi.nlargest(top_k_corr).index

    #print(f"[SELECT] Kept top-{len(corr_keep_cols)} features by |Pearson r| with target (requested {top_k_corr}).")


    X_train_decorr = X_train_vt.loc[:,top_features]

    # RandomForest-based selection, keep top 'rf_keep' features
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features, 
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train_decorr, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train_decorr.columns).sort_values(ascending=False)

    if rf_keep is None or rf_keep <= 0:
        final_cols = importances.index.tolist()
    else:
        final_cols = importances.head(min(rf_keep, len(importances))).index.tolist()
    
    return final_cols