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

SEED = 25
np.random.seed(SEED)

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def feature_engineering_celestin(
    X_train, y_train, X_test,
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
    X_test  = pd.DataFrame(X_test).copy()
    y_train = pd.Series(np.asarray(y_train).reshape(-1), index=X_train.index)


    #median imputation
    med = X_train.median(axis=0)
    X_train_imp = X_train.fillna(med)
    X_test_imp  = X_test.fillna(med)

    miss_before = int(X_train.isna().sum().sum())
    print(f"[IMPUTE] Median imputation. Missing before: {miss_before} -> after: 0")

    # standardize 
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=X_train.columns, index=X_train.index
    )
    X_test_std = pd.DataFrame(
        scaler.transform(X_test_imp), columns=X_test.columns, index=X_test.index
    )

    print("[SCALE] StandardScaler applied (fit on train).")

    #Remove low-variance features
    vt = VarianceThreshold(threshold=low_var_threshold)
    X_train_vt = vt.fit_transform(X_train_std)
    kept_idx_vt = vt.get_support(indices=True)
    vt_cols = X_train_std.columns[kept_idx_vt]
    X_train_vt = pd.DataFrame(X_train_vt, columns=vt_cols, index=X_train_std.index)
    X_test_vt  = pd.DataFrame(vt.transform(X_test_std), columns=vt_cols, index=X_test_std.index)

    dropped_zero_var = [c for c in X_train_std.columns if c not in vt_cols]

    print(f"[FILTER] low-variance features: {len(dropped_zero_var)}")

    #Top-K by absolute Pearson correlation with target (on VT-filtered set)
    corrs = {}
    yv = y_train
    for c in vt_cols:
        x = X_train_vt[c]
        if x.std() == 0:
            corrs[c] = 0.0
            continue
        try:
            corrs[c] = float(np.corrcoef(x, yv)[0, 1])
        except Exception:
            corrs[c] = 0.0
    corr_s = pd.Series(corrs).abs().sort_values(ascending=False)
    corr_keep_cols = corr_s.head(min(top_k_corr, len(corr_s))).index.tolist()

    X_train_corr = X_train_vt[corr_keep_cols].copy()
    X_test_corr  = X_test_vt[corr_keep_cols].copy()

    print(f"[SELECT] Kept top-{len(corr_keep_cols)} features by |Pearson r| with target (requested {top_k_corr}).")


    X_train_decorr = X_train_corr
    X_test_decorr  = X_test_corr
    dropped = []

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

    X_train_final = X_train_decorr[final_cols].copy()
    X_test_final  = X_test_decorr[final_cols].copy()


    print(f"[RF] RandomForest feature selection:")
    print(f"     - kept {len(final_cols)} features (requested {rf_keep})")
    print(f"     - top-10 importances:\n{importances.head(10)}")
    
    return final_cols