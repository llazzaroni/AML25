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
from sklearn.decomposition import PCA

SEED = 25
np.random.seed(SEED)

def remove_outliers_IF(
    X_train,
    y_train,
    X_test,
    contamination=0.8,     
):
    # Median imputation
    med = X_train.median(axis=0)
    Xtr_imp = X_train.fillna(med)
    Xte_imp = X_test.fillna(med)
    y_train = pd.Series(np.asarray(y_train).reshape(-1), index=X_train.index)

    # Scaling
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_std = scaler.fit_transform(Xtr_imp)

    # 2D latent embedding (PCA)
    pca = PCA(n_components=2, random_state=SEED)
    X_proj = pca.fit_transform(Xtr_std)

    # Isolation forest
    iso = IsolationForest(contamination=contamination, random_state=SEED)
    mask = iso.fit_predict(X_proj) == 1

    X_train, y_train = X_train[mask], y_train[mask]
    return X_train, y_train, X_test