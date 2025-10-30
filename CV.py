import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

# Project specific imports
from utils import outliers as outliers
from utils import features as features

SEED = 25
np.random.seed(SEED)

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train = X_train_df.iloc[:, 1:]
    y_train = y_train_df.iloc[:, 1:]
    X_test = X_test_df.iloc[:, 1:]

    X_train_in_i, y_train_in_i, X_test_tmp = outliers.remove_outliers_IF(
        X_train=X_train, y_train=y_train, X_test=X_test,
        contamination=0.045,
    )

    final_cols = features.feature_engineering_celestin(
        X_train=X_train_in_i, y_train=y_train_in_i, X_test=X_test_tmp,
        top_k_corr=200,
        rf_keep=167,
    )

    # Use the knn imputer instead of the median

    # First rescale but without the median
    X_train_feat = X_train_in_i[final_cols]
    X_test_feat = X_test_tmp[final_cols]

    mean = X_train_feat.mean()
    std = X_train_feat.std(ddof=0)

    X_train_scaled = (X_train_feat - mean) / std
    X_test_scaled = (X_test_feat - mean) / std

    imputer = KNNImputer(n_neighbors=10, weights='distance')
    #imputer = IterativeImputer(estimator=SVR(kernel='rbf', C=52, gamma="scale"), initial_strategy='median', max_iter=20)
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)

    X_train_fe_i = pd.DataFrame(X_train_imputed, columns=X_train_feat.columns)
    X_test_fe_i = pd.DataFrame(X_test_imputed, columns=X_test_feat.columns)

    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

    fold_scores = []
    for tr_idx, te_idx in kf.split(X_train_fe_i):
        X_tr, X_te = X_train_fe_i.iloc[tr_idx], X_train_fe_i.iloc[te_idx]
        y_tr, y_te = y_train_in_i.iloc[tr_idx], y_train_in_i.iloc[te_idx]

        model = SVR(kernel='rbf', C=52, gamma="scale")
        #model = GradientBoostingRegressor(n_estimators=200)
        #model = RandomForestRegressor()
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        fold_scores.append(r2_score(y_te, y_hat))
        print("end of split")

    mean_r2 = float(np.mean(fold_scores))
    std_r2  = float(np.std(fold_scores, ddof=1))

    print(mean_r2, std_r2)



if __name__ == "__main__":
    main()