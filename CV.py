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
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels

import lightgbm as lgb

# Project specific imports
from utils import outliers as outliers
from utils import features as features

def spaced_ints(start, stop, num):
    """Return exactly `num` unique, evenly spaced integers in [start, stop]."""
    if start > stop:
        return np.array([], dtype=int)
    grid = np.arange(start, stop + 1)
    num = min(num, len(grid))          # cap if you ask for too many
    chunks = np.array_split(grid, num) # split into num buckets
    return np.array([c[len(c)//2] for c in chunks], dtype=int)  

SEED = 25
np.random.seed(SEED)

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train = X_train_df.iloc[:, 1:]
    y_train = y_train_df.iloc[:, 1:]
    X_test = X_test_df.iloc[:, 1:]

    for contamination in [0.047]:

        # First remove the 0-variace columns
        print(X_train.shape[1])
        eps = 1e-7
        mean = X_train.mean()
        std = X_train.std(ddof=0)
        cols = std[std > eps].index
        X_train = X_train.loc[:,cols]
        X_test = X_test.loc[:,cols]
        print(X_train.shape[1])

        X_train_in_i, y_train_in_i, X_test_tmp = outliers.remove_outliers_IF(
            X_train=X_train, y_train=y_train, X_test=X_test,
            contamination=contamination,
        )

        top_k_corr_loop = spaced_ints(202, 202, 1)

        for top_k_corr in top_k_corr_loop:
            rf_vals = spaced_ints(173, 173, 1)
            for rf_keep in rf_vals:

                final_cols = features.feature_engineering_celestin(
                    X_train=X_train_in_i, y_train=y_train_in_i, X_test=X_test_tmp,
                    top_k_corr=top_k_corr,
                    rf_keep=rf_keep,
                )

                # Use the knn imputer instead of the median

                # First rescale but without the median
                X_train_feat = X_train_in_i[final_cols]
                X_test_feat = X_test_tmp[final_cols]

                mean = X_train_feat.mean()
                std = X_train_feat.std(ddof=0)

                X_train_scaled = (X_train_feat - mean) / std
                X_test_scaled = (X_test_feat - mean) / std

                '''''''''

                imputer = KNNImputer(n_neighbors=10, weights='distance')
                #imputer = IterativeImputer(initial_strategy='median', max_iter=30)
                X_train_imputed = imputer.fit_transform(X_train_scaled)
                X_test_imputed = imputer.transform(X_test_scaled)
                print("second imputer is done")

                

                X_train_fe_i = pd.DataFrame(X_train_imputed, columns=X_train_feat.columns)
                X_test_fe_i = pd.DataFrame(X_test_imputed, columns=X_test_feat.columns)
                '''
                X_train_fe_i = pd.DataFrame(X_train_scaled, columns=X_train_feat.columns)
                X_test_fe_i = pd.DataFrame(X_test_scaled, columns=X_test_feat.columns)
                

                kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

                fold_scores = []
                for tr_idx, te_idx in kf.split(X_train_fe_i):
                    X_tr, X_te = X_train_fe_i.iloc[tr_idx], X_train_fe_i.iloc[te_idx]
                    y_tr, y_te = y_train_in_i.iloc[tr_idx], y_train_in_i.iloc[te_idx]

                    imputer = KNNImputer(n_neighbors=2, weights='distance')
                    #imputer = IterativeImputer(initial_strategy='median', max_iter=30)
                    X_train_imputed = imputer.fit_transform(X_tr)
                    X_test_imputed = imputer.transform(X_te)
                    #print("second imputer is done")


                    #model = SVR(kernel='rbf', C=52, gamma="scale")
                    model = NuSVR(nu=0.5, kernel='rbf', C=55, gamma="scale")
                    #model = AdaBoostRegressor(estimator=NuSVR(nu=0.5, kernel='rbf', C=52, gamma="scale"))
                    #model2 = BaggingRegressor(estimator=NuSVR(nu=0.5, kernel='rbf', C=52, gamma="scale"))
                    #model3 = NuSVR(nu=0.5, kernel='rbf', C=52, gamma="scale")
                    #model = GradientBoostingRegressor(estimator=NuSVR(nu=0.5, kernel='rbf', C=52, gamma="scale"))
                    #kernel1 = kernels.RBF()
                    #kernel2 = kernels.Matern()
                    #kernel3 = kernels.RationalQuadratic()
                    #model = GaussianProcessRegressor(kernel=kernel2)
                    #model = DecisionTreeRegressor(criterion='absolute_error')
                    #model = GradientBoostingRegressor(n_estimators=200)
                    #model = RandomForestRegressor()
                    model.fit(X_train_imputed, y_tr)
                    y_hat = model.predict(X_test_imputed)
                    fold_scores.append(r2_score(y_te, y_hat))
                    print("end of split")

                mean_r2 = float(np.mean(fold_scores))
                std_r2  = float(np.std(fold_scores, ddof=1))

                print(mean_r2, std_r2)
                print("top_k_corr:", top_k_corr, "rf_keep:", rf_keep)
                #print("contamination:", contamination)
                print("######")



if __name__ == "__main__":
    main()