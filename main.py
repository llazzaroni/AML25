from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import NuSVR
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

# Project specific imports
from utils import outliers as outliers
from utils import features as features
from utils import preprocessing as preprocessing
from stack import LabelColumnSelector

SEED = 25
np.random.seed(SEED)

def main() -> None:

    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train, y_train, X_test = preprocessing.preprocess(X_train_df, X_test_df, y_train_df)

    SVR_branch = Pipeline([
        ("sel", LabelColumnSelector(top_k_corr=202, rf_keep=173, selection_mechanism="spearman")),
        ("scale", StandardScaler()),
        ("impute2", KNNImputer(n_neighbors=2, weights='distance')),
        ("model", NuSVR(nu=0.5, kernel='rbf', C=55, gamma="scale"))
    ])
    
    kernel = kernels.ConstantKernel(1.21**2) * kernels.RationalQuadratic(length_scale=10.6, alpha=0.532) \
         + kernels.WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e2))
    
    # 197, 168
    GPR_branch = Pipeline([
        ("sel", LabelColumnSelector(top_k_corr=202, rf_keep=173, selection_mechanism="spearman")),
        ("scale", RobustScaler()),
        ("impute2", KNNImputer(n_neighbors=2, weights="distance")),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=SEED
        ))
    ])

    HGB_branch = Pipeline([
            ("sel", LabelColumnSelector(top_k_corr=197, rf_keep=168)),
            ("scale", StandardScaler()),
            ("impute", KNNImputer(n_neighbors=2, weights="distance")),
            ("model", HistGradientBoostingRegressor()),
        ]
    )

    model1 = StackingRegressor(
        estimators = [("svr", SVR_branch), ("hgb", HGB_branch), ("gpr", GPR_branch)],
        final_estimator = LinearRegression(),
        cv=5
    )

    model1.fit(X_train, y_train.values.ravel())
    y_hat = model1.predict(X_test)
    table = pd.DataFrame({'id': np.arange(0, y_hat.shape[0]), 'y': y_hat.flatten()})
    table.to_csv('submission2.csv', index=False)

if __name__ == "__main__":
    main()