from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import NuSVR
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel, Matern, RBF
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb


# Project specific imports
from utils import outliers as outliers
from utils import features as features
from utils import preprocessing as preprocessing

SEED = 25

# Simple column selector that keeps a branch-specific feature list
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class LabelColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        # DO NOT coerce here — keep exactly what the user passed
        self.columns = columns

    def fit(self, X, y=None):
        # Create learned attributes here
        self.columns_ = list(self.columns)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns in X (first few): {missing[:5]}")
        return self

    def transform(self, X):
        return X.loc[:, self.columns_]

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train, y_train, X_test = preprocessing.preprocess(X_train_df, X_test_df, y_train_df)

    final_cols_SVR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train, X_test=X_test,
        top_k_corr=202,
        rf_keep=173,
    )

    SVR_branch = Pipeline([
        ("sel", LabelColumnSelector(final_cols_SVR)),
        ("scale", StandardScaler()),
        ("impute", KNNImputer(n_neighbors=2, weights='distance')),
        ("model", NuSVR(nu=0.5, kernel='rbf', C=55, gamma="auto"))
    ])

    final_cols_HGBR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train, X_test=X_test,
        top_k_corr=197,
        rf_keep=168,
    )

    HGBR_branch = Pipeline([
        ("sel", LabelColumnSelector(final_cols_HGBR)),
        ("scale", StandardScaler()),
        ("impute", KNNImputer(n_neighbors=2, weights='distance')),
        ("model", HistGradientBoostingRegressor())
    ])

    final_cols_GPR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train, X_test=X_test,
        top_k_corr=197,
        rf_keep=168,
    )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) \
            * RationalQuadratic(length_scale=1.0, alpha=1.0) \
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

    GPR_branch = Pipeline([
        ("sel", LabelColumnSelector(final_cols_GPR)),
        ("scale", RobustScaler()),
        ("impute1", KNNImputer(n_neighbors=2, weights="distance")),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=SEED
        ))
    ])

    final_cols_ABR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train, X_test=X_test,
        top_k_corr=197,
        rf_keep=160,
    )

    ABR_branch = Pipeline([
        ("sel", LabelColumnSelector(final_cols_ABR)),
        ("scale", StandardScaler()),
        ("imp", KNNImputer(n_neighbors=2, weights="distance")),
        ("model", AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=15,              # tune: 2–6
                min_samples_leaf=5,       # tune: 1–20
                random_state=SEED
            ),
            n_estimators=600,             # tune: 200–1500
            learning_rate=0.03,           # tune with n_estimators (smaller lr -> more trees)
            loss="square",                # 'linear' or 'square' are usually better than 'exponential' (less outlier-sensitive)
            random_state=SEED
        ))
    ])

    final_cols_ETR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train, X_test=X_test,
        top_k_corr=197,
        rf_keep=160,
    )

    ETR_branch = Pipeline([
        ("sel", LabelColumnSelector(final_cols_ETR)),
        ("scale", StandardScaler()),
        ("imp", KNNImputer(n_neighbors=2, weights="distance")),
        ("model", ExtraTreesRegressor(
            n_estimators=1000, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", random_state=SEED, n_jobs=-1
        ))
    ])

    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_scores = []
    for tr_idx, te_idx in kf.split(X_train):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

        model = StackingRegressor(
            estimators=[("svr", SVR_branch), ("hgb", HGBR_branch), ("etr", ETR_branch), ("abr", ABR_branch), ("gpr", GPR_branch)],
            final_estimator=LinearRegression(n_jobs=None)
        )

        # Fit the pipeline
        model.fit(X_tr, y_tr.values.ravel())
        y_hat = model.predict(X_te)
        fold_scores.append(r2_score(y_te, y_hat))
        print("end of split")

    mean_r2 = float(np.mean(fold_scores))
    std_r2  = float(np.std(fold_scores, ddof=1))

    print(mean_r2, std_r2)
    print("######")



if __name__ == "__main__":
    main()