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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Project specific imports
from utils import outliers as outliers
from utils import features as features
from utils import preprocessing as preprocessing

SEED = 35

# Simple column selector that keeps a branch-specific feature list
from sklearn.base import BaseEstimator, TransformerMixin


class LabelColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k_corr, rf_keep, selection_mechanism="spearman"):
        self.top_k_corr = top_k_corr
        self.rf_keep = rf_keep
        self.selection_mechanism = selection_mechanism

    def fit(self, X, y=None):
        # Create learned attributes here
        if self.selection_mechanism == "spearman":
            self.columns_ = list(features.feature_engineering_spearman(X_train = X,
                                                                    y_train = y,
                                                                    top_k_corr = self.top_k_corr,
                                                                    rf_keep = self.rf_keep
            ))
        else:
            self.columns_ = list(features.feature_engineering_mi(X_train = X,
                                                                    y_train = y,
                                                                    top_k_corr = self.top_k_corr,
                                                                    rf_keep = self.rf_keep
            ))
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

    # 202, 173
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


    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_scores = []
    i = 1
    for tr_idx, te_idx in kf.split(X_train):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

        model1 = StackingRegressor(
            estimators = [("svr", SVR_branch), ("hgb", HGB_branch), ("gpr", GPR_branch)],
            final_estimator = LinearRegression(),
            cv=5
        )

        # Fit the pipeline
        model1.fit(X_tr, y_tr.values.ravel())
        y_hat = model1.predict(X_te)
        score = r2_score(y_te, y_hat)
        fold_scores.append(score)
        print("Split r2_score:", score)
        print("end of split")
        i += 1

    mean_r2 = float(np.mean(fold_scores))
    std_r2  = float(np.std(fold_scores, ddof=1))

    print(mean_r2, std_r2)
    print("######")



if __name__ == "__main__":
    main()