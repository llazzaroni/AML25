import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, WhiteKernel, Matern, RBF
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor



# Project specific imports
from utils import outliers as outliers
from utils import features as features
from utils import preprocessing as preprocessing

def spaced_ints(start, stop, num):
    """Return exactly `num` unique, evenly spaced integers in [start, stop]."""
    if start > stop:
        return np.array([], dtype=int)
    grid = np.arange(start, stop + 1)
    num = min(num, len(grid))
    chunks = np.array_split(grid, num)
    return np.array([c[len(c)//2] for c in chunks], dtype=int)  

SEED = 25
np.random.seed(SEED)

# 211, 187
# 197, 168

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train, y_train, X_test = preprocessing.preprocess(X_train_df, X_test_df, y_train_df)

    for top_k_corr in spaced_ints(210, 210, 1):
        for rf_keep in spaced_ints(180, 180, 1):
            final_cols_HGBR = features.feature_engineering_spearman(
                X_train=X_train, y_train=y_train, X_test=X_test,
                top_k_corr=top_k_corr,
                rf_keep=rf_keep,
            )

            X_train_feat = X_train[final_cols_HGBR].copy()
            X_test_feat  = X_test[final_cols_HGBR].copy()
            y_train_feat = pd.DataFrame(y_train)


            pipe = Pipeline([
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
                        
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_scores = []
            for tr_idx, te_idx in kf.split(X_train_feat):
                X_tr, X_te = X_train_feat.iloc[tr_idx], X_train_feat.iloc[te_idx]
                y_tr, y_te = y_train_feat.iloc[tr_idx], y_train_feat.iloc[te_idx]

                # Fit the pipeline
                pipe.fit(X_tr, y_tr.values.ravel())
                y_hat = pipe.predict(X_te)
                fold_scores.append(r2_score(y_te, y_hat))
                print("end of split")

            mean_r2 = float(np.mean(fold_scores))
            std_r2  = float(np.std(fold_scores, ddof=1))

            print(mean_r2, std_r2)
            print("top_k_corr:", top_k_corr, "rf_keep:", rf_keep)
            print("######")



if __name__ == "__main__":
    main()