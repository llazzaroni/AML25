import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

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

# 202, 173

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train, y_train, X_test = preprocessing.preprocess(X_train_df, X_test_df, y_train_df)

    final_cols_SVR = features.feature_engineering_spearman(
        X_train=X_train, y_train=y_train,
        top_k_corr=202,
        rf_keep=173,
    )

    X_train_feat = X_train[final_cols_SVR].copy()
    X_test_feat  = X_test[final_cols_SVR].copy()
    y_train_feat = pd.DataFrame(y_train)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("impute", KNNImputer(n_neighbors=2, weights='distance')),
        ("model", NuSVR(nu=0.5, kernel='rbf', C=55, gamma="auto"))
    ])

                
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
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
    print("######")



if __name__ == "__main__":
    main()