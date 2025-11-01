from utils import outliers as outliers

def preprocess(X_train_df, X_test_df, y_train_df):
    X_train = X_train_df.iloc[:, 1:]
    y_train = y_train_df.iloc[:, 1:]
    X_test = X_test_df.iloc[:, 1:]

    # First remove the 0-variance columns
    eps = 1e-7
    mean = X_train.mean()
    std = X_train.std(ddof=0)
    cols = std[std > eps].index
    X_train = X_train.loc[:,cols]
    X_test = X_test.loc[:,cols]

    # Then remove the outliers
    X_train_in_i, y_train_in_i, X_test_tmp = outliers.remove_outliers_IF(
        X_train=X_train, y_train=y_train, X_test=X_test,
        contamination=0.047,
    )

    return X_train_in_i, y_train_in_i, X_test_tmp
