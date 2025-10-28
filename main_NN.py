import utils.outliers as outliers
import utils.features as features
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import torch
from utils.data import get_subsplits
from torch.utils.data import DataLoader
from utils.model import Net
from utils.model import Trainer
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import r2_score

# Nice but doesn't achieve more than 0.65 r2

def main() -> None:
    X_train_df = pd.read_csv('data/X_train.csv', skiprows=1, header=None)
    y_train_df = pd.read_csv('data/y_train.csv', skiprows=1, header=None)
    X_test_df = pd.read_csv('data/X_test.csv', skiprows=1, header=None)

    X_train = X_train_df.iloc[:, 1:]
    y_train = y_train_df.iloc[:, 1:]
    X_test = X_test_df.iloc[:, 1:]

    X_train_in_i, y_train_in_i, X_test_tmp = outliers.remove_outliers_IF(
        X_train=X_train, y_train=y_train, X_test=X_test,
        contamination=0.09,
    )

    X_train_fe_i, X_test_fe_i, final_cols = features.feature_engineering(
        X_train=X_train_in_i, y_train=y_train_in_i, X_test=X_test_tmp,
        top_k_corr=200,
        rf_keep=167,
    )

    # Create the dataset
    train_subsplit, val_subsplit = get_subsplits(X_train_fe_i, y_train_in_i)
    print("Splits have been created")

    batch_size = 128
    num_workers = 1

    train_loader = DataLoader(
        dataset=train_subsplit,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_subsplit,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    model = Net(in_dim=X_train_fe_i.shape[1])

    optimizer = optim.AdamW(params=model.parameters())

    loss = nn.SmoothL1Loss(beta=1.0)
    print("Model, optimizer and loss have been instantiated")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=101,
        eval_interval=5,
        loss=loss
    )

    print("Training has started")
    trainer.train()
    


    
if __name__ == "__main__":
    main()