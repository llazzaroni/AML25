import torch.nn as nn
import numpy as np
import copy
from sklearn.metrics import r2_score
import torch

class Net(nn.Module):
    # Class for the NN to train

    def __init__(self, in_dim):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, 600),
            nn.GELU(),
            nn.Dropout(0.01),
            nn.Linear(600, 300),
            nn.Sigmoid(),
            nn.Dropout(0.01),
            nn.Linear(300, 100),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        x = self.head(x)
        return x
    


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 val_loader,
                 n_epochs,
                 eval_interval,
                 loss
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.loss = loss

    def train(self):
        self.model.train()
        best_model = self.model
        best_precision = -np.inf

        for epoch in range(self.n_epochs):
            if epoch % self.eval_interval == 0 and epoch != 0:
                precision = self.evaluate()

                if precision > best_precision:
                    best_precision = precision
                    best_model = copy.deepcopy(self.model)

                print(f"epoch {epoch} eval set precision: {precision}")

                self.model.train()

            num_samples = 0
            running_loss = 0.0
            for [X,y] in self.train_loader:
                y_out = self.model(X)

                self.optimizer.zero_grad()
                loss = self.loss(y_out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * y.size(0)
                num_samples += y.size(0)
            print(f"epoch: {epoch} / {self.n_epochs} @ loss {running_loss / num_samples}")

        return best_model

    def evaluate(self):
        self.model.eval()
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for [X, y] in self.val_loader:
                outputs = self.model(X).squeeze(-1)
                predictions.append(outputs)
                ground_truth.append(y.squeeze(-1))

        predictions = torch.cat(predictions, dim=0).numpy()
        ground_truth = torch.cat(ground_truth, dim=0).numpy()
        score = r2_score(ground_truth, predictions)
        return score







