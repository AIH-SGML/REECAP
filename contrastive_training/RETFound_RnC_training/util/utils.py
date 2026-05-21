import random
from os.path import join
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class GowerScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.xgower_factor_ = None

    def _to_numpy(self, X):
        if hasattr(X, "detach"):
            return X.detach().cpu().numpy()
        return X

    def fit(self, X, y=None):
        X = self._to_numpy(X)

        self.mean_ = np.mean(X, axis=0)

        a = np.power(X, 2).sum()
        b = X.dot(X.sum(axis=0)).sum()
        self.xgower_factor_ = np.sqrt((a - b / X.shape[0]) / (X.shape[0] - 1))

        return self

    def transform(self, X, y=None):
        X = self._to_numpy(X)

        X_centered = X - self.mean_
        X_scaled = X_centered / self.xgower_factor_

        return X_scaled

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_embeddings(
    Z_train, y_train,
    Z_val, y_val,
    args,
    Z_test=None, y_test=None,
    save_coefficients=False
):
    """
    Train a linear probe on embeddings and evaluate on validation data.
    Optionally evaluate on test data.
    """

    zscaler = GowerScaler()

    Z_train = zscaler.fit_transform(Z_train)
    Z_val = zscaler.transform(Z_val)

    if Z_test is not None:
        Z_test = zscaler.transform(Z_test)

    reg = Ridge(alpha=getattr(args, "ridge_alpha", 1.0))
    reg.fit(Z_train, y_train)

    yp_train = reg.predict(Z_train)
    yp_val = reg.predict(Z_val)

    train_mse = mean_squared_error(y_train, yp_train)
    val_mse = mean_squared_error(y_val, yp_val)

    if save_coefficients:
        np.save(join(args.outdir, "LR_coefficients.npy"), reg.coef_.T)
        np.save(join(args.outdir, "LR_intercept.npy"), reg.intercept_)

    dfp_train = pd.DataFrame({"obs": y_train, "pred": yp_train})
    dfp_val = pd.DataFrame({"obs": y_val, "pred": yp_val})

    if Z_test is not None:

        yp_test = reg.predict(Z_test)
        test_mse = mean_squared_error(y_test, yp_test)

        dfp_test = pd.DataFrame({"obs": y_test, "pred": yp_test})

        return train_mse, val_mse, test_mse, dfp_train, dfp_val, dfp_test

    return train_mse, val_mse, dfp_train, dfp_val

def save_embeddings_to_csv(file_paths, embeddings, output_dir, filename):
    
    df = pd.DataFrame({"file_path": file_paths})
    embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i+1}" for i in range(embeddings.shape[1])])
    df = pd.concat([df, embeddings_df], axis=1)
    output_path = join(output_dir, filename)
    df.to_csv(output_path, index=False)