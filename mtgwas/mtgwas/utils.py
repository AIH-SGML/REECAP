import pandas as pd
import numpy as np
import scipy.stats as st


def df_match(dfs, keys=None):
    if keys is None:
        keys = [None] * len(dfs)
    dfidxs = []
    for _i, (df, key) in enumerate(zip(dfs, keys)):
        _id = df[key].values if key is not None else df.index.values
        _dfidx = {'ID': _id, f'id{_i}': np.arange(df.shape[0])}
        dfidxs.append(pd.DataFrame(_dfidx))
    dfidx = dfidxs[0].copy()
    for _i in range(1, len(dfidxs)):
        dfidx = dfidx.merge(dfidxs[_i], how='inner', on='ID')
    out = [dfidx[f'id{_i}'].values for _i in range(len(dfs))]
    return out


def prepare_covariates(df_covar):
    # Prepare covariates
    array_keys = ["Axiom", "BiLEVE"]
    keys_to_remove = ["FID", "IID"]
    if "batch_3" in df_covar.keys():
        keys_to_remove.append("batch_3")
    if "batch_0" in df_covar.keys():
        array_keys.append("batch_0")
        array_keys.append("batch_1")
        array_keys.append("batch_2")

    nonarray_keys = np.setdiff1d(df_covar.keys(), array_keys + keys_to_remove)
    try:
        F1 = df_covar[array_keys].values
        F2 = df_covar[nonarray_keys].values
        F2 = F2[:, F2.std(0) > 0]
        F2 = (F2 - F2.mean(0)) / F2.std(0)
        F = np.concatenate([F1, F2], 1)
    except:
        F = df_covar["cov"].values[:, None]
    return F

def gaussianize(Y):
    """
    Gaussianize X: [samples x phenotypes]
    - each phentoype is converted to ranks and transformed back to normal using the inverse CDF
    """
    N, P = Y.shape

    YY = toRanks(Y)
    quantiles = (np.arange(N) + 0.5) / N
    gauss = st.norm.isf(quantiles)
    Y_gauss = np.zeros((N, P))
    for i in range(P):
        Y_gauss[:, i] = gauss[YY[:, i]]
    Y_gauss *= -1
    return Y_gauss

def toRanks(A):
    """
    converts the columns of A to ranks
    """
    AA = np.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:, i] = st.rankdata(A[:, i])
    AA = np.array(np.around(AA), dtype="int") - 1
    return AA