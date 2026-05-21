import numpy as np
import scipy as sp
from limix_core.util.preprocess import regressOut
from .ourstructLMM import davies_pvalue
from tqdm import trange

def combine_gwas_pvalues(Z, Y, F):
    """
    Combine p-values from GWAS studies for multiple traits, accounting for the correlation among these traits,
    with adjustment for covariates used in the GWAS.

    Parameters
    ----------
    Z : ndarray
        A matrix of z-scores with dimensions (#SNPs x #Traits).
    Y : ndarray
        Phenotype matrix with dimensions (#Individuals x #Traits).
    F : ndarray
        Covariate matrix used in the GWAS.

    Returns
    -------
    tuple of ndarrays
        Tuple containing an array of combined p-values and an array of quadratic forms for each SNP.
    """

    # Regress out the effects of covariates from Y
    Yr = regressOut(Y, F)

    # Compute the covariance matrix of the residuals
    R = np.corrcoef(Yr.T)

    # Number of SNPs
    num_snps = Z.shape[0]

    # Compute eigenvalues of the covariance matrix
    eigenvalues = np.linalg.eigvalsh(R)

    # Compute all quadratic forms: z^T R z for each SNP
    q_values = np.einsum('ij,jk,ik->i', Z, R, Z)

    # Array to store combined p-values
    combined_pvalues = np.zeros(num_snps)

    # Compute combined p-value for each SNP using Davies' method
    for i in trange(num_snps):
        combined_pvalues[i] = davies_pvalue(q_values[i], eigenvalues)

    return combined_pvalues, q_values

def acat_test(pvalues, weights=None):
    '''acat_test()
    Aggregated Cauchy Assocaition Test
    A p-value combination method using the Cauchy distribution.
    
    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R
    
    Author: Ryan Neff
    
    Inputs:
        pvalues: <list or numpy array>
            The p-values you want to combine.
        weights: <list or numpy array>, default=None
            The weights for each of the p-values. If None, equal weights are used.
    
    Returns:
        pval: <float>
            The ACAT combined p-value.
    '''
    if any(np.isnan(pvalues)):
        raise Exception("Cannot have NAs in the p-values.")
    if any([(i>1)|(i<0) for i in pvalues]):
        raise Exception("P-values must be between 0 and 1.")
    if any([i==1 for i in pvalues])&any([i==0 for i in pvalues]):
        raise Exception("Cannot have both 0 and 1 p-values.")
    if any([i==0 for i in pvalues]):
        print("Warn: p-values are exactly 0.")
        return 0
    if any([i==1 for i in pvalues]):
        print("Warn: p-values are exactly 1.")
        return 1
    if weights==None:
        weights = [1/len(pvalues) for i in pvalues]
    elif len(weights)!=len(pvalues):
        raise Exception("Length of weights and p-values differs.")
    elif any([i<0 for i in weights]):
        raise Exception("All weights must be positive.")
    else:
        weights = [i/len(weights) for i in weights]
    
    pvalues = np.array(pvalues)
    weights = np.array(weights)
    
    if any([i<1e-16 for i in pvalues])==False:
        cct_stat = sum(weights*np.tan((0.5-pvalues)*np.pi))
    else:
        is_small = [i<(1e-16) for i in pvalues]
        is_large = [i>=(1e-16) for i in pvalues]
        cct_stat = sum((weights[is_small]/pvalues[is_small])/np.pi)
        cct_stat += sum(weights[is_large]*np.tan((0.5-pvalues[is_large])*np.pi))
    
    if cct_stat>1e15:
        pval = (1/cct_stat)/np.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)
    
    return pval