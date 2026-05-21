from time import time
import numpy as np
import scipy.linalg as la
import scipy.stats as st
from .gwas import GWAS
from limix_core.util.preprocess import regressOut 


class MTGWAS(GWAS):
    r"""
    Multi-trait linear model for association testing `P` traits and `S` inputs,
    where each input is tested in isolation (`S` test performed).
    This model implrements the approximation proposed in Korte et al 2014 and
    Lippert et al 2014, where trait covariances are not retrained under the alternative
    but only the scale of the trait covariance is fitted. This approximation is conservative

    Parameters
    ----------
    Y : (`N`, `P`) ndarray
        outputs
    F : (`N`, `K`) ndarray
        covariates. If not specified, an intercept is assumed.
        
    """

    def __init__(self, Y, F=None):
        if F is None:
            F = np.ones((Y.shape[0], 1))
        self.Y = Y
        self.F = F

        # rotate pheno
        _Yrot = regressOut(Y, F)
        _Sigma = np.cov(_Yrot.T)
        _Sc, _Uc = la.eigh(_Sigma)
        _Yrot = _Yrot.dot(_Uc) / np.sqrt(_Sc[np.newaxis, :])
        self._Sc = _Sc
        self._Uc = _Uc

        # core computations happen in LMM
        self.core = GWAS(_Yrot, F)

    def process(self, G, verbose=False):
        r"""
        Fit genotypes one-by-one.
        
        Parameters
        ----------
        G : (`N`, `S`) ndarray
            inputs
        verbose : bool
            verbose flag.
        """
        # core compute
        self.core.process(G, verbose)

        # transforms relevant results
        self.lrt = self.core.lrt.sum(1)
        self.pv = st.chi2(self.Y.shape[1]).sf(self.lrt)
        self.beta_g = np.dot(self.core.beta_g, self._Uc) * np.sqrt(self._Sc[np.newaxis, :])

    def getBetaSNPste(self):
        """
        get standard errors on betas
        
        Returns
        -------
        beta_ste : ndarray
        """
        beta = self.getBetaSNP()
        pv = self.getPv()
        z = np.sign(beta) * np.sqrt(st.chi2(1).isf(pv[:, np.newaxis]))  # Shape: (SNPs, Traits)
        ste = beta / z # Shape: (SNPs, Traits)
        return ste



