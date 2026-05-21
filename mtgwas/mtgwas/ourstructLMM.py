import logging

import numpy as np
import torch
from limix_core.covar import FreeFormCov
from limix_core.gp import GP2KronSumLR
from limix_lmm import LMM
from tqdm import tqdm
import scipy.linalg as la
from numpy import asarray, atleast_1d
from chiscore._davies import _pvalue_lambda

def davies_pvalue(q, w, return_info=False):
    """
    Joint significance of statistics derived from chi2-squared distributions.
    Parameters
    ----------
    q : float
        Test statistics.
    w : array_like
        Weights of the linear combination.
    Returns
    -------
    float
        Estimated p-value.
    """

    q = asarray(atleast_1d(q), float)
    w = asarray(w, float)
    maxq = q.max()
    if maxq > 0:
        q = q / maxq
        w = w / maxq
    re = _pvalue_lambda(w, q)
    if return_info:
        return re["p_value"][0], re
    return re["p_value"][0]


logger = logging.getLogger(__name__)


# util function to compute eigenvalues using torch
def compute_eigenvals(Lambda):
    Lambdat = torch.Tensor(Lambda.astype(np.float32))
    lambdas = torch.linalg.eigvalsh(Lambdat)
    return lambdas.data.numpy().astype(np.float64)


class OurStructLMM:
    """Faster version of StructLMM."""

    def __init__(self, y, E, F, verbose=False):
        self.y = y
        self.E = E
        self.F = F

        self.verbose = verbose

    def interaction_test(self, G, exact=False):
        if exact or G.shape[1] == 1:
            if not exact:
                logger.info("Exact test not requested, but only one variant provided. Using exact test.")

            iterator = range(G.shape[1])
            if self.verbose:
                iterator = tqdm(iterator, desc="Exact test")

            self.pvs = np.array([self.single_interaction_test(G[:, [i]]) for i in iterator])
            return self.pvs

        # learn a covariance on the null model (no variant effect; this is a hack, should be changed)
        gp = GP2KronSumLR(Y=self.y, Cn=FreeFormCov(1), G=self.E, F=self.F, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)  # noqa: F841

        # fit null
        self.lmm = LMM(self.y, self.F, gp.covar.solve)
        self.lmm.process(G)
        pv = self.lmm.getPv()  # noqa: F841
        beta = self.lmm.getBetaSNP()  # noqa: F841
        beta_ste = self.lmm.getBetaSNPste()  # noqa: F841
        lrt = self.lmm.getLRT()  # noqa: F841

        # make interaction test
        Yhat = self.F.dot(self.lmm.beta_F) - G * beta
        Yr = self.y - Yhat
        PY = gp.covar.solve(Yr) / self.lmm.s2

        # score statistics
        W = np.einsum("ns,nk->nsk", G, self.E)
        WPY = np.einsum("nsk,ns->sk", W, PY)
        Q = np.einsum("sk,sk->s", WPY, WPY)

        # eigenvalues
        PW = np.zeros_like(W)
        for i in range(W.shape[2]):
            PW[:, :, i] = (
                gp.covar.solve(W[:, :, i] - Yhat) / self.lmm.s2
            )  # added the denominator here, which was missing (was a bug. Probably not influencial)
        Lambda = np.einsum("nsk,nsl->skl", W, PW)
        lambdas = compute_eigenvals(Lambda)

        self.pvs = np.array([davies_pvalue(Q[i], lambdas[i]) for i in range(G.shape[1])])
        self.G = G
        return self.pvs

    
    def _P(self,X,gp):
        KiX = gp.covar.solve(X)
        FtKiX = gp.mean.W.T.dot(KiX)
        Areml_inv = la.inv(gp.mean.W.T.dot(gp.covar.solve(gp.mean.W)))
        KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(Areml_inv.dot(FtKiX)))
        #    KiFAiFtKiX = gp.covar.solve(gp.mean.W.dot(gp.Areml.solve(FtKiX)))
        out = KiX - KiFAiFtKiX
        return out
    
    def single_interaction_test(self, g):

        # fit exact null model
        F1 = np.concatenate([self.F, g], 1)
        gp = GP2KronSumLR(Y=self.y, Cn=FreeFormCov(1), G=self.E, F=F1, A=np.ones((1, 1)))
        gp.covar.Cr.setCovariance(0.5 * np.ones((1, 1)))
        gp.covar.Cn.setCovariance(0.5 * np.ones((1, 1)))
        self.info_opt = gp.optimize(verbose=False)

        # make interaction test
        PY = self._P(self.y,gp)

        # score statistics
        W = g * self.E
        WPY = W.T.dot(PY)
        Q = (WPY**2).sum()

        # eigenvalues
        PW = self._P(W,gp)
        Lambda = W.T.dot(PW)
        lambdas = compute_eigenvals(Lambda)
        return davies_pvalue(Q, lambdas)
    
