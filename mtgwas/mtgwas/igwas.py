from mtgwas import GWAS
import numpy as np
import scipy.stats as st
from tqdm import tqdm, trange
import matplotlib.pyplot as pl
import scipy.stats as st

class IGWAS(GWAS):

    def __init__(self, Y: np.ndarray, P: np.ndarray, F: np.ndarray = None):
        if F is None:
            F = np.ones((Y.shape[0], 1))
        self.Y = Y
        self.P = P
        self.Fp = np.append(F, P, axis=1)
        self.gwas0 = GWAS(self.Y, self.Fp)
        self.lik0 = self.compute_llik(self.gwas0)
        self.tolerance = 1e-4

    def process(self, G:np.ndarray, verbose:bool =True):
        '''
        H1: y = F + PRS + g
        H2: y = F + PRS + g + g*PRS
        Pv_association: H2 vs H0
        Pv_interaction: H2 vs H1
        '''
        self.lik1 = np.ones(G.shape[1])
        self.beta_g_h1 = np.zeros([1, G.shape[1]])
        F1 = np.append(self.Fp, np.zeros([G.shape[0], 1]), axis=1)

        self.lik2 = np.ones(G.shape[1])
        self.beta_g_h2 = np.zeros([2, G.shape[1]])
        F2 = np.append(self.Fp, np.zeros([G.shape[0], 2]), axis=1)
        correlated = []
        null_burdens = []
        queue = range(G.shape[1]) if not verbose else tqdm(range(G.shape[1]))
        for p in queue:
            F1[:, -1] = G[:, p]
            F2[:, -2] = G[:, p]
            F2[:, -1] = G[:, p] * self.P.ravel()
            '''
            if np.unique(G[:,p]).shape[0]==1:
                self.lik2[p] = 1
                self.lik1[p] = 1
                null_burdens.append(p)
                continue
            '''
            corr,_= st.spearmanr(F2[:,-2], F2[:,-1])
            corr = np.abs(corr)
            if ((1-corr)<self.tolerance) or corr==np.nan: 
                correlated.append(p)
                gwas_h1 = GWAS(self.Y, F1)
                self.lik1[p] = self.compute_llik(gwas_h1)
                self.beta_g_h1[:, p] = gwas_h1.beta_F0[-1:, 0]
                self.lik2[p] = self.lik1[p] 
            else:
                gwas_h1 = GWAS(self.Y, F1)
                gwas_h2 = GWAS(self.Y,F2)
                self.lik1[p] = self.compute_llik(gwas_h1)
                self.lik2[p] = self.compute_llik(gwas_h2)
                self.beta_g_h1[:, p] = gwas_h1.beta_F0[-1:, 0]
                self.beta_g_h2[:, p] = gwas_h2.beta_F0[-2:, 0]
        self.lrt_ass = 2 * (self.lik2 - self.lik0)
        self.pv_ass = st.chi2(2).sf(self.lrt_ass)
        self.lrt_int = 2 * (self.lik2 - self.lik1)
        self.pv_int = st.chi2(1).sf(self.lrt_int)
        self.correlated = np.asarray(correlated)
        self.null_burdens = np.asarray(null_burdens)
        self.lrt_h1_h0 = 2 * (self.lik1 - self.lik0)
        self.pv_h1_h0 = st.chi2(1).sf(self.lrt_h1_h0)
        
        
    def compute_llik(self, gwas:GWAS):
        mean = np.dot(gwas.F, gwas.beta_F0)
        return st.norm.logpdf(gwas.Y, loc=mean, scale=np.sqrt(gwas.s20)).sum()

    def getPv(self):
        return self.pv_ass, self.pv_int,self.pv_h1_h0

    def getLRT(self):
        return self.lrt_ass, self.lrt_int,self.lrt_h1_h0

    def getBetaSNP(self):
        return self.beta_g_h2, self.beta_g_h1
    
    def getCorrelated(self):
        return self.correlated
    
    def getNullBurdens(self):
        return self.null_burdens
    

def generate_pheno(X, P, Sc=5, v_prs=0.20, v_g=0.01, v_int=1e-3):
    """
    X : genetics
    Sc : number of causal variants
    v_g : variance explained by causal variants
          (total phenotypic variance of 1 is assumed)
    """
    # prs
    P = (P - P.mean(0)) / P.std(0)
    P *= np.sqrt(v_prs)
    # burden
    idxc = np.sort(np.random.choice(X.shape[1], Sc, replace=False))
    Xc = X[:, idxc]
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    b = np.asarray([-1 if np.random.uniform(0, 1) < 0.5 else 1 for _ in range(Sc)])[:, None]
    G = Xc.dot(b)
    G = (G - G.mean(0)) / G.std(0)
    G *= np.sqrt(v_g)

    # interaction
    I = G * P
    #pdb.set_trace()
    if I.std(0)==0:
        I = (I - I.mean(0))/1
    else:
        I = (I - I.mean(0)) / I.std(0)
    b = np.asarray([-1 if np.random.uniform(0, 1) < 0.5 else 1 for _ in range(P.shape[0])])[:, None]
    I = I*b
    if I.std(0) == 0:
        I = (I - I.mean(0)) / 1
    else:
        I = (I - I.mean(0)) / I.std(0)
    I *= np.sqrt(v_int)

    # noise
    N = np.random.randn(P.shape[0], 1)
    N = (N - N.mean(0)) / N.std(0)
    N *= np.sqrt(1 - v_prs - v_g - v_int)
    return P + G + I + N, idxc


