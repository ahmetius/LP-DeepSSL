# Author: Dmytro Mishkin, 2018.
# This is simple partial re-implementation of papers 
# Fast Spectral Ranking for Similarity Search, CVPR 2018. https://arxiv.org/abs/1703.06935
# and  Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations, CVPR 2017 http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html

import pdb
import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg

def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal())
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn

def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 20, tol = 1e-6):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    return ranks, out_sims