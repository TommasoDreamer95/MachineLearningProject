# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:32:11 2021

@author: tommy
"""

import numpy
import scipy.linalg
from matrixUtilities import mcol, compute_num_classes





"""
----BELOW LDA solution of Luca-----
"""
"""
input:
1) data matrix 
2) matrix with labels
3) number of samples
4) mean
output:
the SB matrix of the LDA algorithm
"""
def computeSB(D, L, N, mu):
    SB = 0.0
    k = compute_num_classes(L)
    for i in range(k):
        Di = D[:, L==i]
        muc = Di.mean(axis=1)
        muc = mcol(muc)
        nc = Di.shape[1]
        SB += nc * numpy.dot((muc-mu), (muc-mu).T)
    SB = SB/N
    return SB

"""
input:
1) data matrix 
2) matrix with labels
3) number of samples
output:
the SW matrix of the LDA algorithm
"""
def computeSW(D, L, N):
    SW = 0.0
    k = compute_num_classes(L)
    for i in range(k):
        Di = D[:, L==i]
        nc = Di.shape[1]
        muc = Di.mean(axis=1)
        muc = mcol(muc)
        SW += numpy.dot((Di-muc), (Di-muc).T)
    SW /= N
    return SW

"""
input:
1) data matrix
2) matrix with labels
3) number of dimensions
output:
data matrix with m dimensions, obtained applying LDA algorithm
"""
def compute_data_LDA(D, L, m):
    mu = D.mean(1) 
    mu = mcol(mu)
    N = D.shape[1]
    SB = computeSB(D, L, N, mu)
    # --- COMPUTE SW -----
    SW = computeSW(D, L, N)
    s, U = scipy.linalg.eigh(SB, SW)
    #--- W is the matrix of the LDA
    W = U[:, ::-1][:, 0:m]
    y = numpy.dot(W.T, D)
    return y
    