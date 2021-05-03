# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:32:11 2021

@author: tommy
"""

import numpy
import scipy.linalg
from matrixUtilities import mcol, compute_num_classes


"""#input = data matrix and labels vector"""    
def computeSB(D, L):
    mu = D.mean(1)
    betweenClassCovarianceMatrix = 0
    for c in range(0,2): 
        Dc = D[:, L==c] #Dc = for each plant (columns) his 4 attributes (rows - petal length, petal width etc)
        muc = Dc.mean(1) #row vector of the mean of the 4 attributes (should be a column vector)    
        betweenClassCovarianceMatrix = betweenClassCovarianceMatrix + numpy.dot( mcol(muc - mu), mcol(muc-mu).T ) * Dc.shape[1]       
    betweenClassCovarianceMatrix = betweenClassCovarianceMatrix / float(D.shape[1])
    return betweenClassCovarianceMatrix

"""#input = data matrix and labels vector"""    
def computeSW(D,L):     
    withinClassCovarianceMatrix = 0
    for classe in range(0,2): 
        Dc = D[:, L==classe] #Dc = for each plant (columns) his 4 attributes (rows - petal length, petal width etc)
        muc = Dc.mean(1) #row vector of the mean of the 4 attributes (should be a column vector)    
        for i in range(0, Dc.shape[1]):  #each row having i-th attribute of all samples      
            withinClassCovarianceMatrix = withinClassCovarianceMatrix + numpy.dot( mcol(Dc[:, i] - muc), mcol(Dc[:, i] - muc).T)
    withinClassCovarianceMatrix = withinClassCovarianceMatrix / float(D.shape[1])
    return withinClassCovarianceMatrix

"""#input = Between and Within class covariance matrixes and number of eighenvectors we want to take"""    
def LDA_byGeneralizedEigenvalueProblem(SB, SW, m):
    eigenvalues, eigenvectorsMatrix = scipy.linalg.eigh(SB, SW)
    directionMaximizingVariabilityRatio = eigenvectorsMatrix[:, ::-1][:, 0:m]
    return directionMaximizingVariabilityRatio #eigenvectors rapresenting that direction


def LDA(DTR, LTR, m):
    SW = computeSW(DTR,LTR)
    SB = computeSB(DTR, LTR)
    W = LDA_byGeneralizedEigenvalueProblem(SB, SW, m)
    DP = numpy.dot(W.T, DTR)
    return DP


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
def computeSB_Luca(D, L, N, mu):
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
def computeSW_Luca(D, L, N):
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
def compute_data_LDA_Luca(D, L, m):
    mu = D.mean(1) 
    mu = mcol(mu)
    N = D.shape[1]
    SB = computeSB_Luca(D, L, N, mu)
    # --- COMPUTE SW -----
    SW = computeSW_Luca(D, L, N)
    s, U = scipy.linalg.eigh(SB, SW)
    #--- W is the matrix of the LDA
    W = U[:, ::-1][:, 0:m]
    y = numpy.dot(W.T, D)
    return y
    