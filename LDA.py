# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:32:11 2021

@author: tommy
"""

import numpy
import scipy.linalg
from matrixUtilities import mcol


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