# -*- coding: utf-8 -*-
"""
Created on Sat May  1 09:24:45 2021

@author: tommy
"""
from matrixUtilities import mcol
import numpy

def covariance(D):
    mu = D.mean(1) #array con la media delle colonne della matrice
    DC = D - mcol(mu)  #matrice centrata, in cui ad ogni colonna sottraggo la media
    C = numpy.dot(DC, DC.T) / float(DC.shape[1]) # C = 1/N * Dc * Dc Trasposta
    return C 

"""input = 
    1) data matrix
    2) #resulting dimentions
output:
    1) projected Data                  """
def PCA(D, resultingDimentions): 
    CovarianceMatrix = covariance(D)
    eigenvectorsMatrix, eigenvalues, Vh = numpy.linalg.svd(CovarianceMatrix)
    mLeadingEigenvectors = eigenvectorsMatrix[:, 0:resultingDimentions]
    DataProjected = numpy.dot(mLeadingEigenvectors.T, D) #apply projection to matrix of samples
    return DataProjected 

def PCAforDTE(DTR, DTE, resultingDimentions): 
    D = DTR
    CovarianceMatrix = covariance(D)
    eigenvectorsMatrix, eigenvalues, Vh = numpy.linalg.svd(CovarianceMatrix)
    mLeadingEigenvectors = eigenvectorsMatrix[:, 0:resultingDimentions]
    DTRProjected = numpy.dot(mLeadingEigenvectors.T, D) #apply projection to matrix of samples
    DTEProjected = numpy.dot(mLeadingEigenvectors.T, DTE) #apply projection to matrix of samples
    return DTRProjected, DTEProjected