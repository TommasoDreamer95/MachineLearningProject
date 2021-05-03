# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:58:16 2021

@author: tommy
"""
import numpy
from matrixUtilities import mcol
from PCA import PCA



def covariance(D):
    mu = D.mean(1) #array con la media delle colonne della matrice
    
    DC = D - mcol(mu)  #matrice centrata, in cui ad ogni colonna sottraggo la media
    
    C = numpy.dot(DC, DC.T) / float(DC.shape[1]) # C = 1/N * Dc * Dc Trasposta
    return C   

"""spots if an element out of the diagonal is greater than 1*10^-4
returns true if the covariance matrix shows dependency between dimentions
false otherwise """
def spotDependency(sigma):
    for i in range(0, sigma.shape[0]):
        for j in range(0, sigma.shape[1]):
            if i!=j:
                if abs(sigma[i, j]) > 0.0001:
                    return True
    return False

def DataIndependecyAfterPCA(D):
    m = 0
    for m in range(2, D.shape[0]+1):
        DTRPCA = PCA(D, m)
        sigma = covariance(DTRPCA)
        """
        print(m)
        print(sigma)
        print(spotDependency(sigma))
        print()
        """
    
    sigma=covariance(D)    
    """
    print(D.shape[0]+1)
    print(sigma)
    print(spotDependency(sigma))
    """
