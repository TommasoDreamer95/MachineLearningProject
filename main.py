# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA
from LDA import LDA
import testModel
from testModel import testModel_MVG, testModel_Naive_Bayes
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, create_sigma_diagonal
#import classificatori

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
        DTRPCA = PCA(DTR, m)
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
    
if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = loadData.load_data()
    #plot_hist(DTR, LTR);

    """applico PCA e LDA"""
    #m=4
    #DTRPCA = PCA(DTR, m)
    #DTRPCAplusLDA = LDA(DTRPCA, LTR, m)
    
    DataIndependecyAfterPCA(DTR)
    
    #plot_hist(DTRPCA, LTR);
    
    """applico il classificatore opportuno"""
    """MVG"""
    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
    
    """Naive Bayes"""
    sigma_diag = create_sigma_diagonal(sigma)
    acc_Naive, err_Naive, CM_Naive = testModel_Naive_Bayes(mu, sigma, DTE, LTE)
    print("Error rate Naive Bayes without PCA: " + str(format(err_Naive * 100, ".2f")) + "%\n")
    """Tied"""  
    #mu, sigma = computeMeanAndCovarianceForTied(DTR, LTR)

    """calcolo accuratezza, errore e confusion matrix sui dati di test. Per ora il costo non è considerato"""
    acc_MVG, err_MVG, CM_MVG = testModel_MVG(mu, sigma, DTE, LTE)
    print("Error rate MVG without PCA: " + str(format(err_MVG * 100, ".2f")) + "%\n")
    #print(CM)   
    #print(err)