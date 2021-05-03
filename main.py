# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA
from LDA import LDA, compute_data_LDA_Luca
import testModel
from testModel import testModel
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes
from spotDataDependency import DataIndependecyAfterPCA
from split import leaveOneOutSplit
from compareAlgorithms import compareAlgorithmsAndDimentionalityReduction


    
if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = loadData.load_data()
    #plot_hist(DTR, LTR);

    """applico PCA e LDA"""
    #m=4
    #DTRPCA = PCA(DTR, m)
    #DTRPCAplusLDA = LDA(DTRPCA, LTR, m)
<<<<<<< HEAD
   # D_before_LDA = compute_data_LDA_Luca(DTR,LTR, m)
    
    
    
    
    DataIndependecyAfterPCA(DTR)
=======
    #DataIndependecyAfterPCA(DTR)
>>>>>>> TommasoBranch
    
    #plot_hist(DTRPCA, LTR);
    
    #compareAlgorithmsAndDimentionalityReduction(DTR, LTR)
    
    """applico il classificatore opportuno"""
    """MVG"""
    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
    """Naive Bayes"""
    _ ,sigma_diag = computeMeanAndCovarianceForNaiveBayes(DTR, LTR)
    """Tied"""  
    _, sigma_tied = computeMeanAndCovarianceForTied(DTR, LTR)

    """calcolo accuratezza, errore e confusion matrix sui dati di test. Per ora il costo non è considerato"""
    acc_MVG, err_MVG, CM_MVG = testModel(mu, sigma, DTE, LTE)
    print("Error rate MVG without PCA: " + str(format(err_MVG * 100, ".2f")) + "%\n")
    acc_Naive, err_Naive, CM_Naive = testModel(mu, sigma_diag, DTE, LTE)
    print("Error rate Naive Bayes without PCA: " + str(format(err_Naive * 100, ".2f")) + "%\n")
    acc_Tied, err_Tied, CM_Tied = testModel(mu, sigma_tied, DTE, LTE)
    print("Error rate Tied without PCA: " + str(format(err_Tied * 100, ".2f")) + "%\n")
    
    #print(CM)   
    #print(err)