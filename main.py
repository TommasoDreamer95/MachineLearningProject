# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA
#from LDA import LDA, compute_data_LDA_Luca
from LDA import compute_data_LDA
import testModel
from testModel import testModel
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes
from spotDataDependency import DataIndependecyAfterPCA
from split import leaveOneOutSplit
from compareAlgorithms import compareAlgorithmsAndDimentionalityReduction, \
    compareLDA, compare_PCA_before_LDA, applyAndTestModels, kFold


    
if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = loadData.load_data()
    """plot histograms"""
    #plot_hist(DTR, LTR);

    """applico PCA e LDA"""
    #m=4
    #DTRPCA = PCA(DTR, m)
    #DTRPCAplusLDA = LDA(DTRPCA, LTR, m)

   # D_before_LDA = compute_data_LDA_Luca(DTR,LTR, m)
    
    
    
    
    

    #DataIndependecyAfterPCA(DTR)

    
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
    
    """Compare LDA with different algorithms"""
    #compareLDA(DTR, LTR)
    #compare_PCA_before_LDA(DTR, LTR)
    
    """compute error rate in the evaluation set with LDA with m=3 using Naive Bayes classifier"""
    m = 3
    DTRLDA = compute_data_LDA(DTR, LTR, 3)
    DTELDA = compute_data_LDA(DTE, LTE, 3)
    acc_LDA, err_LDA = applyAndTestModels(DTRLDA, LTR, DTELDA, LTE, 1)
    print("Accuracy with test data with LDA m={}: ".format(str(m)) + str(format(err_LDA * 100, ".2f")) + "%\n")