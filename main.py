# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA
#from LDA import LDA, compute_data_LDA_Luca
from LDA import compute_data_LDA
import testModel
from testModel import testModel, testLogisticRegression, testLinearSVM, testGMM
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes, computeParametersForLogisticRegression, \
        computeParametersForLinearSVM, computeGMMs
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
    #mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
    """Naive Bayes"""
    #_ ,sigma_diag = computeMeanAndCovarianceForNaiveBayes(DTR, LTR)
    """Tied"""  
    #_, sigma_tied = computeMeanAndCovarianceForTied(DTR, LTR)
    """Logistic Regression"""
    l = 1/1000 #we should try different parameters for l such as [0, 1/1000000, 1/1000, 1]
    #w, b = computeParametersForLogisticRegression(DTR, LTR, l)
    """Linear SVM"""
    k = 1
    C = 0.1 # altri possibili valori Cvalues = [0.1, 1.0, 10.0]    Kvalues = [1, 10]
    #wHatStar = computeParametersForLinearSVM(DTR, LTR, k, C)
    """Gaussian Mixture Model"""
    DeltaL = 10e-6
    finalImpl = "standard"#other possible values ["standard", "diagonal", "tied"]:   
    finalGmms = 8 #other possible values [1,2,4,8,16]
    GMM = computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl)
        
    
    """calcolo accuratezza, errore e confusion matrix sui dati di test. Per ora il costo non è considerato"""
    """
    acc_MVG, err_MVG, CM_MVG = testModel(mu, sigma, DTE, LTE)
    print("Error rate MVG without PCA: " + str(format(err_MVG * 100, ".2f")) + "%\n")
    acc_Naive, err_Naive, CM_Naive = testModel(mu, sigma_diag, DTE, LTE)
    print("Error rate Naive Bayes without PCA: " + str(format(err_Naive * 100, ".2f")) + "%\n")
    acc_Tied, err_Tied, CM_Tied = testModel(mu, sigma_tied, DTE, LTE)
    print("Error rate Tied without PCA: " + str(format(err_Tied * 100, ".2f")) + "%\n")
    acc_LogReg, err_LogReg = testLogisticRegression(w, b, DTE, LTE)
    print("Error rate Logistic Regression without PCA: " + str(format(err_LogReg * 100, ".2f")) + "%\n")
    acc_LinSVM , err_LinSVM = testLinearSVM(wHatStar, k, DTE,LTE)
    print("Error rate Linear SVM without PCA: " + str(format(err_LinSVM * 100, ".2f")) + "%\n")
    """
    acc_GMM, err_GMM = testGMM(GMM, DTE, LTE)
    print("Error rate GMM without PCA: " + str(format(err_GMM * 100, ".2f")) + "%\n")
    
    """Compare LDA with different algorithms"""
    #compareLDA(DTR, LTR)
    #compare_PCA_before_LDA(DTR, LTR)
    
    """compute error rate in the evaluation set with LDA with m=3 using Naive Bayes classifier"""
    """
    m = 9
    n = 8
    DTRPCA = PCA(DTR, m)
    DTRLDA = compute_data_LDA(DTRPCA, LTR, n)
    DTELDA = compute_data_LDA(DTE, LTE, n)
    acc_LDA, err_LDA = applyAndTestModels(DTRLDA, LTR, DTELDA, LTE, 2)
    print("Accuracy with test data with LDA m={}: ".format(str(m)) + str(format(err_LDA * 100, ".2f")) + "%\n")
    """
    