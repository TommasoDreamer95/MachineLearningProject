# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA, PCAforDTE
import testModel
from testModel import testModel, testLogisticRegression, testLinearSVM, testGMM, testKernelSVM
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes, computeParametersForLogisticRegression, \
        computeParametersForLinearSVM, computeGMMs, computeParameterForKernelPolynomialSVM, computeParameterForKernelRBFSVM, \
            compute_matrix_Z_kernel_SVM, compute_polynomial_kernel, compute_RBF_kernel
from spotDataDependency import DataIndependecyAfterPCA
from split import leaveOneOutSplit
from compareAlgorithms import compareAlgorithmsAndDimentionalityReduction, \
    applyAndTestModels, kFold


    
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
    
    compareAlgorithmsAndDimentionalityReduction(DTR, LTR) 
    
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
    """kernel SVM"""
    """polynomial Kernel SVM"""
    K_poly = 1
    C_poly = 0.1
    d_poly = 2
    c_poly = 0
    #Z_kernel_SVM = compute_matrix_Z_kernel_SVM(DTR, LTR) # matrix Z is common for polynomial kernel and RBF kernel
    #alfa_polynomial_SVM = computeParameterForKernelPolynomialSVM(DTR, LTR, K_poly, C_poly, d_poly, c_poly)
    """
     TODO: for the computation of the scores, compute polynomial kernel for DTR, DTE amd Z matrix 
     the computation of Z can be done inside the fucntion that computes the scores
    """
    """RBF kernel SVM"""
    K_RBF = 1
    C_RBF = 0.1
    gamma_RBF = 1.0
    #alfa_RBF_kernel = computeParameterForKernelRBFSVM(DTR, LTR, K_RBF, C_RBF, gamma_RBF)
    """Gaussian Mixture Model"""
    DeltaL = 10e-6
    finalImpl = "standard"#other possible values ["standard", "diagonal", "tied"]:   
    finalGmms = 8 #other possible values [1,2,4,8,16]
    #GMM = computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl)
        
    
    """calcolo accuratezza, errore e confusion matrix sui dati di test. Per ora il costo non è considerato"""
    """
    acc_MVG, err_MVG, CM_MVG = testModel(mu, sigma, DTE, LTE)
    print("Error rate MVG without PCA: " + str(format(err_MVG * 100, ".2f")) + "%\n")
    acc_Naive, err_Naive, CM_Naive = testModel(mu, sigma_diag, DTE, LTE)
    print("Error rate Naive Bayes without PCA: " + str(format(err_Naive * 100, ".2f")) + "%\n")
    acc_Tied, err_Tied, CM_Tied = testModel(mu, sigma_tied, DTE, LTE)
    print("Error rate Tied without PCA: " + str(format(err_Tied * 100, ".2f")) + "%\n")
    acc_LogReg, err_LogReg, scores_LogReg = testLogisticRegression(w, b, DTE, LTE)
    print("Error rate Logistic Regression without PCA: " + str(format(err_LogReg * 100, ".2f")) + "%\n")
    acc_LinSVM , err_LinSVM, scores_LinSVM = testLinearSVM(wHatStar, k, DTE,LTE)
    print("Error rate Linear SVM without PCA: " + str(format(err_LinSVM * 100, ".2f")) + "%\n")
    acc_GMM, err_GMM, scores_GMM = testGMM(GMM, DTE, LTE)
    print("Error rate GMM without PCA: " + str(format(err_GMM * 100, ".2f")) + "%\n")
    ###-- compute scores for polynomial kernel --###
    poly_kernel_DTR_DTE = compute_polynomial_kernel(DTR, DTE, c_poly, d_poly, K_poly)
    acc_poly_kernel, err_poly_kernel, scores_poly_kernel = testKernelSVM(alfa_polynomial_SVM, Z_kernel_SVM, poly_kernel_DTR_DTE, LTE)
    ###-- compute scores for RBF kernel --###
    RBF_kernel_DTR_DTE = compute_RBF_kernel(DTR, DTE, gamma_RBF, K_RBF)
    acc_RBF_kernel, err_RBF_kernel, scores_RBF_kernel = testKernelSVM(alfa_RBF_kernel, Z_kernel_SVM, RBF_kernel_DTR_DTE, LTE)
    """
    
    """examples of testing and training with PCA"""
    """
    m=8
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)

    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTRPCA, LTR)
    acc_MVG, err_MVG, CM_MVG = testModel(mu, sigma, DTEPCA, LTE)
    print("Error rate MVG with PCA: " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
    DeltaL = 10e-6
    finalImpl = "standard"#other possible values ["standard", "diagonal", "tied"]:   
    finalGmms = 8 #other possible values [1,2,4,8,16]
    GMM = computeGMMs(DTRPCA, LTR, DeltaL, finalGmms, finalImpl)
    acc_GMM, err_GMM, scores_GMM = testGMM(GMM, DTEPCA, LTE)
    print("Error rate GMM with PCA: " + str(format(err_GMM * 100, ".2f")) + "%\n")
    """

    