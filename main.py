# -*- coding: utf-8 -*-

import numpy #, scipy.special, scipy.linalg
from matrixUtilities import plot_hist, plot_scatter
import loadData
from PCA import PCAforDTE
import testModel
from testModel import testLogisticRegression, testModel, testKernelSVM, testLinearSVM, testGMM
from classificatori import computeParametersForLogisticRegression, computeMeanAndCovarianceForNaiveBayes, \
    computeMeanAndCovarianceForMultiVariate, computeMeanAndCovarianceForTied, compute_matrix_Z_kernel_SVM, \
        computeParameterForKernelRBFSVM, compute_RBF_kernel, computeParametersForLinearSVM, \
            computeParameterForKernelPolynomialSVM, compute_polynomial_kernel, computeGMMs
from compareAlgorithms import compareAlgorithmsAndDimentionalityReduction, compute_min_dcf_prior


    
if __name__ == '__main__':
    (DTR, LTR) = loadData.load_data_TR()
    
    """plot histograms of training data"""
    plot_hist(DTR, LTR);
    plot_scatter(DTR, LTR)
    
    
    
    """ Cross validation:
        trains various models with different parameters and different PCA applied on the training set,
        evaluates each model with a 3-fold strategy and stores the results on the output file "AlgorithmsOutput.txt".
        The execution takes about 3-4 hours
        N.B. only the training data are passed as input to this function, 
        the function is unaware of what the test data look like. DTE variables inside the compareAlgorithmsAndDimentionalityReduction function 
        refer to validation data produced via splitting of DTR with the k-fold
    """
    compareAlgorithmsAndDimentionalityReduction(DTR, LTR) 
    
    
    
    """based on the compare algorithms output the best algorithm should be logistic regression, 
        applied to either unprocessed data or to data preprocessed with PCA in order to obtain 10 dimentions out of the initial 11 ones
        and a value of the parameter lambda equal to 1/1000.
        That will be the model we deliver.
        Therefore we train the model on the training data and test it on the test data.
        N.B. the following lines of code where only added after having analyzed the the output of 
        the compareAlgorithmsAndDimentionalityReduction function. The test data were not used for cross validation
    """
    
    (DTE, LTE) = loadData.load_data_TE()
    
    """Logistic regression with PCA"""
    """
    m=10
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    l = 1/1000
    w, b = computeParametersForLogisticRegression(DTRPCA, LTR, l)
    acc, err, scores = testLogisticRegression(w, b, DTEPCA, LTE)
    scores = numpy.hstack(scores)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate with test data (with PCA): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF with test data with p = 0.5 (with PCA): " + str(format(min_DCF, ".3f")) + "\n")
    """
    
    """MVG classifier with PCA(m=8) on test set"""
    m = 8
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    #DTRPCA = PCA(DTR, m)
    #acc_MVG,err_MVG, min_DCF_MVG, optimal_threshold_MCG = kFold(DTRPCA, LTR, 0)
    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTRPCA, LTR)
    acc, err, scores = testModel(mu, sigma, DTEPCA, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate on test set MVG with PCA (m=" + str(m) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) MVG with PCA (m={}) on test set: {}\n".format( str(m), str(format(min_DCF, ".3f") )))
    
    
    
    """Naive Bayes with PCA(m=7) on test set"""
    m = 7
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    mu , sigma = computeMeanAndCovarianceForNaiveBayes(DTRPCA, LTR)
    acc, err, scores = testModel(mu, sigma, DTEPCA, LTE)
    scores = numpy.hstack(scores)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate with test data Naive Bayes: " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) on test set with Naive with PCA (m={}): {}\n".format(str(m), str(format(min_DCF, ".3f") )))
    
    """Tied gaussian classifier with PCA(m=9) on test set""" 
    m = 9
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    mu, sigma= computeMeanAndCovarianceForTied(DTRPCA, LTR)
    acc, err, scores = testModel(mu, sigma, DTEPCA, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate Tied on test set with PCA (m=" + str(m) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) Tied on test set with PCA (m={}): {}\n".format( str(m), str(format(min_DCF, ".3f") )))
    
    
    """RBF SVM with k=0, C=1, gamma=1 with PCA(m=6) on test set"""
    m = 6
    K_RBF = 0
    C_RBF = 0
    gamma_RBF = 1
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    Z_kernel_SVM = compute_matrix_Z_kernel_SVM(DTRPCA, LTR)
    alfa_RBF_kernel = computeParameterForKernelRBFSVM(DTRPCA, LTR, K_RBF, C_RBF, gamma_RBF)
    RBF_kernel_DTR_DTE = compute_RBF_kernel(DTRPCA, DTEPCA, gamma_RBF, K_RBF)
    acc, err, scores = testKernelSVM(alfa_RBF_kernel, Z_kernel_SVM, RBF_kernel_DTR_DTE, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate RBF SVM with PCA (m=" + str(m) + ", k = " + str(K_RBF) + ", C = " + str(C_RBF) + ", gamma = " + str(gamma_RBF) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5)  RBF SVM with PCA (m={}, k = {}, C = {}, gamma = {} ) on test set: {}\n".format(str(m), str(K_RBF), str(C_RBF), str(gamma_RBF) , str(format(min_DCF, ".3f") )))    
       
    
    
    """Linear SVM (k = 10, C = 0.1) with PCA(m=9) on test set"""
    m = 9
    k = 10
    C = 9
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    wHatStar = computeParametersForLinearSVM(DTRPCA, LTR, k, C)
    acc , err, scores = testLinearSVM(wHatStar, k, DTEPCA, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate Linear SVM on test set with PCA (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) Linear SVM with PCA (m={}, k = {}, C = {} ): {}\n".format(str(m), str(k), str(C) , str(format(min_DCF, ".3f") )))    
    
    
    """Polinomial SVM (k = 0, C = 1, c = 1, d = 1) with PCA(m=10)"""
    m = 10
    k = 0
    C = 1
    c = 1
    d = 1
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    Z_kernel_SVM = compute_matrix_Z_kernel_SVM(DTRPCA, LTR) # matrix Z is common for polynomial kernel and RBF kernel
    alfa_polynomial_SVM = computeParameterForKernelPolynomialSVM(DTRPCA, LTR, k, C, d, c)
    poly_kernel_DTR_DTE = compute_polynomial_kernel(DTRPCA, DTEPCA, c, d, k)
    acc, err, scores = testKernelSVM(alfa_polynomial_SVM, Z_kernel_SVM, poly_kernel_DTR_DTE, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate Polinomial SVM on test set with PCA (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + ", c = " + str(c) + ", d = " + str(d) + "): " + str(format(err * 100, ".2f")) + "%\n")    
    print("min DCF(prior p1=0.5) on test set with Polinomial SVM with PCA (m={}, k = {}, C = {}, c = {}, d = {} ): {}\n".format( str(m), str(k), str(C), str(c), str(d), str(format(min_DCF, ".3f") )))    
    
    
    """standard GMM 8 components on test set without PCA"""
    
    DeltaL = 10e-6
    finalGmms = 8
    finalImpl = "standard"
    GMM = computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl)
    acc, err, scores = testGMM(GMM, DTE, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate Gaussian Mixture Model on test set (impl = " + finalImpl + ", GMMs = " + str(finalGmms) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) Gaussian Mixture Model without PCA (impl = {}, GMMs = {} ): {}\n".format( finalImpl, str(finalGmms), str(format(min_DCF, ".3f") )))    
    
    
    """diagonal GMM 1 component on test set with PCA (m=7)"""
    m = 7
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    DeltaL = 10e-6
    finalImpl = "diagonal"
    finalGmms = 1
    GMM = computeGMMs(DTRPCA, LTR, DeltaL, finalGmms, finalImpl)
    acc, err, scores = testGMM(GMM, DTEPCA, LTE)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    print("Error rate Gaussian Mixture Model on test set with PCA (m=" + str(m) + " impl = " + finalImpl + ", GMMs = " + str(finalGmms) + "): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) Gaussian Mixture Model with PCA (m={}, impl = {}, GMMs = {} ): {}\n".format(str(m), finalImpl, str(finalGmms), str(format(min_DCF, ".3f") )))    
    
    
    
    l = 1/1000
    optimal_threshold_kfold = -1.19007
    w, b = computeParametersForLogisticRegression(DTR, LTR, l)
    acc, err, scores = testLogisticRegression(w, b, DTE, LTE)
    scores = numpy.hstack(scores)
    min_DCF, _ = compute_min_dcf_prior(scores, LTE)
    actual_DCF, _ = compute_min_dcf_prior(scores, LTE, optimal_threshold_kfold)
    print("Error rate with test data Logistic Regression (NO PCA): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF(prior p1=0.5) Logistic Regression on Evaluation set (l={}): {}\n".format( str(l), str(format(min_DCF, ".3f") )))
    print("actual DCF(prior p1=0.5, threshold={}) Logistic Regression on Evaluation set (l={}): {}\n".format(str(format(optimal_threshold_kfold, ".5f")), str(l), str(format(min_DCF, ".3f") )))


    