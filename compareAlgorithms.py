# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:10:29 2021

@author: tommy
"""



import numpy
from PCA import PCA

from testModel import testModel, testLogisticRegression, testLinearSVM, testGMM, testKernelSVM, testPolinomialSVM
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes, computeParametersForLogisticRegression, \
        computeParametersForLinearSVM, computeGMMs, computeParameterForKernelPolynomialSVM, computeParameterForKernelRBFSVM, \
            compute_matrix_Z_kernel_SVM, compute_polynomial_kernel, compute_RBF_kernel, computeParametersForPolinomialSVM
from split import leaveOneOutSplit, kFoldSplit


"""applica e testa il modello selezionato sulla singola iterazione 
input:
    1) matrice dati di training (#variates, #samples)
    2) vettore label di training (#samples)
    3) matrice dati di test (#variates, #samples) con #samples = 1 se leaveOneOut
    4) vettore label di test (#samples) con #samples = 1 se leaveOneOut
output:
    1)errore
    2) accuratezza                 """
def applyAndTestModels(DTR, LTR, DTE, LTE, model, params):
    acc = 0
    err = 0
    if model == 0:
        """MVG"""
        mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
        acc, err, _ = testModel(mu, sigma, DTE, LTE)
        #print(err)
    
    elif model == 1:
        """Naive Bayes"""  
        mu , sigma = computeMeanAndCovarianceForNaiveBayes(DTR, LTR)
        acc, err, _ = testModel(mu, sigma, DTE, LTE)
        #print(err)
        
    elif model == 2: 
        """Tied"""    
        mu, sigma= computeMeanAndCovarianceForTied(DTR, LTR)
        acc, err, _ = testModel(mu, sigma, DTE, LTE)
        #print(err)
    elif model == 3: 
        """Logistic Regression"""    
        l = params[0] #we should try different parameters for l such as [0, 1/1000000, 1/1000, 1]
        w, b = computeParametersForLogisticRegression(DTR, LTR, l)
        acc, err, scores = testLogisticRegression(w, b, DTE, LTE)
        #print(err)
    elif model == 4: 
        """GMM"""    
        DeltaL = params[0]
        finalImpl = params[1]   
        finalGmms = params[2]
        GMM = computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl)
        acc, err, scores = testGMM(GMM, DTE, LTE)
    elif model == 5: 
        """Linear SVM"""    
        k=params[0]
        C = params[1]
        wHatStar = computeParametersForLinearSVM(DTR, LTR, k, C)
        acc , err, scores = testLinearSVM(wHatStar, k, DTE,LTE)
    elif model == 6:
        """Polinomial kernel SVM"""
        K_poly = params[0]
        C_poly = params[1]
        d_poly = params[2]
        c_poly = params[3]
        Z_kernel_SVM = compute_matrix_Z_kernel_SVM(DTR, LTR) # matrix Z is common for polynomial kernel and RBF kernel
        alfa_polynomial_SVM = computeParameterForKernelPolynomialSVM(DTR, LTR, K_poly, C_poly, d_poly, c_poly)
        poly_kernel_DTR_DTE = compute_polynomial_kernel(DTR, DTE, c_poly, d_poly, K_poly)
        acc, err, scores = testKernelSVM(alfa_polynomial_SVM, Z_kernel_SVM, poly_kernel_DTR_DTE, LTE)
    elif model == 7:
        """RBF kernel SVM"""
        K_RBF = params[0]
        C_RBF = params[1]
        gamma_RBF = params[2]
        Z_kernel_SVM = compute_matrix_Z_kernel_SVM(DTR, LTR)
        alfa_RBF_kernel = computeParameterForKernelRBFSVM(DTR, LTR, K_RBF, C_RBF, gamma_RBF)
        RBF_kernel_DTR_DTE = compute_RBF_kernel(DTR, DTE, gamma_RBF, K_RBF)
        acc, err, scores = testKernelSVM(alfa_RBF_kernel, Z_kernel_SVM, RBF_kernel_DTR_DTE, LTE)
    elif model == 8:
        """Polinomial kernel SVM 2nd impl"""
        k = params[0]
        C = params[1]
        d = params[2]
        c = params[3]
        optimalAlpha = computeParametersForPolinomialSVM(DTR, LTR, k, C, d, c)
        acc, err, S = testPolinomialSVM(optimalAlpha, LTR, DTR, DTE, LTE, c, d, k)     
        
    return acc, err

"""esegue il leave one out split
input:
    1) matrice dati di training (#variates, #samples)
    2) vettore label di training (#samples)
    3) modello da addestrare (0=MVG, 1= Naive, 2= Tied)
Output:
    1)errore
    2) accuratezza"""
def kFold(D, L, model, params=[]):
    errors = []
    accuracies = []
    k = 3
    """
    for leftOut in range(D.shape[1]):
        (DTR, LTR), (DTE, LTE) = leaveOneOutSplit(D, L, leftOut)
    """
    
    for foldIndex in range (0, k):
        (DTR, LTR), (DTE, LTE) = kFoldSplit(D, L, foldIndex, k)
        acc, err = applyAndTestModels(DTR, LTR, DTE, LTE, model, params)
        accuracies.append( acc )
        errors.append(  err )
        
    
    acc = sum(accuracies) / len(accuracies)
    err = sum(errors) / len(errors)
    return acc, err

"""
esegue tutti i test possibili
input:
    1) matrice dati di training (#variates, #samples)
    2) vettore label di training (#samples)
"""

"""returns the previous PCA if already computed, DTR if m == 1 and compute PCA otherwise """
def compute_PCA_if_needed(DTR, DTRPCA, m):
    if DTRPCA.shape[0] == m:
        return DTRPCA
    elif m == 11:
        print("No PCA")
        return DTR
    else:
        return PCA(DTR, m)


def compareAlgorithmsAndDimentionalityReduction(DTR, LTR):
    
    """con m = 11 DTRPCA = DTR"""
    DTRPCA = numpy.zeros((1,1), dtype="float64")
    m = 0
    minDimentionsTested = 5
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):       
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        #DTRPCA = PCA(DTR, m)
        acc_MVG,err_MVG = kFold(DTRPCA, LTR, 0)
        print("Error rate MVG (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        acc_Naive, err_Naive = kFold(DTRPCA, LTR, 1)
        print("Error rate Naive Bayes (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        acc_Tied, err_Tied = kFold(DTRPCA, LTR, 2)
        print("Error rate Tied (m=" + str(m) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)    
        for l in [0, 1/1000000, 1/1000, 1]:
            params = []
            params.append(l)
            acc_LogReg, err_LogReg = kFold(DTRPCA, LTR, 3, params)
            print("Error rate Logistic Regression (m=" + str(m) + ", l = " + str(l) + "): " + str(format(err_LogReg * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)    
        DeltaL = 10e-6
        for finalImpl in ["standard", "diagonal"]:#, "tied"]:
            for finalGmms in [1,2,4,8,16]:
                params = []
                params.append(DeltaL)
                params.append(finalImpl)
                params.append(finalGmms)
                acc_GMM, err_GMM = kFold(DTRPCA, LTR, 4, params)
                print("Error rate Gaussian Mixture Model (m=" + str(m) + ", impl = " + finalImpl + ", GMMs = " + str(finalGmms) + "): " + str(format(err_GMM * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)       
        for k in [1, 10]:
            for C in [0.1, 1.0, 10.0]:
                params = []
                params.append(k)
                params.append(C)
                acc_LinSVM, err_LinSVM = kFold(DTRPCA, LTR, 5, params)
                print("Error rate Linear SVM (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + "): " + str(format(err_LinSVM * 100, ".2f")) + "%\n")
    
        #too slow, see other impl
    #for m in range(minDimentionsTested, DTR.shape[0]+1):        
    #    DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)    
    #    d = 2
    #    C = 1
    #    for k in [0, 1]:
    #        for c in [0, 1]:
    #            params = []
    #            params.append(k)
    #            params.append(C)
    #            params.append(d)
    #            params.append(c)
    #            acc_PoliSVM, err_PoliSVM = kFold(DTRPCA, LTR, 6, params)
    #            print("Error rate Polinomial SVM (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + ", c = " + str(c) + ", d = " + str(d) + "): " + str(format(err_PoliSVM * 100, ".2f")) + "%\n")
    
    print("\n")           
    for m in range(minDimentionsTested, DTR.shape[0]+1):
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        C = 1
        for gamma in [1, 10]:
            for k in [0, 1]:
                params = []
                params.append(k)
                params.append(C)
                params.append(gamma)
                acc_PoliRBF, err_PoliRBF = kFold(DTRPCA, LTR, 7, params)
                print("Error rate RBF SVM (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + ", gamma = " + str(gamma) + "): " + str(format(err_PoliRBF * 100, ".2f")) + "%\n")
    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)    
        d = 2
        C = 1
        for k in [0, 1]:
            for c in [0, 1]:
                params = []
                params.append(k)
                params.append(C)
                params.append(d)
                params.append(c)
                acc_PoliSVM2, err_PoliSVM2 = kFold(DTRPCA, LTR, 8, params)
                print("Error rate Polinomial SVM (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + ", c = " + str(c) + ", d = " + str(d) + "): " + str(format(err_PoliSVM2 * 100, ".2f")) + "%\n")    
      