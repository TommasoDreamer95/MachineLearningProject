# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:10:29 2021

@author: tommy
"""



import numpy
from PCA import PCA
import sys

from testModel import testModel, testLogisticRegression, testLinearSVM, testGMM, testKernelSVM, testPolinomialSVM
from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes, computeParametersForLogisticRegression, \
        computeParametersForLinearSVM, computeGMMs, computeParameterForKernelPolynomialSVM, computeParameterForKernelRBFSVM, \
            compute_matrix_Z_kernel_SVM, compute_polynomial_kernel, compute_RBF_kernel, computeParametersForPolinomialSVM
from split import leaveOneOutSplit, kFoldSplit

def compute_confusion(predicted_labels, LTE):
    list_conf_matrix = []
    #num_classes = len(numpy.unique(predicted_labels))
    num_classes = len(numpy.unique(LTE))#prova
    for pred_class in range(num_classes):
        #predicted class i
        predictions_class_loop = (predicted_labels == pred_class)
        list_row_confusion_matrix = []
        for correct_class in range(num_classes):
            # features of class j
            features_class_loop = (LTE == correct_class)
            # compute element (i, j) of the matrix
            num_features_prediction_label = (predictions_class_loop & features_class_loop).sum()
            list_row_confusion_matrix.append(num_features_prediction_label)
        list_conf_matrix.append(list_row_confusion_matrix)
    confusion_matrix = numpy.array(list_conf_matrix)
    return confusion_matrix

def compute_decision_given_threshold(log_ratios, threshold):
    list_predictions = []
    for index in range(log_ratios.shape[0]):
        if(log_ratios[index] > threshold):
            list_predictions.append(1)
        else:
            list_predictions.append(0)
    return numpy.array(list_predictions)

def compute_bayes_risk(confusion_matrix, p1, cfn, cfp):
    FN = confusion_matrix[0][1]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[0][0]
    FNR = FN/(FN + TP)
    FPR = FP/(FP + TN)
    return p1*cfn * FNR + (1 - p1)*cfp * FPR

def compute_normalized_DCF(p1, cfn, cfp, bayes_risk):
    Bdummy = min(p1*cfn, (1-p1)*cfp)
    return  bayes_risk/Bdummy

def compute_min_dcf(log_likelihood_ratios, labels, p1, cfn, cfp):
    list_thresholds = sorted(log_likelihood_ratios) #sort in ascending order
    list_thresholds.insert(0, sys.float_info.min)
    list_thresholds.append(sys.float_info.max)
    min_DCF = sys.float_info.max
    for threshold in list_thresholds:
        decisions = compute_decision_given_threshold(log_likelihood_ratios, threshold)
        confusion_matrix = compute_confusion(decisions, labels)
        bayes_risk = compute_bayes_risk(confusion_matrix, p1, cfn, cfp)
        # compute normalized DCF
        DCF = compute_normalized_DCF(p1, cfn, cfp, bayes_risk)
        #print(DCF)
        if DCF < min_DCF:
            min_DCF = DCF
    return min_DCF

"""
compute the min dcf given different values of priors (0.5, 0.1, 0.9) and return a list 
in which each element is the min dcf of the prior
"""
def compute_min_dcf_different_priors(log_likelihood_ratios, labels):
    list_p1 = [0.5, 0.1, 0.9]
    cfn = 1
    cfp = 1
    min_dcf_priors = []
    for p1 in list_p1:
        min_dcf_p1 = compute_min_dcf(log_likelihood_ratios, labels, p1, cfn, cfp)
        min_dcf_priors.append([p1, min_dcf_p1])
    return min_dcf_priors

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
        acc, err, scores = testModel(mu, sigma, DTE, LTE)
        #print(err)
    
    elif model == 1:
        """Naive Bayes"""  
        mu , sigma = computeMeanAndCovarianceForNaiveBayes(DTR, LTR)
        acc, err, scores = testModel(mu, sigma, DTE, LTE)
        #print(err)
        
    elif model == 2: 
        """Tied"""    
        mu, sigma= computeMeanAndCovarianceForTied(DTR, LTR)
        acc, err, scores = testModel(mu, sigma, DTE, LTE)
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
        acc, err, scores = testPolinomialSVM(optimalAlpha, LTR, DTR, DTE, LTE, c, d, k)     
        
    return acc, err, scores

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
    scores = []
    k = 3
    """
    for leftOut in range(D.shape[1]):
        (DTR, LTR), (DTE, LTE) = leaveOneOutSplit(D, L, leftOut)
    """
    
    for foldIndex in range (0, k):
        (DTR, LTR), (DTE, LTE) = kFoldSplit(D, L, foldIndex, k)
        acc, err, scores_k = applyAndTestModels(DTR, LTR, DTE, LTE, model, params)
        scores.append(scores_k)
        accuracies.append( acc )
        errors.append(  err )
        
    scores = numpy.hstack(scores)
    min_DCF_priors = compute_min_dcf_different_priors(scores, L)
    
    acc = sum(accuracies) / len(accuracies)
    err = sum(errors) / len(errors)
    return acc, err, min_DCF_priors

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
        acc_MVG,err_MVG, min_DCF_MVG_priors = kFold(DTRPCA, LTR, 0)
        print("Error rate MVG with PCA (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
        for list_min_DCF in min_DCF_MVG_priors:
            p1 = list_min_DCF[0]
            min_DCF_MVG = list_min_DCF[1]
            print("min DCF(prior p1={}) MVG with PCA (m={}): {}\n".format(str(p1), str(m), str(format(min_DCF_MVG, ".3f") )))

    
    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        acc_Naive, err_Naive, min_DCF_Naive_priors = kFold(DTRPCA, LTR, 1)
        print("Error rate Naive Bayes with PCA (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
        for list_min_DCF in min_DCF_Naive_priors:
            p1 = list_min_DCF[0]
            min_DCF_Naive = list_min_DCF[1]
            print("min DCF(prior p1={}) Naive with PCA (m={}): {}\n".format(str(p1), str(m), str(format(min_DCF_Naive, ".3f") )))

    print("\n")
    for m in range(minDimentionsTested, DTR.shape[0]+1):        
        DTRPCA = compute_PCA_if_needed(DTR, DTRPCA, m)
        acc_Tied, err_Tied, min_DCF_tied_priors = kFold(DTRPCA, LTR, 2)
        print("Error rate Tied with PCA (m=" + str(m) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")
        for list_min_DCF in min_DCF_tied_priors:
            p1 = list_min_DCF[0]
            min_DCF_Tied = list_min_DCF[1]
            print("min DCF(prior p1={}) Tied with PCA (m={}): {}\n".format(str(p1), str(m), str(format(min_DCF_Tied, ".3f") )))
    
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
                print("Error rate Polinomial SVM with PCA (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + ", c = " + str(c) + ", d = " + str(d) + "): " + str(format(err_PoliSVM2 * 100, ".2f")) + "%\n")    
         

