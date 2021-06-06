# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:10:29 2021

@author: tommy
"""




from PCA import PCA
from LDA import compute_data_LDA

from testModel import testModel, testLogisticRegression, testLinearSVM, testGMM

from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes, computeParametersForLogisticRegression, \
        computeParametersForLinearSVM, computeGMMs
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
        acc, err = testLogisticRegression(w, b, DTE, LTE)
        #print(err)
    elif model == 4: 
        """GMM"""    
        DeltaL = params[0]
        finalImpl = params[1]   
        finalGmms = params[2]
        GMM = computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl)
        acc, err = testGMM(GMM, DTE, LTE)
    elif model == 5: 
        """Linear SVM"""    
        k=params[0]
        C = params[1]
        wHatStar = computeParametersForLinearSVM(DTR, LTR, k, C)
        acc , err = testLinearSVM(wHatStar, k, DTE,LTE)
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
def compareAlgorithmsAndDimentionalityReduction(DTR, LTR):
    
    m = 0
    for m in range(5, DTR.shape[0]+1):
        
        DTRPCA = PCA(DTR, m)
        
        
        if m == 11:
            acc_MVG,err_MVG = kFold(DTR, LTR, 0)
            print("Error rate MVG NO PCA : " + str(format(err_MVG * 100, ".2f")) + "%\n")
        else:
            acc_MVG,err_MVG = kFold(DTRPCA, LTR, 0)
            print("Error rate MVG with PCA (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
        if m == 11:
            acc_Naive, err_Naive = kFold(DTRPCA, LTR, 1)
            print("Error rate Naive Bayes NO PCA : " + str(format(err_Naive * 100, ".2f")) + "%\n")
        else:        
            acc_Naive, err_Naive = kFold(DTRPCA, LTR, 1)
            print("Error rate Naive Bayes with PCA (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
    
        if m == 11:
            acc_Tied, err_Tied = kFold(DTRPCA, LTR, 2)
            print("Error rate Tied NO PCA : " + str(format(err_Tied * 100, ".2f")) + "%\n")
        else:
            acc_Tied, err_Tied = kFold(DTRPCA, LTR, 2)
            print("Error rate Tied with PCA (m=" + str(m) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")
      
        for l in [0, 1/1000000, 1/1000, 1]:
            params = []
            params.append(l)
            if m == 11:
                acc_LogReg, err_LogReg = kFold(DTR, LTR, 3, params)
                print("Error rate Logistic Regression NO PCA (l = " + str(l) + "): " + str(format(err_LogReg * 100, ".2f")) + "%\n")
            else:
                acc_LogReg, err_LogReg = kFold(DTRPCA, LTR, 3, params)
                print("Error rate Logistic Regression with PCA (m=" + str(m) + ", l = " + str(l) + "): " + str(format(err_LogReg * 100, ".2f")) + "%\n")
        
        DeltaL = 10e-6
        for finalImpl in ["standard", "diagonal"]:#, "tied"]:
            for finalGmms in [1,2,4,8,16]:
                params = []
                params.append(DeltaL)
                params.append(finalImpl)
                params.append(finalGmms)
                if m == 11:
                    acc_GMM, err_GMM = kFold(DTR, LTR, 4, params)
                    print("Error rate Gaussian Mixture Model NO PCA (impl = " + finalImpl + ", GMMs = " + str(finalGmms) + "): " + str(format(err_GMM * 100, ".2f")) + "%\n")
                else:
                    acc_GMM, err_GMM = kFold(DTRPCA, LTR, 4, params)
                    print("Error rate Gaussian Mixture Model with PCA (m=" + str(m) + ", impl = " + finalImpl + ", GMMs = " + str(finalGmms) + "): " + str(format(err_GMM * 100, ".2f")) + "%\n")
        
        for k in [1, 10]:
            for C in [0.1, 1.0, 10.0]:
                params = []
                params.append(k)
                params.append(C)
                if m == 11:
                    acc_LinSVM, err_LinSVM = kFold(DTR, LTR, 5, params)
                    print("Error rate Linear SVM NO PCA (k = " + str(k) + ", C = " + str(C) + "): " + str(format(err_LinSVM * 100, ".2f")) + "%\n")
                else:
                    acc_LinSVM, err_LinSVM = kFold(DTRPCA, LTR, 5, params)
                    print("Error rate Linear SVM with PCA (m=" + str(m) + ", k = " + str(k) + ", C = " + str(C) + "): " + str(format(err_LinSVM * 100, ".2f")) + "%\n")
                
        
def compareLDA(DTR, LTR):
    m = 0
    for m in range(2, DTR.shape[0]+1):
        #DTRPCA = PCA(DTR, m)
        DTRLDA = compute_data_LDA(DTR, LTR, m)
        #acc_MVG,err_MVG = kFold(DTRPCA, LTR, 0)
        acc_MVG,err_MVG = kFold(DTRLDA, LTR, 0)
        #print("Error rate MVG with PCA (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
        print("Error rate MVG with LDA (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
    m = 0
    for m in range(2, DTR.shape[0]+1):
        #DTRPCA = PCA(DTR, m)
        DTRLDA = compute_data_LDA(DTR, LTR, m)
        #acc_Naive, err_Naive = kFold(DTRPCA, LTR, 1)
        acc_Naive, err_Naive = kFold(DTRLDA, LTR, 1)
        #print("Error rate Naive Bayes with PCA (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
        print("Error rate Naive Bayes with LDA (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
    
    m = 0
    for m in range(2, DTR.shape[0]+1):
        DTRLDA = compute_data_LDA(DTR, LTR, m)
        acc_Tied, err_Tied = kFold(DTRLDA, LTR, 2)
        print("Error rate Tied with LDA (m=" + str(m) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")

"""compare PCA and after LDA"""
def compare_PCA_before_LDA(DTR, LTR):
    
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTRPCA, LTR, m_LDA)
            acc_MVG,err_MVG = kFold(DTRLDA, LTR, 0)
            print("Error rate MVG with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
    
    # compute statistics for Naive Bayes
    """
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTRPCA, LTR, m_LDA)
            acc_Naive, err_Naive = kFold(DTRLDA, LTR, 1)
            print("Error rate Naive Bayes with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
    """
    
    # compute statistics fot Tied classifier
    """
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTRPCA, LTR, m_LDA)
            acc_Tied ,err_Tied = kFold(DTRLDA, LTR, 2)
            print("Error rate Tied with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")
   """ 