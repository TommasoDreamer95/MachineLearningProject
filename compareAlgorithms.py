# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:10:29 2021

@author: tommy
"""




from PCA import PCA
from LDA import compute_data_LDA

from testModel import testModel

from classificatori import computeMeanAndCovarianceForMultiVariate, \
    computeMeanAndCovarianceForTied, computeMeanAndCovarianceForNaiveBayes
from split import leaveOneOutSplit


"""applica e testa il modello selezionato sulla singola iterazione 
input:
    1) matrice dati di training (#variates, #samples)
    2) vettore label di training (#samples)
    3) matrice dati di test (#variates, #samples) con #samples = 1 se leaveOneOut
    4) vettore label di test (#samples) con #samples = 1 se leaveOneOut
output:
    1)errore
    2) accuratezza                 """
def applyAndTestModels(DTR, LTR, DTE, LTE, model):
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
    
    return acc, err

"""esegue il leave one out split
input:
    1) matrice dati di training (#variates, #samples)
    2) vettore label di training (#samples)
    3) modello da addestrare (0=MVG, 1= Naive, 2= Tied)
Output:
    1)errore
    2) accuratezza"""
def kFold(D, L, model):
    errors = []
    accuracies = []
    
    for leftOut in range(D.shape[1]):
        (DTR, LTR), (DTE, LTE) = leaveOneOutSplit(D, L, leftOut)
        acc, err = applyAndTestModels(DTR, LTR, DTE, LTE, model)
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
    for m in range(2, DTR.shape[0]+1):
        DTRPCA = PCA(DTR, m)
        acc_MVG,err_MVG = kFold(DTRPCA, LTR, 0)
        print("Error rate MVG with PCA (m=" + str(m) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    
    m = 0
    for m in range(2, DTR.shape[0]+1):
        DTRPCA = PCA(DTR, m)
        acc_Naive, err_Naive = kFold(DTRPCA, LTR, 1)
        print("Error rate Naive Bayes with PCA (m=" + str(m) + "): " + str(format(err_Naive * 100, ".2f")) + "%\n")
    
    m = 0
    for m in range(2, DTR.shape[0]+1):
        DTRPCA = PCA(DTR, m)
        acc_Tied, err_Tied = kFold(DTRPCA, LTR, 2)
        print("Error rate Tied with PCA (m=" + str(m) + "): " + str(format(err_Tied * 100, ".2f")) + "%\n")
        
        
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
    """
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTR, LTR, m_LDA)
            acc_MVG,err_MVG = kFold(DTRLDA, LTR, 0)
            print("Error rate MVG with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    """
    
    # compute statistics for Naive Bayes
    """
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTR, LTR, m_LDA)
            acc_MVG,err_MVG = kFold(DTRLDA, LTR, 1)
            print("Error rate Naive Bayes with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    """
    
    # compute statistics fot Tied classifier
    m_PCA = 0
    for m_PCA in range(2, DTR.shape[0]):
        DTRPCA = PCA(DTR, m_PCA)
        for m_LDA in range(2, m_PCA):
            DTRLDA = compute_data_LDA(DTR, LTR, m_LDA)
            acc_MVG,err_MVG = kFold(DTRLDA, LTR, 2)
            print("Error rate Tied with PCA (m=" + str(m_PCA) + ") and then LDA(m = " + str(m_LDA) + "): " + str(format(err_MVG * 100, ".2f")) + "%\n")
    