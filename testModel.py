# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:19:02 2021

@author: tommy
"""

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol

"""computing log-density, 
input : 
    1) test data matrix(#variates, #sample test), 
    2) mean vector(#variates)
    3) covariance matrix sigma(#variates, #variates)
output:
    1) vettore delle densità (#samples test)
        """  
def logpdf_GAU_ND(XND, mu, C):
    M = XND.shape[0]
    sigma = C
    _, logdet = numpy.linalg.slogdet(sigma)
    pref_dens = -1 * M/2 * numpy.log(2 * numpy.pi) - 1/2 * logdet
    sigma_inverse = numpy.linalg.inv(sigma)
    list_values = []
    for i in range(XND.shape[1]):
        list_values.append( -1/2 * numpy.dot(numpy.dot((XND[:, i:i+1]-mu).T, sigma_inverse), (XND[:, i:i+1] - mu)))
    
    log_density = numpy.vstack(list_values)
    log_density += pref_dens
    log_density = log_density.reshape(log_density.shape[0])

    return log_density
"""calcola errore e accuratezza della previsione
input:
    1) vettore di class posterior probabilities (#classi, #samples test)
    2) matrice di label di test (#sample test)
output:
    1) accuratezza da 0 a 1
    2) errore da 0 a 1
"""
def computeError(SPost, LTE):
    predictedLabels = SPost.argmax(0)
    correct = 0
    for i in range(0, LTE.shape[0]):
        if predictedLabels[i] == LTE[i]:
               correct = correct + 1
               
    acc = correct / LTE.shape[0]
    err = 1 - acc
    return acc, err

"""computes confusion matrix for a problem of 2 classes (predicted class, actual class)
input:
    1) vettore di class posterior probabilities (#classi, #samples test)
    2) matrice di label di test (#sample test)
output: Confusion Matrix (#classes, #classes)    """
def computeConfusionMatrix(SPost, LTE):
    predictedLabels = SPost.argmax(0)
    CM = numpy.zeros( (2, 2), dtype="int32") 
    for i in range(0, LTE.shape[0]):
        predictionClass = predictedLabels[i]
        actualClass = LTE[i]
        CM[predictionClass , actualClass ] = CM[predictionClass , actualClass ] + 1
    return CM



"""testa il modello (rappresentato dalla media e dalla covarianza sui dati di test
input:
    1) vettore di vettori rappresentante la media del modello (#classi, #variates)
    2) vettore di matrici rappresentante la covarianza del modello (#classi, #variates, #variates)
    3) matrice di dati di test (#classi, #samples)
    4) vettore di label di test (#samples)
output:
    1) accuratezza da 0 a 1
    2) errore da 0 a 1
    3) matrice di confusione (#classi, #classi)"""
def testModel(mu, sigma, DTE, LTE):   
    S = numpy.zeros( (2, DTE.shape[1]), dtype="float32")  #exponential of the log densities Matrix (each row a class)    
    Sjoint = numpy.zeros( (2, DTE.shape[1]), dtype="float32")
    SPost =  numpy.zeros( (2, DTE.shape[1]), dtype="float32")
    for c in range(0,2):
        S[c] = logpdf_GAU_ND(DTE, mcol(mu[c]), sigma[c]) #log densities
        Sjoint[c] = S[c] + numpy.log(1/2)       # joint log densities
    
    SPost = Sjoint - scipy.special.logsumexp(Sjoint, 0) #joint log-densities / marginal log-densities
    
    acc, err = computeError(SPost, LTE)
    CM = computeConfusionMatrix(SPost, LTE)
    return acc, err, CM
    
