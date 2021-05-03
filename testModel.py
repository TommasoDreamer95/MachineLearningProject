# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:19:02 2021

@author: tommy
"""

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, compute_num_classes

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



"""
compute di gaussian density for a scalar
inpute:
1) point for which compute the gaussian density
2) mean 
3)variance 
"""
def GAU_scalar(x, mu, var):
    N = 1/(numpy.sqrt(2*numpy.pi * var)) * numpy.exp(-1* (x-mu)**2/(2*var))
    return N

"""
input: 
1) List of labels
2) matrix of data
3) mean for all classes
4) sigma matrix for all classes
                                 
Output: matrix S with class conditional probabilities 
"""
def compute_naiveBayes(L, D, mu, sigma):
    list_score = [] 
    for index_class in range(sigma.shape[0]):
        list_score_class = []
        sigma_class = sigma[index_class, :]
        sigma_vector_diag = numpy.diag(sigma_class)
        mu_class = mcol(mu[index_class, :])
        for index_sample in range(D.shape[1]):  
            sample = mcol(D[:, index_sample])
            list_components_prob = []
            for index_component in range(sample.shape[0]):       
                mu_component = mu_class[index_component]
                sample_component = sample[index_component, 0]
                sigma_component = sigma_vector_diag[index_component]
                probab_component = GAU_scalar(sample_component, mu_component, sigma_component)
                list_components_prob.append(probab_component)
            # --multiply the probabiltiy of all components of the sample
            array_components = numpy.array(list_components_prob)
            list_score_class.append(array_components.prod())
        prediction_class = numpy.hstack(list_score_class)
        list_score.append(prediction_class)
    return numpy.vstack(list_score)

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
def testModel_MVG(mu, sigma, DTE, LTE):   
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
"""
Test the Naive Bayes classifier
"""
def testModel_Naive_Bayes(mu, sigma, DTE, LTE):
    S = compute_naiveBayes(LTE, DTE, mu, sigma)
    probab_classes = 1/compute_num_classes(LTE)
    Sjoint = probab_classes * S
    SPost = Sjoint / Sjoint.sum(axis=0)
    acc, err = computeError(SPost, LTE)
    CM = computeConfusionMatrix(SPost, LTE)
    return acc, err, CM
    
