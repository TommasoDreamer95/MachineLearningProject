# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:46:22 2021

@author: tommy
"""
import numpy , scipy.special
from matrixUtilities import mcol

"""calcola la media mu e la covarianza sigma del modello MVG
input: 
    1) Matrice di training (#variates, #sample di training)
    2) Vettore label di training (#sample di training)
output: 
    1) vettore contenente #classi di vettori da (#variabili casuali del sample) elementi
    2) vettore contenente #classi di matrici quadrate da (#variabili casuali del sample) elementi per lato""" 
def computeMeanAndCovarianceForMultiVariate(DTR, LTR):
    mu = numpy.zeros( (2, DTR.shape[0]))
    SigmaC = numpy.zeros( (2, DTR.shape[0], DTR.shape[0] ) )
    for c in range(0,2):
        Dc = DTR[:, LTR==c] 
        mu[c] = Dc.mean(1)  
        DC = Dc - mcol(mu[c])  #matrice centrata, in cui ad ogni colonna sottraggo la media
        SigmaC[c] = numpy.dot(DC, DC.T) / float(DC.shape[1]) # C = 1/N * Dc * Dc Trasposta
    return mu, SigmaC

"""calcola la media mu e la covarianza sigma del modello Naive Bayes
input: 
    1) Matrice di training (#variates, #sample di training)
    2) Vettore label di training (#sample di training)
output: 
    1) vettore contenente #classi di vettori da (#variabili casuali del sample) elementi
    2) vettore contenente #classi di matrici diagonali e quadrate da (#variabili casuali del sample) elementi per lato""" 
def computeMeanAndCovarianceForNaiveBayes(DTR, LTR):
    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
    sigma = sigma * numpy.eye(DTR.shape[0])
    return mu, sigma

"""calcola la media mu e la covarianza sigma del modello Tied Covariance
input: 
    1) Matrice di training (#variates, #sample di training)
    2) Vettore label di training (#sample di training)
output: 
    1) vettore contenente #classi di vettori da (#variabili casuali del sample) elementi
    2) vettore contenente #classi di matrici quadrate da (#variabili casuali del sample) elementi per lato""" 
def computeMeanAndCovarianceForTied(DTR, LTR):
    mu = numpy.zeros( (2, DTR.shape[0]))
    sigma = numpy.zeros( (2, DTR.shape[0], DTR.shape[0] ) )
    for c in range(0,2):
        mu[c] = DTR[:, LTR==c].mean(1)
        sigma[c] = computeSW(DTR, LTR)
    return mu, sigma    

"""calcola la matrice di covarianza sigma del modello Tied
input : 
    1) Data Matrix (#variates, #samples)
    2) Label vector (#samples)
output :
    1) matrice di Covarianza (#variates, #variates) 
"""    
def computeSW(D,L): 
    withinClassCovarianceMatrix = 0
    for classe in range(0,2):
        Dc = D[:, L==classe] 
        muc = Dc.mean(1)  
        for i in range(0, Dc.shape[1]):  
            withinClassCovarianceMatrix = withinClassCovarianceMatrix + numpy.dot( mcol(Dc[:, i] - muc), mcol(Dc[:, i] - muc).T)
    withinClassCovarianceMatrix = withinClassCovarianceMatrix / float(D.shape[1])
    #print(withinClassCovarianceMatrix)
    return withinClassCovarianceMatrix

"""
compute sigma with only diagonal elements defined, for each class
"""
def create_sigma_diagonal(sigma):
    num_classes = sigma.shape[0]
    list_sigma = []
    for index_class in range(num_classes):
        sigma_class = sigma[index_class, :]
        sigma_class = sigma_class * numpy.identity(sigma_class.shape[0])
        list_sigma.append(sigma_class)
    return numpy.array(list_sigma)


def logreg_obj(v, DTR, LTR, l):
    w, b = v[0:-1], v[-1]    
    firstTerm = (l * numpy.linalg.norm(w)**2 ) / 2     
    totalSum = 0
    n = DTR.shape[1]
    for i in range(0, n):
        zi = -1
        if LTR[i] == 1:
            zi = 1
        prod = numpy.dot(w.T, DTR[: , i])
        thisSum = numpy.log1p( numpy.exp( (- zi)*(prod + b) ) )
        totalSum = totalSum + thisSum
    secondTerm = totalSum / n
    result = firstTerm + secondTerm 
    return result

def computeParametersForLogisticRegression(DTR, LTR, l):     
    v = numpy.zeros(DTR.shape[0] + 1) 
    (x, f, _) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v, args=(DTR, LTR, l), approx_grad=True)  
    w, b = x[0:-1], x[-1]
    return w, b