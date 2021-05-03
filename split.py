# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:08:25 2021

@author: tommy
"""
import numpy

"""singola iterazione del leaveOneOut, funzione chiamata n volte
input : 
    1) Data Matrix (#variates, #samples)
    2) Label vector (#samples)
    3) #sample che farà parte del campione di test
output : 
    1) Matrice di Training (#variates, #samples-1) 
    2) Vettore Label di training (#samples-1) 
    3) Matrice di test (#variates, 1) 
    4) Vettore Label di test (1)"""
def leaveOneOutSplit(D, L, testSampleIndex):
    i = testSampleIndex
    DTR = numpy.delete(D, [testSampleIndex], axis=1)
    DTE = D[:, i : i+1]
    LTR = numpy.delete(L, testSampleIndex)
    LTE = L[i : i+1]
    return (DTR, LTR), (DTE, LTE)