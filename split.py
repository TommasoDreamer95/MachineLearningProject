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


def kFoldSplit(D, L, foldIndex, k):
    samplesPerFold = int(L.shape[0] / k)
    startIndex = foldIndex * samplesPerFold
    endIndex = (foldIndex +1 ) * samplesPerFold
    DTR = D
    LTR = L
    for i in range (startIndex, endIndex):
        DTR = numpy.delete(DTR, [startIndex], axis=1)
        LTR = numpy.delete(LTR, startIndex)
    DTE = D[:, startIndex : endIndex]    
    LTE = L[startIndex : endIndex]
    return (DTR, LTR), (DTE, LTE)

if __name__ == '__main__':
    D = numpy.zeros((2,3))
    L = numpy.zeros((3))
    for i in range (0,3):
        D[:, i] = i
        L[i]= i
    (DTR, LTR), (DTE, LTE) = kFoldSplit(D, L, 0, 3)
    
    print(D)
    print(DTR)
    print(DTE)