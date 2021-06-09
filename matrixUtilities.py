# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:31:01 2021

@author: luca
"""

import matplotlib.pyplot as plt
import numpy

"""
transform a matrix in a matrix with one column
"""
def mcol(v):
    return v.reshape((v.size, 1))

"""
compute the number of classes, given the matrix of labels L
"""
def compute_num_classes(L):
    # compute number of classes given vector L
    return len(numpy.unique(L))

"""
plot histogram for some dimensions of the features
"""
def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }

    for dIdx in range(0, D.shape[0]):
        plt.figure()
        if D.shape[0] == 11:
            plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Low quality wine')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'High quality wine')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('../hist_%d.pdf' % dIdx)
    plt.show()
    
def plot_scatter(D, L):
    D0 = D[:, L==0]
    D1 = D[:, L==1]  
    
    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }
    for idx in range(0, D.shape[0]):
        for i in range(0, D.shape[0]):
            if i != idx:
                plt.figure()
                plt.xlabel(hFea[idx]);
                plt.ylabel(hFea[i])
                plt.scatter(D0[idx], D0[i])
                plt.scatter(D1[idx], D1[i])
                plt.show();