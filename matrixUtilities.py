# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:31:01 2021

@author: luca
"""

import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

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

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Low quality wine')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'High quality wine')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('../hist_%d.pdf' % dIdx)
    plt.show()