# -*- coding: utf-8 -*-

import numpy , scipy.special, scipy.linalg
from matrixUtilities import mcol, plot_hist
import loadData
from PCA import PCA
from LDA import LDA
from testModel import TestModel
from classificatori import computeMeanAndCovarianceForMultiVariate, computeMeanAndCovarianceForTied





if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = loadData.load_data()
    #plot_hist(DTR, LTR);

    """applico PCA e LDA"""
    m=4
    DTRPCA = PCA(DTR, m)
    DTRPCAplusLDA = LDA(DTRPCA, LTR, m)
    
    
    #mu, sigma = computeMeanAndCovarianceForMultiVariate(DTRPCA, LTR)
    #print(sigma)
    
    #plot_hist(DTRPCA, LTR);
    
    """applico il classificatore opportuno"""
    """MVG"""
    mu, sigma = computeMeanAndCovarianceForMultiVariate(DTR, LTR)
    
    """Tied"""  
    #mu, sigma = computeMeanAndCovarianceForTied(DTR, LTR)

    """calcolo accuratezza, errore e confusion matrix sui dati di test. Per ora il costo non è considerato"""
    acc, err, CM = TestModel(mu, sigma, DTE, LTE)
    #print(CM)   
    #print(err)