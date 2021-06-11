# -*- coding: utf-8 -*-

import numpy #, scipy.special, scipy.linalg
from matrixUtilities import plot_hist, plot_scatter
import loadData
from PCA import PCAforDTE
import testModel
from testModel import testLogisticRegression
from classificatori import computeParametersForLogisticRegression
from compareAlgorithms import compareAlgorithmsAndDimentionalityReduction, compute_min_dcf_prior


    
if __name__ == '__main__':
    (DTR, LTR) = loadData.load_data_TR()
    
    """plot histograms of training data"""
    #plot_hist(DTR, LTR);
    #plot_scatter(DTR, LTR)
    
    
    
    """ Cross validation:
        trains various models with different parameters and different PCA applied on the training set,
        evaluates each model with a 3-fold strategy and stores the results on the output file "AlgorithmsOutput.txt".
        The execution takes about 3-4 hours
        N.B. only the training data are passed as input to this function, 
        the function is unaware of what the test data look like. DTE variables inside the compareAlgorithmsAndDimentionalityReduction function 
        refer to evaluation data produced via splitting of DTR
    """
    compareAlgorithmsAndDimentionalityReduction(DTR, LTR) 
    
    
    
    """based on the compare algorithms output the best algorithm should be logistic regression, 
        applied to either unprocessed data or to data preprocessed with PCA in order to obtain 10 dimentions out of the initial 11 ones
        and a value of the parameter lambda equal to 1/1000.
        That will be the model we deliver.
        Therefore we train the model on the training data and test it on the test data.
        N.B. the following lines of code where only added after having analyzed the the output of 
        the compareAlgorithmsAndDimentionalityReduction function. The test data were not used for cross validation
    """
    
    (DTE, LTE) = loadData.load_data_TE()
    
    m=10
    DTRPCA, DTEPCA = PCAforDTE(DTR, DTE, m)
    l = 1/1000
    w, b = computeParametersForLogisticRegression(DTRPCA, LTR, l)
    acc, err, scores = testLogisticRegression(w, b, DTEPCA, LTE)
    scores = numpy.hstack(scores)
    min_DCF = compute_min_dcf_prior(scores, LTE)
    print("Error rate with test data (with PCA): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF with test data with p = 0.5 (with PCA): " + str(format(min_DCF, ".3f")) + "%\n")
    
    l = 1/1000
    w, b = computeParametersForLogisticRegression(DTR, LTR, l)
    acc, err, scores = testLogisticRegression(w, b, DTE, LTE)
    scores = numpy.hstack(scores)
    min_DCF = compute_min_dcf_prior(scores, LTE)
    print("Error rate with test data (NO PCA): " + str(format(err * 100, ".2f")) + "%\n")
    print("min DCF with test data with p = 0.5 (NO PCA): " + str(format(min_DCF, ".3f")) + "%\n")
    

    