# -*- coding: utf-8 -*-

import numpy , scipy
from matrixUtilities import mcol, plot_hist
import loadData



def covariance(D):
    mu = D.mean(1) #array con la media delle colonne della matrice
    DC = D - mcol(mu)  #matrice centrata, in cui ad ogni colonna sottraggo la media
    C = numpy.dot(DC, DC.T) / float(DC.shape[1]) # C = 1/N * Dc * Dc Trasposta
    return C 

"""input = 
    1) data matrix
    2) #resulting dimentions
output:
    1) projected Data                  """
def PCA(D, resultingDimentions): 
    CovarianceMatrix = covariance(D)
    eigenvectorsMatrix, eigenvalues, Vh = numpy.linalg.svd(CovarianceMatrix)
    mLeadingEigenvectors = eigenvectorsMatrix[:, 0:resultingDimentions]
    DataProjected = numpy.dot(mLeadingEigenvectors.T, D) #apply projection to matrix of samples
    return DataProjected 

"""#input = data matrix and labels vector"""    
def computeSB(D, L):
    mu = D.mean(1)
    betweenClassCovarianceMatrix = 0
    for c in range(0,2): 
        Dc = D[:, L==c] #Dc = for each plant (columns) his 4 attributes (rows - petal length, petal width etc)
        muc = Dc.mean(1) #row vector of the mean of the 4 attributes (should be a column vector)    
        betweenClassCovarianceMatrix = betweenClassCovarianceMatrix + numpy.dot( mcol(muc - mu), mcol(muc-mu).T ) * Dc.shape[1]       
    betweenClassCovarianceMatrix = betweenClassCovarianceMatrix / float(D.shape[1])
    return betweenClassCovarianceMatrix

"""#input = data matrix and labels vector"""    
def computeSW(D,L):     
    withinClassCovarianceMatrix = 0
    for classe in range(0,2): 
        Dc = D[:, L==classe] #Dc = for each plant (columns) his 4 attributes (rows - petal length, petal width etc)
        muc = Dc.mean(1) #row vector of the mean of the 4 attributes (should be a column vector)    
        for i in range(0, Dc.shape[1]):  #each row having i-th attribute of all samples      
            withinClassCovarianceMatrix = withinClassCovarianceMatrix + numpy.dot( mcol(Dc[:, i] - muc), mcol(Dc[:, i] - muc).T)
    withinClassCovarianceMatrix = withinClassCovarianceMatrix / float(D.shape[1])
    return withinClassCovarianceMatrix

"""#input = Between and Within class covariance matrixes and number of eighenvectors we want to take"""    
def LDA_byGeneralizedEigenvalueProblem(SB, SW, m):
    eigenvalues, eigenvectorsMatrix = scipy.linalg.eigh(SB, SW)
    directionMaximizingVariabilityRatio = eigenvectorsMatrix[:, ::-1][:, 0:m]
    return directionMaximizingVariabilityRatio #eigenvectors rapresenting that direction


def LDA(DTR, LTR, m):
    SW = computeSW(DTR,LTR)
    SB = computeSB(DTR, LTR)
    W = LDA_byGeneralizedEigenvalueProblem(SB, SW, m)
    DP = numpy.dot(W.T, DTR)
    return DP

if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = loadData.load_data()
    plot_hist(DTR, LTR);
    """
    m=4  
    DTRPCA = PCA(DTR, m)
    DTRPCAplusLDA = LDA(DTRPCA, LTR, m)
    """
    