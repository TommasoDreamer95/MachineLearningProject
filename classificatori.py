# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:46:22 2021

@author: tommy
"""
import scipy.optimize
import numpy , scipy.special, math
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


"""From here on: Logistic Regression"""

"""
called many times during logistic regression, in order to get the minimizer parameters
"""
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

"""from here on, linear SVM"""
def buildExtendedMatrix (D, k):
    Dhat = numpy.zeros( (D.shape[0]+1, D.shape[1]), dtype="float64")
    for i in range(0,D.shape[0]):
        Dhat[i] = D[i]
    Dhat[D.shape[0]] = numpy.zeros( (D.shape[1]), dtype="float64") + k
    return Dhat
def computeHhatLinear(Dhat, LTR):
    Ghat = numpy.dot(Dhat.T, Dhat)
    z = numpy.equal.outer(LTR, LTR)
    z = z * 2 -1
    Hhat = z * Ghat 
    return Hhat

def findDualSolutionByMinimization(computeNegativeObj, alpha, Hhat, C, DTR):
    boundaries = numpy.zeros((DTR.shape[1], 2), dtype="float64")
    boundaries[:, 1] = boundaries[:, 1] + C
    (optimalAlpha, dualSolution, _) = scipy.optimize.fmin_l_bfgs_b(computeNegativeObj, alpha, args=(Hhat,), approx_grad=False
                                                        , bounds = boundaries, factr = 1.0, maxfun = 15000, maxiter=15000)
    return (optimalAlpha, dualSolution)

def computeNegativeObj(alpha, Hhat):
    Ones = numpy.zeros((alpha.shape[0]), dtype="float64" ) +1
    L = numpy.dot(numpy.dot(alpha.T, Hhat), alpha) / 2 - numpy.dot(alpha.T,Ones)
    gradientL = numpy.dot(Hhat, alpha) - Ones
    return L, gradientL

def computeWfromAlpha(alpha, LTR, Dhat):
    w = 0
    for i in range(0, LTR.shape[0]):
        zi = LTR[i]
        if zi == 0:
            zi = -1
        w = w + alpha[i] * zi * Dhat[:, i] 
    return w

def computeParametersForLinearSVM(DTR, LTR, k, C):
    Dhat = buildExtendedMatrix(DTR, k)
    Hhat = computeHhatLinear(Dhat, LTR)
    
    alpha = numpy.zeros((DTR.shape[1]), dtype="float64")
        
    (optimalAlpha, dualSolution) = findDualSolutionByMinimization(computeNegativeObj, alpha, Hhat, C, DTR)    
  
    
    wHatStar = computeWfromAlpha(optimalAlpha, LTR, Dhat)
    
    return wHatStar

"""Polinomial SVM (versone 2)"""
def polynomialKernel(D1, D2, c, d, k):
    firstTerm = numpy.dot(D1.T, D2)
    P = (firstTerm + c) ** d
    xi = k ** 2
    return P + xi


def computeHhatNonLinearPolynomial(DTR, LTR, c, d, k):
    z = numpy.equal.outer(LTR, LTR)
    z = z * 2 -1
    Hhat = z * polynomialKernel(DTR, DTR, c, d, k)
    return Hhat


def computeParametersForPolinomialSVM(DTR, LTR, k, C, d, c):
    Hhat = computeHhatNonLinearPolynomial(DTR, LTR, c, d, k)
    
    alpha = numpy.zeros((DTR.shape[1]), dtype="float64")
    
    (optimalAlpha, dualSolution) = findDualSolutionByMinimization(computeNegativeObj, alpha, Hhat, C, DTR)    
    return optimalAlpha

"""from here on GMM"""

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

def logpdf_GMM(X, gmm):
    N = X.shape[1] #samples
    D = X.shape[0] #variates
    M = len(gmm) #models
    S = numpy.zeros( (M, N), dtype="float64")
    for g in range (0, len(gmm)):
        mu = gmm[g][1]
        C = gmm[g][2]
        logdensities = logpdf_GAU_ND(X, mu, C)
        S[g, :] = logdensities
        S[g, :] += numpy.log(gmm[g][0])
    logdens = scipy.special.logsumexp(S, axis=0)
    return S, logdens


def constrain_eigenvalues(covNew):
    psi = 0.01
    U, s, _ = numpy.linalg.svd(covNew)
    s[s<psi] = psi
    covNew = numpy.dot(U, mcol(s)*U.T)
    return covNew


def E_part(previousGMM, X):
    #E
    #print(previousGMM[0][2])
    (S, logdens) = logpdf_GMM(X, previousGMM)
    
    ll = logdens.sum()
    N = X.shape[1] #samples
    avg_ll = ll / N
    #print(avg_ll)
    loggamma = S - logdens
    gamma = numpy.exp(loggamma)
    
    return gamma, avg_ll
 
def M_part(X, previousGMM, DeltaL, finalImpl, gamma):
    #M
    nextGMM = []
    gmmTuple = []
    N = X.shape[1] #samples
    
    Z = numpy.zeros( (len(previousGMM)), dtype="float64" )
    w = numpy.zeros( (len(previousGMM)), dtype="float64" )
    mu = numpy.zeros( (len(previousGMM), X.shape[0]), dtype="float64" )
    C = numpy.zeros( (len(previousGMM), X.shape[0], X.shape[0]), dtype="float64" )
    
    for g in range (0, len(previousGMM)):
        Z[g] = gamma[g].sum()
        Fg = numpy.zeros((X.shape[0]), dtype="float64")
        Sg = numpy.zeros((X.shape[0], X.shape[0]), dtype="float64")
        
        for i in range (0, X.shape[1]):
            Fg += (gamma[g][i] * X[:, i])
            Sg += (gamma[g][i] * numpy.dot(mcol(X[:, i]) , mcol(X[:, i]).T))
        
        mu[g] = Fg / Z[g]
        C[g] = Sg/Z[g] -numpy.dot(mcol(mu[g]), mcol(mu[g]).T)#numpy.dot(mu[g], mu[g].T)
        if finalImpl == "diagonal":
            C[g] = C[g] * numpy.eye(C[g].shape[0]) #diagonal update  
        C[g] = constrain_eigenvalues(C[g])
    
    if finalImpl == "tied":
        Ccopy = C
    for g in range (0, len(previousGMM)):
        w[g] = Z[g]/ (Z.sum() ) 
        if finalImpl == "tied":        
            newCg = numpy.zeros( (X.shape[0], X.shape[0]), dtype="float64" )
            for g1 in range (0, len(previousGMM)):
                newCg += Ccopy[g] * Z[g]
            C[g] = newCg / N
        gmmTuple = (w[g], mcol(mu[g]), C[g])
        #print(gmmTuple[0])
        nextGMM.append(gmmTuple)
        
    return nextGMM


def EM_wrapper(X, previousGMM, DeltaL, finalImpl):
    
    prev_avg_ll = "primo valore"
    
    while(True):
        (gamma, avg_ll) = E_part(previousGMM, X)
        #if prev_avg_ll != "primo valore":
        #    print (abs(avg_ll - prev_avg_ll))
        if prev_avg_ll != "primo valore" and abs(avg_ll - prev_avg_ll) < DeltaL :
            return previousGMM   
        
        prev_avg_ll = avg_ll
        
        previousGMM = M_part(X, previousGMM, DeltaL, finalImpl, gamma)        

def LBG_algorithm(previousGMM):
    
    newGMM = []
    gmmTuple = []
    alpha = 0.1
    for g in range (0, len(previousGMM)):
        wg = previousGMM[g][0]
        mug = previousGMM[g][1]
        Sigma_g = previousGMM[g][2]
        Sigma_g = constrain_eigenvalues(Sigma_g)
        
        U, s, Vh = numpy.linalg.svd(Sigma_g)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        
        newWg = wg / 2
        #1st
        newMug =mcol( mug) + d
        gmmTuple = (newWg, newMug, Sigma_g)
        newGMM.append(gmmTuple)
        
        #2nd
        newMug = mcol(mug) - d
        gmmTuple = (newWg, newMug, Sigma_g)
        newGMM.append(gmmTuple)
    
    return newGMM


def LBG(DTR, DeltaL, mu, C, finalImpl, finalGmms):
    
    
    lbgIterations = math.log(finalGmms, 2)
    GMM = [(1.0, mu, C)]
    for i in range (0, int(lbgIterations)):
        EM = EM_wrapper( DTR, GMM, DeltaL, finalImpl)
        GMM = LBG_algorithm(EM)
        
    EM_final = EM_wrapper(DTR, GMM, DeltaL, finalImpl)
    return EM_final


def computeGMMs(DTR, LTR, DeltaL, finalGmms, finalImpl):
    
    mu = numpy.zeros( (3, DTR.shape[0]))
    SigmaC = numpy.zeros( (3, DTR.shape[0], DTR.shape[0] ) )
    GMMs = []
    for c in range(0,2):
        Dc = DTR[:, LTR==c] 
        mu[c] = Dc.mean(1)  
        DC = Dc - mcol(mu[c])  #matrice centrata, in cui ad ogni colonna sottraggo la media
        SigmaC[c] = numpy.dot(DC, DC.T) / float(DC.shape[1]) # C = 1/N * Dc * Dc Trasposta
        
        GMM = LBG(Dc, DeltaL, mcol(mu[c]), SigmaC[c], finalImpl, finalGmms)
        GMMs.append(GMM)
        
        
    return GMMs

"""
computation of dual j kernel for SVM
"""
def dual_J_kernel(alfa, DTR, LTR, K, kernel):
    list_k = []
    for i in range(DTR.shape[1]):
        list_k.append(K)
        #list_k.append(1)
    Dc = numpy.vstack([DTR, list_k])
    Gc = numpy.dot(Dc.T, Dc)
    # compute matrix Z of products zi * zj
    list_zi = []
    for i in range(len(LTR)):
        if LTR[i] == 1:
            zi = 1
            #zi = -1
        else:
            zi = -1
            #zi = 1
        list_zi.append(zi)
    row_z = numpy.vstack(list_zi)
    Z = numpy.dot(row_z, row_z.T)
    # compute matrix Hc
    Hc = Z * kernel
    vstack_ones = mcol(numpy.ones(len(LTR)))
    Lt =  1/2 * numpy.dot(numpy.dot(alfa.T, Hc), alfa) - numpy.dot(alfa.T, vstack_ones)[0]
    grad_alfa = numpy.dot(Hc, alfa) -1
    return Lt, grad_alfa
    #return Lt

""" 
computation of polynomial kernel for SVM
"""
def compute_polynomial_kernel(x1, x2, c, d, psi):
    poly_kern = (numpy.dot(x1.T, x2) + c)**d + psi
    return poly_kern

def compute_RBF_kernel(m1, m2, gamma, psi):
    list_vertical_rows = []
    for i in range(m1.shape[1]):
        xi = m1[:, i]
        list_row_kernel = []
        for j in range(m2.shape[1]):
            xj = m2[:, j]
            kij = numpy.exp(-1 * gamma * numpy.linalg.norm(xi - xj)**2) + psi
            list_row_kernel.append(kij)
        row_kernel = numpy.hstack(list_row_kernel)
        list_vertical_rows.append(row_kernel)
    return numpy.vstack(list_vertical_rows)

def compute_matrix_Z_kernel_SVM(DTR, LTR):
    Z = []
    for i in range(DTR.shape[1]):
        if LTR[i] == 1:
            zi = 1
        else:
            zi = -1
        Z.append(zi)
    Z = numpy.vstack(Z)
    return Z

def compute_bounds_list_kernel_SVM(DTR, C):
    bounds_list = []
    #for i in range(DTR.shape[0] + 1 ):
    for i in range(DTR.shape[1]):
        bounds_list.append((0, C))
    return bounds_list
    

def computeParameterForKernelPolynomialSVM(DTR, LTR, K, C, d, c):
    psi = K ** 2
    bounds_list = compute_bounds_list_kernel_SVM(DTR, C)
    # --- compute dual J
    x0_dual = mcol(numpy.zeros(DTR.shape[1]))
    kc_polynomial = compute_polynomial_kernel(DTR, DTR, c, d, psi)
    alfa_polynomial, _, _ = scipy.optimize.fmin_l_bfgs_b(dual_J_kernel, x0_dual, args=(DTR, LTR, K, kc_polynomial), factr=1.0, bounds=bounds_list)
    return alfa_polynomial

def computeParameterForKernelRBFSVM(DTR, LTR, K, C, gamma):
    psi = K ** 2
    bounds_list = compute_bounds_list_kernel_SVM(DTR, C)
    x0_dual = mcol(numpy.zeros(DTR.shape[1]))
    kernel_RBF = compute_RBF_kernel(DTR, DTR, gamma, psi)
    alfa_RBF, _, _ = scipy.optimize.fmin_l_bfgs_b(dual_J_kernel, x0_dual, args=(DTR, LTR, K, kernel_RBF), factr=1.0, bounds=bounds_list)
    return alfa_RBF

def compute_polynomial_kernel(x1, x2, c, d, K_RBF):
    psi = K_RBF ** 2
    poly_kern = (numpy.dot(x1.T, x2) + c)**d + psi
    
    return poly_kern

def compute_RBF_kernel(m1, m2, gamma, K):
    psi = K ** 2
    list_vertical_rows = []
    for i in range(m1.shape[1]):
        xi = m1[:, i]
        list_row_kernel = []
        for j in range(m2.shape[1]):
            xj = m2[:, j]
            kij = numpy.exp(-1 * gamma * numpy.linalg.norm(xi - xj)**2) + psi
            list_row_kernel.append(kij)
        row_kernel = numpy.hstack(list_row_kernel)
        list_vertical_rows.append(row_kernel)
    return numpy.vstack(list_vertical_rows)