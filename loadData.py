# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:08:26 2021

@author: luca
"""
import numpy
import matplotlib
from matrixUtilities import mcol

def load_data():
    DTE, LTE = load("./data/Test.txt")
    DTR, LTR = load("./data/Train.txt")
    return (DTE, LTE) , (DTR, LTR)

def load_data_TR():
    DTR, LTR = load("./data/Train.txt")
    return (DTR, LTR)

def load_data_TE():
    DTE, LTE = load("./data/Test.txt")
    return (DTE, LTE)

"""input = file name
output : 
    1) Data Matrix (#variates, #samples)
    2) Labels vector (#samples) """
def load(fileName):
    f = open(fileName)
    lista = []
    classes = []  
    for line in f:
        if line != '\n':
            numbers = [float(x) for x in line.split(",")[0:11]]
            characteristics = mcol(numpy.array(numbers))
            lista.append(characteristics)         
            classNumber = line.split(",")[11]
            classes.append(classNumber)         
    matrix = numpy.hstack(lista) 
    classLabels = numpy.array(classes, dtype=numpy.int32) 
    return matrix, classLabels
    

    