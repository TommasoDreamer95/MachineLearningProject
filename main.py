# -*- coding: utf-8 -*-

import numpy 

def mcol(v):
    return v.reshape((v.size, 1))

def load_data():
    DTE, LTE = load("./data/Test.txt")
    DTR, LTR = load("./data/Train.txt")
    return (DTE, LTE) , (DTR, LTR)
    
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

if __name__ == '__main__':
    (DTE, LTE) , (DTR, LTR) = load_data()
    
    