import numpy as np

def transCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])

def transPolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
    else:
        innerProduct = vectoriseInnerProduct
    angle = np.arccos(innerProduct/(computeVectorNorm(vector1) * computeVectorNorm(vector2)))
    return angle

def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(axis = 1), 0.5)
