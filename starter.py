#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from numpy import linalg as LA

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    # Your implementation here
    #LOSS FUNCTION
    #L = LD+ LW
    ld = 0
    for n in range(1,len(x)):
        difference = np.transpose(W) * x[n,1] + b - y[n]
        twonorm = LA.norm(difference) * LA.norm(difference)
        sum = twonorm / (2 * len(x))
        ld = ld + sum

    lw = (reg/2) * (LA.norm(W)*LA.norm(W))
    l = ld + lw

    return l


def gradMSE(W, b, x, y, reg):
    # Your implementation here
    wgradient = (1/len(x)) * (np.transpose(x)) * (x*W + b -y)  + reg * W
    bgradient = (1/len(x)) * (x*W + b -y)
    return wgradient, bgradient


# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here
#
# def gradCE(W, b, x, y, reg):
#     # Your implementation here

#
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    m = len(trainingLabels)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))

    for i in range(iterations)
        prediction = np.dot(trainData, W)

        W = W - (1/m) * learni



# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
#     # Your implementation here
#

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# print trainData
# print np.shape(trainData)
# print trainTarget[0]
weight = np.empty(28)
b = np.empty(28)
for i in range(28):
    weight[i] = 1
    b[i] = 0

mse = MSE(weight, b, trainData, trainTarget, 0)
gradientW, gradientB = gradMSE(weight, b, trainData, trainTarget, 0)
print gradientW
