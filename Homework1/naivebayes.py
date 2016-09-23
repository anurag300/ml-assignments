import numpy as np
import math
import matplotlib.pyplot as plt
import collections


def getMean(data):
    return np.mean(data)


def getStandardDeviation(data):
    X_mu = np.mean(data)
    return math.sqrt(sum([(x - X_mu)**2 for x in data])/(len(data)-1))


def getSummary(dataset):
    return getMean(dataset),getStandardDeviation(dataset)


def plot(data):
    plt.plot(data)
    plt.show()


def calculateClassProb(data):
    counter = collections.Counter(data)
    sum = reduce(lambda x,y:x+y , counter.values())
    for key,value in counter.items():
        counter[key] = float(value)/sum
    return counter


def calculateGaussianProb(x, mu, sigma):
    if sigma == 0 and x == 0:
        return 1
    elif sigma == 0 and x:
        return 0
    exponent = math.exp(-(math.pow(x - mu,2)/(2*math.pow(sigma,2))))
    return (1/(math.sqrt(2*math.pi)*sigma))*exponent


def calculateNaiveBayesProb(inputVec, summary, key, classProbDict):
    prob = 0
    for input, params in zip(inputVec, summary):
        returnVal = calculateGaussianProb(input,*params)
        prob+= math.log(returnVal) if returnVal!=0 else -99999

    if key in classProbDict:
        prob+= math.log(classProbDict[key])
    else:
        prob+=-99999
    return prob


