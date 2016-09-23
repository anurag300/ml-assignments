import numpy as np
import math
import operator

def getDistance(vec1, vec2, norm):
    if norm == 'l2':
        return math.sqrt(sum((x-y)**2 for x,y in zip(vec1,vec2)))
    if norm == 'l1':
        return sum(abs(x - y) for x, y in zip(vec1, vec2))


def nearestNeighbours(trainingSet, testInstance, k,norm):
    distances = []
    for train_inst in trainingSet:
        dist = getDistance(train_inst[:-1], testInstance, norm)
        distances.append((train_inst[-1:],train_inst[:-1],dist))
    distances.sort(key = operator.itemgetter(2))

    neighbour = []
    for x in xrange(k):
        neighbour.append((distances[x][0][0],distances[x][1]))
    return neighbour

def predictClass(neighbours):
    classdict = {}
    for cat,vec in neighbours:
        if cat in classdict:
            classdict[cat]+=1
        else:
            classdict[cat] = 1
    return max(classdict, key = classdict.get)

def knn(train_data, test_data, k, norm):
    Y_predict = []
    for test_inst in test_data:
        Y_predict.append(predictClass(nearestNeighbours(train_data,test_inst,k, norm)))
    return Y_predict

def standardize(data,params):
    std_data = []
    for data_point, param in zip(data,params):
        std_data.append(float(data_point - param[0])/param[1])
    return std_data


def mean(data):
    return np.mean(data)

def standardDeviation(data):
    X_mu = np.mean(data)
    return math.sqrt(sum([(x - X_mu)**2 for x in data])/(len(data)-1))

def getDataStats(train_data):
    data_stats = []
    for feature in zip(*train_data):
        data_stats.append((mean(feature), standardDeviation(feature)))
    return data_stats




