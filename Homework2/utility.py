import numpy as np
import math
import matplotlib.pyplot as plt

def pearsonCorr(X,Y):
    num = mean(np.array(X)*np.array(Y)) - mean(X)*mean(Y)
    demon  = sigma(X)*sigma(Y)
    return num/demon

def standardize(X,mu,sigma):
    return map(lambda x : (x-mu)/sigma, X)

def mean(sample):
    return np.mean(sample)

def sigma(sample):
    mean = np.mean(sample)
    return math.sqrt(sum([(x - mean)**2 for x in sample])/(len(sample)-1))

def splitdata(data):
    indices = [i for i in xrange(0,len(data))]
    data_test = map(lambda x : data[x], filter(lambda y : y%7==0,indices))
    data_train = map(lambda x : data[x], filter(lambda y : y%7!=0, indices))
    return data_train,data_test

def plothist(data,feature_name):
    bins = np.linspace(min(data),max(data),11)
    plt.hist(data,bins = bins)
    plt.title(feature_name)
    plt.xlabel('Value')
    plt.ylabel('frequency')
    plt.savefig(feature_name)
    plt.show()
    plt.clf()


def plotXY(X,Y):
    lst = zip(X,Y)
    lst.sort(key = lambda x : x[0])
    plt.plot(zip(*lst)[0],zip(*lst)[1])
    plt.title('Lambda vs MSE for 10 fold CV')
    plt.xlabel('lambda')
    plt.ylabel('mean squared error')
    plt.show()

def calculateMSL(Y_true,Y_predict):

    try:
        if len(Y_predict)== len(Y_true):
            return sum(math.pow(y_true-y_predict,2) for y_true,y_predict in zip(Y_true,Y_predict))/(len(Y_true))
        else:
            raise Exception('Length of vectors does not match')
    except Exception as e:
        print e.message

def k_fold(size,**options):
    n_arrays = range(size)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    n_folds= options.pop('n_folds',None)
    shuffle = options.pop('shuffle',None)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    if n_folds is None:
        n_folds = 10
    if shuffle:
        np.random.shuffle(n_arrays)

    shuffled_list = []

    set_size = size/n_folds
    remainder = size%n_folds
    list_ranges = []

    start = 0
    for i in range(n_folds):
        if remainder:
            list_ranges.append((start,start+set_size))
            start += + set_size+ 1
            remainder -= 1
        else:
            list_ranges.append((start,start+set_size-1))
            start += set_size

    for i,j in list_ranges:
        test_indices = n_arrays[i:j+1]
        train_indces = n_arrays[0:i] + n_arrays[j+1:]
        shuffled_list.append((sorted(train_indces),sorted(test_indices)))

    return shuffled_list

