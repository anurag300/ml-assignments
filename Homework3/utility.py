import math
import matplotlib.pyplot as plt
import numpy as np


def generate_normal_data(mu=0, sigma=0.1, size=None):
    return np.random.normal(mu, sigma, size)


def generate_uniform_data(low=-1, high=1, size=None):
    return np.random.uniform(low, high, size)


def standardize(X, mu, sigma):
    return map(lambda x: (x - mu) / sigma if sigma != 0 else 0, X)


def mean(sample):
    return np.mean(sample)


def sigma(sample):
    mean = np.mean(sample)
    return math.sqrt(sum([(x - mean) ** 2 for x in sample]) / (len(sample) - 1))


def plothist(data, feature_name, sample_size = 10):
    bins = np.linspace(min(data), max(data), 11)
    plt.hist(data, bins=bins)
    plt.title('Sample Size = {s} Mean Squared Error for {f}'.format(s=sample_size, f=feature_name))
    plt.xlabel('MSE')
    plt.savefig(feature_name + str(sample_size))
    plt.show()
    plt.clf()


def plotXY(X, Y):
    lst = zip(X, Y)
    lst.sort(key=lambda x: x[0])
    plt.plot(zip(*lst)[0], zip(*lst)[1])
    plt.title('Lambda vs MSE for 10 fold CV')
    plt.xlabel('lambda')
    plt.ylabel('mean squared error')
    plt.show()


def calculate_sse(Y_true, Y_predict):
    try:
        if len(Y_predict) == len(Y_true):
            return sum(math.pow(y_true - y_predict, 2) for y_true, y_predict in zip(Y_true, Y_predict))
        else:
            raise Exception('Length of vectors does not match')
    except Exception as e:
        print e.message


def feature_transform(x):
    feature_vec = []
    x = zip(*x)
    three_feat = [2,7,8,14,15,26,29]

    for i, col in enumerate(x):
        val_set = sorted(list(set(col)))
        if i+1 in three_feat:
            val_set = [-1,0,1]
            feature_vec.append([[1 if v == val else 0 for v in val_set] for val in col])
        elif len(val_set) == 2 and 0 in val_set and 1 in val_set:
            feature_vec.append([[val] for val in col])
        elif len(val_set) == 2 and -1 in val_set and 1 in val_set:
            feature_vec.append([[1 if val == 1 else 0] for val in col])

    x = map(lambda l: reduce(lambda l1, l2: l1 + l2, l), zip(*feature_vec))
    return x


def get_param(start, end, ratio):
    term = start
    while term <= end:
        yield term
        term *= ratio
