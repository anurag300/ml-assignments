import numpy as np
from utils import *
from scipy.stats import multivariate_normal


def gmm_em(X, Y, K):
    data = np.array([X, Y]).transpose()
    N, D = data.shape
    likelihoodVec = {}

    # initializing parameters
    bestmu = []
    bestsigma = []
    bestindex = []

    for m in range(5):
        print 'Running iteration {i}'.format(i=m)
        # Initialization
        # Initializing means
        mu = np.zeros((K, D))
        randNos = np.random.choice(range(N), K)
        mean_points = [data[r] for r in randNos]
        for i in range(K):
            for j in range(D):
                mu[i, j] = mean_points[i][j]

        # Initializing covariance matrix
        sigma2 = np.zeros((D, D, K))
        for k in range(K):
            sigma2[:, :, k] = np.identity(D)

        # Initializing weights
        pc = np.ones((1, K)) * (1.0 / K)

        # Responsobility matrix
        wnk = np.zeros((N, K))

        iter = 0
        mu_old = np.zeros((N, 1))
        mu_new = []
        ll = []
        llmax = -99999999999

        while not np.array_equal(mu_old, mu_new) and iter < 500:
            iter += 1
            mu_old = mu_new

            # E-Step
            for p in range(K):
                wnk[:, p] = pc[0, p] * multivariate_normal.pdf(data, mu[p, :], sigma2[:, :, p], allow_singular=True)

            temp = np.sum(wnk, axis=1)
            temp = temp.reshape(len(temp), 1)
            temp = np.repeat(temp, K, axis=1)
            wnk = np.divide(wnk, temp)

            # M-Step
            mu = np.divide(np.asarray(np.asmatrix(wnk).transpose() * np.asmatrix(data)),
                           np.repeat(np.sum(wnk, axis=0).reshape(1, 3).transpose(), D, axis=1))
            for i in range(K):
                numerator = np.zeros(D)
                for j in range(N):
                    numerator = numerator + np.asarray(
                        wnk[j, i] * (np.asmatrix(data[j, :]) - np.asmatrix(mu[i, :])).transpose() * (
                            np.asmatrix(data[j, :]) - np.asmatrix(mu[i, :])))

                sigma2[:, :, i] = np.divide(numerator, np.sum(wnk[:, i]))

            pc = (np.sum(wnk, axis=0) / N).reshape(1, K)

            # Log likelihood
            sdist, sindex = np.amax(wnk, axis=1), np.argmax(wnk, axis=1)
            mu_new = mu
            ll.append(np.sum(np.log(temp[:, 0])))

            if np.array_equal(mu_old, mu_new):
                likelihoodVec[m] = ll
                if ll[-1] > llmax:
                    llmax = ll[-1]
                    bestmu = mu
                    bestsigma = sigma2
                    bestindex = sindex

    clusters = form_clusters(X, Y, bestindex)
    return likelihoodVec, bestsigma, bestmu, clusters, llmax


def form_clusters(X, Y, index):
    clusters = {}
    data_points = zip(X, Y)
    for i, clus in enumerate(index):
        if clus in clusters:
            clusters[clus].append(data_points[i])
        else:
            clusters[clus] = []
            clusters[clus].append(data_points[i])
    return clusters
