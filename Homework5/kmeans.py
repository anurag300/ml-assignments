import numpy as np


def find_clusters(points, mu):
    clusters = {}
    for point in points:
        bestmu = \
            min([(i[0], np.linalg.norm(np.array(point) - np.array(i[1]))) for i in enumerate(mu)], key=lambda t: t[1])[
                0]
        if bestmu not in clusters:
            clusters[bestmu] = []
        clusters[bestmu].append(point)
    return clusters


def recalculate_mu(clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for key in keys:
        newmu.append(np.mean(clusters[key], axis=0))
    return newmu


def has_converged(mu1, mu2):
    return set([tuple(i) for i in mu1]) == set([tuple(i) for i in mu2])


def initialize_mu(points, K):
    idx_old = np.random.choice(range(len(points)), K)
    idx = np.random.choice(range(len(points)), K)
    old_mu = []
    for k, v in enumerate(points):
        if k in idx_old:
            old_mu.append(v)

    mu = []
    for k, v in enumerate(points):
        if k in idx:
            mu.append(v)
    return old_mu, mu


def k_means(X, Y, K):
    # Initialize to K random centers
    points = zip(X, Y)
    old_mu, mu = initialize_mu(points, K)
    while not has_converged(old_mu, mu):
        old_mu = mu
        clusters = find_clusters(points, mu)
        mu = recalculate_mu(clusters)
    return (mu, clusters)


def changed(clusters1, clusters2):
    pass


def kernel_k_means(X, Y, K, C):
    points = zip(X, Y)
    N = len(points)
    vec1 = np.add(np.square(np.array(X)), np.square(np.array(Y)))
    vec2 = np.add(np.square(np.array(X)), np.square(np.array(Y)))

    # Calculating Kernal Matrix
    K_mat = np.asmatrix(vec1).transpose() * np.asmatrix(vec2)

    # Randomly assigning initial clusters
    randNos = np.random.choice(range(N), K)
    rnk = np.zeros(N)
    for i in range(K):
        rnk[randNos[i]] = i + 1
    updatedRnk = rnk
    changed = True
    count = 0
    # Calculating distance
    print 'Iterations = {c}'.format(c=C)
    while changed and count < C:
        print 'Running iteration {i}'.format(i=count + 1)
        rnk = updatedRnk.copy()
        changed = False
        term3_map = {}
        for i in range(N):
            dist = np.zeros(K)
            for j in range(1, K + 1):
                clusterJidx = np.where(rnk == j)[0]
                sumTerm1 = K_mat[i, i]
                sumTerm2 = 0
                sumTerm3 = 0
                for jidx in range(len(clusterJidx)):
                    sumTerm2 += K_mat[i, clusterJidx[jidx]]

                if j in term3_map:
                    sumTerm3 = term3_map[j]
                else:
                    for jidx in range(len(clusterJidx)):
                        for lidx in range(len(clusterJidx)):
                            sumTerm3 += K_mat[clusterJidx[jidx], clusterJidx[lidx]]
                    term3_map[j] = sumTerm3

                dist[j - 1] = sumTerm1 - 2 * ((sumTerm2) / len(clusterJidx)) + sumTerm3 / (len(clusterJidx)) ** 2
                if dist[j - 1] < 0:
                    raise Exception("Distance less than zero")
            minIdx = np.where(dist == min(dist))[0]
            if updatedRnk[i] != minIdx[0] + 1:
                updatedRnk[i] = minIdx[0] + 1
                changed = True
        count += 1
    print 'Converged in {i} iterations'.format(i=count)
    clusters = {}
    for idx, cluster in enumerate(updatedRnk):
        if cluster in clusters:
            clusters[cluster].append(points[idx])
        else:
            clusters[cluster] = []
            clusters[cluster].append(points[idx])
    return clusters
