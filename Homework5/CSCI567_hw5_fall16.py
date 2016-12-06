from utils import *
from kmeans import *
from gmm_em import *


def main():
    df_blob, df_circle = load_data()
    for k in [2, 3, 5]:
        print 'Running k-means for blob dataset with k = {k}'.format(k=k)
        mu, clusters = k_means(list(df_blob.d1), list(df_blob.d2), k)
        visualize(k, clusters, 'k_means_blob')

    print '\n'
    for k in [2, 3, 5]:
        print 'Running k-means for circle dataset with k = {k}'.format(k=k)
        mu, clusters = k_means(list(df_circle.d1), list(df_circle.d2), k)
        visualize(k, clusters, 'k_means_circle')

    print '\n'
    for k in [2]:
        print 'Running kernelized k-means for circle dataset with k = {k}'.format(k=k)
        clusters = kernel_k_means(list(df_circle.d1), list(df_circle.d2), k, 10)
        visualize(k, clusters, 'kernel_k_means_circle')

    print '\n'
    for k in [3]:
        print 'Running EM algorithm for Gaussian Mixture Model on blob dataset'
        likelihoodVec, bsigma, bu, clusters, llmax = gmm_em(list(df_blob.d1), list(df_blob.d2), 3)
        visualize(k, clusters, 'GMM_circle')
        plot(likelihoodVec, 'EM_plot')
        print '\n'
        print 'Maximum Likelihood for best run : {l}'.format(l=llmax)
        print 'GMM Parameters :'
        for i in range(k):
            print 'Covariance Matrix for cluster {c}'.format(c=i + 1)
            print bsigma[:, :, i]
            print 'Mean for cluster {c}'.format(c=i + 1)
            print bu[i, :]
            print '\n'


if __name__ == '__main__':
    main()
