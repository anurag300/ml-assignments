import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset1 = 'hw5_blob.csv'
dataset2 = 'hw5_circle.csv'


def load_data():
    df_blob = pd.read_csv(dataset1, names=['d1', 'd2'])
    df_circle = pd.read_csv(dataset2, names=['d1', 'd2'])
    return df_blob, df_circle


def visualize(K, clusters, name):
    colors = "bgrcmykw"
    markers = ['x', '+', 'o', '*', '^']
    index = 0
    for i, points in enumerate(clusters.values()):
        X, Y = zip(*points)
        plt.scatter(list(X), list(Y), c=colors[index], marker=markers[index])
        index += 1
    plt.savefig(str(K)+str('_')+name)
    plt.show()
    plt.clf()


def plot(x, name):
    colors = "bgrcmykw"
    markers = ['x', '+', 'o', '*', '^']
    for i, data in x.items():
        param = str(colors[i])
        plt.plot(data[:100], param)
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.savefig(name)
    plt.show()
    plt.clf()


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
