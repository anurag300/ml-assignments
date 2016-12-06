import numpy as np
from numpy.linalg import inv,pinv


class LinearRegression(object):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def _create_matrices(self):
        self.X_mat = np.matrix(self.X)
        self.Y_mat = np.matrix(self.Y).getT()
        self.X_mat = np.insert(self.X_mat, 0, 1, axis=1)

    def _calc_params(self):
        self.W = pinv(self.X_mat.getT()*self.X_mat)*self.X_mat.getT()*self.Y_mat

    def getparam(self):
        self._create_matrices()
        self._calc_params()
        return self.W

    def predict(self,X):
        X = np.matrix(X)
        X = np.insert(X,0,1,axis  = 1)
        return X*self.W

class RidgeRegression(LinearRegression):
    def __init__(self, X, Y,L):
        super(RidgeRegression, self).__init__(X,Y)
        self.X = X
        self.Y = Y
        self.L = L
        self.dim = len(self.X[0])

    def _calc_params(self):
        self.W = inv(self.X_mat.getT()*self.X_mat + self.L*np.identity(self.dim+1))*self.X_mat.getT()*self.Y_mat

    def getparam(self):
        self._create_matrices()
        self._calc_params()
        return self.W


