import import_path
from oracle import Oracle
import numpy as np
import pandas as pd
from load_data import read_W
from model import *


class Without_teacher():
    def __init__(self, min_w, eta=0.01, lambd=0.01):
        self.eta = eta
        self.lambd = lambd
        self.w_star = None
        self.W = None
        self.min_w = min_w.copy()

    def estimate_w_star(self, X, W, training_epochs=10):
        N, D = X.shape
        J = W.shape[0]

        Y = self.makeY(W.copy(), X.copy(deep=True))
        W_ = self.duplicate_W(W.copy(), N)
        X = self.remake_X(X.copy(deep=True), J)
        model = W_star_model(self.min_w, self.eta, self.lambd, W_)

        self.w_star, W_ = model.learn(X, Y, training_epochs)
        # self.W = self.resize_W(W_, N)
        return W_

    def duplicate_W(self, W, N):
        """
        duplicate W (J,D) -> (J*N,D)

        Parameters
        ----------
        W : numpy
            shape = (J,D)
        N : int
            The number of questions

        Returns
        -------
        W_
            shape = (J*N,D)
        """
        J, D = W.shape
        W_ = np.zeros((N * J, D))
        for j in range(J):
            tmp = W[j].copy()
            for n in range(N):
                W_[N*j+n] = tmp
        return W_

    def makeY(self, W, X):
        """ make worker’s answer

        Parameters
        ----------
        W : numpy
            shape = (J,D)
        X : pandas
            shape = (N,D)

        Returns
        -------
        numpy
            shape = (N*J,)
        """
        J, D = W.shape
        N = X.shape[0]
        Y = np.zeros((N * J))
        for j in range(J):
            for n in range(N):
                logit = np.dot(W[j, :], X.iloc[n])
                p_1 = 1/(1+np.exp(-logit))
                Y[N * j + n] = np.random.choice(2, p=[1 - p_1, p_1])
        return Y

    def remake_X(self, X, J):
        """remake X (N,D) -> (N*J,D)

        Parameters
        ----------
        X : pandas
            shape = (N,D), feature matrix
        J : int
            The number of workers

        Returns
        -------
        X_
            shape = (N*J,D)
        """
        N, D = X.shape
        X_ = np.zeros((N * J, D))
        for i in range(N * J):
            X_[i, :] = X.iloc[i % N]
        return X_

    def resize_W(self, W_, N):
        J = int(W_.shape[0] / N)
        D = W_.shape[1]
        W = np.zeros((J, D))
        for j in range(J):
            W[j] = W_[N * j].copy()

        return W
