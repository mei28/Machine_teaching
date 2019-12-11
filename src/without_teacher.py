import import_path
from oracle import Oracle
import numpy as np
import pandas as pd
from load_data import read_W
from model import *


class Without_teacher():
    def __init__(self, w, W, eta=0.01, lambd=0.01):
        """[summary]

        Parameters
        ----------
        min_w : [type]
            [description]
        eta : float, optional
            [description], by default 0.01
        lambd : float, optional
            [description], by default 0.01
        alpha : float, optional
            [description], by default 0.01
        """
        self.eta = eta
        self.lambd = lambd
        self.w_star = w
        self.W = W

    def learn(self, X, training_epochs=10):
        """
        learn and update w_star and W

        Parameters
        ----------
        X : pandas
            shape = (N,D)
        training_epochs : int, optional
            training epochs, by default 10
        """
        N, D = X.shape
        J = self.W.shape[0]
        print('start: w_star and W')
        for i in range(training_epochs):
            Y = self.makeY(self.W.copy(), X.copy())
            W_ = self.duplicate_W(self.W.copy(), N)
            X_ = self.remake_X(X.copy(), J)
            self.update_w_star(X_, Y, W_, training_epochs)
            self.update_W(X_, Y, training_epochs)
        print('end: w_star and W')

    def update_w_star(self, X, Y, W_, training_epochs=10):
        """
        update w_star

        Parameters
        ----------
        X : numpy
            shape = (J*N,D)
        Y : numpy
            shape = (J*N,)
        W_ : numpy
            shape = (J*N,D)
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------
        return w_star
        """
        model = W_star_model(self.w_star, self.eta, self.lambd, self.W)
        self.w_star = model.learn_w_star(X, Y, W_, training_epochs)
        return self.w_star

    def update_W(self, X, Y, training_epochs=10):
        """
        update_W

        Parameters
        ----------
        X : numpy
            shape = (J*N,D)
        Y : numpy
            shape = (J*N,)
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------
        return self.W
        """
        model = W_star_model(self.w_star, self.eta, self.lambd, self.W)
        self.W = model.learn_W(X, Y, training_epochs=training_epochs)
        return self.W

    def estimate_w_star(self, X, W, training_epochs=10):
        """
        estimate w_star from worker's ans

        Parameters
        ----------
        X : pandas
            shape = (N,D)
        W : numpy
            worker's model parameter
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------

        """
        N, D = X.shape
        J = W.shape[0]

        Y = self.makeY(W.copy(), X.copy(deep=True))
        W_ = self.duplicate_W(W.copy(), N)
        X = self.remake_X(X.copy(deep=True), J)
        model = W_star_model(self.min_w, self.eta, self.lambd, W_)

        self.w_star = model.learn(X, Y, training_epochs)

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

        return Y
        for j in range(J):
            for n in range(N):
                logit = np.dot(W[j, :], X.iloc[n, :])
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
        """
        resize shape (J*N,D) → (J,D)

        Parameters
        ----------
        W_ : numpy
            shape = (J*N,D)
        N : int
            the number of problem

        Returns
        -------
        numpy 
            shape = (J,D)
        """
        J = int(W_.shape[0] / N)
        D = W_.shape[1]
        W = np.zeros((J, D))
        for j in range(J):
            W[j] = W_[N * j].copy()

        return W

    def predict_y(self, X, w):
        """
        return predicted y
        Parameters
        ----------
        X : pandas
        w : numpy
            model parameter

        Returns
        -------
        return predicted y numpy
        """
        N, D = X.shape
        y = np.zeros(N)
        for n in range(N):
            logit = np.dot(X.iloc[n], w)
            p_1 = 1 / (1 + np.exp(-logit))
            y[n] = np.random.choice(2, p=[1 - p_1, p_1])

        return y
