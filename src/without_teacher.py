import import_path
from oracle import Oracle
import numpy as np
import pandas as pd
from load_data import read_W
from model import *


class Without_teacher():
    def __init__(self, min_w, eta=0.01, lambd=0.01, alpha=0.01):
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
        self.w_star = None
        self.W = None
        self.alpha = p = alpha
        self.min_w = min_w.copy()

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
        # self.W = self.resize_W(W_, N)

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

    def update_W(self, W, X):
        """
        to update W

        Parameters
        ----------
        W : numpy
            worker's model parameter
        X : pandas
            text book pool

        Returns
        -------
        W   numpy
            updated W
        """
        N, D = X.shape
        J = W.shape[0]
        W_new = np.zeros(shape=(J, D))

        for j in range(J):
            w_j_old = W[j, :].copy()
            y = self.predict_y(X, w_j_old)
            H = self.hessian_function(X, y, w_j_old)
            H_inv = np.linalg.inv(H)
            g = self.grad_wj(X, y, w_j_old)
            W_new[j, :] = w_j_old - (self.alpha * H_inv * g).sum()

        return W_new

    def grad_wj(self, X, y, w_j):
        """
        calc grad wj        
        Parameters
        ----------
        X : pandas
        y : pandas
        w_j : numpy
            worker's model parameter

        Returns
        -------
        return grad w_j numpy
        """
        g = self.grad_wj_function()
        g_ = np.array(g(X, y, w_j, self.w_star)).flatten()
        return np.array(g_).flatten()

    def hessian_function(self, X, y, w_j):
        """
        make hessian matrix

        Parameters
        ----------
        X : pandas
            textbook pool
        y : numpy
            worker's answer
        w_j : numpy
            worker's model parameter

        Returns
        -------
        return hessian matrix
        """
        K, L = w_j.shape[0], w_j.shape[0]
        I = X.shape[0]
        hes = 0
        hessian = np.zeros(shape=(L, K))
        for l in range(L):
            for k in range(K):
                for i in range(I):
                    logit = 1 / (1 + np.exp(-np.dot(X.iloc[i], w_j)))
                    hes += (1-logit)*(logit)*X.iat[i, k]*X.iat[i, l]
                hessian[k, l] = hes
                hes = 0
        hessian = hessian + self.lambd*np.eye(K)
        return hessian

    def grad_wj_function(self):
        """
        return grad wj function

        Returns
        -------
        inputs = [X,y,w_j,w_star]
        outputs = [calc]
        """
        X = T.matrix(name='X')
        y = T.vector(name='y')
        w_j = T.vector(name='w_j')
        w_star = T.vector(name='w_star')

        logit = (y - T.nnet.sigmoid(T.dot(X, w_j)))
        calc = T.dot(logit, X) - self.lambd*(w_j - w_star)

        function = theano.function(
            inputs=[X, y, w_j, w_star],
            outputs=[calc],

        )
        return function

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
