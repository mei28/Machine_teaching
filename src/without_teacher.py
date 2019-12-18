import import_path
from oracle import Oracle
import numpy as np
import pandas as pd
from load_data import read_W
from model import *
from omniscient_teacher import Omniscient
from utils import *


class Without_teacher():
    def __init__(self, w, W, N, eta=0.01, lambd=0.01, alpha=0.01):
        self.eta = eta
        self.lambd = lambd
        self.w_star = w
        self.W = W
        self.alpha = alpha
        self.W_star = np.random.normal(
            loc=self.w_star,
            scale=lambd,
            size=W.shape
        )
        J, D = W.shape
        self.mask = np.full((J, N), True, dtype=bool)

    def learn(self, X, training_epochs=10, loops=10):
        """update w_star and W_star

        Parameters
        ----------
        X : pandas
            shape = (N,D)
        training_epochs : int, optional
            the number of all training epochs, by default 10
        loops : int, optional
            the number of each optimization epochs, by default 10

        Returns
        -------
        self.w_star and self.W_star
            [description]
        """
        N, D = X.shape
        J = self.W.shape[0]
        Y = self.make_Y(self.W, X)

        for i in range(training_epochs):
            print('{:>4}: {}'.format(i, rmse_W(self.W, self.W_star)))
            model = W_star_model(
                self.w_star, self.W_star, self.eta, self.lambd)
            self.w_star = model.learn_w_star(X, Y, training_epochs=loops)
            self.W_star = model.learn_W_star(X, Y, training_epochs=loops)
            # self.w_star, self.W_star = model.learn(X, Y, training_epochs=loops)

        return self.w_star, self.W_star

    def make_Y(self, W, X):
        """
        make Y which is worker's answers

        Parameters
        ----------
        W : numpy
            shape = (J,D), worker's model parameter matrix
        X : pandas
            shape = (N,D), feature matrix

        Returns
        -------
        numpy   
            shape = (J,N) →　(J*N)
        """
        J, D = W.shape
        N = X.shape[0]
        Y = np.zeros(shape=J*N)

        logit = np.dot(W, X.T).flatten()
        p_1_list = 1 / (1 + np.exp(-logit))
        for i, p_1 in enumerate(p_1_list):
            Y[i] = np.random.choice(2, p=[1-p_1, p_1])
        return Y

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
            W[j] = W_[N * j:N*(j+1)].mean()

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

    def return_textbook_omni(self, X, y, w_j, drop=True):
        omt = Omniscient(self.w_star, alpha=self.alpha)
        X_t, y_t, index = omt.return_textbook(X, y, w_j, self.w_star, drop)
        return X_t, y_t, index

    def update_wj_by_omni(self, X_t, y_t, w_j):
        omt = Omniscient(self.w_star, alpha=self.alpha)
        w_j = omt.update_w_j(X_t, y_t, w_j)
        return w_j

    def show_textbook(self, X, y, N, drop=True):
        J, D = self.W.shape
        X_pool = []
        y_pool = []
        for j in range(J):
            X_copy = X.copy()
            y_copy = y.copy()
            for n in range(N):
                w_j_star = self.W_star[j, :]
                X_t, y_t, index = self.return_textbook_omni(
                    X_copy, y_copy, w_j_star, drop)

                w_j = self.W[j, :]
                w_j_new = self.update_wj_by_omni(X_t, y_t, w_j)
                # print('{}: {}'.format(j, index))
                # print(w_j_new)
                self.W[j, :] = w_j_new
            X_pool.append(X_copy)
            y_pool.append(y_copy)

        return X_pool, y_pool

    def rebuild_pool(self, X_pool, y_pool):
        new_X = pd.DataFrame(X_pool[0])
        new_y = pd.DataFrame(y_pool[0])
        for x, y in zip(X_pool[1:], y_pool[1:]):
            X_tmp = pd.concat([new_X, x])
            y_tmp = pd.concat([new_y, y], axis=1)

            new_X = X_tmp[X_tmp.duplicated()]
            new_y = y_tmp[y_tmp.duplicated()]

        return new_X, new_y
