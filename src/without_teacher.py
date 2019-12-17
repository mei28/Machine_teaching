import import_path
from oracle import Oracle
import numpy as np
import pandas as pd
from load_data import read_W
from model import *
from omniscient_teacher import Omniscient
from utils import *


class Without_teacher():
    def __init__(self, w, W, eta=0.01, lambd=0.01, alpha=0.01):
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
        self.alpha = alpha
        self.W_star = np.random.normal(
            loc=self.w_star,
            scale=lambd,
            size=W.shape
        )

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

    def return_text_book_omni(self, X, y, W):
        J, D = W.shape

        om_teacher = Omniscient(self.w_star, alpha=self.alpha)
        X_pool = np.zeros(shape=(J, D))
        y_pool = np.zeros(shape=(J))
        index_set = set()
        for j in range(J):
            w_j = W[j, :]
            X_t, y_t, index = om_teacher.return_textbook(
                X, y, w_j, self.w_star, drop=True)
            X_pool[j, :] = X_t
            y_pool[j] = y_t
            index_set.add(index)
        # om_teacher.drop_textbook(X, y, index_set)
        X_pool = pd.DataFrame(X_pool)
        y_pool = pd.DataFrame(y_pool)
        return X_pool, y_pool, index_set
