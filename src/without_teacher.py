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
        """
        constructor

        Parameters
        ----------
        w : numpy
            w_star
        W : numpy
            true workers' model parameters
        N : int
            the number of text book pool
        eta : float, optional
            true w parameter, by default 0.01
        lambd : float, optional
            W's parameter, by default 0.01
        alpha : float, optional
            learning rate, by default 0.01
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
        self.N = N
        self.J, self.D = W.shape
        self.mask = np.full((self.J, self.N), True, dtype=bool)

    def learn(self, X, training_epochs=10, loops=10):
        """estimate w_star and W_star

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
        Y = self.make_Y(self.W, X)

        for i in range(training_epochs):
            # print('{:>4}: {}'.format(i, rmse_W(self.W, self.W_star)))
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
        N, D = X.shape
        Y = np.zeros(shape=self.J*N)

        logit = np.dot(W, X.T).flatten()
        p_1_list = 1 / (1 + np.exp(-logit))
        for i, p_1 in enumerate(p_1_list):
            Y[i] = np.random.choice(2, p=[1-p_1, p_1])
        return Y

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
        return predicted y pandas
        """
        N, D = X.shape
        y = np.zeros(N)
        for n in range(N):
            logit = np.dot(X.iloc[n], w)
            p_1 = 1 / (1 + np.exp(-logit))
            y[n] = 1 if p_1 > 0.5 else 0
        y = pd.Series(y)
        return y

    def return_textbook_omni(self, X, y, w_j):
        """
        return textbook

        Parameters
        ----------
        X : pandas
            text book pool
        y : pandas
            True label
        w_j : numpy
            worker's parameter

        Returns
        -------
        return X_t, y_t, index
        """
        N, D = X.shape
        omt = Omniscient(self.w_star, self.W_star, N=N, alpha=self.alpha)
        X_t, y_t = omt.return_textbook(X, y, w_j, self.w_star)
        return X_t, y_t

    def update_wj_by_omni(self, X_t, y_t, w_j):
        """
        to update w_j parameter by text book

        Parameters
        ----------
        X_t : pandas
            a text book
        y_t : pandas
            a true label
        w_j : numpy
            worker's parameter

        Returns
        -------
            return updated w_j
        """
        omt = Omniscient(self.w_star, self.W_star, N=self.N, alpha=self.alpha)
        w_j = omt.update_w_j(X_t, y_t, w_j)
        return w_j

    def show_textbook(self, X, y=None, N=1, option='None'):
        """
        show text book for each worker. and update their parameter

        Parameters
        ----------
        X : pandas 
            textbook pool
        y : pandas 
            true label
        N : int
            the number of textbook to show
        """
        J, D = self.J, self.D

        if y is not None:
            # print('use y')
            y = y
        else:
            if option == 'mix':
                # print('use mix')
                y = self.decision_Y_by_mix(X, self.W_star)
            elif option == 'majority':
                # print('use majority')
                y = self.decision_Y_by_majority(X, self.W_star)
            elif option == 'prob':
                # print('use prob')
                y = self.decision_Y_by_prob(X, self.W_star)
            elif option == 'w_star':
                y = self.predict_y(X, self.w_star)
            else:
                print('default: w_star')
                y = self.predict_y(X, self.w_star)

        for j in range(J):
            w_j_star = self.W_star[j, :]
            for n in range(N):
                mask = self.mask[j]
                X_j, y_j = X[mask], y[mask]

                X_t, y_t = self.return_textbook_omni(
                    X_j, y_j, w_j_star)
                index = np.where(X == X_t)[0][0]
                self.mask[j, index] = False

                w_j = self.W[j, :]
                w_j_new = self.update_wj_by_omni(X_t, y_t, w_j)
                # print('{}: {}'.format(j, index))
                self.W[j, :] = w_j_new

    def decision_Y_by_majority(self, X, W):
        """
        return label from worker decision

        Parameters
        ----------
        X : pandas
            text book pool

        Returns
        -------
        y pandas
            decision by majority
        """
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=(N))
        Y = return_answer_matrix(W, X, J=J)

        for n in range(N):
            Y_n = Y[n, :]
            y[n] = return_mode(Y_n)
        y = pd.Series(y)
        return y

    def decision_Y_by_prob(self, X, W):
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=N)
        Y = return_answer_matrix(W, X, J)

        for i, tmp in enumerate(Y):
            y[i] = np.random.choice(tmp)
        y = pd.Series(y)
        return y

    def decision_Y_by_mix(self, X, W):
        N = X.shape[0]
        J = self.J

        y = np.zeros(shape=N)
        Y = return_answer_matrix(W, X, J)

        threshhold = 0.2
        for i, tmp in enumerate(Y):
            sum_num = tmp.sum()
            if J * threshhold < sum_num and sum_num < (1 - threshhold) * J:
                y[i] = np.random.choice(tmp)
            else:
                y[i] = return_mode(tmp)
        y = pd.Series(y)
        return y
