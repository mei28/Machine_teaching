from model import *
import import_path
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
# theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"


class Oracle():
    min_w = None

    def __init__(self, eta=0.01, lambd=0.01):
        """[summary]

        Parameters
        ----------
        eta : float, optional
            min_w's parameter, by default 0.01
        lambd : float, optional
            W's parameter, by default 0.01
        """
        self.eta = eta
        self.lambd = lambd

    def estimate_min_w(self, X, y, training_epochs=10):
        w_init = np.random.normal(
            loc=0,
            scale=self.eta,
            size=X.shape[1]
        )
        logistic_model = Logistic_model(w_init, self.eta)
        logistic_model.learn(X, y, training_epochs)
        self.min_w = logistic_model.w
        return self.min_w

    def make_W_init(self, J):
        """make first W

        Parameters
        ----------
        J : int
            the number of worker

        Returns
        -------
        return W numpy
            shape = (J,D)
        """
        W = np.zeros((J, self.min_w.shape[0]))

        for j in range(J):
            W[j, :] = np.random.normal(
                loc=self.min_w,
                scale=self.lambd,
                size=self.min_w.shape[0]
            )
        return W
