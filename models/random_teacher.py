import import_path
from .teacher import Teacher
import numpy as np
import pandas as pd
import theano
import theano.tensor as T


class Random(Teacher):
    def __init__(self, min_w, W, N, alpha=0.01):
        super().__init__(min_w, W, N, alpha=alpha)

    def return_textbook(self, X, y, w_t, w_):
        """
        return text book

        Parameters
        ----------
        X : pandas
            text book pool
        y : pandas
            goal

        Returns
        -------
        X_t, y_t, index
        """
        N, D = X.shape
        index = np.random.randint(0, N)
        X_t, y_t = X.iloc[index], y.iloc[index]

        return X_t, y_t
