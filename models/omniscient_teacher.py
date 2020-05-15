import import_path
import numpy as np
import pandas as pd
from .teacher import Teacher
import theano
import theano.tensor as T


class Omniscient(Teacher):
    def __init__(self, min_w, W, N, alpha=0.01):
        super().__init__(min_w, W, N, alpha=alpha)

    def make_loss_function(self):
        """
        return omniscient teacher function

        Parameters
        ----------
        alpha : float, optional
            w_t parameter, by default 0.01

        Returns
        -------
        return function
            inputs = [grad_loss,w_t,w_]
            outputs = [loss]
        """
        grad_loss = T.matrix(name='grad_loss')
        w_ = T.vector(name='w_')
        w_t = T.vector(name='w_t')

        first = (self.alpha ** 2) * (grad_loss ** 2).sum()
        second = -2 * self.alpha * (T.dot(grad_loss, w_t - w_))
        loss = first + second
        function = theano.function(
            inputs=[grad_loss, w_t, w_],
            outputs=[loss],
            on_unused_input='ignore'
        )
        return function

    def return_textbook(self, X, y, w_t, w_):
        """
        return text book

        Parameters
        ----------
        X : pandas
            text book pool
        y : pandas
            goal
        w_t : numpy
            worker's parameter when t times
        w_ : numpy
            true model parameter

        Returns
        -------
        X_t,y_t,index
        """
        grad_loss = self.make_grad_loss_matrix(X, y, w_t)
        choicer = self.make_loss_function()
        loss_matrix = choicer(grad_loss, w_t, w_)
        index = self.return_argmin_index(loss_matrix)
        # print('omni: {}'.format(index))
        X_t, y_t = X.iloc[index], y.iloc[index]

        return X_t, y_t
