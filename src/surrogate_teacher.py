from teacher import Teacher
import numpy as np
import pandas as pd
import theano
import theano.tensor as T


class Surrogate(Teacher):
    def __init__(self, min_w, alpha=0.01):
        """
        constructor

        Parameters
        ----------
        Teacher : base model class
        min_w : numpy
            true model parameter            
        alpha : float, optional
            learning rate, by default 0.01
        """
        super().__init__(min_w, alpha=alpha)

    def make_loss_function(self):
        """
        return loss function

        Returns
        -------
        theano.function
            inputs=[grad_loss,loss_t,loss__]
            outputs=[loss]
        """
        grad_loss_ = T.matrix(name='grad_loss')
        loss_t = T.vector(name='loss_t')
        loss__ = T.vector(name='loss__')
        first = (self.alpha**2)*(grad_loss_ ** 2).sum()
        second = -2*self.alpha*(loss_t-loss__)
        loss = first + second
        t = theano.function(
            inputs=[grad_loss_, loss_t, loss__],
            outputs=[loss],
        )
        return t

    def return_textbook(self, X, y, w_t, w_, drop=True):
        """
        return text book

        Parameters
        ----------
        X : pandas
            text book pool
        y : pandas
            goal
        w_t : numpy

        w_ : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        grad_loss = self.make_grad_loss_matrix(X, y, w_t)
        loss = self.loss_function()
        loss_t = np.array(loss(X, y, w_t)).flatten()
        loss__ = np.array(loss(X, y, w_)).flatten()

        choicer = self.make_loss_function()
        loss_matrix = choicer(grad_loss, loss_t, loss__)
        index = self.return_argmin_index(loss_matrix)
        # print('surr: {}'.format(index))
        X_t, y_t = X.iloc[index], y.iloc[index]
        if drop:
            self.drop_textbook(X, y, index)

        return X_t, y_t, index
