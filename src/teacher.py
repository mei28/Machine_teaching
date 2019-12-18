import numpy as np
import pandas as pd
import theano
import theano.tensor as T


class Teacher():
    def __init__(self, min_w, alpha=0.01):
        """
        teacher base class

        Parameters
        ----------
        min_w : numpy
            true model parameter
        alpha : float, optional
            learning rate, by default 0.01
        """
        super().__init__()
        self.min_w = min_w.copy()
        self.alpha = alpha

    def make_grad_loss_matrix(self, X, y, w):
        """return grad loss matrix

        Parameters
        ----------
        X : pandas matrix
            feature matrix
        y : pandas vector
            predict
        w : numpy 
            model parameter    
        Returns
        -------
        numpy matrix
            grad loss matrix
        """
        grad_loss_matrix = np.zeros((X.shape[0], w.shape[0]))
        grad_loss = self.grad_loss_function()
        for i in range(X.shape[0]):
            grad_loss_ = grad_loss(X.iloc[i], y.iloc[i], w)
            a = np.array(grad_loss_).flatten()
            grad_loss_matrix[i] = a

        return grad_loss_matrix

    def grad_loss_function(self):
        """
        return grad loss function

        Returns
        -------
        grad loss function
        inputs = [X,y,w]
        outputs = [grad_loss]
        """
        X = T.vector(name='X')
        y = T.scalar(name='y')
        w = T.vector(name='w')

        logit = y * T.dot(X, w)
        loss = T.log(1+T.exp(-logit)).mean()
        grad_loss = T.grad(loss, w)

        grad_loss_ = theano.function(
            inputs=[X, y, w],
            outputs=[grad_loss],
            on_unused_input='ignore'
        )

        return grad_loss_

    def loss_function(self):
        """
        return loss function 

        Returns
        -------
        return loss function
        inputs = [X,y,w]
        outputs = [loss]
        """
        X = T.matrix(name='X')
        y = T.vector(name='y')
        w = T.vector(name='w')

        logit = y * T.dot(X, w)
        loss = T.log(1+T.exp(-logit))
        loss_ = theano.function(
            inputs=[X, y, w],
            outputs=[loss],
            on_unused_input='ignore'
        )
        return loss_

    def return_argmin_index(self, X):
        """
        return one argmin index

        Parameters
        ----------
        X : list
            loss list

        Returns
        -------
        index
            return one argmin index with random
        """
        X = np.array(X).flatten()
        index_list = np.where(X == np.min(X))[0]
        # if len(index_list) > 1:
        #     print('random choiced')
        index = np.random.choice(index_list)
        return index

    def update_w_j(self, X_t, y_t, w_j):
        grad_loss = self.grad_loss_function()
        grad_loss_ = grad_loss(X_t, y_t, w_j)[0]

        w_j = w_j - self.alpha * grad_loss_
        return w_j
