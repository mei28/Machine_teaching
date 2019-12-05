from collections import OrderedDict
import theano
import theano.tensor as T
import pandas as pd
import numpy as np


class Model():
    def __init__(self, w_init):
        self.w = w_init

    def learn(self, X, y):
        pass

    def response(self, X):
        pass

    def make_model(self):
        pass


class Logistic_model(Model):
    def __init__(self, w_init, lambd):
        super().__init__(w_init)
        self.lambd = lambd

    def make_loss_function(self):
        """return logistic model
        Returns
        -------
            inputs=[X, y],
            outputs=[loss, w],
            updates=updates,
        """
        X = T.matrix(name="X")
        y = T.vector(name="y")
        w = theano.shared(self.w, name="w")

        logit = T.nnet.sigmoid(T.dot(X, w))
        xent = T.nnet.binary_crossentropy(logit, y)
        loss = xent.mean() + self.lambd * (w ** 2).sum()/2

        params = [w]
        updates = SGD(params=params).updates(loss)

        print('start: compile model')

        train = theano.function(
            inputs=[X, y],
            outputs=[loss, w],
            updates=updates,
            on_unused_input='ignore'
        )

        print('complete: compile model')

        return train

    def learn(self, X, y, training_epochs=10):
        """to train

        Parameters
        ----------
        X : pandas 
            feature matrix, shape = (N,D)
        y : pandas
            goal vectore shape = (N,)
        training_epochs : int, optional
            the number of training epochs, by default 10
        """
        model = self.make_loss_function()
        print('start: learning')
        for i in range(training_epochs):
            loss, self.w = model(X, y)
        print('end: learning')

    def response(self, X):
        logit = np.dot(X, self.w)
        pred_y = T.nnet.sigmoid(logit).eval()
        return pred_y


class Optimizer(object):
    def __init__(self, params=None):
        if params is None:
            return NotImplementedError()
        self.params = params

    def updates(self, loss=None):
        if loss is None:
            return NotImplementedError()

        self.updates = OrderedDict()
        self.gparams = [T.grad(loss, param) for param in self.params]


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = 0.01

    def updates(self, loss=None):
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates
