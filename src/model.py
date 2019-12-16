from collections import OrderedDict
import theano
import theano.tensor as T
import pandas as pd
import numpy as np
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"


class Model():
    """
    base learning model
    """

    def __init__(self, w_init):
        """
        constructor

        Parameters
        ----------
        w_init : numpy
            leaner's parameter
        """
        self.w = w_init

    def make_loss_function(self):
        """
        define loss function
        """
        pass

    def learn(self, X, y):
        """
        learn and update w

        Parameters
        ----------
        X : pandas
            feature matrix
        y : pandas
            goal feature
        """
        pass

    def response(self, X):
        """return predict by w

        Parameters
        ----------
        X : pandas
            feature matrix
        """
        pass

    def make_model(self):
        """
        to make learn function
        """
        pass


class W_star_model():
    def __init__(self, w_init, W_, eta, lambd):
        self.w_star = w_init
        self.W_ = W_
        self.eta = eta
        self.lambd = lambd

    def model(self):
        X = T.matrix(name="X")
        y = T.vector(name="y")
        w_star = theano.shared(self.w_star, name="w_0")
        W_ = theano.shared(self.W_, name="W_")
        lambd = self.lambd
        eta = self.eta

        first = lambd * ((W_-w_star)**2).sum()/2
        second = eta * (w_star ** 2).sum()/2

        p_1 = T.nnet.nnet.sigmoid((W_*X).sum(axis=1))
        xent = T.nnet.nnet.binary_crossentropy(p_1, y)
        third = xent.sum()

        loss = first + second + third
        params = [w_star, W_]
        updates = SGD(params=params).updates(loss)

        print('start: compile model')
        train = theano.function(
            inputs=[X, y],
            outputs=[loss, w_star, W_],
            updates=updates,
            on_unused_input='ignore'
        )
        print('end: compile model')

        return train

    def learn(self, X, Y, training_epochs=10):
        train = self.model()
        for i in range(training_epochs):
            loss, w_star, W_ = train(X, Y)
            if i % (training_epochs / 100) == 1:
                print('{}: loss:{}'.format(i, loss))
        return w_star, W_


class Logistic_model(Model):
    def __init__(self, w_init, lambd):
        """
        logistic learning model

        Parameters
        ----------
        Model : base class
        w_init : model parameter
        lambd : float
            learning rate
        """
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
