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


class W_star_model(Model):
    def __init__(self, w_init, eta, lambd, W):
        """
        constructor

        Parameters
        ----------
        Model : base 
            base class
        w_init : numpy
            model parameter
        eta : float
            w_star's parameter
        lambd : float
            W's parameter
        W_ : numpy
            W parameters
        """
        super().__init__(w_init)
        self.eta = eta
        self.lambd = lambd
        self.W = W.copy()

    def make_w_star_loss_function(self):
        """
        make w_star loss function

        Returns
        -------
        theano.function
        inputs=[X,y]
        outputs=[loss,w_star]
        """
        X = T.matrix(name='X')
        y = T.vector(name='y')
        w_star = theano.shared(self.w, name='w_0')
        W_ = T.matrix(name='W_')

        first = self.lambd * ((W_ - w_star) ** 2).sum() / 2
        second = self.eta * (w_star ** 2).sum() / 2

        p_1 = T.nnet.nnet.sigmoid((W_*X).sum(axis=1))
        xent = T.nnet.nnet.binary_crossentropy(p_1, y)
        third = xent.mean()

        loss = first + second + third
        params = [w_star]
        updates = SGD(params=params).updates(loss)

        # print('start: compile estimate w* model')
        model = theano.function(
            inputs=[X, y, W_],
            outputs=[loss, w_star],
            updates=updates,
            on_unused_input='ignore'
        )
        # print('end: compile estimate w* moddel')
        return model

    def make_wj_loss_function(self, w_j):
        """
        make w_j loss function for updating W

        Parameters
        ----------
        w_j : numpy
            shape=(D,) worker:J's model parameter

        Returns
        -------
        theano.function
            inputs=[X,y,w_star]
            outputs=[loss,w_j]
        """
        X = T.matrix(name='X')
        y = T.vector(name='y')
        w_star = T.vector(name='w_star')
        w_j = theano.shared(w_j, name='w_j')

        first = self.lambd * ((w_j - w_star) ** 2).sum() / 2
        second = self.eta * (w_star ** 2).sum() / 2

        p_1 = T.nnet.sigmoid(T.dot(X, w_j))
        xent = T.nnet.binary_crossentropy(p_1, y)
        third = xent.mean()

        loss = first + second + third

        params = [w_j]
        updates = SGD(params=params).updates(loss)
        function = theano.function(
            inputs=[X, y, w_star],
            outputs=[loss, w_j],
            updates=updates,
            on_unused_input='ignore'
        )
        return function

    def learn_w_star(self, X, y, W_, training_epochs=10):
        """
        learn w_star and update

        Parameters
        ----------
        X : numpy
            shape = (J*N,D)
        y : numpy
            shape = (J*N,)
        W_ : numpy
            shape = (J*N,D)
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------
        self.w
            it means w_star
        """
        model = self.make_w_star_loss_function()
        # print('start: learning')
        for i in range(training_epochs):
            loss, self.w = model(X, y, W_)
        # print('end: learning')
        return self.w

    def learn_W(self, X_, Y, training_epochs=10):
        """
        learn and update W

        Parameters
        ----------
        X_ : numpy
            shape = (J*N,D)
        Y : numpy
            shape = (J*N,)
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------
        self.W
        """
        J = self.W.shape[0]
        N_, D = X_.shape
        N = N_ // J
        # print('start: learning')
        for j in range(J):
            w_j = self.W[j, :]
            model = self.make_wj_loss_function(w_j)
            y = Y[N * j: N * (j + 1)]
            X = X_[N * j: N * (j + 1)]
            for i in range(training_epochs):
                loss, w_j_new = model(X, y, self.w)

            self.W[j, :] = np.array(w_j_new).flatten()

        # print('end: learning')
        return self.W

    def response(self, X):
        """
        predict by logistic

        Parameters
        ----------
        X : pandas

        Returns
        -------
        return predict
        """
        logit = np.dot(X, self.w)
        pred_y = T.nnet.sigmoid(logit).eval()
        return pred_y


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
