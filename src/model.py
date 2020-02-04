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
    def __init__(self, w_init, W, eta, lambd):
        """
        constructor

        Parameters
        ----------
        w_init : numpy
            w_star
        W : numpy
            W_star, worker's model parameter matrix
        eta : float
            w_star's parameter
        lambd : float
            W_star's parameter
        """
        self.w_star = w_init
        self.W = W
        self.eta = eta
        self.lambd = lambd

    def model(self):
        X = T.matrix(name="X")
        y = T.vector(name="y")
        w_star = theano.shared(self.w_star, name="w_0")
        W = theano.shared(self.W, name='W')

        first = self.lambd * ((W-w_star)**2).sum()/2
        second = self.eta * (w_star ** 2).sum()/2

        logit = T.flatten(T.dot(W, X.T))
        p_1 = T.nnet.nnet.sigmoid(logit)
        xent = T.nnet.nnet.binary_crossentropy(p_1, y)
        third = xent.sum()

        loss = first + second + third
        params = [w_star, W]
        updates = SGD(params=params).updates(loss)

        # print('start: compile model')
        train = theano.function(
            inputs=[X, y],
            outputs=[loss, w_star, W],
            updates=updates,
            on_unused_input='ignore'
        )
        # print('end: compile model')

        return train

    def model_w_star(self):
        """
        make a model to estimate w_star 

        Returns
        -------
        inputs = [X,y,W]
        outputs = [loss,w_star]
        """
        X = T.matrix(name="X")
        y = T.vector(name="y")
        w_star = theano.shared(self.w_star, name="w_0")
        W = T.matrix(name='W')

        first = self.lambd * ((W-w_star)**2).sum()/2
        second = self.eta * (w_star ** 2).sum()/2

        logit = T.flatten(T.dot(W, X.T))
        p_1 = T.nnet.nnet.sigmoid(logit)
        xent = T.nnet.nnet.binary_crossentropy(p_1, y)
        third = xent.sum()

        loss = first + second + third
        params = [w_star]
        updates = SGD(params=params).updates(loss)

        # print('start: compile model')
        train = theano.function(
            inputs=[X, y, W],
            outputs=[loss, w_star],
            updates=updates,
            on_unused_input='ignore'
        )
        # print('end: compile model')

        return train

    def model_W_star(self, w_j):
        """
        make a model to estimate W_star(w_j)

        Parameters
        ----------
        w_j : numpy
            worker's model parameter

        Returns
        -------
        inputs =[X,y,w_star]
        outputs=[loss,w_j]
        """
        X = T.matrix(name="X")
        y = T.vector(name="y")
        w_star = T.vector(name='w_star')
        w_j = theano.shared(w_j, name="w_j")

        first = self.lambd * ((w_j-w_star)**2).sum()/2
        second = self.eta * (w_star ** 2).sum()/2

        logit = T.dot(X, w_j)
        p_1 = T.nnet.nnet.sigmoid(logit)
        xent = T.nnet.nnet.binary_crossentropy(p_1, y)
        third = xent.sum()

        loss = first + second + third
        params = [w_j]
        updates = SGD(params=params).updates(loss)

        # print('start: compile model')
        train = theano.function(
            inputs=[X, y, w_star],
            outputs=[loss, w_j],
            updates=updates,
            on_unused_input='ignore'
        )
        # print('end: compile model')

        return train

    def learn(self, X, Y, training_epochs=10):
        train = self.model()
        for i in range(training_epochs):
            loss, w_star, W = train(X, Y)

            # if i % (training_epochs / 100) == 0:
            #     print('{}: loss:{}'.format(i, loss))

        self.w_star = w_star
        self.W = W
        return w_star, W

    def learn_w_star(self, X, Y, training_epochs=10):
        """
        estimate w_star

        Parameters
        ----------
        X : pandas
            text book pool, shape=(N,D)
        Y : numpy
            worker's answers, shape=(J*N)
        training_epochs : int, optional
            training_epochs, by default 10

        Returns
        -------
        estimated w_star
        """
        train = self.model_w_star()
        for i in range(training_epochs):
            loss, w_star = train(X, Y, self.W)

            # if i % (training_epochs / 100) == 0:
            #     print('w{}: loss:{}'.format(i, loss))

        self.w_star = w_star
        return w_star

    def learn_W_star(self, X, Y, training_epochs=10):
        """
        estimate W_star

        Parameters
        ----------
        X : pandas
            text book pool, shape=(N*D)
        Y : numpy
            worker's answers, shape=(J,N)
        training_epochs : int, optional
            training epochs, by default 10

        Returns
        -------
        estimated W_star
            [description]
        """
        J, D = self.W.shape
        N = X.shape[0]
        W = np.zeros_like(self.W)
        y = np.zeros(shape=N)
        for j in range(J):
            y = Y[N*j: N*(j+1)]
            w_j = W[j, :]
            train = self.model_W_star(w_j)
            for i in range(training_epochs):
                loss, w_j_new = train(X, y, self.w_star)
                # if i % (training_epochs / 100) == 0:
                # print('w_j{}: loss:{}'.format(i, loss))
            W[j, :] = w_j_new

        self.W = W
        return W


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
