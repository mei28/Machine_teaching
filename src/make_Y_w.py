

# %%

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import theano.tensor as T
import theano
import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
# %%


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
# %%


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = 0.01

    def updates(self, loss=None):
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates


# %%


def makeY(W, X):
    """ make worker’s answer

    Parameters
    ----------
    W : numpy
        shape = (J,D)
    X : pandas
        shape = (N,D)

    Returns
    -------
    numpy
        shape = (N*J,)
    """
    J, D = W.shape
    N = X.shape[0]
    Y = np.zeros((N * J))
    for j in range(J):
        for n in range(N):
            logit = np.dot(W[j, :], X.iloc[n])
            p_1 = 1/(1+np.exp(-logit))
            Y[N * j + n] = np.random.choice(2, p=[1 - p_1, p_1])
    return Y
# %%


def makeW_w_init(eta, lambd, J, D):
    """make w_init and W

    Parameters
    ----------
    eta : float
        w_init's parameter
    lambd : float
            W's parameter
    J : int
        the number of workers
    D : int
        feature dimensions

    Returns
    -------
    w_init
        shape = (D,)
    W
        shape = (J,D)
    """

    w_init = np.random.normal(
        loc=0,
        scale=eta,
        size=D
    )
    W = np.zeros((J, D))

    for j in range(J):
        W[j] = np.random.normal(
            loc=w_init,
            scale=lambd,
            size=w_init.shape
        )
        # W[j] = w_init + 0.01 * np.random.randn(D)
    return W, w_init
# %%


def duplicate_W(W, N):
    """
    duplicate W (J,D) -> (J*N,D)

    Parameters
    ----------
    W : numpy
        shape = (J,D)
    N : int
        The number of questions

    Returns
    -------
    W_
        shape = (J*N,D)
    """
    J, D = W.shape
    W_ = np.zeros((N * J, D))
    for j in range(J):
        tmp = np.tile(W[j], (N, 1))
        W_[j * N: (j + 1) * N] = tmp

    return W_
# %%


def estimate_w_model(w_, W, eta, lambd):
    """
    return model which is to estimate w* from worker's answer

    Parameters
    ----------
    w_ : numpy vector
         w*
    W : numpy matrix
        worker's model parameters
    eta : float
          w_init parameter
    lambd : float
          W parameter

    Returns
    -------
    model
        model which is to estimate w*
    """
    X = T.matrix(name='X')
    y = T.vector(name='y')
    w_0 = theano.shared(w_, name='w_0')
    W_ = theano.shared(W, name='W')

    first = lambd * ((W_ - w_0) ** 2).sum() / 2
    second = eta * (w_0 ** 2).sum() / 2

    p_1 = T.nnet.nnet.sigmoid((W_*X).sum(axis=1))
    xent = T.nnet.nnet.binary_crossentropy(p_1, y)
    third = xent.mean()

    loss = first + second + third
    params = [w_0, W_]
    updates = SGD(params=params).updates(loss)

    print('start: compile estimate w* model')
    model = theano.function(
        inputs=[X, y],
        outputs=[loss, w_0, W_],
        updates=updates,
        on_unused_input='ignore'

    )
    print('end: compile estimate w* moddel')
    return model
# %%


def remake_X(X, J):
    """remake X (N,D) -> (N*J,D)

    Parameters
    ----------
    X : pandas
        shape = (N,D), feature matrix
    J : int
        The number of workers

    Returns
    -------
    X_
        shape = (N*J,D)
    """
    N, D = X.shape
    X_ = np.zeros((N * J, D))
    for i in range(N * J):
        X_[i, :] = X.iloc[i % N]
    return X_
# %%


def estimate_w(X, eta=0.01, lambd=0.01, J=10, epochs=10):
    """ return estimated w*

    Parameters
    ----------
    X : pandas
        shape = (N*J,D)
    eta : float, optional
        w_init's parameter, by default 0.01
    lambd : float, optional
        W's parameter, by default 0.01
    J : int, optional
        The number of workers, by default 10
    epochs : int, optional
        The number of training epochs, by default 10

    Returns
    -------
    min_w
        estimated w*
    w_init
        the first w
    """
    N, D = X.shape
    W, w_init = makeW_w_init(eta, lambd, J, D=X.shape[1])
    Y = makeY(W, X)
    W = duplicate_W(W, N)
    X = remake_X(X, J)

    model = estimate_w_model(w_=w_init, W=W, eta=eta, lambd=lambd)

    min_w = 9999999
    min_loss = 999999999

    for t in range(epochs):
        loss, w_0, W_ = model(X, Y)
        # print('{}: loss:{}'.format(t, loss))

        if loss < min_loss:
            min_w = w_0
            min_loss = loss
        else:
            break
    return min_w, w_init
# %%


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))
# %%


def main():

    J = 10
    df = pd.read_csv('output/weebil_vespula.csv')
    X = df.drop('Spe', axis=1)
    # ms = MinMaxScaler()
    # X = ms.fit_transform(X)
    # X = pd.DataFrame(X)
    y = df['Spe']

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True
    )

    # Wの生成
    eta = 0.01
    lambd = 0.01

    min_w, w_init = estimate_w(X, eta, lambd, J=J, epochs=9999)

    print('-' * 20)
    print(w_init)
    print('w_init: {}'.format(predict(test_X, test_y, w_init)))
    print(min_w)
    print('w_*_train: {}'.format(predict(train_X, train_y, min_w)))
    print('w_*_test: {}'.format(predict(test_X, test_y, min_w)))


# main()
# %%
if __name__ == "__main__":
    main()
