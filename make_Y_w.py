# %%
import sys
sys.path.append('src')

from Machine_teaching import predict
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import theano.tensor as T
import theano
import pandas as pd
import numpy as np
import math

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
    J, D = W.shape
    N = X.shape[0]
    Y = np.zeros((N * J))
    for j in range(J):
        for n in range(N):
            logit = np.dot(W[j], X.iloc[n])
            p_1 = 1/(1 + math.exp(-logit))
            Y[J * j + n] = np.random.choice(2, p=[1-p_1,p_1])
    return Y
# %%


def makeW_w0(eta, lambd, J, X):
    N, D = X.shape
    w_0 = np.random.normal(
        loc=0,
        scale=eta,
        size=D
    )
    W = np.zeros((J, D))

    for j in range(J):
        W[j] = np.random.normal(
            loc=w_0,
            scale=lambd,
            size=w_0.shape
        )
    return W, w_0
# %%


def duplicate_W(W, N):
    J, D = W.shape
    W_ = np.zeros((N * J, D))
    for j in range(J):
        tmp = np.tile(W[j], (N, 1))
        W_[j * N: (j + 1) * N] = tmp

    return W_
# %%


def estimate_w_model(w_, W, eta, lambd):
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
    N, D = X.shape
    X_ = np.zeros((N * J, D))
    for i in range(N * J):
        X_[i, :] = X.iloc[i % N]
    return X_
# %%


def estimate_w(X, y, w_, W, eta, lambd, epochs=10):
    J, D = W.shape
    N = X.shape[0]

    W = duplicate_W(W, N)
    X = remake_X(X, J)

    model = estimate_w_model(w_=w_, W=W, eta=eta, lambd=lambd)

    min_w = 9999999
    min_loss = 999999999

    for t in range(epochs):
        loss, w_0, W_ = model(X, y)
        # print('{}: loss:{}'.format(t, loss))

        if loss < min_loss:
            min_w = w_0
            min_loss = loss
        else:
            break
    return min_w
# %%


def main():

    J = 10
    df = pd.read_csv('output/weebil_vespula.csv')
    X = df.drop('Spe', axis=1)
    y = df['Spe']

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True
    )
    N, D = train_X.shape

    np.random.seed(11)
    # Wの生成
    eta = 0.01
    lambd = 0.01

    W, w_0 = makeW_w0(eta, lambd, J, X)
    train_Y = makeY(W, train_X)

    min_w = estimate_w(train_X, train_Y, w_0, W, eta, lambd, epochs=99)
    print(min_w)

    print('-' * 20)
    print('w_init: {}'.format(predict(test_X, test_y, w_0)))

    print('w_*: {}'.format(predict(test_X, test_y, min_w)))


main()
# %%
if __name__ == "__main__":
    main()

