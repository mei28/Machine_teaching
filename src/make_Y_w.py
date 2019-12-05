

# %%

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import theano.tensor as T
import theano
import pandas as pd
import numpy as np
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


def makeW(min_w, J, lambd=0.01):
    """
    make W 

    Parameters
    ----------
    min_w : numpy
        shape = (D,)
    lambd : float, optional
        W's parameter, by default 0.01

    Returns
    -------
    return W 
        shape = (J,D) worker's model parameter
    """
    D = min_w.shape[0]
    W = np.zeros((J, D))

    for j in range(J):
        W[j] = np.random.normal(
            loc=min_w,
            scale=lambd,
            size=min_w.shape
        )
        # W[j] = w_init + 0.01 * np.random.randn(D)
    return W
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
        tmp = W[j].copy()
        for n in range(N):
            W_[N*j+n] = tmp
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


def estimate_w_star(X, min_w, eta=0.01, lambd=0.01, J=10, training_epochs=10):
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
    w_star
        shape = (D,)
    """
    N, D = X.shape
    W = makeW(min_w, J, lambd)
    Y = makeY(W, X)
    W = duplicate_W(W, N)
    X = remake_X(X, J)

    model = estimate_w_model(w_=min_w, W=W, eta=eta, lambd=lambd)

    w_star = 9999999
    min_loss = 999999999

    for t in range(training_epochs):
        loss, w_0, W_ = model(X, Y)
        # print('{}: loss:{}'.format(t, loss))

        if loss < min_loss:
            w_star = w_0
            min_loss = loss
        else:
            break
    return w_star
# %%


def predict(X, y, w):
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))
# %%


def estimate_min_w(X, y, eta, training_epochs=10):
    """return min_w

    Parameters
    ----------
    X : pandas
        shape = (N,D)
    y : pandas
        shape = (N,)
    eta : int
        w's parameter
    training_epochs : int, optional
        training epochs, by default 10

    Returns
    -------
    min_w
        shape = (D,)
    """
    w_init = np.random.normal(
        loc=0,
        scale=eta,
        size=X.shape[1]
    )
    train = model(eta, w_init)

    min_w = 9999
    for t in range(training_epochs):
        loss, w = train(X, y)
        # print('{}: loss: {}'.format(t, loss))
        min_w = w

    return min_w
# %%


def model(eta, w_init):
    """
    return model (to estimate min_w)

    Parameters
    ----------
    eta : float
        w_init's parameter
    w_init : numpy vector
        model parameter

    Returns
    -------
    model 
        inputs = [X,y]
        outputs = [loss,w]
    """

    X = T.matrix(name="X")
    y = T.vector(name="y")
    w = theano.shared(w_init, name="w")

    logit = T.nnet.sigmoid(T.dot(X, w))
    xent = T.nnet.binary_crossentropy(logit, y)
    loss = xent.mean() + eta * (w ** 2).sum()/2

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
# %%


def main():

    J = 1
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
    epochs = 50
    min_w = estimate_min_w(train_X, train_y, eta, training_epochs=epochs)
    epochs = 500
    w_star = estimate_w_star(X, min_w, eta, lambd, J=J, training_epochs=epochs)

    print('-' * 20)
    print(min_w)
    print('min_w: {}'.format(predict(test_X, test_y, min_w)))
    print(w_star)
    print('w_*_train: {}'.format(predict(train_X, train_y, w_star)))
    print('w_*_test: {}'.format(predict(test_X, test_y, w_star)))


if __name__ == "__main__":
    main()
