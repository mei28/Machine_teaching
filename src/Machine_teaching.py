# %%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import theano.tensor as T
import theano
from sklearn.metrics import roc_auc_score
from make_Y_w import estimate_w_star, estimate_min_w
import sys
sys.path.append('src')
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


def make_grad_loss_matrix(X, y, w):
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
    grad_loss = grad_loss_function()
    for i in range(X.shape[0]):
        grad_loss_ = grad_loss(X.iloc[i], y.iloc[i], w)
        a = np.array(grad_loss_).flatten()
        grad_loss_matrix[i] = a

    return grad_loss_matrix


# %%


def return_argmin_index(X):
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

# %%


def grad_loss_function():
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

# %%


def loss_function():
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

# %%


def omniscient_teacher(w_t, w_, X, y, eta=0.01):
    """
    return X_t, y_t by omniscient teacher

    Parameters
    ----------
    w_t : numpy vector
        worker's model parameter
    w_ : numpy vector 
        w*
    X : pandas matrix
        feature matrix
    y : pandas vector
        predict vector
    eta : float, optional
        w_t parameter, by default 0.01

    Returns
    -------
    return X_t, y_t, index
    """
    grad_loss_ = make_grad_loss_matrix(X, y, w_t)
    choicer = omniscient_teacher_function(eta)
    loss = choicer(grad_loss_, w_t, w_)
    index = return_argmin_index(loss)
    print("omni: {}".format(index))
    X_t = X.iloc[index]
    y_t = y.iloc[index]
    return X_t, y_t, index
# %%


def omniscient_teacher_function(eta=0.01):
    """
    return omniscient teacher function

    Parameters
    ----------
    eta : float, optional
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

    first = (eta ** 2) * (grad_loss ** 2).sum(axis=1)
    second = -2 * eta * (T.dot(grad_loss, w_t - w_))
    loss = first + second
    function = theano.function(
        inputs=[grad_loss, w_t, w_],
        outputs=[loss],
        on_unused_input='ignore'
    )
    return function

# %%


def surrogate_teacher(w_t, w_, X, y, eta=0.01):
    """
    return X_t,y_t by surrogate teacher

    Parameters
    ----------
    w_t : numpy vector
        worker's parameter
    w_ : numpy vector
        w*
    X : pandas matrix
        feature matrix
    y : pandas vector
        predict vector
    eta : float, optional
        w_t's parameter, by default 0.01

    Returns
    -------
    X_t, y_t, index
    """
    grad_loss_ = make_grad_loss_matrix(X, y, w_t)
    loss = loss_function()
    loss_t = np.array(loss(X, y, w_t)).flatten()
    loss__ = np.array(loss(X, y, w_)).flatten()

    choicer = surrogate_teacher_function(eta)
    loss = choicer(grad_loss_, loss_t, loss__)
    index = return_argmin_index(loss)

    X_t = X.iloc[index]
    y_t = y.iloc[index]
    print("surr: {}".format(index))
    return X_t, y_t, index

# %%


def surrogate_teacher_function(eta=0.01):
    """
    return surrogate teacher function

    Parameters
    ----------
    eta : float, optional
        w_t's parameter, by default 0.01

    Returns
    -------
    surrogate teacher function
        inputs = [gras_loss,loss_t,loss_*],
        outputs = [loss]
    """
    grad_loss_ = T.matrix(name='grad_loss')
    loss_t = T.vector(name='loss_t')
    loss__ = T.vector(name='loss__')
    first = (eta**2)*(grad_loss_ ** 2).sum()
    second = -2*eta*(loss_t-loss__)
    loss = first + second
    t = theano.function(
        inputs=[grad_loss_, loss_t, loss__],
        outputs=[loss],
    )
    return t

# %%


def model(lambd, w_init):
    """
    return model (to estimate w*)

    Parameters
    ----------
    lambd : float
        w's parameter
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
    loss = xent.mean() + lambd * (w ** 2).sum()/2

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


def student_model(lambd, w_init):
    """
    return student model
    updates by X_t,y_t

    Parameters
    ----------
    lambd : float
        w_t parameter    
     w_init : 
        model parameter

    Returns
    -------
        student
        inputs = [X,y]
        outputs = [loss,w]
    """
    X = T.vector(name='X')
    y = T.scalar(name='y')
    w = theano.shared(w_init, name='w')

    logit = T.nnet.sigmoid(T.dot(X, w))
    xent = T.nnet.binary_crossentropy(logit, y)
    loss = xent + lambd * (w ** 2).sum() / 2

    print('start: compile student model')
    params = [w]
    updates = SGD(params=params).updates(loss)

    student = theano.function(
        inputs=[X, y],
        outputs=[loss, w],
        updates=updates,
        on_unused_input='ignore'
    )

    print('complete: compile student model')

    return student
# %%


def predict(X, y, w):
    """
    return roc auc score

    Parameters
    ----------
    X : pandas matrix
        feature matrix
    y : pandas vector
        true predict
    w : numpy vector 
        model parameter
    """
    logit = np.dot(X, w)
    pred_y = T.nnet.sigmoid(logit).eval()
    # pred_y = [1 if i > 0.5 else 0 for i in pred_y]
    return(roc_auc_score(y, pred_y))
    # print(roc_auc_score(y, pred_y))
# %%


def main():
    # ワーカ数
    J = 10
    # df = pd.read_csv('output/weebil_vespula_with_cut.csv')
    df = pd.read_csv('output/weebil_vespula.csv')

    X = df.drop('Spe', axis=1)
    y = df['Spe']

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, shuffle=True
    )

    train_X.reset_index(drop=True, inplace=True)
    train_y.reset_index(drop=True, inplace=True)

    # np.random.seed(seed=28)
    lambd = 0.01
    eta = 0.01
    training_epochs = 10
    print("training epochs: {}".format(training_epochs))

    min_w = estimate_min_w(train_X, train_y, eta, training_epochs)

    w_star = estimate_w_star(
        train_X, min_w, eta, lambd, J, training_epochs)

    print('-' * 20)

    teach_epochs = 5
    w_init = np.random.normal(
        loc=0,
        scale=eta,
        size=X.shape[1]
    )

    w_t = w_init.copy()
    surrogate_student = student_model(lambd, w_t)
    surrogate_w = 0
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        X_t, y_t, index = surrogate_teacher(
            w_t, min_w, train_X_, train_y_, eta=0.01)
        loss, w = surrogate_student(X_t, y_t)
        # print('{}: loss: {}'.format(t, loss))
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        surrogate_w = w

    random_w = 0
    random_select = student_model(lambd, w_init)
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        index = np.random.randint(0, train_X_.shape[0])
        _, random_w = random_select(train_X_.iloc[index], train_y_.iloc[index])
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)

    w_t = w_init.copy()
    omni_student = student_model(lambd, w_t)
    omni_w = 99999
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        X_t, y_t, index = omniscient_teacher(
            w_t, min_w, train_X_, train_y_, eta=0.01)
        loss, w = omni_student(X_t, y_t)
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        omni_w = w

    print('teach epochs: {}'.format(teach_epochs))
    print('{}min_w{}'.format('-'*20, '-'*20))
    print('w_init: {}'.format(predict(test_X, test_y, w_init)))
    print('min_w: {}'.format(predict(test_X, test_y, min_w)))
    print('random_w: {}'.format(predict(test_X, test_y, random_w)))
    print('surrogate_w: {}'.format(predict(test_X, test_y, surrogate_w)))
    print('omni_w: {}'.format(predict(test_X, test_y, omni_w)))

    print('-'*20)
    teach_epochs = 5

    w_t = w_init.copy()
    surrogate_student = student_model(lambd, w_t)
    surrogate_w = 0
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        X_t, y_t, index = surrogate_teacher(
            w_t, w_star, train_X_, train_y_, eta=0.01)
        loss, w = surrogate_student(X_t, y_t)
        # print('{}: loss: {}'.format(t, loss))
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        surrogate_w = w

    random_w = 0
    random_select = student_model(lambd, w_init)
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        index = np.random.randint(0, train_X_.shape[0])
        _, random_w = random_select(train_X_.iloc[index], train_y_.iloc[index])
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)

    w_t = w_init.copy()
    omni_student = student_model(lambd, w_t)
    omni_w = 99999
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        X_t, y_t, index = omniscient_teacher(
            w_t, w_star, train_X_, train_y_, eta=0.01)
        loss, w = omni_student(X_t, y_t)
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        omni_w = w

    print('teach epochs: {}'.format(teach_epochs))
    print('{}w*{}'.format('-'*20, '-'*20))
    print('w_init: {}'.format(predict(test_X, test_y, w_init)))
    print('w*: {}'.format(predict(test_X, test_y, w_star)))
    print('random_w: {}'.format(predict(test_X, test_y, random_w)))
    print('surrogate_w: {}'.format(predict(test_X, test_y, surrogate_w)))
    print('omni_w: {}'.format(predict(test_X, test_y, omni_w)))

    print('-'*20)
    teach_epochs = 5

    w_t = w_init.copy()
    surrogate_student = student_model(lambd, w_t)
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    surrogate_w = 0
    for t in range(teach_epochs):
        X_t, y_t, index = surrogate_teacher(
            w_t, w_init, train_X_, train_y_, eta=0.01)
        loss, w = surrogate_student(X_t, y_t)
        # print('{}: loss: {}'.format(t, loss))
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        surrogate_w = w

    random_w = 0
    random_select = student_model(lambd, w_init)
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        index = np.random.randint(0, train_X_.shape[0])
        _, random_w = random_select(train_X_.iloc[index], train_y_.iloc[index])
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)

    w_t = w_init.copy()
    omni_student = student_model(lambd, w_t)
    omni_w = 99999
    train_X_ = train_X.copy(deep=True)
    train_y_ = train_y.copy(deep=True)
    for t in range(teach_epochs):
        X_t, y_t, index = omniscient_teacher(
            w_t, w_init, train_X_, train_y_, eta=0.01)
        loss, w = omni_student(X_t, y_t)
        train_X_.drop(index, inplace=True)
        train_y_.drop(index, inplace=True)
        omni_w = w

    print('teach epochs: {}'.format(teach_epochs))
    print('{}w_init{}'.format('-'*20, '-'*20))
    print('w_init: {}'.format(predict(test_X, test_y, w_init)))
    print('random_w: {}'.format(predict(test_X, test_y, random_w)))
    print('surrogate_w: {}'.format(predict(test_X, test_y, surrogate_w)))
    print('omni_w: {}'.format(predict(test_X, test_y, omni_w)))


main()
# %%

if __name__ == "__main__":
    main()
